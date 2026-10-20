#!/usr/bin/env python3
"""The obstacle monitor: keeps the obstacle grid up to date from the LiDARs and answers
CheckMotion requests from the trajectory server.

    ros2 run stretch_obstacle_grid obstacle_monitor_node.py [--ros-args -p publish_labels:=true]

Inputs: /lidar_points_left and /lidar_points_right (Hesai PointCloud2), /joint_states,
/wheel_odom (for telling static obstacles from moving ones while the base turns), and TF for
the LiDAR mounts.

Outputs:
  check_motion (stretch_motion_planning_interfaces/CheckMotion)
  obstacle_grid/timing (diagnostic_msgs/DiagnosticArray, every report_period s): how long each
      step takes, from the end of a LiDAR sweep to the grid being ready, and per check
  obstacle_grid/{occupied,static,shadow,contaminated} with publish_labels, as grid_viz_node.py
      does, and obstacle_grid/robot_shapes (MarkerArray): the shapes the grid takes as the robot
      (for RViz; costs a few ms per frame)

Every frame from either LiDAR (each sweeps at 10 Hz, about 35 ms apart) goes into the grid as
it arrives. The labels and tables are rebuilt after every frame, or with
update_every_frame:=false once every LiDAR has sent a new frame (about half the work, but the
first LiDAR's frame waits for the second's). Each LiDAR's subscription keeps only its newest
frame, so frames that arrive while the grid is busy are dropped (and counted) rather than
queued behind it.

Parameters:
  latency (0.15): s from a check's answer to the robot starting to slow (the braking test
      measured under 0.14 s from the desktop, network included). The age of the LiDAR data
      is measured per check and added on top.
  obstacle_speed (1.5): m/s a moving obstacle is assumed to approach at.
  stale_after (0.5): s; a check fails if the oldest frame behind the grid is older than this
      (a LiDAR stopped), so the server treats everything as unknown.
  contamination_reach (0.2), static_after (3.0): see ObstacleGridParams.
  update_every_frame (true): see above.
  threads (0): OpenMP threads for the grid's parallel loops, 0 for every core. Idle threads
      sleep (OMP_WAIT_POLICY=passive), so they don't take CPU from the drivers between loops.
  publish_labels (false), report_period (5.0)
  ring_elevations_file (""): a LiDAR calibration CSV (Elevation column, deg, one row per ring)
      to take ring elevations from instead of measuring them from the first frames. For the
      MuJoCo sim, whose upward rings return nothing in an open room:
      stretch4_mujoco/models/stretch_4/lidar_calibration.csv.
  azimuth_bins (720): range image columns per turn. The real Hesai fires 901 times a turn;
      the sim traces 250, so use 250 there (empty columns read as unseen space).
  calibration_file (""): a stretch_calibration_values.yaml with LiDAR joint origins to use
      instead of TF's mounts, for bags recorded before the robot was calibrated.
"""

import collections
import os
import threading
import time

# OpenMP's worker threads spin between parallel loops by default, taking CPU the LiDAR driver
# needs. Must be set before the grid module loads.
os.environ.setdefault("OMP_WAIT_POLICY", "passive")

import numpy as np
import rclpy
import tf2_ros
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from nav_msgs.msg import Odometry
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import JointState, PointCloud2
from std_msgs.msg import ColorRGBA, Header
from visualization_msgs.msg import Marker, MarkerArray

import stretch_obstacle_grid as og
from grid_setup import (GRID_FRAME, LIDARS, RobotShapes, cloud_fields, joint_vector, load_calibration,
                        load_ring_elevations, make_grid, odom_pose, step_braking, transform_matrix)
from grid_viz_node import PUBLISHED_LABELS, xyz_cloud
from stretch_motion_planning.collision_model import JOINT_NAMES
from stretch_motion_planning_interfaces.srv import CheckMotion

FRAME_DURATION = 0.1  # s per LiDAR sweep
# Only the newest frame: an old one is worth less than the time it takes.
NEWEST_FRAME_ONLY = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
DEBUG_LABELS = ("occupied", "static", "shadow", "contaminated")
ADD_FRAME_STEPS = ("bin_points", "nearest", "clusters", "classify", "veil", "empty_rays")


def stamp_seconds(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def shape_markers(header, shapes):
    """The grid's robot shapes as translucent markers."""
    markers = MarkerArray()
    kinds = {"box": Marker.CUBE, "cylinder": Marker.CYLINDER, "capsule": Marker.CYLINDER, "sphere": Marker.SPHERE}
    for i, shape in enumerate(shapes):
        pose = np.asarray(shape.pose, dtype=float)
        d = np.asarray(shape.dims, dtype=float)
        m = Marker(header=header, ns="robot_shapes", id=i, type=kinds[shape.kind], action=Marker.ADD,
                   color=ColorRGBA(r=0.2, g=0.6, b=1.0, a=0.35))
        m.pose.position.x, m.pose.position.y, m.pose.position.z = (float(v) for v in pose[:3, 3])
        q = rotation_to_quaternion(pose[:3, :3])
        m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z, m.pose.orientation.w = q
        if shape.kind == "box":
            m.scale.x, m.scale.y, m.scale.z = (float(v) for v in 2 * d)
        elif shape.kind == "sphere":
            m.scale.x = m.scale.y = m.scale.z = float(2 * d[0])
        else:  # cylinder; a capsule as its bounding cylinder
            m.scale.x = m.scale.y = float(2 * d[0])
            m.scale.z = float(2 * d[1] + (2 * d[0] if shape.kind == "capsule" else 0.0))
        markers.markers.append(m)
    return markers


def rotation_to_quaternion(r):
    """(x, y, z, w) of a rotation matrix."""
    w = np.sqrt(max(0.0, 1.0 + r[0, 0] + r[1, 1] + r[2, 2])) / 2
    x = np.copysign(np.sqrt(max(0.0, 1.0 + r[0, 0] - r[1, 1] - r[2, 2])) / 2, r[2, 1] - r[1, 2])
    y = np.copysign(np.sqrt(max(0.0, 1.0 - r[0, 0] + r[1, 1] - r[2, 2])) / 2, r[0, 2] - r[2, 0])
    z = np.copysign(np.sqrt(max(0.0, 1.0 - r[0, 0] - r[1, 1] + r[2, 2])) / 2, r[1, 0] - r[0, 1])
    return float(x), float(y), float(z), float(w)


class Stats:
    """Recent samples of named timings, summarized for the timing report."""

    def __init__(self):
        self.samples = collections.defaultdict(lambda: collections.deque(maxlen=500))
        self.counts = collections.Counter()
        self.lock = threading.Lock()  # added to from the cloud and service threads

    def add(self, **values):
        with self.lock:
            for name, value in values.items():
                self.samples[name].append(value)

    def count(self, name, n=1):
        with self.lock:
            self.counts[name] += n

    def summary(self):
        with self.lock:
            samples = {name: np.array(values) for name, values in self.samples.items()}
            counts = dict(self.counts)
        out = {}
        for name, values in samples.items():
            v = values * 1e3
            out[name] = f"median {np.median(v):.1f} ms, p95 {np.percentile(v, 95):.1f}, max {v.max():.1f} (n={len(v)})"
        out.update({name: str(count) for name, count in counts.items()})
        return out


class ObstacleMonitor(Node):
    def __init__(self):
        super().__init__("obstacle_monitor")
        self.reach = self.declare_parameter("contamination_reach", 0.2).value
        self.static_after = self.declare_parameter("static_after", 3.0).value
        self.threads = self.declare_parameter("threads", 0).value
        self.update_every_frame = self.declare_parameter("update_every_frame", True).value
        self.stale_after = self.declare_parameter("stale_after", 0.5).value
        self.publish_labels = self.declare_parameter("publish_labels", False).value
        report_period = self.declare_parameter("report_period", 5.0).value
        calibration_file = self.declare_parameter("calibration_file", "").value
        ring_file = self.declare_parameter("ring_elevations_file", "").value
        self.ring_elevations = load_ring_elevations(ring_file) if ring_file else None
        self.azimuth_bins = self.declare_parameter("azimuth_bins", 720).value
        self.calibration = load_calibration(calibration_file) if calibration_file else {}
        limits = og.MotionLimits()
        limits.latency = self.declare_parameter("latency", 0.15).value
        limits.obstacle_speed = self.declare_parameter("obstacle_speed", 1.5).value
        self.checker = og.MotionCheck(limits)

        # Posing the shapes changes the model in place (the arm box stretches), so the cloud
        # thread and the check_motion thread each have their own.
        self.robot = RobotShapes()
        self.planned_robot = RobotShapes()
        resolution = self.robot.config["planner"]["check_resolution"]
        self.resolution = np.array([resolution[name] for name in JOINT_NAMES])
        self.names = self.robot.shape_names()

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.topics = list(LIDARS)
        self.first_clouds = {}
        self.grid = None
        self.centers = None
        self.lock = threading.Lock()  # the grid: updates and checks take turns
        self.joint_state = None
        self.world_from_base = np.eye(4, dtype=np.float32)
        self.headings = collections.deque(maxlen=200)  # (stamp, yaw) from odometry, for theta slack
        self.last_stamp = {}
        self.pending = {}  # LiDAR topic -> sweep end of its frame added since the last update
        self.last_update = None  # sweep end of the newest frame in the last update
        self.stats = Stats()

        sensors = ReentrantCallbackGroup()
        self.create_subscription(JointState, "joint_states", self.on_joint_state, 10, callback_group=sensors)
        self.create_subscription(Odometry, "wheel_odom", self.on_odom, 10, callback_group=sensors)
        clouds = MutuallyExclusiveCallbackGroup()
        for topic in self.topics:
            self.create_subscription(PointCloud2, topic, lambda msg, t=topic: self.on_cloud(t, msg),
                                     NEWEST_FRAME_ONLY, callback_group=clouds)
        self.create_service(CheckMotion, "check_motion", self.check_motion,
                            callback_group=MutuallyExclusiveCallbackGroup())
        self.timing_pub = self.create_publisher(DiagnosticArray, "obstacle_grid/timing", 1)
        self.label_pubs = {name: self.create_publisher(PointCloud2, f"obstacle_grid/{name}", 1)
                           for name in DEBUG_LABELS} if self.publish_labels else {}
        self.shapes_pub = self.create_publisher(MarkerArray, "obstacle_grid/robot_shapes", 1) if self.publish_labels else None
        self.static_shapes = self.robot.static_model().shapes()
        self.create_timer(report_period, self.report, callback_group=sensors)
        self.get_logger().info("Waiting for a frame from each LiDAR and their mounts in TF...")

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    # --- inputs ----------------------------------------------------------------------------

    def on_joint_state(self, msg):
        if "lift_joint" in msg.name:
            self.joint_state = msg

    def on_odom(self, msg):
        self.world_from_base = odom_pose(msg).astype(np.float32)
        yaw = np.arctan2(self.world_from_base[1, 0], self.world_from_base[0, 0])
        self.headings.append((stamp_seconds(msg.header.stamp), float(yaw)))

    def heading_at(self, t):
        if not self.headings:
            return 0.0
        stamps, yaws = zip(*self.headings)
        return float(np.interp(t, stamps, np.unwrap(yaws)))

    def on_cloud(self, topic, msg):
        received = self.now()
        stamp = stamp_seconds(msg.header.stamp)
        if self.grid is None:
            self.set_up(topic, msg)
            return
        if self.joint_state is None:
            return
        previous = self.last_stamp.get(topic)
        if previous is not None and stamp - previous > 1.5 * FRAME_DURATION:
            self.stats.count(f"dropped frames {topic}", round((stamp - previous) / FRAME_DURATION) - 1)
        self.last_stamp[topic] = stamp

        start = time.perf_counter()
        xyz, ring = cloud_fields(msg)
        unpacked = time.perf_counter()
        moving, _, _ = self.robot.moving_model(self.joint_state)
        posed = time.perf_counter()
        sweep_end = stamp + FRAME_DURATION
        self.pending[topic] = sweep_end
        relabel = (self.update_every_frame or len(self.pending) == len(self.topics) or
                   self.last_update is None or sweep_end - self.last_update > 1.5 * FRAME_DURATION)
        with self.lock:
            locked = time.perf_counter()
            self.grid.add_frame(self.topics.index(topic), xyz, ring, stamp, moving)
            if relabel:
                self.grid.update(moving, self.world_from_base)
            # update() leaves add_frame's timings alone; its own are stale unless it just ran.
            steps = {k: v for k, v in self.grid.timing().items() if relabel or k in ADD_FRAME_STEPS}
        done = self.now()
        self.stats.add(delivery=received - sweep_end, grid_update=time.perf_counter() - start,
                       update_unpack=unpacked - start, update_pose=posed - unpacked,
                       update_wait_for_check=locked - posed, **{f"update_{k}": v * 1e-3 for k, v in steps.items()})
        if relabel:
            # From the end of the oldest sweep not yet in the labels.
            self.stats.add(sweep_end_to_grid_ready=done - min(self.pending.values()))
            self.last_update = sweep_end
            self.pending.clear()
        if self.publish_labels:
            labels = self.grid.labels()
            header = Header(stamp=msg.header.stamp, frame_id=GRID_FRAME)
            for name, publisher in self.label_pubs.items():
                publisher.publish(xyz_cloud(header, self.centers[labels == int(PUBLISHED_LABELS[name])]))
            self.shapes_pub.publish(shape_markers(header, self.static_shapes + moving.shapes()))

    def set_up(self, topic, msg):
        """Build the grid once there's a frame from every LiDAR and TF has their mounts."""
        self.first_clouds.setdefault(topic, [cloud_fields(msg)])
        if len(self.first_clouds) < len(self.topics):
            return
        try:
            mounts = [self.mount(LIDARS[t]) for t in self.topics]
        except tf2_ros.TransformException as e:
            self.get_logger().warn(f"No LiDAR mount in TF yet: {e}", throttle_duration_sec=2.0)
            return
        grid, _ = make_grid(mounts, [self.first_clouds[t] for t in self.topics], level=False,
                            contamination_reach=self.reach, log=self.get_logger().info,
                            overrides={"static_after": self.static_after, "threads": self.threads,
                                       **({"table_threads": self.threads} if self.threads else {})},
                            ring_elevations_table=self.ring_elevations, azimuth_bins=self.azimuth_bins)
        self.centers = grid.grid.centers() if self.publish_labels else None
        self.grid = grid
        self.get_logger().info(f"Grid ready ({grid.grid.size:,} voxels); serving check_motion")

    def mount(self, lidar_frame):
        """base_footprint from a LiDAR's frame: from TF, or its calibrated joint origin."""
        if lidar_frame in self.calibration:
            parent, parent_from_lidar = self.calibration[lidar_frame]
            return transform_matrix(self.tf_buffer.lookup_transform(GRID_FRAME, parent, Time()).transform) @ parent_from_lidar
        return transform_matrix(self.tf_buffer.lookup_transform(GRID_FRAME, lidar_frame, Time()).transform)

    # --- checks ------------------------------------------------------------------------------

    def check_motion(self, request, response):
        start = time.perf_counter()
        points = request.trajectory.points
        if self.grid is None or self.joint_state is None:
            response.success, response.message = False, "the grid isn't ready yet"
            return response
        if len(points) < 2:
            response.success, response.message = False, "a trajectory needs at least two points"
            return response

        # Each point as a planner configuration, filling in joints the trajectory leaves out;
        # the gripper box fits the fingers as they're open now.
        q_now = joint_vector(self.joint_state)
        self.planned_robot.set_fingers(self.joint_state)
        index = {name: i for i, name in enumerate(request.trajectory.joint_names)}
        configurations = []
        for point in points:
            q = q_now.copy()
            for j, name in enumerate(JOINT_NAMES):
                if name in index:
                    q[j] = point.positions[index[name]]
            configurations.append(q)

        # Steps of at most the planner's check resolution, each tagged with its segment.
        samples, segment_of_step = [configurations[0]], []
        for segment, (a, b) in enumerate(zip(configurations[:-1], configurations[1:])):
            n = max(1, int(np.ceil(np.max(np.abs(b - a) / self.resolution))))
            for k in range(1, n + 1):
                samples.append(a + (b - a) * (k / n))
                segment_of_step.append(segment)
        bodies = [self.planned_robot.planning_body(q) for q in samples]
        braking = step_braking(samples)
        posed = time.perf_counter()

        with self.lock:
            oldest = self.grid.oldest_stamp()
            age = self.now() - oldest
            if age > self.stale_after:
                response.success = False
                response.message = f"the grid is {age:.2f} s old (a LiDAR may have stopped)"
                return response
            theta_slack = abs(self.heading_at(self.now()) - self.heading_at(oldest))
            steps = self.checker.check(self.grid, bodies, age, theta_slack, braking)
        checked = time.perf_counter()

        n_segments = len(points) - 1
        min_duration = np.zeros(n_segments)
        speed = np.full(n_segments, np.inf)
        shape = [""] * n_segments
        clear_occupied = np.full(n_segments, np.inf)
        clear_moving = np.full(n_segments, np.inf)
        clear_unsafe = np.full(n_segments, np.inf)
        blocked = [False] * n_segments
        unsafe = [False] * n_segments
        slowest_step = np.full(n_segments, -np.inf)
        for step, segment in zip(steps, segment_of_step):
            min_duration[segment] += step.min_duration
            clear_occupied[segment] = min(clear_occupied[segment], step.clearance_occupied)
            clear_moving[segment] = min(clear_moving[segment], step.clearance_moving)
            clear_unsafe[segment] = min(clear_unsafe[segment], step.clearance_unsafe)
            blocked[segment] |= step.blocked
            unsafe[segment] |= step.unsafe
            if step.shape >= 0 and step.min_duration > slowest_step[segment]:
                slowest_step[segment] = step.min_duration
                speed[segment] = step.speed
                shape[segment] = self.names[step.shape]

        response.success = True
        response.message = f"{len(steps)} steps, data age {age:.3f} s, theta slack {theta_slack:.4f} rad"
        response.min_duration = min_duration.tolist()
        response.limiting_speed = np.where(np.isfinite(speed), speed, 0.0).tolist()
        response.limiting_shape = shape
        response.clearance_occupied = clear_occupied.tolist()
        response.clearance_moving = clear_moving.tolist()
        response.clearance_unsafe = clear_unsafe.tolist()
        response.blocked = blocked
        response.unsafe = unsafe
        response.grid_stamp = Time(seconds=oldest).to_msg()
        response.grid_age = float(age)
        self.stats.add(check_posing=posed - start, check_grid=checked - posed,
                       check_total=time.perf_counter() - start, check_grid_age=age)
        return response

    # --- reporting ----------------------------------------------------------------------------

    def report(self):
        summary = self.stats.summary()
        if not summary:
            return
        status = DiagnosticStatus(name="obstacle_grid timing", level=DiagnosticStatus.OK,
                                  values=[KeyValue(key=k, value=v) for k, v in summary.items()])
        self.timing_pub.publish(DiagnosticArray(header=Header(stamp=self.get_clock().now().to_msg()), status=[status]))
        self.get_logger().info("Timing: " + "; ".join(f"{k}: {v}" for k, v in summary.items()))


def main():
    rclpy.init()
    node = ObstacleMonitor()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
