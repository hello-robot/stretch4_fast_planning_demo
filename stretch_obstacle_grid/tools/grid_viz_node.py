#!/usr/bin/env python3
"""Debug view of the obstacle grid in RViz: runs the grid on the LiDAR clouds and publishes
each label's voxels as its own PointCloud2 (base_footprint), for checking by eye.

    ros2 run stretch_obstacle_grid grid_viz_node.py [--ros-args -p show_unknown:=true]
    rviz2 -d $(ros2 pkg prefix stretch_obstacle_grid)/share/stretch_obstacle_grid/rviz/stretch4_obstacle_grid.rviz

Topics: /obstacle_grid/{occupied,static,self,shadow,contaminated,unknown}. Unknown space is most of
the grid, so it's off unless show_unknown is set. /obstacle_grid/points_{left,right} are the
LiDAR clouds as the grid sees them: in base_footprint with the leveled mounts, so they line
up with the voxels (the raw clouds use TF's un-leveled mounts, a few cm off).

Parameters:
  level_to_floor (true): correct each LiDAR's mount so the floor is at z = 0, fitted from
      the first `setup_frames` clouds (the TF mounts are off by about 1.5 deg).
  setup_frames (10): clouds per LiDAR to collect before starting.
  contamination_reach (0.2): m into a shadow from where it touches unknown space or an obstacle.
  show_unknown (false): also publish unknown voxels.
  world_frame ("odom"): a frame fixed in the world, from TF, for telling static obstacles from
      moving ones while the base turns. If TF doesn't have it, the base counts as not moving.
  calibration_file (""): a stretch_calibration_values.yaml with LiDAR joint origins (e.g. from
      calibrate_lidars.py) to use instead of TF's mounts, before it's installed on the robot.
      Set level_to_floor:=false with it, since the calibration already includes the floor.

Not for use on the robot's control path: it's Python, copies every label to ROS messages,
and is meant for watching a bag play back.
"""

import numpy as np
import rclpy
import tf2_ros
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import JointState, PointCloud2, PointField
from std_msgs.msg import Header

import stretch_obstacle_grid as og
from grid_setup import GRID_FRAME, LIDARS, RobotShapes, cloud_fields, load_calibration, make_grid, transform_matrix

PUBLISHED_LABELS = {"occupied": og.OCCUPIED, "static": og.STATIC, "self": og.SELF, "shadow": og.SHADOW,
                    "contaminated": og.CONTAMINATED_SHADOW, "unknown": og.UNKNOWN}
XYZ_FIELDS = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1) for i, n in enumerate("xyz")]


def xyz_cloud(header, points):
    """A PointCloud2 of float32 x, y, z."""
    points = np.ascontiguousarray(points, dtype=np.float32)
    return PointCloud2(header=header, height=1, width=len(points), fields=XYZ_FIELDS, is_bigendian=False,
                       point_step=12, row_step=12 * len(points), is_dense=True, data=points.tobytes())


class GridVizNode(Node):
    def __init__(self):
        super().__init__("obstacle_grid_viz")
        self.level = self.declare_parameter("level_to_floor", True).value
        self.setup_frames = self.declare_parameter("setup_frames", 10).value
        self.reach = self.declare_parameter("contamination_reach", 0.2).value
        self.show_unknown = self.declare_parameter("show_unknown", False).value
        self.publish_points = self.declare_parameter("publish_points", True).value
        self.world_frame = self.declare_parameter("world_frame", "odom").value
        calibration_file = self.declare_parameter("calibration_file", "").value
        self.calibration = load_calibration(calibration_file) if calibration_file else {}

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.topics = list(LIDARS)
        self.first_clouds = {topic: [] for topic in self.topics}
        self.grid = None
        self.mounts = None  # leveled base_from_lidar per LiDAR
        self.centers = None
        self.robot = RobotShapes()
        self.joint_state = None
        self.frames = 0
        self.timings = []

        self.publishers_ = {name: self.create_publisher(PointCloud2, f"obstacle_grid/{name}", 1)
                            for name in PUBLISHED_LABELS if name != "unknown" or self.show_unknown}
        self.point_publishers = {topic: self.create_publisher(PointCloud2, f"obstacle_grid/points_{topic.rsplit('_', 1)[-1]}", 1)
                                 for topic in self.topics} if self.publish_points else {}
        self.create_subscription(JointState, "joint_states", self.on_joint_state, 10)
        for topic in self.topics:
            self.create_subscription(PointCloud2, topic, lambda msg, t=topic: self.on_cloud(t, msg),
                                     qos_profile_sensor_data)
        self.get_logger().info(f"Waiting for {self.setup_frames} clouds from each LiDAR and their TF...")

    def on_joint_state(self, msg):
        if "lift_joint" in msg.name:
            self.joint_state = msg

    def on_cloud(self, topic, msg):
        if self.grid is None:
            self.collect(topic, msg)
            return
        if self.joint_state is None:
            return
        xyz, ring = cloud_fields(msg)
        moving, _, _ = self.robot.moving_model(self.joint_state)
        self.grid.add_frame(self.topics.index(topic), xyz, ring,
                            msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9, moving)
        bin_ms = self.grid.timing()["bin_points"]
        self.grid.update(moving, self.world_from_base(msg.header.stamp))
        self.publish(msg.header.stamp)
        if self.publish_points:
            mount = self.mounts[self.topics.index(topic)]
            finite = xyz[np.isfinite(xyz).all(axis=1) & (np.abs(xyz).sum(axis=1) > 0)]
            in_base = finite @ mount[:3, :3].T.astype(np.float32) + mount[:3, 3].astype(np.float32)
            self.point_publishers[topic].publish(xyz_cloud(Header(stamp=msg.header.stamp, frame_id=GRID_FRAME), in_base))

        self.timings.append(bin_ms + sum(v for k, v in self.grid.timing().items() if k != "bin_points"))
        self.frames += 1
        if self.frames % 50 == 0:
            t = np.array(self.timings[-50:])
            self.get_logger().info(f"{self.frames} frames; grid update {np.median(t):.1f} ms median, {t.max():.1f} max")

    def world_from_base(self, stamp):
        """base_footprint in world_frame at `stamp` (or the latest), as float32 4x4."""
        try:
            t = self.tf_buffer.lookup_transform(self.world_frame, GRID_FRAME, Time.from_msg(stamp))
        except tf2_ros.TransformException:
            try:
                t = self.tf_buffer.lookup_transform(self.world_frame, GRID_FRAME, Time())
            except tf2_ros.TransformException:
                return np.eye(4, dtype=np.float32)
        return transform_matrix(t.transform).astype(np.float32)

    def collect(self, topic, msg):
        """Gather the first clouds, then build the grid once TF has the LiDAR mounts."""
        if len(self.first_clouds[topic]) < self.setup_frames:
            self.first_clouds[topic].append(cloud_fields(msg))
        if min(len(c) for c in self.first_clouds.values()) < self.setup_frames:
            return
        try:
            mounts = []
            for t in self.topics:
                if LIDARS[t] in self.calibration:
                    parent, parent_from_lidar = self.calibration[LIDARS[t]]
                    base_from_parent = transform_matrix(self.tf_buffer.lookup_transform(GRID_FRAME, parent, Time()).transform)
                    mounts.append(base_from_parent @ parent_from_lidar)
                else:
                    mounts.append(transform_matrix(self.tf_buffer.lookup_transform(GRID_FRAME, LIDARS[t], Time()).transform))
        except tf2_ros.TransformException as e:
            self.get_logger().warn(f"No LiDAR mount in TF yet: {e}", throttle_duration_sec=2.0)
            return
        self.grid, self.mounts = make_grid(mounts, [self.first_clouds[t] for t in self.topics], level=self.level,
                                           contamination_reach=self.reach, log=self.get_logger().info)
        self.centers = self.grid.grid.centers()
        self.first_clouds = None
        self.get_logger().info(f"Grid ready: {self.grid.grid.size:,} voxels. Publishing on obstacle_grid/*")

    def publish(self, stamp):
        labels = self.grid.labels()
        header = Header(stamp=stamp, frame_id=GRID_FRAME)
        for name, publisher in self.publishers_.items():
            publisher.publish(xyz_cloud(header, self.centers[labels == int(PUBLISHED_LABELS[name])]))


def main():
    rclpy.init()
    node = GridVizNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
