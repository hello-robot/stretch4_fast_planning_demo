#!/usr/bin/env python3
"""ROS 2 service for planning collision-free joint trajectories on Stretch 4.

Offers plan_joint_trajectory (stretch_motion_planning_interfaces/PlanJointTrajectory).
Plans for mobile_base_rotation_joint, lift_joint, arm_joint and the three wrist joints,
avoiding self-collision, the floor, obstacles given in the request and (optionally) the
collision objects in MoveIt's planning scene. Only plans: send the returned trajectory
to the driver's follow_joint_trajectory action to run it.
"""

import os
import threading

import numpy as np
import pinocchio as pin
import rclpy
import rclpy.parameter
import rclpy.time
import tf2_ros
from ament_index_python.packages import get_package_share_directory
from builtin_interfaces.msg import Duration
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter_client import AsyncParameterClient
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from stretch_motion_planning_interfaces.srv import PlanJointTrajectory
from stretch_motion_planning.collision_model import (
    JOINT_NAMES, StretchCollisionModel, joint_vector, load_config,
)
from stretch_motion_planning.planner import THETA, StretchPlanner
from stretch_motion_planning.timing import time_path

PRIMITIVE_TYPES = {
    SolidPrimitive.BOX: "box",
    SolidPrimitive.SPHERE: "sphere",
    SolidPrimitive.CYLINDER: "cylinder",
    SolidPrimitive.CONE: "cone",
}

# /joint_states reports the arm as its four telescoping segments.
ARM_SEGMENT_JOINTS = ["arm_l1_joint", "arm_l2_joint", "arm_l3_joint", "arm_l4_joint"]

# Joints whose limits come from the driver's joint_limit.<joint>.{lower,upper} parameters.
DRIVER_LIMIT_JOINTS = [name for name in JOINT_NAMES if name != "mobile_base_rotation_joint"]


def pose_to_se3(pose):
    q = pose.orientation
    rotation = pin.Quaternion(q.w, q.x, q.y, q.z).normalized().matrix()
    return pin.SE3(rotation, np.array([pose.position.x, pose.position.y, pose.position.z]))


def transform_to_se3(transform):
    t, q = transform.translation, transform.rotation
    rotation = pin.Quaternion(q.w, q.x, q.y, q.z).normalized().matrix()
    return pin.SE3(rotation, np.array([t.x, t.y, t.z]))


def seconds_to_duration(seconds):
    sec = int(seconds)
    return Duration(sec=sec, nanosec=int(round((seconds - sec) * 1e9)))


class PlanningError(Exception):
    pass


class TrajectoryPlannerNode(Node):
    def __init__(self):
        super().__init__("stretch_trajectory_planner")

        default_config = os.path.join(
            get_package_share_directory("stretch_motion_planning"), "config",
            "stretch4_collision_primitives.yaml",
        )
        self.declare_parameter("config_file", default_config)
        # Frame obstacles are planned in: the base at the start of the trajectory.
        self.declare_parameter("planning_frame", "base_footprint")
        # Driver node to read joint limits from ('' to use the config and URDF only).
        self.declare_parameter("driver_node", "stretch_driver")
        self.declare_parameter("planning_scene_service", "/get_planning_scene")

        config = load_config(self.get_parameter("config_file").value)
        self.collision = StretchCollisionModel(config)
        self.planner = StretchPlanner(self.collision, config)
        self.configured_limit_joints = set(config.get("joint_limits", {}))
        self.max_velocity = joint_vector(config["timing"]["max_velocity"])
        self.max_acceleration = joint_vector(config["timing"]["max_acceleration"])
        self.waypoint_spacing = joint_vector(config["timing"]["waypoint_spacing"])
        self.planning_frame = self.get_parameter("planning_frame").value

        self.latest_joint_state = None
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Planning runs one request at a time; everything else can run alongside it.
        background = ReentrantCallbackGroup()
        self.create_subscription(JointState, "joint_states", self.joint_state_callback,
                                 qos_profile_sensor_data, callback_group=background)
        self.create_service(PlanJointTrajectory, "plan_joint_trajectory", self.plan_callback,
                            callback_group=MutuallyExclusiveCallbackGroup())

        self.planning_scene_client = None
        self.scene_callback_group = background
        self._start_reading_driver_limits(self.get_parameter("driver_node").value, background)

        self.get_logger().info("Ready to plan on plan_joint_trajectory")

    # --- inputs -----------------------------------------------------------------------

    def joint_state_callback(self, msg):
        self.latest_joint_state = msg

    def _start_reading_driver_limits(self, driver_node, callback_group):
        """Read the driver's joint limits once it's up (it may start after this node)."""
        if not driver_node:
            return
        self.driver_parameters = AsyncParameterClient(self, driver_node)
        self.driver_limits_pending = False
        self.driver_limits_timer = self.create_timer(
            2.0, self._request_driver_limits, callback_group=callback_group)

    def _request_driver_limits(self):
        if self.driver_limits_pending or not self.driver_parameters.services_are_ready():
            return
        names = [f"joint_limit.{joint}.{end}" for joint in DRIVER_LIMIT_JOINTS for end in ("lower", "upper")]
        self.driver_limits_pending = True
        self.driver_parameters.get_parameters(names).add_done_callback(
            lambda future: self._apply_driver_limits(names, future))

    def _apply_driver_limits(self, names, future):
        self.driver_limits_pending = False
        if future.result() is None:
            return
        values = dict(zip(names, (rclpy.parameter.parameter_value_to_python(v) for v in future.result().values)))
        limits = {}
        for joint in DRIVER_LIMIT_JOINTS:
            lower, upper = values[f"joint_limit.{joint}.lower"], values[f"joint_limit.{joint}.upper"]
            if joint not in self.configured_limit_joints and lower is not None and upper is not None:
                limits[joint] = [lower, upper]
        self.collision.set_joint_limits(limits)
        self.driver_limits_timer.cancel()
        self.get_logger().info(f"Using the driver's joint limits: {limits}")

    def current_configuration(self):
        """The robot's configuration from /joint_states, with base rotation 0."""
        state = self.latest_joint_state
        if state is None:
            raise PlanningError("No /joint_states received yet")
        positions = dict(zip(state.name, state.position))
        q = np.zeros(len(JOINT_NAMES))
        for i, name in enumerate(JOINT_NAMES):
            if name == "mobile_base_rotation_joint":
                continue
            if name == "arm_joint" and name not in positions:
                if not all(segment in positions for segment in ARM_SEGMENT_JOINTS):
                    raise PlanningError("/joint_states has no arm position")
                q[i] = sum(positions[segment] for segment in ARM_SEGMENT_JOINTS)
            elif name in positions:
                q[i] = positions[name]
            else:
                raise PlanningError(f"/joint_states has no {name}")
        return q

    @staticmethod
    def _apply_named_positions(q, joint_state, what):
        q = q.copy()
        for name, value in zip(joint_state.name, joint_state.position):
            if name not in JOINT_NAMES:
                raise PlanningError(f"Unknown {what} joint '{name}' (expected one of {list(JOINT_NAMES)})")
            q[JOINT_NAMES.index(name)] = value
        return q

    def _transform_from(self, frame):
        """Pose of `frame` in the planning frame."""
        if not frame or frame == self.planning_frame:
            return pin.SE3.Identity()
        try:
            transform = self.tf_buffer.lookup_transform(self.planning_frame, frame, rclpy.time.Time())
        except tf2_ros.TransformException as e:
            raise PlanningError(f"Can't transform obstacles from {frame}: {e}")
        return transform_to_se3(transform.transform)

    @staticmethod
    def _obstacle(primitive, pose):
        if primitive.type not in PRIMITIVE_TYPES:
            raise PlanningError(f"Unsupported obstacle primitive type {primitive.type}")
        return {"type": PRIMITIVE_TYPES[primitive.type], "dims": list(primitive.dimensions), "pose": pose}

    def request_obstacles(self, request):
        if len(request.obstacles) != len(request.obstacle_poses):
            raise PlanningError("obstacles and obstacle_poses must be the same length")
        frame = self._transform_from(request.obstacle_frame)
        return [self._obstacle(primitive, frame * pose_to_se3(pose))
                for primitive, pose in zip(request.obstacles, request.obstacle_poses)]

    def planning_scene_obstacles(self):
        """Primitive collision objects from MoveIt's planning scene (meshes are skipped)."""
        try:
            from moveit_msgs.msg import PlanningSceneComponents
            from moveit_msgs.srv import GetPlanningScene
        except ImportError:
            raise PlanningError("use_planning_scene needs moveit_msgs, which isn't installed")

        if self.planning_scene_client is None:
            self.planning_scene_client = self.create_client(
                GetPlanningScene, self.get_parameter("planning_scene_service").value,
                callback_group=self.scene_callback_group)
        if not self.planning_scene_client.wait_for_service(timeout_sec=1.0):
            raise PlanningError("MoveIt's planning scene service isn't available")

        request = GetPlanningScene.Request()
        request.components.components = PlanningSceneComponents.WORLD_OBJECT_GEOMETRY
        done = threading.Event()
        future = self.planning_scene_client.call_async(request)
        future.add_done_callback(lambda _: done.set())
        if not done.wait(timeout=2.0) or future.result() is None:
            raise PlanningError("MoveIt's planning scene service didn't respond")

        obstacles = []
        for obj in future.result().scene.world.collision_objects:
            if obj.meshes:
                self.get_logger().warn(f"Skipping the meshes of collision object '{obj.id}'",
                                       throttle_duration_sec=10.0)
            object_pose = self._transform_from(obj.header.frame_id) * pose_to_se3(obj.pose)
            for primitive, pose in zip(obj.primitives, obj.primitive_poses):
                obstacles.append(self._obstacle(primitive, object_pose * pose_to_se3(pose)))
        return obstacles

    # --- planning ---------------------------------------------------------------------

    def plan_callback(self, request, response):
        try:
            self._plan(request, response)
        except PlanningError as e:
            response.success = False
            response.message = str(e)
        if not response.success:
            self.get_logger().warn(f"Planning failed: {response.message}")
        return response

    def _plan(self, request, response):
        start = np.zeros(len(JOINT_NAMES))
        if set(request.start.name) != set(JOINT_NAMES):
            start = self.current_configuration()
        start = self._apply_named_positions(start, request.start, "start")
        goal = self._apply_named_positions(start, request.goal, "goal")

        # The base only rotates if the goal asks for a base rotation.
        active = np.ones(len(JOINT_NAMES), bool)
        active[THETA] = "mobile_base_rotation_joint" in request.goal.name

        obstacles = self.request_obstacles(request)
        if request.use_planning_scene:
            obstacles += self.planning_scene_obstacles()
        self.collision.set_obstacles(obstacles)

        result = self.planner.plan(start, goal, active, timeout=request.timeout)
        response.success = result.success
        response.message = result.message
        response.method = result.method
        response.planning_time = result.planning_time
        if not result.success:
            return

        positions, times = time_path(result.path, self.max_velocity, self.max_acceleration,
                                     self.waypoint_spacing)
        rotates = any(abs(q[THETA] - start[THETA]) > 1e-9 for q in result.path)
        joints = [i for i in range(len(JOINT_NAMES)) if i != THETA or rotates]
        response.trajectory = JointTrajectory(
            joint_names=[JOINT_NAMES[i] for i in joints],
            points=[
                JointTrajectoryPoint(positions=[float(q[i]) for i in joints],
                                     time_from_start=seconds_to_duration(t))
                for q, t in zip(positions, times)
            ],
        )
        self.get_logger().info(
            f"Planned with {result.method} in {result.planning_time * 1000:.1f} ms: "
            f"{len(result.path) - 1} straight-line segments, {times[-1]:.1f} s long")


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryPlannerNode()
    executor = MultiThreadedExecutor()
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
