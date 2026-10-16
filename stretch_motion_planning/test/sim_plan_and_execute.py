#!/usr/bin/env python3
"""Plan with the planner node and run the result on a driver (the MuJoCo sim, or the robot
with --driver stretch_driver and the robot_* scenarios).

THIS MOVES THE ROBOT. Check that only the sim driver is on the ROS graph
(`ros2 node list --no-daemon`: /stretch_mujoco_driver, no /stretch_driver) first.

For each scenario: call plan_joint_trajectory, send the trajectory to
follow_joint_trajectory, record /joint_states and /wheel_odom while it runs, and report:
  - the action result and how long it took against the planned duration
  - the final error at the goal
  - how far the executed motion strayed from the planned path (weighted joint distance)
  - whether any executed configuration was in collision (primitives, no padding)

    python3 sim_plan_and_execute.py [--driver stretch_mujoco_driver] [--mode MODE] [--only NAME ...]

--mode sets trajectory_server.mode (and the matching joint modes: velocity for
adaptive_velocity, position otherwise) for the run, then restores target_priority.
"""

import argparse
import os
import time

import numpy as np
import rclpy
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Pose
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.parameter_client import AsyncParameterClient
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive

from stretch_motion_planning_interfaces.srv import PlanJointTrajectory
from stretch_motion_planning.collision_model import (
    JOINT_NAMES, StretchCollisionModel, joint_vector, load_config,
)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "stretch4_collision_primitives.yaml")
ARM_SEGMENTS = ["arm_l1_joint", "arm_l2_joint", "arm_l3_joint", "arm_l4_joint"]
DRIVER_JOINTS = ["lift_joint", "arm_joint", "wrist_yaw_joint", "wrist_pitch_joint", "wrist_roll_joint"]

TABLE = {"type": "box", "dims": [0.5, 1.0, 0.04], "xyz": [0.55, 0.0, 0.72]}

# name: (goal {joint: position}, obstacles, allow_rotation, expect_success)
SCENARIOS = {
    "reach_down": ({"lift_joint": 0.5, "arm_joint": 0.3, "wrist_pitch_joint": 0.6}, [], False, True),
    "above_table": ({"lift_joint": 0.95, "arm_joint": 0.3, "wrist_pitch_joint": 1.0}, [], False, True),
    "under_table": ({"lift_joint": 0.3, "arm_joint": 0.3, "wrist_pitch_joint": 0.0}, [TABLE], False, True),
    "back_over_table": ({"lift_joint": 0.95, "arm_joint": 0.3, "wrist_pitch_joint": 1.0}, [TABLE], False, True),
    "stow": ({"lift_joint": 0.3, "arm_joint": 0.0, "wrist_yaw_joint": 3.14,
              "wrist_pitch_joint": 0.0, "wrist_roll_joint": 0.0}, [], False, True),
    "rotate_blocked": ({"mobile_base_rotation_joint": 0.8, "lift_joint": 0.6}, [], False, False),
    "rotate": ({"mobile_base_rotation_joint": 0.8, "lift_joint": 0.6, "arm_joint": 0.1,
                "wrist_yaw_joint": 0.0}, [], True, True),
    "rotate_back": ({"mobile_base_rotation_joint": -0.8}, [], True, True),
    # A short, conservative sequence for a first real-robot check (run with --only robot_*).
    "robot_reach": ({"lift_joint": 0.6, "arm_joint": 0.2, "wrist_yaw_joint": 0.0,
                     "wrist_pitch_joint": 0.3, "wrist_roll_joint": 0.0}, [], False, True),
    "robot_stow": ({"lift_joint": 0.4, "arm_joint": 0.0, "wrist_yaw_joint": 3.14,
                    "wrist_pitch_joint": 0.0, "wrist_roll_joint": 0.0}, [], False, True),
    "robot_rotate": ({"mobile_base_rotation_joint": 0.3}, [], True, True),
    "robot_rotate_back": ({"mobile_base_rotation_joint": -0.3}, [], True, True),
    "robot_ready": ({"lift_joint": 0.468, "arm_joint": 0.009, "wrist_yaw_joint": 0.6,
                     "wrist_pitch_joint": 0.504, "wrist_roll_joint": 0.0}, [], False, True),
}


def odom_heading(odom):
    q = odom.pose.pose.orientation
    return np.arctan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def distance_to_path(q, waypoints, weights):
    """Weighted distance from q to the nearest point on the polyline through waypoints."""
    best = np.inf
    for a, b in zip(waypoints[:-1], waypoints[1:]):
        d = b - a
        denom = np.dot(weights * d, weights * d)
        t = 0.0 if denom == 0 else np.clip(np.dot(weights * (q - a), weights * d) / denom, 0.0, 1.0)
        best = min(best, np.linalg.norm(weights * (q - (a + t * d))))
    return best


class SimTester(Node):
    def __init__(self, driver):
        super().__init__("planner_sim_test")
        self.plan_client = self.create_client(PlanJointTrajectory, "plan_joint_trajectory")
        self.action_client = ActionClient(self, FollowJointTrajectory, "follow_joint_trajectory")
        self.driver_parameters = AsyncParameterClient(self, driver)
        self.create_subscription(JointState, "joint_states", self.on_joint_states, qos_profile_sensor_data)
        self.create_subscription(Odometry, "wheel_odom", self.on_odom, qos_profile_sensor_data)
        self.joint_state = None
        self.odom = None
        self.recording = None
        self.heading_zero = 0.0

    def on_joint_states(self, msg):
        self.joint_state = msg
        if self.recording is not None and self.odom is not None:
            self.recording.append(self.configuration())

    def on_odom(self, msg):
        self.odom = msg

    def configuration(self):
        positions = dict(zip(self.joint_state.name, self.joint_state.position))
        heading = np.remainder(odom_heading(self.odom) - self.heading_zero + np.pi, 2 * np.pi) - np.pi
        return np.array([heading, positions["lift_joint"], sum(positions[s] for s in ARM_SEGMENTS),
                         positions["wrist_yaw_joint"], positions["wrist_pitch_joint"],
                         positions["wrist_roll_joint"]])

    def wait(self, future, timeout):
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        if not future.done():
            raise TimeoutError("timed out")
        return future.result()

    def set_allow_rotation(self, allow):
        self.wait(self.driver_parameters.set_parameters(
            [Parameter("trajectory_server.allow_rotation", Parameter.Type.BOOL, allow)]), 5.0)

    def set_trajectory_mode(self, mode):
        joint_mode = "velocity" if mode == "adaptive_velocity" else "position"
        self.wait(self.driver_parameters.set_parameters(
            [Parameter("trajectory_server.mode", Parameter.Type.STRING, mode)]
            + [Parameter(f"joint_mode.{joint}", Parameter.Type.STRING, joint_mode) for joint in DRIVER_JOINTS]), 5.0)

    def run(self, name, goal, obstacles, allow_rotation, expect_success, collision):
        print(f"\n=== {name} ===")
        self.set_allow_rotation(allow_rotation)
        while self.joint_state is None or self.odom is None:
            rclpy.spin_once(self, timeout_sec=0.1)

        request = PlanJointTrajectory.Request()
        request.goal = JointState(name=list(goal), position=[float(v) for v in goal.values()])
        for obstacle in obstacles:
            request.obstacles.append(SolidPrimitive(type=SolidPrimitive.BOX, dimensions=obstacle["dims"]))
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = obstacle["xyz"]
            pose.orientation.w = 1.0
            request.obstacle_poses.append(pose)
        plan = self.wait(self.plan_client.call_async(request), 10.0)
        print(f"plan: success={plan.success} method={plan.method} {plan.planning_time * 1000:.1f} ms  {plan.message}")
        if not plan.success:
            return False

        trajectory = plan.trajectory
        planned_duration = trajectory.points[-1].time_from_start.sec + trajectory.points[-1].time_from_start.nanosec * 1e-9
        indices = [JOINT_NAMES.index(n) for n in trajectory.joint_names]
        start = self.configuration()
        start[0] = 0.0
        waypoints = []
        for point in trajectory.points:
            q = start.copy()
            q[indices] = point.positions
            waypoints.append(q)

        collision.set_obstacles([
            {"type": o["type"], "dims": o["dims"], "pose": np.block([[np.eye(3), np.array(o["xyz"])[:, None]], [np.zeros(3), 1.0]])}
            for o in obstacles])

        self.heading_zero = odom_heading(self.odom)
        self.recording = []
        t0 = time.monotonic()
        goal_handle = self.wait(self.action_client.send_goal_async(FollowJointTrajectory.Goal(trajectory=trajectory)), 5.0)
        if not goal_handle.accepted:
            print("action: goal rejected")
            self.recording = None
            return not expect_success
        result = self.wait(goal_handle.get_result_async(), planned_duration + 30.0).result
        elapsed = time.monotonic() - t0
        for _ in range(20):  # let it settle a moment and record it
            rclpy.spin_once(self, timeout_sec=0.05)
        executed, self.recording = self.recording, None

        print(f"action: error_code={result.error_code} '{result.error_string}'  "
              f"took {elapsed:.1f} s (planned {planned_duration:.1f} s), {len(executed)} samples")
        if result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
            if executed:
                print(f"stopped at: {np.round(executed[-1], 3)}")
                print(f"goal:       {np.round(waypoints[-1], 3)}")
            return not expect_success

        final_error = executed[-1] - waypoints[-1]
        weights = joint_vector(self.config["planner"]["distance_weights"])
        deviation = max(distance_to_path(q, waypoints, weights) for q in executed)
        in_collision = [q for q in executed if collision.in_collision(q)]
        print("final error: " + ", ".join(
            f"{n}={e:+.3f}" for n, e in zip(JOINT_NAMES, final_error) if JOINT_NAMES.index(n) in indices or abs(e) > 0.01))
        print(f"max deviation from planned path (weighted): {deviation:.3f}")
        print(f"executed configurations in collision: {len(in_collision)} of {len(executed)}")
        if in_collision:
            print(f"  first: {np.round(in_collision[0], 3)} {collision.colliding_pairs(in_collision[0])}")
        return expect_success and not in_collision


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--driver", default="stretch_mujoco_driver")
    parser.add_argument("--mode", default="target_priority",
                        choices=["target_priority", "time_priority", "adaptive_velocity"])
    parser.add_argument("--only", nargs="*", help=f"scenarios to run (default all: {list(SCENARIOS)})")
    args = parser.parse_args()

    rclpy.init()
    tester = SimTester(args.driver)
    tester.config = load_config(CONFIG_PATH)
    collision = StretchCollisionModel({**tester.config, "padding": 0.0})
    assert tester.plan_client.wait_for_service(timeout_sec=10.0), "planner node isn't running"
    assert tester.action_client.wait_for_server(timeout_sec=10.0), "driver's follow_joint_trajectory isn't up"

    outcomes = []
    try:
        tester.set_trajectory_mode(args.mode)
        for name in args.only or SCENARIOS:
            outcomes.append((name, tester.run(name, *SCENARIOS[name], collision)))
    finally:
        tester.set_allow_rotation(False)
        tester.set_trajectory_mode("target_priority")
    print("\n" + "\n".join(f"{'PASS' if ok else 'FAIL'}  {name}" for name, ok in outcomes))
    tester.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
