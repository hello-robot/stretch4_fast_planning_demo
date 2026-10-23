"""A FollowJointTrajectory server that slows down, and stops short, for obstacles.

It runs inside the Stretch 4 driver next to the default server, loaded with the driver
parameter extra_trajectory_servers:="stretch_dynamic_trajectory.server:DynamicTrajectoryActionServer",
and serves follow_joint_trajectory_dynamic with parameters under dynamic_trajectory_server.*.

The goal's path is followed as given, at a speed scale between 0 and 1 (see timing.py): while
it runs, a background thread asks the obstacle monitor's check_motion about the next
check_horizon seconds of the trajectory, about check_rate times a second, and the scale is
kept low enough to be down to each segment's allowed speed when the robot gets there, to
stop before a blocked segment, and to stop by the end of what's been checked. Without a fresh
answer (check_timeout) the robot slows to a stop. Once the checks stop the robot for a blocked
path, it stays stopped for resume_delay s, then creeps toward the block at no more than
creep_speed (at any point of the robot) while the path just ahead is clear, stopping
creep_standoff short of the block, and goes back to full speed once resume_clear_checks checks in a row see
nothing blocked and resume_delay s have passed. The scale never rises faster than
max_scale_up_rate. Each check's details (which
segment and shape limit the motion, clearances, blocked segments) are published on
dynamic_trajectory_server_diagnostics/check, and the path being followed on
dynamic_trajectory_server_diagnostics/plan (JointTrajectory, latched) every time it changes:
for now once per goal, and later on every replan. Its stamp is when its base rotation is
measured from (the base's heading then is 0). The obstacle monitor draws it in RViz. Joints are commanded in velocity mode, like
the default server's adaptive_velocity mode: the scaled planned velocity plus kp times the
position error, with the joint's maximum acceleration so the firmware follows each command.

While the motion just ahead sweeps unsafe space (unknown, or contaminated shadow) the
driver's guarded-contact sensitivity is raised (unsafe_sensitivity), and set back
(safe_sensitivity) once it's clear. A wrist joint running well over its commanded speed (the
Feetech servos sometimes keep a stale speed) is quick-stopped and the goal aborted.

A rise in a moving joint's effort over its free-motion level (contact.py) means contact: the
robot stops, the joint that touched backs off back_off_distance along its own axis at
back_off_speed (both m at the contact point), and holds; never limp. Then contact_action:
"abort" ends the goal, reporting the contact; "resume" goes back along the plan to where the
robot backed off to, holds contact_hold s, and carries on (into the same thing again, if it's
still there), ending the goal after contact_resume_limit contacts in it.

Not yet: replanning around a blocked path, and dodging while holding a position.
"""

import threading
import time

import numpy as np
from builtin_interfaces.msg import Duration as DurationMsg
from control_msgs.action import FollowJointTrajectory
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from hello_helpers.stretch4_ros_api import StretchTrajectoryActionServer
from stretch_motion_planning_interfaces.srv import CheckMotion

from stretch_dynamic_trajectory.contact import CONTACT_SETTINGS, ContactDetector
from stretch_dynamic_trajectory.timing import (CheckResult, ResumeGate, Trajectory, blocked_anywhere, check_times,
                                               creeping_limits, rewind, scale_deceleration, segment_caps, step_scale,
                                               stopping_for_block, target_scale)

WRIST_JOINTS = ("wrist_yaw_joint", "wrist_pitch_joint", "wrist_roll_joint")
# m a point of the robot moves per unit of each joint's motion, at most, for bounding the
# fastest point's speed while creeping: the robot's reach for base rotation (arm out 0.55 m plus
# the gripper), the gripper's length for the wrist joints, and 1 for everything else.
CREEP_LEVERS = {"mobile_base_rotation_joint": 1.0, **{j: 0.3 for j in WRIST_JOINTS}}


def duration_msg(seconds):
    # Round to whole nanoseconds first: rounding only the fraction turns 0.9999999999 s into 0 s.
    sec, nanosec = divmod(int(round(seconds * 1e9)), 1000000000)
    return DurationMsg(sec=sec, nanosec=nanosec)


class DynamicTrajectoryActionServer(StretchTrajectoryActionServer):
    def __init__(self, driver):
        self._goal_lock = threading.Lock()  # one goal at a time; a new goal ends the running one
        self._superseded = threading.Event()
        self._check = None  # the latest CheckResult
        self._check_lock = threading.Lock()
        self._state = None  # (tau, positions) shared with the check thread
        self._held = False  # whether the resume gate holds the robot, for the check thread
        self._sensitivity = None
        super().__init__(driver, action_name="follow_joint_trajectory_dynamic",
                         param_prefix="dynamic_trajectory_server",
                         diagnostics_namespace="dynamic_trajectory_server_diagnostics")
        self._check_client = driver.create_client(CheckMotion, "check_motion",
                                                  callback_group=MutuallyExclusiveCallbackGroup())
        self._pub_timing = driver.create_publisher(JointState, "dynamic_trajectory_server_diagnostics/timing", 10)
        self._pub_check = driver.create_publisher(DiagnosticArray, "dynamic_trajectory_server_diagnostics/check", 10)
        self._pub_plan = driver.create_publisher(
            JointTrajectory, "dynamic_trajectory_server_diagnostics/plan",
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))

    def declare_params(self):
        super().declare_params()

        def add_param(name, value):
            self.driver.declare_parameter(f"{self.param_prefix}.{name}", value)

        # s of trajectory time ahead to check, how often, and in what steps
        add_param("check_horizon", 1.0)
        add_param("check_rate", 10.0)
        add_param("check_step", 0.1)
        # s without a fresh check answer before slowing to a stop
        add_param("check_timeout", 0.3)
        # the most the speed scale changes per s, whatever the joints could do; and the most it
        # rises per s (0 to full speed in 0.5 s), so starting and resuming are gentle
        add_param("max_scale_rate", 5.0)
        add_param("max_scale_up_rate", 2.0)
        # After the checks stop the robot for a blocked path, keep it stopped for resume_delay
        # s, then let it creep toward the block at no more than creep_speed (m/s at any point of
        # the robot), speeding up and braking for the block at creep_acceleration (m/s^2), and
        # back to full speed once resume_clear_checks checks in a row see nothing blocked and
        # resume_delay s have passed since the last one that did.
        add_param("resume_clear_checks", 3)
        add_param("resume_delay", 0.5)
        add_param("creep_speed", 0.02)
        add_param("creep_acceleration", 0.1)
        # Creeping stops this far (m, at the fastest point) short of a block: more than the grid's
        # self-filter padding (3 cm), within which the robot can't see an obstacle, plus a voxel.
        add_param("creep_standoff", 0.05)
        # Contact (see contact.py): on a rise in a moving joint's effort, stop, back that joint
        # off back_off_distance (m at the contact point: the gripper for a wrist joint) at
        # back_off_speed (m/s), hold, then contact_action: "abort" or "resume" (after
        # contact_hold s).
        add_param("contact_detection", True)
        # Multiplies contact.py's effort thresholds: below 1 stops on a lighter touch.
        add_param("contact_threshold_scale", 1.0)
        add_param("contact_action", "abort")
        add_param("back_off_distance", 0.05)
        add_param("back_off_speed", 0.02)
        add_param("contact_hold", 2.0)
        add_param("contact_resume_limit", 3)
        # While held, the first creep_check_fine_horizon s of each check go in steps of
        # creep_check_step, so a block near the robot is placed closely enough to brake for gently.
        add_param("creep_check_step", 0.02)
        add_param("creep_check_fine_horizon", 0.3)
        # Progress along the path slows as the robot falls behind it, and stops once a joint is
        # this many of its tolerances behind (it can't keep up, or something holds it), so the
        # desired state never runs away from the robot.
        add_param("max_lag_tolerances", 3.0)
        # The position correction (kp * error) may add at most this fraction of the scaled
        # planned speed, plus catch_up_floor (m/s or rad/s): the obstacle check allowed the
        # planned speed at this scale, not a catch-up sprint.
        add_param("catch_up_fraction", 0.2)
        add_param("catch_up_floor", 0.01)
        add_param("switch_sensitivity", True)
        add_param("unsafe_sensitivity", "high_sensitivity_manipulation")
        add_param("safe_sensitivity", "default")
        # s of trajectory time ahead that counts for the sensitivity
        add_param("sensitivity_lookahead", 0.5)
        # stop a wrist moving faster than factor * |command| + floor (rad/s) for this long (s)
        add_param("wrist_overspeed.factor", 2.5)
        add_param("wrist_overspeed.floor", 0.15)
        add_param("wrist_overspeed.time", 0.1)
        # the driver declares per-joint tolerances for the default server; use the same ones
        for name, value in self.driver.get_parameters_by_prefix("trajectory_server.trajectory_tolerance").items():
            if not self.driver.has_parameter(f"{self.param_prefix}.trajectory_tolerance.{name}"):
                add_param(f"trajectory_tolerance.{name}", value.value)

    # --- goals ---------------------------------------------------------------------------------

    def handle_accepted_callback(self, goal_handle):
        # A new goal ends the running one (which stops its joints), then runs.
        self._superseded.set()
        super().handle_accepted_callback(goal_handle)

    def _check_for_interrupt(self, goal_handle):
        result = super()._check_for_interrupt(goal_handle)
        if result is None and self._superseded.is_set():
            return self._abort(goal_handle, FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED,
                               "Superseded by a new goal")
        return result

    def execute_callback(self, goal_handle):
        with self._goal_lock:
            self._superseded.clear()
            return self._execute(goal_handle)

    def _execute(self, goal_handle):
        trajectory = goal_handle.request.trajectory
        trajectory.joint_names = [self.driver.resolve_joint_name(n) for n in trajectory.joint_names]
        joints = list(trajectory.joint_names)
        if self.BASE_ROTATION_JOINT in joints:
            if not self.get_param("allow_rotation"):
                return self._abort(goal_handle, FollowJointTrajectory.Result.INVALID_JOINTS,
                                   f"Trajectory includes {self.BASE_ROTATION_JOINT} but "
                                   f"{self.param_prefix}.allow_rotation is false")
            if self.driver.last_odom is None:
                return self._abort(goal_handle, FollowJointTrajectory.Result.INVALID_GOAL,
                                   f"No odometry yet for {self.BASE_ROTATION_JOINT}")
        result = self._check_joints_in_mode(joints, self.driver.velocity_mode, goal_handle)
        if result is not None:
            return result
        if not self._check_client.service_is_ready():
            return self._abort(goal_handle, FollowJointTrajectory.Result.INVALID_GOAL,
                               "No check_motion service (is the obstacle monitor running?)")

        stop_checks = threading.Event()
        checker = None
        try:
            if self.BASE_ROTATION_JOINT in joints:
                self._start_base_rotation()
            self.active_joints = joints
            self.direct_command_preempted = False
            self.preempted_joints = set()
            plan = self._plan(trajectory)
            self._publish_plan(plan, joints)
            with self._check_lock:
                self._check = None
            self._state = (0.0, self._positions(joints))
            self._held = False
            checker = threading.Thread(target=self._check_loop, args=(plan, joints, stop_checks), daemon=True)
            checker.start()
            result = self._follow(plan, trajectory, goal_handle)
        except Exception as e:
            self.driver.get_logger().error(f"Dynamic trajectory failed: {e}")
            result = self._abort(goal_handle, FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED, f"Error {e}")
        finally:
            stop_checks.set()
            if checker is not None:
                checker.join(timeout=1.0)
            self.stop_vel_joints(joints)
            self.active_joints = None
            self._set_sensitivity(self.get_param("safe_sensitivity"))
        return result

    def _plan(self, trajectory):
        """The goal as a Trajectory from the robot's current state at trajectory time 0."""
        joints = list(trajectory.joint_names)
        times = [self._msg_seconds(p.time_from_start) for p in trajectory.points]
        positions = [list(p.positions) for p in trajectory.points]
        given = all(len(p.velocities) == len(joints) for p in trajectory.points)
        velocities = [list(p.velocities) for p in trajectory.points] if given else None
        now = self._positions(joints)
        if times[0] > 1e-6:
            times = [0.0] + times
            positions = [list(now)] + positions
            if velocities is not None:
                velocities = [[0.0] * len(joints)] + velocities
        else:
            positions[0] = list(now)  # start from where the robot is
        return Trajectory(times, positions, velocities)

    def _publish_plan(self, plan, joints, step=0.05, max_points=400):
        """The path being followed, sampled every `step` s of trajectory time (the waypoints
        alone would miss the curves between them), stamped with when the base's rotation in it
        is measured from."""
        times = np.linspace(0.0, plan.duration, int(np.clip(np.ceil(plan.duration / step), 1, max_points - 1)) + 1)
        msg = JointTrajectory(header=Header(stamp=self.driver.get_clock().now().to_msg()), joint_names=joints)
        msg.points = [JointTrajectoryPoint(positions=[float(v) for v in plan.sample(t)[0]], time_from_start=duration_msg(t))
                      for t in times]
        self._pub_plan.publish(msg)

    def _positions(self, joints):
        return np.array([self._joint_position(j) for j in joints], dtype=float)

    def _limits(self, joints, kind):
        """joint_limit.<joint>.<kind> (velocity or acceleration) per joint; the base's from
        base_rotation.max_velocity and joint_limit.omnibase.angular.acceleration."""
        values = []
        for joint in joints:
            if joint != self.BASE_ROTATION_JOINT:
                value = self.driver.get_parameter_or(f"joint_limit.{joint}.{kind}").value
            elif kind == "velocity":
                value = self.get_param("base_rotation.max_velocity")
            else:
                value = self.driver.get_parameter_or("joint_limit.omnibase.angular.acceleration").value
            values.append(np.inf if value is None else float(value))
        return np.array(values)

    # --- following -----------------------------------------------------------------------------

    def _follow(self, plan, trajectory, goal_handle):
        joints = list(trajectory.joint_names)
        base_i = joints.index(self.BASE_ROTATION_JOINT) if self.BASE_ROTATION_JOINT in joints else None
        rate = self.driver.create_rate(self.get_param("loop_rate"))
        kp, base_kp = self.get_param("kp"), self.get_param("base_rotation.kp")
        max_velocity = self._limits(joints, "velocity")
        max_acceleration = self._limits(joints, "acceleration")
        tolerances = np.array([self._tolerance(j) for j in joints])
        wrists = [i for i, j in enumerate(joints) if j in WRIST_JOINTS]
        overspeed_since = {i: None for i in wrists}
        max_rate, max_up_rate = self.get_param("max_scale_rate"), self.get_param("max_scale_up_rate")
        gate = ResumeGate(self.get_param("resume_clear_checks"), self.get_param("resume_delay"))
        gated_stamp = None  # the last check counted by the gate
        creep_speed, creep_acceleration = self.get_param("creep_speed"), self.get_param("creep_acceleration")
        creep_standoff = self.get_param("creep_standoff")
        levers = np.array([CREEP_LEVERS.get(j, 1.0) for j in joints])
        check_timeout = self.get_param("check_timeout")
        max_lag = self.get_param("max_lag_tolerances")
        threshold_scale = self.get_param("contact_threshold_scale")
        detectors = ({i: ContactDetector(CONTACT_SETTINGS[j], threshold_scale=threshold_scale)
                      for i, j in enumerate(joints) if j in CONTACT_SETTINGS}
                     if self.get_param("contact_detection") else {})
        commands = np.zeros(len(joints))  # last loop's, which the efforts now respond to
        hold_until = None  # after a contact, with contact_action "resume"
        contact = 0.0  # the last contact's effort rise, for the timing topic
        contacts = 0
        catch_up_fraction, catch_up_floor = self.get_param("catch_up_fraction"), self.get_param("catch_up_floor")

        tau, scale = 0.0, 0.0
        last = time.monotonic()
        start = self.driver.get_clock().now()
        while tau < plan.duration:
            interrupt = self._check_for_interrupt(goal_handle)
            if interrupt is not None:
                return interrupt
            now = time.monotonic()
            dt, last = now - last, now

            # How fast the plan may go now.
            with self._check_lock:
                check = self._check
            if check is not None and now - check.stamp > check_timeout:
                check = None  # stale: slow to a stop
            horizon_end = min(plan.duration, tau + self.get_param("check_horizon"))
            rate_limit = scale_deceleration(plan, tau, horizon_end, max_acceleration, max_rate)
            target = target_scale(tau, check, rate_limit, plan.duration)
            if check is not None and check.stamp != gated_stamp:
                gated_stamp = check.stamp
                gate.update(check.stamp, stopping_for_block(check, target), blocked_anywhere(check))
                self._held = gate.held
            up_rate = max_up_rate
            if gate.state == ResumeGate.STOPPED or (hold_until is not None and now < hold_until):
                target = 0.0
            elif gate.state == ResumeGate.CREEPING:
                creep_target, creep_rate = creeping_limits(plan, tau, horizon_end, check, creep_speed,
                                                           creep_acceleration, levers, creep_standoff)
                target, up_rate = min(target, creep_target), min(up_rate, creep_rate)
            scale = step_scale(scale, target, rate_limit, dt, up_rate=up_rate)

            actual = self._positions(joints)
            touched = self._detect_contact(detectors, joints, commands, now)
            if touched is not None:
                i, rise = touched
                contact, contacts = rise, contacts + 1
                backed_off = self._back_off(joints, i, commands[i], levers[i], goal_handle, rate)
                if self.get_param("contact_action") != "resume" or contacts >= self.get_param("contact_resume_limit"):
                    return self._abort(goal_handle, FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED,
                                       f"contact on {joints[i]} (effort {rise:.0f} over free motion); "
                                       f"backed off {backed_off:.3f}")
                actual = self._positions(joints)
                tau = rewind(plan, tau, i, actual[i])
                scale, hold_until = 0.0, time.monotonic() + self.get_param("contact_hold")
                commands = np.zeros(len(joints))
                last = time.monotonic()
                continue

            # Advance along the plan more slowly the further the robot lags behind it, and not
            # at all past max_lag tolerances.
            desired, _, _ = plan.sample(tau)
            lag = np.max(np.abs(desired - actual) / tolerances) / max_lag
            moving = scale * float(np.clip(1.0 - lag, 0.0, 1.0))
            tau = min(plan.duration, tau + moving * dt)
            self._state = (tau, actual)
            desired, planned_velocity, _ = plan.sample(tau)
            gains = np.full(len(joints), kp)
            if base_i is not None:
                gains[base_i] = base_kp
            feedforward = moving * planned_velocity
            room = catch_up_fraction * np.abs(feedforward) + catch_up_floor
            correction = np.clip(gains * (desired - actual), -room, room)
            commands = np.clip(feedforward + correction, -max_velocity, max_velocity)

            stopped_wrist = self._watch_wrists(joints, wrists, commands, overspeed_since, now)
            if stopped_wrist is not None:
                return self._abort(goal_handle, FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED,
                                   f"{stopped_wrist} ran faster than commanded; quick-stopped it")

            with self._server_commands():
                for i, joint in enumerate(joints):
                    if i == base_i:
                        self._set_base_rotation_rate(commands[i])
                    else:
                        self.driver.check_and_set_vel(joint, float(commands[i]), a=float(max_acceleration[i])
                                                      if np.isfinite(max_acceleration[i]) else None)
            self._publish(goal_handle, joints, tau, scale, target, check, now, desired, actual, commands, start,
                          gate.state, contact)
            rate.sleep()

        # At the end of the path: settle on the last waypoint as the default server does.
        last_point = JointTrajectoryPoint(positions=list(plan.positions[-1]))
        final = JointTrajectory(joint_names=joints, points=[last_point])
        result = self._converge_on_final_point(final, goal_handle, rate, base_i)
        if result is None and base_i is not None:
            result = self._settle_base_rotation(plan.positions[-1][base_i], goal_handle, rate)
        if result is not None:
            return result
        goal_handle.succeed()
        result = FollowJointTrajectory.Result()
        result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
        return result

    def _detect_contact(self, detectors, joints, commands, now):
        """(joint index, effort rise) of the first joint in contact, or None."""
        for i, detector in detectors.items():
            effort = self._joint_effort(joints[i])
            if effort is None:
                continue
            rise = detector.update(now, effort, commands[i])
            if rise is not None:
                return i, rise
        return None

    def _joint_effort(self, joint):
        """A command joint's effort from the latest joint state (the arm's segments each carry
        the arm's), or None."""
        state = self.driver.last_known_state
        if state is None:
            return None
        for name in getattr(self.driver, "SPLIT_JOINTS", {}).get(joint, [joint]):
            if name in state.name:
                k = state.name.index(name)
                return float(state.effort[k]) if k < len(state.effort) else None
        return None

    def _back_off(self, joints, i, command, lever, goal_handle, rate):
        """After contact: stop every joint, then move joint i back the way it came by
        back_off_distance at back_off_speed (m at the contact point; lever m per unit of the
        joint), and hold. Returns how far it went (joint units)."""
        direction = -np.sign(command) if command else 0.0
        speed = self.get_param("back_off_speed") / lever
        distance = self.get_param("back_off_distance") / lever
        self.driver.get_logger().warn(f"Contact on {joints[i]}: backing off {distance:.3f} at {speed:.3f}/s")
        start = self._joint_position(joints[i])
        deadline = time.monotonic() + 2.0 * distance / max(speed, 1e-6) + 1.0
        moved = 0.0
        while direction and moved < distance and time.monotonic() < deadline:
            if self._check_for_interrupt(goal_handle) is not None:
                break
            with self._server_commands():
                for k, joint in enumerate(joints):
                    self._command_velocity(joint, direction * speed if k == i else 0.0)
            rate.sleep()
            moved = abs(self._joint_position(joints[i]) - start)
        with self._server_commands():
            for joint in joints:
                self._command_velocity(joint, 0.0)
        return moved

    def _command_velocity(self, joint, velocity):
        if joint == self.BASE_ROTATION_JOINT:
            self._set_base_rotation_rate(velocity)
        else:
            self.driver.check_and_set_vel(joint, float(velocity))

    def _watch_wrists(self, joints, wrists, commands, overspeed_since, now):
        """Quick-stop a wrist that's been running well over its command (a stale servo speed);
        returns its name, or None."""
        factor, floor = self.get_param("wrist_overspeed.factor"), self.get_param("wrist_overspeed.floor")
        for i in wrists:
            speed = self._joint_velocity(joints[i]) or 0.0
            if abs(speed) > factor * abs(commands[i]) + floor:
                overspeed_since[i] = overspeed_since[i] or now
                if now - overspeed_since[i] > self.get_param("wrist_overspeed.time"):
                    with self._server_commands():
                        self.driver.check_and_set_vel(joints[i], 0.0)  # exactly 0: quick_stop
                    return joints[i]
            else:
                overspeed_since[i] = None
        return None

    def _publish(self, goal_handle, joints, tau, scale, target, check, now, desired, actual, commands, start, gate,
                 contact=0.0):
        feedback = FollowJointTrajectory.Feedback(joint_names=joints)
        feedback.desired = JointTrajectoryPoint(positions=list(desired), time_from_start=duration_msg(tau))
        feedback.actual = JointTrajectoryPoint(positions=list(actual), time_from_start=duration_msg(
            (self.driver.get_clock().now() - start).nanoseconds * 1e-9))
        goal_handle.publish_feedback(feedback)
        stamp = self.driver.get_clock().now().to_msg()
        self.pub_desired_state.publish(JointState(name=joints, position=list(desired)))
        self.pub_actual_state.publish(JointState(name=joints, position=list(actual)))
        self.pub_commands.publish(JointState(name=joints, velocity=[float(c) for c in commands]))
        # tau and the scale, as a JointState so it plots with the rest; gate is the ResumeGate
        # state (0 free, 1 stopped, 2 creeping); contact the last contact's effort rise (0: none)
        self._pub_timing.publish(JointState(
            header=Header(stamp=stamp), name=["tau", "scale", "target_scale", "check_age", "gate", "contact"],
            position=[tau, scale, target, (now - check.stamp) if check is not None else -1.0, float(gate),
                      float(contact)]))

    # --- checks ----------------------------------------------------------------------------------

    def _check_loop(self, plan, joints, stop):
        """Ask check_motion about the motion ahead, check_rate times a second, until stopped."""
        period = 1.0 / self.get_param("check_rate")
        while not stop.is_set():
            started = time.monotonic()
            try:
                result = self._check_ahead(plan, joints)
                if result is not None:
                    with self._check_lock:
                        self._check = result
                    self._update_sensitivity(result)
            except Exception as e:
                self.driver.get_logger().warn(f"check_motion failed: {e}", throttle_duration_sec=2.0)
            stop.wait(max(0.0, period - (time.monotonic() - started)))

    def _check_ahead(self, plan, joints):
        """One check of the next check_horizon s of the plan; a CheckResult, or None if there
        was no good answer in time."""
        tau, actual = self._state
        horizon, step = self.get_param("check_horizon"), self.get_param("check_step")
        end = min(plan.duration, tau + horizon)
        taus = (check_times(tau, end, step, self.get_param("creep_check_step"),
                            tau + self.get_param("creep_check_fine_horizon"))
                if self._held else check_times(tau, end, step))
        if end - tau < 1e-6:
            return CheckResult(starts=np.array([tau]), ends=np.array([plan.duration]), caps=np.array([1.0]),
                               stamp=time.monotonic())
        # The first point is where the robot is; the base's rotation is relative to its heading now.
        base_i = joints.index(self.BASE_ROTATION_JOINT) if self.BASE_ROTATION_JOINT in joints else None
        request = CheckMotion.Request(trajectory=JointTrajectory(joint_names=joints))
        first = list(actual)
        if base_i is not None:
            first[base_i] = 0.0
        request.trajectory.points.append(JointTrajectoryPoint(positions=[float(v) for v in first],
                                                              time_from_start=duration_msg(0.0)))
        for t in taus:
            positions = plan.sample(t)[0].copy()
            if base_i is not None:
                positions[base_i] -= actual[base_i]
            request.trajectory.points.append(JointTrajectoryPoint(positions=[float(v) for v in positions],
                                                                  time_from_start=duration_msg(t - tau)))
        stamp = time.monotonic()
        response = self._call(request, timeout=self.get_param("check_timeout"))
        if response is None or not response.success:
            if response is not None:
                self.driver.get_logger().warn(f"check_motion: {response.message}", throttle_duration_sec=2.0)
            return None
        starts = np.array([tau] + taus[:-1])
        ends = np.array(taus)
        caps = segment_caps(ends - starts, response.min_duration, response.blocked)
        self._publish_check(response, tau, starts, caps, time.monotonic() - stamp)
        return CheckResult(starts=starts, ends=ends, caps=caps, unsafe=np.array(response.unsafe, dtype=bool),
                           stamp=stamp)

    def _publish_check(self, response, tau, starts, caps, round_trip):
        """Why the motion ahead is limited: the most limiting segment (lowest cap) and the first
        blocked one, with their shapes and clearances."""
        def segment(prefix, i):
            return [KeyValue(key=f"{prefix}_{name}", value=value) for name, value in [
                ("segment", str(i)), ("tau", f"{starts[i]:.3f}"), ("cap", f"{caps[i]:.3f}"),
                ("shape", response.limiting_shape[i]), ("speed", f"{response.limiting_speed[i]:.3f}"),
                ("clearance_occupied", f"{response.clearance_occupied[i]:.3f}"),
                ("clearance_moving", f"{response.clearance_moving[i]:.3f}"),
                ("clearance_unsafe", f"{response.clearance_unsafe[i]:.3f}")]]

        blocked = [i for i, b in enumerate(response.blocked) if b]
        values = [KeyValue(key="tau", value=f"{tau:.3f}"), KeyValue(key="round_trip", value=f"{round_trip:.4f}"),
                  KeyValue(key="grid_age", value=f"{response.grid_age:.3f}"),
                  KeyValue(key="blocked", value=" ".join(map(str, blocked))),
                  KeyValue(key="unsafe", value=" ".join(str(i) for i, u in enumerate(response.unsafe) if u))]
        values += segment("limiting", int(np.argmin(caps)))
        if blocked:
            values += segment("first_blocked", blocked[0])
        self._pub_check.publish(DiagnosticArray(
            header=Header(stamp=self.driver.get_clock().now().to_msg()),
            status=[DiagnosticStatus(name="dynamic_trajectory_server check", values=values)]))

    def _call(self, request, timeout):
        done = threading.Event()
        future = self._check_client.call_async(request)
        future.add_done_callback(lambda _: done.set())
        if not done.wait(timeout):
            future.cancel()
            return None
        return future.result()

    def _update_sensitivity(self, check):
        if not self.get_param("switch_sensitivity"):
            return
        tau = self._state[0]
        soon = (check.starts < tau + self.get_param("sensitivity_lookahead")) & (check.ends > tau)
        unsafe = bool(np.any(check.unsafe[soon])) if len(check.unsafe) == len(check.starts) else False
        self._set_sensitivity(self.get_param("unsafe_sensitivity" if unsafe else "safe_sensitivity"))

    def _set_sensitivity(self, value):
        if not self.get_param("switch_sensitivity") or value == self._sensitivity:
            return
        result = self.driver.set_parameters([Parameter("sensitivity", value=value)])[0]
        if result.successful:
            self._sensitivity = value
            self.driver.get_logger().info(f"Guarded-contact sensitivity: {value}")
        else:
            self.driver.get_logger().warn(f"Couldn't set sensitivity to {value}: {result.reason}",
                                          throttle_duration_sec=5.0)
