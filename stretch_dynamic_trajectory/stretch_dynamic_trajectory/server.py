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
at the goal's start and on every switch to a detour. Its base rotation is measured from the
base's heading at its stamp, when it was published (not the goal's start: a reader that needs
the heading at a stamp minutes old may not have it). The obstacle monitor draws it in RViz. Joints are commanded in velocity mode, like
the default server's adaptive_velocity mode: the scaled planned velocity plus kp times the
position error, with the joint's maximum acceleration so the firmware follows each command.

While the motion just ahead sweeps unsafe space (unknown, or contaminated shadow) the
driver's guarded-contact sensitivity is raised (unsafe_sensitivity), and set back
(safe_sensitivity) once it's clear. A wrist joint running well over its commanded speed (the
Feetech servos sometimes keep a stale speed) is quick-stopped and the goal aborted.

A rise in a moving joint's effort over its free-motion level (contact.py) means contact: the
robot stops, and the joint that touched backs off back_off_distance along its own axis at
back_off_speed (both m at the contact point). A held arm or lift joint that's pushed (its
effort beyond its holding band; contact.py) yields: it moves with the push, as fast as the
push says up to yield_max_speed (0.02 m/s), until the push eases, it runs into something
else, or it has gone yield_max_distance (0.1 m) (Elaine, 2026-10-23: not holding firm
until the firmware guard trips and leaves it limp). After a contact or a push the plan holds
for contact_hold s, still yielding to pushes, then contact_action: "abort" ends the goal,
reporting what happened; "resume" goes back along the plan to where the robot is and carries
on (into the same thing again, if it's still there), ending the goal after
contact_resume_limit contacts and pushes in it.

Replanning (replan, on by default; needs the planner node): while the checks show a block
ahead, or the robot is on a detour, the server asks the planner node's replan_around, at most
every replan_period s, one at a time, for detours leaving the plan replan_lookahead s ahead (at
the current speed) and rejoining the goal's trajectory at several points: from rejoin_margin s
past the blocked stretch (or, on a detour, from just past where it left the goal's trajectory,
so going straight back is one option), a second apart. The planner plans each and predicts
its time to the goal on one snapshot of the obstacle grid. Which one the server takes, and
when, depends on replan_mode:
- "match_path" (default): the goal's path matters, not just where it ends (Elaine, 2026-10-23,
  after a detour on a two-loop circle went straight to the goal, which was where the circle
  started; "we don't want to turn our circle into a polygon by cutting off the curves"). The
  detour that strays least from the goal's path (replan.path_deviation: points matched by
  progress along each path, not by time), if it strays match_margin m less than the plan being
  followed: so a clear plan is never swapped for a shortcut, and a detour goes back to the
  path as soon as it can. The goal is a rejoin point only if it's within those few seconds.
- "reach_goal": only the goal matters. The fastest detour, which may rejoin at the goal
  itself, if it's switch_margin s faster than the plan being followed.
Either way it switches whenever the plan being followed is blocked. replan.py splices the current plan up to the leaving point, the detour and the
rest of the goal's trajectory, starting at the current velocity, and the new plan is
published. Detours move only the goal's joints, unless replan_joints is "all": then the server
takes over every joint (the others held where they are, in velocity mode; base rotation only
with allow_rotation) and a detour may move them, back where they were by the time it rejoins.
Without a detour the robot stops and creeps as before. blocked_timeout (s, 0 for none) ends a
goal that stays blocked that long.

Not yet: dodging while holding a position.
"""

import math
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
from stretch_motion_planning_interfaces.srv import CheckMotion, ReplanAround

from stretch_dynamic_trajectory.contact import (CONTACT_SETTINGS, YIELD_SETTINGS, ContactDetector, ContactSettings,
                                                Yielding)
from stretch_dynamic_trajectory.replan import (FollowedPlan, blocked_run, detour_start, left_at, path_deviation,
                                              rejoin_candidates, splice)
from stretch_dynamic_trajectory.timing import (CheckResult, ResumeGate, Trajectory, blocked_anywhere, check_times,
                                               creeping_limits, rewind, scale_deceleration, segment_caps, step_scale,
                                               stopping_for_block, target_scale)

WRIST_JOINTS = ("wrist_yaw_joint", "wrist_pitch_joint", "wrist_roll_joint")
STILL_SPEED = 1e-3  # planned joint speed (units/s) under which a joint counts as held
# The joints detours may move with replan_joints "all" (the planner's).
REPLAN_JOINTS = ("mobile_base_rotation_joint", "lift_joint", "arm_joint", "wrist_yaw_joint", "wrist_pitch_joint",
                 "wrist_roll_joint")
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
        self._state = None  # (FollowedPlan, tau, positions) shared with the check thread
        self._check_now = threading.Event()  # wakes the check thread (a new plan to check)
        self._held = False  # whether the resume gate holds the robot, for the check thread
        self._sensitivity = None
        super().__init__(driver, action_name="follow_joint_trajectory_dynamic",
                         param_prefix="dynamic_trajectory_server",
                         diagnostics_namespace="dynamic_trajectory_server_diagnostics")
        self._check_client = driver.create_client(CheckMotion, "check_motion",
                                                  callback_group=MutuallyExclusiveCallbackGroup())
        self._replan_client = driver.create_client(ReplanAround, "replan_around",
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
        # back_off_speed (m/s), hold contact_hold s (yielding to pushes), then contact_action:
        # "abort" or "resume". contact_threshold.<joint>: effort points over the free level.
        add_param("contact_detection", True)
        for joint, settings in CONTACT_SETTINGS.items():
            add_param(f"contact_threshold.{joint}", settings.threshold)
        # Multiplies the contact thresholds: below 1 stops on a lighter touch.
        add_param("contact_threshold_scale", 1.0)
        add_param("contact_action", "abort")
        add_param("back_off_distance", 0.02)
        add_param("back_off_speed", 0.02)
        add_param("contact_hold", 2.0)
        add_param("contact_resume_limit", 3)
        # Replanning (see the module docstring): requests at most every replan_period s, detours
        # leaving replan_lookahead s of trajectory (at the current speed) ahead, rejoining from
        # rejoin_margin s past the block, planned within replan_timeout s; switch_margin s
        # faster than the current plan to switch. replan_joints: "goal" or "all".
        # replan_mode: "match_path" (keep to the goal's path; switch for match_margin m less
        # deviation) or "reach_goal" (only its end matters; switch for switch_margin s).
        add_param("replan", True)
        add_param("replan_period", 0.5)
        add_param("replan_lookahead", 0.6)
        add_param("switch_margin", 0.5)
        add_param("replan_joints", "goal")
        add_param("replan_mode", "match_path")
        add_param("match_margin", 0.01)
        add_param("rejoin_margin", 0.5)
        add_param("replan_timeout", 1.0)
        add_param("blocked_timeout", 0.0)
        # Yielding to a push on a held arm or lift: yield_threshold.<joint> is effort points
        # beyond its holding band; it moves with the push at up to yield_max_speed (m/s).
        add_param("yield_to_pushes", True)
        for joint, settings in YIELD_SETTINGS.items():
            add_param(f"yield_threshold.{joint}", settings.threshold)
        add_param("yield_max_speed", 0.02)
        # A yield ends after this far (m at the contact point), whatever the effort says.
        add_param("yield_max_distance", 0.1)
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
        if self.get_param("replan") and self.get_param("replan_joints") == "all":
            trajectory = self._with_all_joints(trajectory)
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
            plan = FollowedPlan(self._plan(trajectory))
            self._publish_plan(plan.trajectory, joints)
            with self._check_lock:
                self._check = None
            self._state = (plan, 0.0, self._positions(joints))
            self._held = False
            checker = threading.Thread(target=self._check_loop, args=(joints, stop_checks), daemon=True)
            checker.start()
            result = self._follow(plan, trajectory, goal_handle)
        except Exception as e:
            self.driver.get_logger().error(f"Dynamic trajectory failed: {e}")
            result = self._abort(goal_handle, FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED, f"Error {e}")
        finally:
            stop_checks.set()
            self._check_now.set()
            if checker is not None:
                checker.join(timeout=1.0)
            self.stop_vel_joints(joints)
            self.active_joints = None
            self._set_sensitivity(self.get_param("safe_sensitivity"))
        return result

    def _with_all_joints(self, trajectory):
        """The goal with every joint a detour may move added, held where it is now (base
        rotation only if allowed)."""
        extra = [j for j in REPLAN_JOINTS if j not in trajectory.joint_names
                 and (j != self.BASE_ROTATION_JOINT or self.get_param("allow_rotation"))]
        if not extra:
            return trajectory
        if self.BASE_ROTATION_JOINT in extra:
            self._start_base_rotation()
        held = [float(self._joint_position(j)) for j in extra]
        out = JointTrajectory(header=trajectory.header, joint_names=list(trajectory.joint_names) + extra)
        for point in trajectory.points:
            out.points.append(JointTrajectoryPoint(
                positions=list(point.positions) + held,
                velocities=list(point.velocities) + [0.0] * len(extra) if point.velocities else [],
                time_from_start=point.time_from_start))
        return out

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
        alone would miss the curves between them), stamped now, with the base's rotation from
        its heading now (the plan's is from the goal's start)."""
        times = np.linspace(0.0, plan.duration, int(np.clip(np.ceil(plan.duration / step), 1, max_points - 1)) + 1)
        offset = np.zeros(len(joints))
        if self.BASE_ROTATION_JOINT in joints:
            offset[joints.index(self.BASE_ROTATION_JOINT)] = self._joint_position(self.BASE_ROTATION_JOINT)
        msg = JointTrajectory(header=Header(stamp=self.driver.get_clock().now().to_msg()), joint_names=joints)
        msg.points = [JointTrajectoryPoint(positions=[float(v) for v in plan.sample(t)[0] - offset],
                                           time_from_start=duration_msg(t)) for t in times]
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

    def _follow(self, followed, trajectory, goal_handle):
        joints = list(trajectory.joint_names)
        reference, plan = followed.trajectory, followed.trajectory
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
        detectors, yielders = self._contact_watchers(joints, levers)
        commands = np.zeros(len(joints))  # last loop's, which the efforts now respond to
        feedforward = np.zeros(len(joints))  # last loop's planned motion: zero for held joints
        hold_until = None  # the plan holds until then after a contact or a push
        pending = None  # (joint index, what happened) after a contact or a push, until hold_until
        yielding = None  # the joint index moving with a push
        yield_from = 0.0  # where it started
        contact = 0.0  # the last contact's effort rise, for the timing topic
        contacts = 0  # contacts and pushes in this goal
        catch_up_fraction, catch_up_floor = self.get_param("catch_up_fraction"), self.get_param("catch_up_floor")
        replan, last_replan = None, -np.inf  # the detour request in flight, and when the last went out
        blocked_since = None

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
            if check is not None and (now - check.stamp > check_timeout or check.plan_id != followed.id):
                check = None  # stale, or of a plan left behind: slow to a stop until a fresh one

            # A detour from the planner, if one's ready: switch to it.
            if replan is not None and replan.done.is_set():
                detour = self._best_detour(replan, followed, reference, tau, joints)
                replan = None
                if detour is not None:
                    followed, plan, tau = detour, detour.trajectory, 0.0
                    self._state = (followed, tau, self._positions(joints))
                    gate = ResumeGate(self.get_param("resume_clear_checks"), self.get_param("resume_delay"))
                    self._held, check, blocked_since = False, None, None
                    self._publish_plan(plan, joints)
                    self._check_now.set()
            horizon_end = min(plan.duration, tau + self.get_param("check_horizon"))
            rate_limit = scale_deceleration(plan, tau, horizon_end, max_acceleration, max_rate)
            target = target_scale(tau, check, rate_limit, plan.duration)
            if check is not None and check.stamp != gated_stamp:
                gated_stamp = check.stamp
                gate.update(check.stamp, stopping_for_block(check, target), blocked_anywhere(check))
                self._held = gate.held
                # Blocked ahead, or on a detour: ask for detours (at most every replan_period s,
                # one request at a time).
                on_detour = followed.is_detour and tau < followed.detour_end
                if ((blocked_anywhere(check) or on_detour) and replan is None and self.get_param("replan")
                        and now - last_replan >= self.get_param("replan_period")):
                    replan = self._request_detours(followed, reference, check, tau, scale, joints)
                    last_replan = now
            blocked_since = (blocked_since or now) if gate.held else None
            blocked_timeout = self.get_param("blocked_timeout")
            if blocked_timeout > 0.0 and blocked_since is not None and now - blocked_since > blocked_timeout:
                return self._abort(goal_handle, FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED,
                                   f"blocked for over {blocked_timeout:.1f} s")
            up_rate = max_up_rate
            if gate.state == ResumeGate.STOPPED or (hold_until is not None and now < hold_until):
                target = 0.0
            elif gate.state == ResumeGate.CREEPING:
                creep_target, creep_rate = creeping_limits(plan, tau, horizon_end, check, creep_speed,
                                                           creep_acceleration, levers, creep_standoff)
                target, up_rate = min(target, creep_target), min(up_rate, creep_rate)
            scale = step_scale(scale, target, rate_limit, dt, up_rate=up_rate)

            actual = self._positions(joints)
            efforts = {i: self._joint_effort(joints[i]) for i in set(detectors) | set(yielders)}

            # Moving with a push: nothing else moves until it eases.
            if yielding is not None:
                effort = efforts.get(yielding)
                velocity = yielders[yielding].update(now, effort) if effort is not None else 0.0
                if abs(actual[yielding] - yield_from) >= self.get_param("yield_max_distance") / levers[yielding]:
                    self.driver.get_logger().warn(f"{joints[yielding]}: yielded yield_max_distance; holding")
                    yielders[yielding].stop()
                    velocity = 0.0
                commands = np.zeros(len(joints))
                commands[yielding] = velocity
                with self._server_commands():
                    for k, joint in enumerate(joints):
                        self._command_velocity(joint, commands[k])
                if not yielders[yielding].active:
                    self.driver.get_logger().info(f"{joints[yielding]}: the push eased; holding")
                    yielding, hold_until = None, now + self.get_param("contact_hold")
                scale = 0.0
                self._publish(goal_handle, joints, tau, scale, 0.0, check, now, plan.sample(tau)[0], actual,
                              commands, start, gate.state, contact)
                rate.sleep()
                continue

            # Running into something: back off, then hold (yielding to pushes).
            touched = self._detect_contact(detectors, commands, efforts, now)
            if touched is not None:
                i, rise = touched
                contact, contacts = rise, contacts + 1
                backed_off = self._back_off(joints, i, commands[i], levers[i], goal_handle, rate)
                pending = (i, f"contact on {joints[i]} (effort {rise:.0f} over free motion); backed off {backed_off:.3f}")
                scale, hold_until = 0.0, time.monotonic() + self.get_param("contact_hold")
                commands, feedforward = np.zeros(len(joints)), np.zeros(len(joints))
                last = time.monotonic()
                continue

            # Pushed while held: yield.
            for i, yielder in yielders.items():
                if abs(feedforward[i]) < STILL_SPEED and efforts.get(i) is not None and yielder.watch(now, efforts[i]):
                    yielding, yield_from = i, actual[i]
                    self.driver.get_logger().warn(f"{joints[i]} pushed (effort {efforts[i]:.0f}): yielding")
                    if pending is None:
                        contacts += 1
                        pending = (i, f"{joints[i]} pushed")
                    break
            if yielding is not None:
                continue

            # After a contact or a push: hold still where it is (not back toward the plan, which
            # would lead back into whatever it was) for contact_hold s, then end the goal or
            # carry on.
            if pending is not None and now < hold_until:
                commands = np.zeros(len(joints))
                with self._server_commands():
                    for joint in joints:
                        self._command_velocity(joint, 0.0)
                scale = 0.0
                self._publish(goal_handle, joints, tau, scale, 0.0, check, now, plan.sample(tau)[0], actual,
                              commands, start, gate.state, contact)
                rate.sleep()
                continue
            if pending is not None:
                i, what = pending
                pending = None
                if self.get_param("contact_action") != "resume" or contacts >= self.get_param("contact_resume_limit"):
                    return self._abort(goal_handle, FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED, what)
                tau = rewind(plan, tau, i, actual[i])

            # Advance along the plan more slowly the further the robot lags behind it, and not
            # at all past max_lag tolerances.
            desired, _, _ = plan.sample(tau)
            lag = np.max(np.abs(desired - actual) / tolerances) / max_lag
            moving = scale * float(np.clip(1.0 - lag, 0.0, 1.0))
            tau = min(plan.duration, tau + moving * dt)
            self._state = (followed, tau, actual)
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

    # --- replanning ------------------------------------------------------------------------------

    def _request_detours(self, followed, reference, check, tau, scale, joints):
        """Ask the planner, in the background, for detours from a little ahead back onto the
        goal's trajectory; a _Replan to wait on, or None if there's no planner."""
        if not self._replan_client.service_is_ready():
            self.driver.get_logger().warn("No replan_around service (planner node): can't replan around blocks",
                                          throttle_duration_sec=30.0)
            return None
        plan = followed.trajectory
        block = blocked_run(check) if blocked_anywhere(check) else None
        horizon = block[0] if block is not None else plan.duration
        leave = detour_start(tau, scale, self.get_param("replan_lookahead"), horizon)
        match_path = self._match_path()
        rejoins = rejoin_candidates(followed, leave, reference.duration, self.get_param("rejoin_margin"), block,
                                    match_path=match_path)
        # The planner's base rotation is from the heading now; the goal's, from its start.
        base_i = joints.index(self.BASE_ROTATION_JOINT) if self.BASE_ROTATION_JOINT in joints else None
        offset = np.zeros(len(joints))
        if base_i is not None:
            offset[base_i] = self._joint_position(self.BASE_ROTATION_JOINT)
        start = plan.sample(leave)[0]
        request = ReplanAround.Request(joint_names=joints, start=[float(v) for v in start - offset],
                                       rejoin_times=[float(t) for t in rejoins],
                                       timeout=self.get_param("replan_timeout"),
                                       # All of them, to compare paths; else the node's default.
                                       max_detours=len(rejoins) if match_path else 0)
        request.current = self._trajectory_msg(plan, joints, leave, plan.duration, offset)
        request.reference = self._trajectory_msg(reference, joints, 0.0, reference.duration, offset, relative=False)
        replan = _Replan(followed.id, leave, None, offset, start)
        future = self._replan_client.call_async(request)
        future.add_done_callback(lambda f: replan.finish(f.result()))
        return replan

    @staticmethod
    def _trajectory_msg(trajectory, joints, t0, t1, offset, step=0.25, relative=True):
        """A trajectory sampled every `step` s from t0 to t1 (and its waypoints between), minus
        `offset`; times from t0 if relative, else as they are."""
        times = sorted(set(np.r_[np.arange(t0, t1, step), [t1], trajectory.times[(trajectory.times > t0) &
                                                                                (trajectory.times < t1)]]))
        return JointTrajectory(joint_names=joints, points=[
            JointTrajectoryPoint(positions=[float(v) for v in trajectory.sample(t)[0] - offset],
                                 time_from_start=duration_msg(t - t0 if relative else t)) for t in times])

    def _match_path(self):
        mode = self.get_param("replan_mode")
        if mode not in ("match_path", "reach_goal"):
            self.driver.get_logger().warn(f"replan_mode '{mode}' isn't match_path or reach_goal: using match_path",
                                          throttle_duration_sec=30.0)
        return mode != "reach_goal"

    def _best_detour(self, replan, followed, reference, tau, joints):
        """The FollowedPlan to switch to from the planner's answer, or None (also for an answer
        about a plan already left, or when the robot is past where the detours leave).

        match_path: the detour that strays least from the goal's path (replan.path_deviation,
        every option, the plan being followed too, measured from where the detours leave to the
        latest rejoin among them), if it strays match_margin m less than the plan being followed
        (always, if that's blocked). So a clear plan is never swapped for a shortcut, and a
        detour goes back to the goal's path as soon as it can.
        reach_goal: the fastest detour (predicted time to the goal), if it's switch_margin s
        faster than the plan being followed (always, if that's blocked)."""
        response = replan.response
        if response is None or not response.success or not response.detours:
            self.driver.get_logger().info(f"No detour: {response.message if response else 'no answer'}",
                                          throttle_duration_sec=2.0)
            return None
        if replan.plan_id != followed.id or tau > replan.leave + 1e-6:
            return None
        left = left_at(followed, replan.leave)
        detours = [self._detour_plan(response, k, replan, followed, reference, tau, joints, left)
                   for k in range(len(response.detours))]
        blocked = not np.isfinite(response.current_time)
        if self._match_path():
            weights = np.array([CREEP_LEVERS.get(j, 1.0) for j in joints])
            until = max([*response.detour_rejoin_times, followed.rejoin if followed.is_detour else left])
            deviations = [path_deviation(d, replan.leave - tau, reference, left, until, weights) for d in detours]
            current = math.inf if blocked else path_deviation(followed, replan.leave, reference, left, until, weights)
            k = int(np.argmin(deviations))
            if not blocked and deviations[k] > current - self.get_param("match_margin"):
                return None  # the plan being followed matches the path about as well
            why = f"strays {deviations[k] * 100:.1f} cm from the goal's path against {current * 100:.1f} cm"
        else:
            k = 0  # the planner lists them fastest first
            if response.detour_times[k] > response.current_time - self.get_param("switch_margin"):
                return None  # the plan being followed is about as fast
            why = f"predicted {response.detour_times[k]:.1f} s to the goal against {response.current_time:.1f} s"
        self.driver.get_logger().info(
            f"Detour ({response.detour_methods[k]}, {len(response.detours)} found in "
            f"{response.planning_time * 1e3:.0f} ms): {why}; rejoins the goal's trajectory at "
            f"{response.detour_rejoin_times[k]:.2f} s")
        return detours[k]

    def _detour_plan(self, response, k, replan, followed, reference, tau, joints, left):
        """The FollowedPlan that takes the planner's detour k: the plan being followed to where
        the detours leave it, the detour, then the rest of the goal's trajectory."""
        trajectory = response.detours[k]
        names = list(trajectory.joint_names)
        times, positions = [], []
        for point in trajectory.points:
            q = replan.start.copy()
            for i, joint in enumerate(joints):
                if joint in names:
                    q[i] = point.positions[names.index(joint)] + replan.offset[i]
            times.append(self._msg_seconds(point.time_from_start))
            positions.append(q)
        return splice(followed.trajectory, tau, replan.leave, times, positions, reference,
                      response.detour_rejoin_times[k], left=left)

    def _contact_watchers(self, joints, levers):
        """({joint index: ContactDetector}, {joint index: Yielding}) for this goal's joints, as
        the parameters say."""
        detectors, yielders = {}, {}
        if self.get_param("contact_detection"):
            scale = self.get_param("contact_threshold_scale")
            for i, joint in enumerate(joints):
                if joint in CONTACT_SETTINGS:
                    threshold = self.get_param(f"contact_threshold.{joint}")
                    settings = ContactSettings(threshold, None, CONTACT_SETTINGS[joint].signed,
                                               CONTACT_SETTINGS[joint].start_transient)
                    detectors[i] = ContactDetector(settings, threshold_scale=scale)
        if self.get_param("yield_to_pushes"):
            for i, joint in enumerate(joints):
                if joint in YIELD_SETTINGS:
                    yielders[i] = Yielding(YIELD_SETTINGS[joint], self.get_param("yield_max_speed") / levers[i],
                                           self.get_param(f"yield_threshold.{joint}"))
        return detectors, yielders

    def _detect_contact(self, detectors, commands, efforts, now):
        """(joint index, effort rise) of the first joint in contact, or None."""
        for i, detector in detectors.items():
            if efforts.get(i) is None:
                continue
            rise = detector.update(now, efforts[i], commands[i])
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

    def _check_loop(self, joints, stop):
        """Ask check_motion about the motion ahead, check_rate times a second (and at once when
        there's a new plan), until stopped."""
        period = 1.0 / self.get_param("check_rate")
        while not stop.is_set():
            started = time.monotonic()
            self._check_now.clear()
            try:
                result = self._check_ahead(joints)
                if result is not None:
                    with self._check_lock:
                        self._check = result
                    self._update_sensitivity(result)
            except Exception as e:
                self.driver.get_logger().warn(f"check_motion failed: {e}", throttle_duration_sec=2.0)
            self._check_now.wait(max(0.0, period - (time.monotonic() - started)))

    def _check_ahead(self, joints):
        """One check of the next check_horizon s of the plan; a CheckResult, or None if there
        was no good answer in time."""
        followed, tau, actual = self._state
        plan = followed.trajectory
        horizon, step = self.get_param("check_horizon"), self.get_param("check_step")
        end = min(plan.duration, tau + horizon)
        taus = (check_times(tau, end, step, self.get_param("creep_check_step"),
                            tau + self.get_param("creep_check_fine_horizon"))
                if self._held else check_times(tau, end, step))
        if end - tau < 1e-6:
            return CheckResult(starts=np.array([tau]), ends=np.array([plan.duration]), caps=np.array([1.0]),
                               stamp=time.monotonic(), plan_id=followed.id)
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
                           stamp=stamp, plan_id=followed.id)

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
        tau = self._state[1]
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


class _Replan:
    """A detour request in flight: what was asked (the plan it's for, where it leaves that plan,
    where it rejoins the goal's trajectory, the base rotation added back to the answer, and the
    configuration it leaves from), and once done, the planner's response."""

    def __init__(self, plan_id, leave, rejoin, offset, start):
        self.plan_id, self.leave, self.rejoin, self.offset = plan_id, leave, rejoin, offset
        self.start = np.asarray(start, dtype=float)  # where it leaves the plan (the goal's joints)
        self.response = None
        self.done = threading.Event()

    def finish(self, response):
        self.response = response
        self.done.set()

