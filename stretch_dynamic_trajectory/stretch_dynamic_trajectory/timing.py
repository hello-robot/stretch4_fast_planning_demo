"""Following a joint trajectory at a variable speed, without ROS.

The trajectory keeps its own clock, tau: trajectory time, in the seconds of the goal's
time_from_start. tau advances at `scale` times real time, with scale between 0 and 1, so the
path never changes, only how fast the robot moves along it. Slowing down or stopping freezes
tau, so when the robot can go again it picks up where it is, without the desired state having
run ahead and the robot rushing to catch up.

The scale comes from obstacle checks of the motion ahead: each checked segment of trajectory
time has a cap (planned time / the least time the check allows; 0 if the segment is blocked).
The scale must be down to each cap by the time tau reaches that segment, braking no harder
than the joints can, and it never runs past the end of what's been checked. Once the checks
have stopped the robot for a blocked path, it stays stopped for a moment, then may creep
toward the block at a low speed while the path just ahead is clear, and goes back to full
speed only once several checks in a row see nothing blocked (ResumeGate); then it speeds up
gently.
"""

from dataclasses import dataclass, field

import numpy as np


class Trajectory:
    """Cubic Hermite interpolation through waypoints by trajectory time.

    times (n,), positions (n, joints); velocities (n, joints) or None to infer them: the
    average of the neighboring segments' slopes inside, zero at both ends, each clipped to
    3 times the slope of the segments next to it (as the driver's adaptive_velocity mode).
    """

    def __init__(self, times, positions, velocities=None):
        self.times = np.asarray(times, dtype=float)
        self.positions = np.asarray(positions, dtype=float)
        if len(self.times) < 2 or np.any(np.diff(self.times) <= 0.0):
            raise ValueError("a trajectory needs at least two waypoints at increasing times")
        slopes = np.diff(self.positions, axis=0) / np.diff(self.times)[:, None]
        if velocities is None:
            velocities = np.zeros_like(self.positions)
            velocities[1:-1] = 0.5 * (slopes[:-1] + slopes[1:])
            limit = 3.0 * np.minimum(np.abs(slopes[:-1]), np.abs(slopes[1:]))
            velocities[1:-1] = np.clip(velocities[1:-1], -limit, limit)
        self.velocities = np.asarray(velocities, dtype=float)

    @property
    def duration(self):
        return float(self.times[-1])

    def sample(self, tau):
        """(position, velocity, acceleration) per joint at trajectory time tau (clamped to the
        trajectory), in trajectory time."""
        tau = float(np.clip(tau, self.times[0], self.times[-1]))
        i = int(np.clip(np.searchsorted(self.times, tau, side="right") - 1, 0, len(self.times) - 2))
        t0, t1 = self.times[i], self.times[i + 1]
        h = t1 - t0
        s = (tau - t0) / h
        p0, p1 = self.positions[i], self.positions[i + 1]
        m0, m1 = self.velocities[i] * h, self.velocities[i + 1] * h
        h00, h10, h01, h11 = 2 * s**3 - 3 * s**2 + 1, s**3 - 2 * s**2 + s, -2 * s**3 + 3 * s**2, s**3 - s**2
        d00, d10, d01, d11 = 6 * s**2 - 6 * s, 3 * s**2 - 4 * s + 1, -6 * s**2 + 6 * s, 3 * s**2 - 2 * s
        a00, a10, a01, a11 = 12 * s - 6, 6 * s - 4, -12 * s + 6, 6 * s - 2
        position = h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1
        velocity = (d00 * p0 + d10 * m0 + d01 * p1 + d11 * m1) / h
        acceleration = (a00 * p0 + a10 * m0 + a01 * p1 + a11 * m1) / h**2
        return position, velocity, acceleration

    def max_speed(self, tau0, tau1, samples=20):
        """Largest |velocity| per joint over trajectory time [tau0, tau1]."""
        taus = np.linspace(tau0, tau1, samples)
        return np.max(np.abs([self.sample(t)[1] for t in taus]), axis=0)


@dataclass
class CheckResult:
    """One obstacle check of the motion ahead, in trajectory time.

    starts: each segment's start (tau); ends: its end. caps: the most scale the segment allows.
    unsafe: the segment sweeps unsafe space. stamp: when the check was asked (s, monotonic).
    """
    starts: np.ndarray
    ends: np.ndarray
    caps: np.ndarray
    unsafe: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    stamp: float = 0.0

    @property
    def checked_until(self):
        return float(self.ends[-1]) if len(self.ends) else 0.0


def segment_caps(planned_durations, min_durations, blocked):
    """Scale caps from a check: planned / least allowed time, at most 1; 0 where blocked."""
    planned = np.asarray(planned_durations, dtype=float)
    least = np.asarray(min_durations, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        caps = np.where(least > planned, planned / least, 1.0)
    caps = np.where(np.isfinite(caps), caps, 0.0)
    return np.where(np.asarray(blocked, dtype=bool), 0.0, np.clip(caps, 0.0, 1.0))


def target_scale(tau, check, scale_deceleration, trajectory_end):
    """The most scale allowed now: low enough to slow to every segment's cap by the time tau
    reaches it, and to stop by the end of the checked motion (unless that's the trajectory's
    end), braking at scale_deceleration (scale per s). In trajectory time, slowing from s to c
    at rate a takes (s^2 - c^2) / (2a) of tau."""
    if check is None or len(check.caps) == 0:
        return 0.0
    distances = np.maximum(0.0, check.starts - tau)
    # Segments already passed don't limit anything.
    ahead = check.ends > tau
    limits = np.sqrt(check.caps[ahead] ** 2 + 2.0 * scale_deceleration * distances[ahead])
    scale = float(np.min(limits)) if limits.size else 1.0
    if check.checked_until < trajectory_end - 1e-9:
        scale = min(scale, float(np.sqrt(2.0 * scale_deceleration * max(0.0, check.checked_until - tau))))
    return float(np.clip(scale, 0.0, 1.0))


def scale_deceleration(trajectory, tau0, tau1, joint_decelerations, max_rate):
    """How fast the scale can change (per s) without any joint's acceleration from the change
    exceeding its limit: a joint planned at speed v moves at s v, so changing s at rate r
    accelerates it by r v. At most max_rate."""
    speeds = trajectory.max_speed(tau0, max(tau0, tau1))
    with np.errstate(divide="ignore"):
        rates = np.where(speeds > 1e-6, np.asarray(joint_decelerations) / speeds, np.inf)
    return float(min(max_rate, np.min(rates)))


def step_scale(scale, target, rate, dt, up_rate=None):
    """Move the scale toward target by at most rate * dt (and when speeding up, also at most
    up_rate * dt)."""
    up = rate if up_rate is None else min(rate, up_rate)
    return float(np.clip(target, scale - rate * dt, scale + up * dt))


def stopping_for_block(check, target):
    """Whether a check brings the robot to a stop for a blocked segment (not just slowing for
    one further ahead): the target is 0 and a segment ahead is blocked."""
    return target <= 1e-6 and blocked_anywhere(check)


def blocked_anywhere(check):
    """Whether any segment of a check is blocked."""
    return bool(np.any(check.caps == 0.0))


def check_times(tau, end, step, fine_step=None, fine_until=None):
    """Segment ends (trajectory time) for a check of the motion from tau to end: every step s,
    or every fine_step s up to fine_until first. While creeping toward a block, fine steps
    near the robot say more closely where the block starts, so the robot can brake for it
    gently instead of stopping hard when it turns up in the first segment."""
    fine_end = min(end, fine_until) if fine_step and fine_until is not None and fine_until > tau else tau
    times = list(np.arange(tau + fine_step, fine_end - 1e-6, fine_step)) if fine_end > tau else []
    if fine_end > tau:
        times.append(fine_end)
    times += list(np.arange(fine_end + step, end - 1e-6, step))
    if not times or times[-1] < end - 1e-9:
        times.append(end)
    return [float(t) for t in times]


def creep_scale(trajectory, tau0, tau1, speed, levers):
    """The scale at which no point of the robot moves faster than `speed` (m/s) over trajectory
    time [tau0, tau1], bounding the fastest point's planned speed by each joint's top speed
    times its lever (m per unit of joint motion: 1 for the lift and arm, the robot's reach for
    base rotation, the gripper's length for a wrist joint), summed. At most 1."""
    top = float(np.sum(trajectory.max_speed(tau0, max(tau0, tau1)) * np.asarray(levers)))
    return 1.0 if top <= speed else speed / top


def travel(trajectory, tau0, tau1, levers):
    """The most any point of the robot moves between trajectory times tau0 and tau1 (m),
    bounded as in creep_scale: each joint's change times its lever, summed."""
    q0, q1 = trajectory.sample(tau0)[0], trajectory.sample(tau1)[0]
    return float(np.sum(np.abs(q1 - q0) * np.asarray(levers)))


def with_standoff(trajectory, tau, check, standoff, levers, step=0.01):
    """The check with its first block brought standoff m nearer along the path (at the robot's
    fastest point; walked back in steps of `step` s of trajectory time, but no nearer than tau),
    so stopping before it stops that far short of the obstacle."""
    if check is None:
        return None
    blocked = np.flatnonzero((check.caps <= 0.0) & (check.ends > tau))
    if standoff <= 0.0 or not blocked.size:
        return check
    b = int(blocked[0])
    block_at = max(tau, float(check.starts[b]))
    hold, moved = block_at, 0.0
    while hold > tau and moved < standoff:
        earlier = max(tau, hold - step)
        moved += travel(trajectory, earlier, hold, levers)
        hold = earlier
    # Segments before the hold point (cut short at it), then one blocked from it to the block.
    before = np.flatnonzero(check.starts < hold)
    unsafe = check.unsafe if len(check.unsafe) == len(check.caps) else np.zeros(len(check.caps), dtype=bool)
    return CheckResult(
        starts=np.r_[check.starts[before], hold, check.starts[b:]],
        ends=np.r_[np.minimum(check.ends[before], hold), block_at, check.ends[b:]],
        caps=np.r_[check.caps[before], 0.0, check.caps[b:]],
        unsafe=np.r_[unsafe[before], False, unsafe[b:]],
        stamp=check.stamp)


def creeping_limits(trajectory, tau, tau1, check, speed, acceleration, levers, standoff=0.0):
    """(target scale, the most it may rise per s) while creeping toward a block: no point of the
    robot faster than `speed` (m/s), and speeding up and braking for what the check says is
    ahead at no more than `acceleration` (m/s^2 at the fastest point), so each creep is a smooth
    step rather than a jerk (Elaine, 2026-10-22). It stops `standoff` m short of the block: the
    robot can't see anything within its self-filter padding (3 cm) of itself, so creeping right
    up to a block would carry on into it, unseen (robot_lift_hand_12, 2026-10-22). Slowing for a
    block that appears right ahead is still up to the caller's usual rate."""
    cap = creep_scale(trajectory, tau, tau1, speed, levers)
    rate = acceleration * cap / speed
    check = with_standoff(trajectory, tau, check, standoff, levers)
    return min(target_scale(tau, check, rate, trajectory.duration), cap), rate


class ResumeGate:
    """What the robot may do after the checks stop it for a blocked path (Elaine, 2026-10-22:
    stop, then edge slowly toward the obstacle while there's still some clear space, then stop
    very close to it, e.g. for making contact on purpose):

    - STOPPED for delay s from the check that stopped it;
    - then CREEPING (the caller caps the speed), stopping again before the block as usual
      whenever the checks say so;
    - FREE again once clear_checks checks in a row see nothing blocked and delay s have passed
      since the last one that did.

    On the robot (2026-10-20), single checks came back clear between blocked ones while a hand
    was in the way, and each restarted the arm toward the hand at 60-70% speed; on 2026-10-22,
    checks that saw the block a few cm ahead instead of right away did the same at up to 80%.
    """

    FREE, STOPPED, CREEPING = 0, 1, 2

    def __init__(self, clear_checks, delay):
        self.clear_checks = clear_checks
        self.delay = delay
        self.state = self.FREE
        self._clear = 0
        self._stopped_at = None
        self._blocked_at = None

    @property
    def held(self):
        return self.state != self.FREE

    def update(self, stamp, stopping, blocked):
        """Count one new check (stamp: when it was asked, s): whether it stops the robot for a
        block, and whether it sees a block anywhere. Returns the new state."""
        if stopping and not self.held:
            self.state, self._stopped_at = self.STOPPED, stamp
        if not self.held:
            return self.state
        if stopping or blocked:
            self._clear, self._blocked_at = 0, stamp
        else:
            self._clear += 1
        if self._clear >= self.clear_checks and stamp - self._blocked_at >= self.delay:
            self.state = self.FREE
        elif self.state == self.STOPPED and stamp - self._stopped_at >= self.delay:
            self.state = self.CREEPING
        return self.state
