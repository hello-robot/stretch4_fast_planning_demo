"""Following a joint trajectory at a variable speed, without ROS.

The trajectory keeps its own clock, tau: trajectory time, in the seconds of the goal's
time_from_start. tau advances at `scale` times real time, with scale between 0 and 1, so the
path never changes, only how fast the robot moves along it. Slowing down or stopping freezes
tau, so when the robot can go again it picks up where it is, without the desired state having
run ahead and the robot rushing to catch up.

The scale comes from obstacle checks of the motion ahead: each checked segment of trajectory
time has a cap (planned time / the least time the check allows; 0 if the segment is blocked).
The scale must be down to each cap by the time tau reaches that segment, braking no harder
than the joints can, and it never runs past the end of what's been checked.
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


def step_scale(scale, target, rate, dt):
    """Move the scale toward target by at most rate * dt."""
    return float(np.clip(target, scale - rate * dt, scale + rate * dt))
