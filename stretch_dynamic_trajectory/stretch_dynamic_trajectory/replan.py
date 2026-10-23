"""Replanning around a block (no ROS): where to rejoin the original plan, and the plan to switch
to.

The goal's trajectory stays the reference. A detour from the planner leaves the plan being
followed a little ahead of the robot and rejoins the reference just past the block (or ends at
the goal), and the plan switched to is: the current plan from where the robot is to where the
detour leaves it, the detour, then the rest of the reference. It starts at the robot's current
planned velocity, so the switch doesn't jerk. FollowedPlan keeps track of which part of a plan
is the detour, so a block further on maps back to the reference.
"""

import itertools
from dataclasses import dataclass, field

import numpy as np

from stretch_dynamic_trajectory.timing import Trajectory

_ids = itertools.count(1)


@dataclass
class FollowedPlan:
    """A plan being followed: the reference itself (detour_end 0), or a detour that left the
    reference at reference time `left` and rejoins it at reference time `rejoin`, reached at
    plan time detour_end."""
    trajectory: Trajectory
    detour_end: float = 0.0
    rejoin: float = 0.0
    left: float = 0.0
    id: int = field(default_factory=lambda: next(_ids))

    @property
    def is_detour(self):
        return self.detour_end > 0.0

    def reference_time(self, t):
        """The reference time that plan time t follows, or None on the detour itself."""
        if t < self.detour_end - 1e-9:
            return None
        return t - self.detour_end + self.rejoin


def blocked_run(check):
    """(start, end) plan times of the first run of consecutive blocked segments in a check, and
    whether it runs to the end of what was checked (where it really ends isn't known)."""
    blocked = np.asarray(check.caps) <= 0.0
    first = int(np.flatnonzero(blocked)[0])
    last = first
    while last + 1 < len(blocked) and blocked[last + 1]:
        last += 1
    return float(check.starts[first]), float(check.ends[last]), last == len(blocked) - 1


def rejoin_time(plan, block_end, margin, reference_duration):
    """Reference time to rejoin at for a block ending at plan time block_end: margin s past it;
    for a block on a detour, margin past where the detour was going to rejoin. At most the
    reference's end (the detour then ends at the goal)."""
    reference = plan.reference_time(block_end)
    if reference is None:
        reference = plan.rejoin
    return float(min(reference_duration, reference + margin))


def rejoin_candidates(plan, leave, reference_duration, margin, block=None, spacing=1.0, count=5, match_path=False):
    """Reference times to try rejoining the reference at, leaving `plan` at plan time `leave`:
    from just past a block (block: (start, end, open_ended) plan times, as blocked_run gives),
    or, without one, from just past where the plan is (on a detour: just past where it left the
    reference, so going straight back is one of them), every `spacing` s, and the goal.

    A block that runs past what was checked: only the goal, or with `match_path` (the
    reference's path matters, not just where it ends) from just past what was checked like any
    other block, and then the goal only if it's within `count` candidates, so a detour never
    skips more of the reference than it has to."""
    here = plan.reference_time(leave)
    if block is not None:
        _, end, open_ended = block
        first = (reference_duration if open_ended and not match_path else
                 rejoin_time(plan, end, margin, reference_duration))
    elif here is None:
        first = min(reference_duration, plan.left + margin)
    else:
        first = min(reference_duration, here + margin)
    times = [first + k * spacing for k in range(count) if first + k * spacing < reference_duration]
    if match_path and first + count * spacing < reference_duration:
        return times
    return times + [reference_duration]


def detour_start(tau, scale, lookahead, block_start):
    """Plan time to leave the current plan at: lookahead s of trajectory time ahead at the
    current scale (the robot keeps going while the planner works), but before the block."""
    return float(np.clip(tau + scale * lookahead, tau, max(tau, block_start)))


def splice(current, tau, leave, detour_times, detour_positions, reference, rejoin, left=0.0, lead_step=0.05):
    """The FollowedPlan to switch to at plan time tau of `current` (a Trajectory): along
    current to plan time `leave`, then the detour (times from 0 at its start, which is current
    at `leave`), then the reference past reference time `rejoin` (where the detour ends). `left`:
    the reference time the detour counts as leaving the reference at."""
    lead = list(np.arange(tau, leave, lead_step)) + [leave]
    if len(lead) > 1 and lead[-1] - lead[-2] < 1e-6:
        lead.pop(-2)
    times = [t - tau for t in lead]
    positions = [current.sample(t)[0] for t in lead]
    offset = times[-1]
    for t, q in zip(detour_times[1:], detour_positions[1:]):
        times.append(offset + float(t))
        positions.append(np.asarray(q, dtype=float))
    detour_end = times[-1]
    later = reference.times > rejoin + 1e-6
    for t, q in zip(reference.times[later], reference.positions[later]):
        times.append(detour_end + float(t) - rejoin)
        positions.append(q)
    # Velocities as Trajectory infers them, except along the lead-in: the current plan's, so the
    # switch is smooth.
    velocities = Trajectory(times, positions).velocities.copy()
    for k, t in enumerate(lead):
        velocities[k] = current.sample(t)[1]
    return FollowedPlan(Trajectory(times, positions, velocities), detour_end=max(detour_end, 1e-9), rejoin=rejoin,
                        left=left)


def plan_time_of(plan, reference_time):
    """The plan time at which `plan` (a FollowedPlan) reaches reference time `reference_time`,
    which must be at or past where it rejoins the reference."""
    return reference_time - plan.rejoin + plan.detour_end


def _stretch(trajectory, t0, t1, step=0.05):
    """Positions along a Trajectory from t0 to t1: every `step` s and at its waypoints."""
    t1 = max(t0, min(t1, trajectory.duration))
    times = np.r_[np.arange(t0, t1, step), trajectory.times[(trajectory.times > t0) & (trajectory.times < t1)], t1]
    return np.array([trajectory.sample(t)[0] for t in np.unique(times)])


def _resample(points, weights, count):
    """`count` points spread evenly along a polyline by weighted length."""
    lengths = np.linalg.norm(np.diff(points, axis=0) * weights, axis=1)
    along = np.r_[0.0, np.cumsum(lengths)]
    if along[-1] <= 0.0:
        return np.repeat(points[:1], count, axis=0)
    at = np.linspace(0.0, along[-1], count)
    return np.column_stack([np.interp(at, along, points[:, j]) for j in range(points.shape[1])])


def path_deviation(plan, leave, reference, start, until, weights, samples=60):
    """How far `plan` (a FollowedPlan) strays from the reference's path: from plan time `leave`
    until it reaches reference time `until`, against the reference from reference time `start`
    to `until`. Both stretches are spread over `samples` points evenly by length (weights:
    roughly m of motion per unit of each joint), and it's the mean distance between matching
    points (m).

    Points are matched by how far along each path they are, not by time: the robot runs behind
    the reference's schedule whenever it slows, and a path is the same path at any speed. Nor
    by the nearest point on the reference, which would make a shortcut on a reference that
    comes back on itself (a loop ending where it started) look like no deviation at all;
    matching by progress makes skipping part of the reference cost what it skips. Every option
    being compared should run to the same `until`: one that rejoins sooner then follows the
    reference, and counts nothing for that stretch."""
    weights = np.asarray(weights, dtype=float)
    path = _stretch(plan.trajectory, leave, plan_time_of(plan, until))
    original = _stretch(reference, start, until)
    if len(path) < 2:
        path = np.repeat(path, 2, axis=0)
    if len(original) < 2:
        original = np.repeat(original, 2, axis=0)
    a, b = _resample(path, weights, samples), _resample(original, weights, samples)
    return float(np.mean(np.linalg.norm((a - b) * weights, axis=1)))


def block_ahead(check, tau):
    """Plan time from tau to the first blocked segment of a check (0 if it's blocked where the
    robot is), or infinity if nothing's blocked."""
    blocked = np.flatnonzero(np.asarray(check.caps) <= 0.0)
    if not blocked.size:
        return float("inf")
    return max(0.0, float(check.starts[blocked[0]]) - tau)


def worth_switching(detour_check, detour_tau, current_check, tau, gain=0.5):
    """Whether to switch to a detour, from a check of it on the live grid: if it's clear, or its
    first block is at least `gain` s further ahead than the plan being followed's (which counts
    as blocked where the robot is without a check)."""
    ahead = block_ahead(detour_check, detour_tau)
    if ahead == float("inf"):
        return True
    current = block_ahead(current_check, tau) if current_check is not None else 0.0
    return ahead > current + gain


def left_at(plan, leave):
    """The reference time a detour leaving `plan` at plan time `leave` counts as leaving the
    reference at: where plan is on the reference there, or where plan's own detour left it."""
    here = plan.reference_time(leave)
    return plan.left if here is None else here
