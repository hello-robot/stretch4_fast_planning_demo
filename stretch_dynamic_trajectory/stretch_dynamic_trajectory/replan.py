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


def rejoin_candidates(plan, leave, reference_duration, margin, block=None, spacing=1.0, count=5):
    """Reference times to try rejoining the reference at, leaving `plan` at plan time `leave`:
    from just past a block (block: (start, end, open_ended) plan times, as blocked_run gives),
    or, without one, from just past where the plan is (on a detour: just past where it left the
    reference, so going straight back is one of them), every `spacing` s, and the goal."""
    here = plan.reference_time(leave)
    if block is not None:
        _, end, open_ended = block
        first = reference_duration if open_ended else rejoin_time(plan, end, margin, reference_duration)
    elif here is None:
        first = min(reference_duration, plan.left + margin)
    else:
        first = min(reference_duration, here + margin)
    times = [first + k * spacing for k in range(count) if first + k * spacing < reference_duration]
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


def left_at(plan, leave):
    """The reference time a detour leaving `plan` at plan time `leave` counts as leaving the
    reference at: where plan is on the reference there, or where plan's own detour left it."""
    here = plan.reference_time(leave)
    return plan.left if here is None else here
