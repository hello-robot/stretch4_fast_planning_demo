"""Tests for replanning around a block (no ROS)."""

import numpy as np
import pytest

from stretch_dynamic_trajectory.replan import (FollowedPlan, block_ahead, blocked_run, detour_start, left_at,
                                              path_deviation, rejoin_candidates, rejoin_time, splice, worth_switching)
from stretch_dynamic_trajectory.timing import CheckResult
from stretch_dynamic_trajectory.timing import Trajectory


def reference():
    # Two joints: the first goes 0 -> 0.4 over 4 s, the second stays at 0.
    return Trajectory([0.0, 1.0, 2.0, 3.0, 4.0], [[0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [0.3, 0.0], [0.4, 0.0]])


def test_the_switch_starts_where_the_robot_is_moving_as_it_was():
    ref = reference()
    plan = FollowedPlan(ref)
    tau, leave = 1.2, 1.5
    detour_times = [0.0, 1.0, 2.0]
    detour = [ref.sample(leave)[0], [0.2, 0.1], ref.sample(2.5)[0]]  # around something at 0.2
    new = splice(ref, tau, leave, detour_times, detour, ref, 2.5)
    p0, v0, _ = new.trajectory.sample(0.0)
    np.testing.assert_allclose(p0, ref.sample(tau)[0], atol=1e-9)
    np.testing.assert_allclose(v0, ref.sample(tau)[1], atol=1e-9)
    # Through the detour, then on along the reference from where it rejoined, to its end.
    lead = leave - tau
    np.testing.assert_allclose(new.trajectory.sample(lead + 1.0)[0], [0.2, 0.1], atol=1e-9)
    assert new.detour_end == pytest.approx(lead + 2.0)
    np.testing.assert_allclose(new.trajectory.sample(new.trajectory.duration)[0], ref.positions[-1], atol=1e-9)
    assert new.trajectory.duration == pytest.approx(lead + 2.0 + (4.0 - 2.5))
    assert new.is_detour and not plan.is_detour and new.id != plan.id


def test_reference_times_map_through_a_detour():
    ref = reference()
    new = splice(ref, 1.0, 1.0, [0.0, 1.5], [ref.sample(1.0)[0], ref.sample(3.0)[0]], ref, 3.0)
    assert new.reference_time(0.5) is None  # on the detour
    assert new.reference_time(new.detour_end + 0.5) == pytest.approx(3.5)


def test_rejoin_just_past_the_block_or_at_the_goal():
    ref = reference()
    plan = FollowedPlan(ref)
    assert rejoin_time(plan, 2.0, 0.5, ref.duration) == pytest.approx(2.5)
    assert rejoin_time(plan, 3.8, 0.5, ref.duration) == pytest.approx(4.0)
    # Blocked again on a detour: still rejoin past where that detour was headed.
    detour = splice(ref, 1.0, 1.0, [0.0, 1.5], [ref.sample(1.0)[0], ref.sample(3.0)[0]], ref, 3.0)
    assert rejoin_time(detour, 0.5, 0.5, ref.duration) == pytest.approx(3.5)


def test_leave_a_little_ahead_but_before_the_block():
    assert detour_start(1.0, 0.5, 0.6, 3.0) == pytest.approx(1.3)
    assert detour_start(1.0, 1.0, 0.6, 1.2) == pytest.approx(1.2)
    assert detour_start(1.0, 0.0, 0.6, 3.0) == pytest.approx(1.0)  # stopped: from here


def test_the_blocked_run_and_whether_it_goes_past_what_was_checked():
    starts = np.arange(0.0, 1.0, 0.1)
    check = CheckResult(starts=starts, ends=starts + 0.1, caps=np.array([1, 1, 0, 0, 0, 1, 0, 1, 1, 1.0]))
    start, end, open_ended = blocked_run(check)
    assert (start, end, open_ended) == (pytest.approx(0.2), pytest.approx(0.5), False)
    check.caps = np.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 0.0])
    assert blocked_run(check)[2]


def test_rejoin_candidates_past_a_block_or_back_where_the_detour_left():
    ref = reference()
    plan = FollowedPlan(ref)
    # Blocked from 1.0 to 1.5 on the reference: from 2.0 on, and the goal.
    assert rejoin_candidates(plan, 0.5, ref.duration, 0.5, block=(1.0, 1.5, False)) == pytest.approx([2.0, 3.0, 4.0])
    # Blocked past what was checked: only the goal.
    assert rejoin_candidates(plan, 0.5, ref.duration, 0.5, block=(1.0, 1.5, True)) == pytest.approx([4.0])
    # Unless the path matters: from just past what was checked, as for any block.
    assert rejoin_candidates(plan, 0.5, ref.duration, 0.5, block=(1.0, 1.5, True), match_path=True) == \
        pytest.approx([2.0, 3.0, 4.0])
    # On a detour that left at 1.0 for 3.0: going back just past where it left is a candidate.
    detour = splice(ref, 1.0, 1.0, [0.0, 1.5], [ref.sample(1.0)[0], ref.sample(3.0)[0]], ref, 3.0, left=1.0)
    assert rejoin_candidates(detour, 0.5, ref.duration, 0.5)[0] == pytest.approx(1.5)
    assert left_at(detour, 0.5) == pytest.approx(1.0)
    assert left_at(plan, 0.7) == pytest.approx(0.7)



def test_matching_the_path_leaves_out_a_goal_far_beyond_the_rejoin_points():
    # A long reference (a loop, say, that ends where it started): rejoining at its goal would
    # skip most of it, so match_path stops at the candidates just past the block.
    long_ref = Trajectory([0.0, 30.0, 60.0], [[0.0, 0.0], [0.4, 0.0], [0.0, 0.0]])
    plan = FollowedPlan(long_ref)
    times = rejoin_candidates(plan, 5.0, long_ref.duration, 0.5, block=(6.0, 7.0, True), match_path=True)
    assert times == pytest.approx([7.5, 8.5, 9.5, 10.5, 11.5])
    assert rejoin_candidates(plan, 5.0, long_ref.duration, 0.5, block=(6.0, 7.0, True)) == pytest.approx([60.0])


def circle(turns=1.0, period=8.0, radius=0.5, points=65):
    """Two joints tracing a circle (as if both were meters), `turns` times around in
    turns * period s, starting and ending at (radius, 0) when turns is whole."""
    t = np.linspace(0.0, turns * period, points)
    angle = 2.0 * np.pi * t / period
    return Trajectory(t, np.column_stack([radius * np.cos(angle), radius * np.sin(angle)]))


def detour(reference, leave, rejoin, middle=None, duration=1.0):
    """A FollowedPlan from the reference at `leave` straight (or through `middle`) to the
    reference at `rejoin`, then along the reference."""
    a, b = reference.sample(leave)[0], reference.sample(rejoin)[0]
    times, positions = [0.0, duration], [a, b]
    if middle is not None:
        times, positions = [0.0, 0.5 * duration, duration], [a, np.asarray(middle, dtype=float), b]
    return splice(reference, leave, leave, times, positions, reference, rejoin, left=leave)


WEIGHTS = np.ones(2)


def test_following_the_path_at_any_speed_does_not_stray():
    ref = circle()
    assert path_deviation(FollowedPlan(ref), 0.0, ref, 0.0, 8.0, WEIGHTS) == pytest.approx(0.0, abs=1e-6)
    # The same path at half speed, as a detour that reaches the reference's end (8 s) at 16 s.
    slow = FollowedPlan(Trajectory(ref.times * 2.0, ref.positions), detour_end=16.0, rejoin=8.0)
    assert path_deviation(slow, 0.0, ref, 0.0, 8.0, WEIGHTS) == pytest.approx(0.0, abs=2e-3)


def test_a_chord_across_the_circle_strays_more_than_the_arc():
    ref = circle()
    chord = detour(ref, 0.0, 2.0)  # a quarter of the way round, straight across
    arc = path_deviation(FollowedPlan(ref), 0.0, ref, 0.0, 2.0, WEIGHTS)
    across = path_deviation(chord, 0.0, ref, 0.0, 2.0, WEIGHTS)
    assert arc == pytest.approx(0.0, abs=1e-6)
    assert across > 0.05  # the arc bulges about 0.15 m from the chord at its middle


def test_skipping_a_loop_back_to_its_start_counts_as_skipping_it():
    # Two loops ending where they began: from 1 s in, straight to the goal (right next to where
    # the robot is) strays far more than going round, though it's far shorter.
    ref = circle(turns=2.0)
    shortcut = detour(ref, 1.0, 16.0)
    assert path_deviation(shortcut, 0.0, ref, 1.0, 16.0, WEIGHTS) > 0.2
    assert path_deviation(FollowedPlan(ref), 1.0, ref, 1.0, 16.0, WEIGHTS) == pytest.approx(0.0, abs=1e-6)


def test_going_back_to_the_path_sooner_strays_less():
    ref = circle()
    # Both bulge 0.1 m outward from the reference at 1.5 s; one is back at 2 s, one at 4 s.
    out = 1.2 * ref.sample(1.5)[0]
    soon = detour(ref, 1.0, 2.0, middle=out)
    late = detour(ref, 1.0, 4.0, middle=out, duration=3.0)
    assert path_deviation(soon, 0.0, ref, 1.0, 4.0, WEIGHTS) < path_deviation(late, 0.0, ref, 1.0, 4.0, WEIGHTS)


def check_of(caps, start=0.0, step=0.5):
    caps = np.asarray(caps, dtype=float)
    starts = start + step * np.arange(len(caps))
    return CheckResult(starts=starts, ends=starts + step, caps=caps)


def test_a_detour_is_taken_if_its_live_check_is_clear_or_blocked_further_on():
    blocked_soon = check_of([1.0, 0.0, 0.0], start=2.0)  # the plan now: blocked 0.5 s ahead
    assert block_ahead(blocked_soon, 2.0) == pytest.approx(0.5)
    assert block_ahead(check_of([1.0, 1.0]), 0.0) == float("inf")
    assert worth_switching(check_of([1.0, 1.0, 1.0]), 0.0, blocked_soon, 2.0)
    # Blocked as soon as the plan now: no better. Blocked 1.5 s on: better by 1 s.
    assert not worth_switching(check_of([1.0, 0.0, 1.0]), 0.0, blocked_soon, 2.0)
    assert worth_switching(check_of([1.0, 1.0, 1.0, 0.0]), 0.0, blocked_soon, 2.0)
    # Without a check of the plan now, it counts as blocked where the robot is.
    assert worth_switching(check_of([1.0, 1.0, 0.0]), 0.0, None, 2.0)
    assert not worth_switching(check_of([0.0, 1.0]), 0.0, None, 2.0)
