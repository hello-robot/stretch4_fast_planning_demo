"""Tests for replanning around a block (no ROS)."""

import numpy as np
import pytest

from stretch_dynamic_trajectory.replan import FollowedPlan, blocked_run, detour_start, rejoin_time, splice
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

