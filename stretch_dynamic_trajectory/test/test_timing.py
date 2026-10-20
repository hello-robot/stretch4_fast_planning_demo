"""Tests for following a trajectory at a variable speed (no ROS)."""

import numpy as np
import pytest

from stretch_dynamic_trajectory.timing import (CheckResult, Trajectory, scale_deceleration, segment_caps, step_scale,
                                               target_scale)


def line(duration=2.0, distance=0.3):
    """One joint moving `distance` over `duration`, through a midpoint."""
    return Trajectory([0.0, duration / 2, duration], [[0.0], [distance / 2], [distance]])


def test_hermite_hits_waypoints_with_smooth_velocity():
    trajectory = line()
    for tau, expected in [(0.0, 0.0), (1.0, 0.15), (2.0, 0.3)]:
        assert trajectory.sample(tau)[0][0] == pytest.approx(expected)
    assert trajectory.sample(0.0)[1][0] == pytest.approx(0.0)  # starts and ends at rest
    assert trajectory.sample(2.0)[1][0] == pytest.approx(0.0)
    assert trajectory.sample(1.0)[1][0] == pytest.approx(0.15)  # the average slope in the middle
    # Velocity is the derivative of position.
    for tau in (0.3, 0.9, 1.4):
        numeric = (trajectory.sample(tau + 1e-5)[0] - trajectory.sample(tau - 1e-5)[0]) / 2e-5
        assert trajectory.sample(tau)[1][0] == pytest.approx(numeric[0], rel=1e-4)


def test_given_velocities_are_used():
    trajectory = Trajectory([0.0, 1.0], [[0.0], [1.0]], velocities=[[1.0], [1.0]])
    assert trajectory.sample(0.5)[0][0] == pytest.approx(0.5)
    assert trajectory.sample(0.5)[1][0] == pytest.approx(1.0)


def test_caps_from_a_check():
    caps = segment_caps([0.1, 0.1, 0.1, 0.1], [0.05, 0.2, np.inf, 0.0], [False, False, False, True])
    assert caps.tolist() == pytest.approx([1.0, 0.5, 0.0, 0.0])


def check(starts, caps, checked_until=None, unsafe=None):
    starts = np.asarray(starts, dtype=float)
    ends = np.r_[starts[1:], checked_until if checked_until is not None else starts[-1] + 0.1]
    return CheckResult(starts=starts, ends=ends, caps=np.asarray(caps, dtype=float),
                       unsafe=np.zeros(len(starts), bool) if unsafe is None else np.asarray(unsafe))


def test_target_scale_brakes_in_time_for_a_cap_ahead():
    # Free for 0.5 s of trajectory time, then capped at 0.2.
    result = check(np.arange(0.0, 1.0, 0.1), [1.0] * 5 + [0.2] * 5, checked_until=1.0)
    a = 2.0
    assert target_scale(0.0, result, a, trajectory_end=1.0) == pytest.approx(1.0)
    # 0.1 s of tau before the cap: s^2 <= 0.2^2 + 2 * 2 * 0.1.
    assert target_scale(0.4, result, a, trajectory_end=1.0) == pytest.approx(np.sqrt(0.04 + 0.4))
    assert target_scale(0.55, result, a, trajectory_end=1.0) == pytest.approx(0.2)


def test_target_scale_stops_by_the_end_of_what_was_checked():
    result = check([0.0, 0.1, 0.2], [1.0, 1.0, 1.0], checked_until=0.3)
    # The trajectory goes on past 0.3 s, which nobody has checked.
    assert target_scale(0.3, result, 2.0, trajectory_end=2.0) == 0.0
    assert target_scale(0.25, result, 2.0, trajectory_end=2.0) == pytest.approx(np.sqrt(2 * 2.0 * 0.05))
    # Unless that's the trajectory's end.
    assert target_scale(0.25, result, 2.0, trajectory_end=0.3) == pytest.approx(1.0)
    assert target_scale(0.0, None, 2.0, trajectory_end=2.0) == 0.0


def simulate(trajectory, cap_at, a, dt=0.02, seconds=6.0, horizon=1.0, step=0.1):
    """Run the scale and tau forward with checks every 0.1 s; cap_at(t, tau) gives the cap at
    trajectory time tau at wall time t, and a segment gets the tightest cap anywhere in it (as
    a real check sweeps the whole segment). Returns arrays of (t, tau, scale)."""
    tau, scale, rows, result = 0.0, 0.0, [], None
    for k in range(int(seconds / dt)):
        t = k * dt
        if k % 5 == 0:
            starts = np.arange(tau, min(tau + horizon, trajectory.duration), step)
            ends = np.r_[starts[1:], min(tau + horizon, trajectory.duration)]
            caps = [min(cap_at(t, u) for u in np.linspace(s, e, 11)[:-1]) for s, e in zip(starts, ends)]
            result = CheckResult(starts=starts, ends=ends, caps=np.array(caps))
        target = target_scale(tau, result, a, trajectory.duration)
        scale = step_scale(scale, target, a, dt)
        tau = min(trajectory.duration, tau + scale * dt)
        rows.append((t, tau, scale))
    return np.array(rows)


def test_a_blocked_stretch_stops_the_robot_before_it_and_resumes_without_a_surge():
    trajectory = line(duration=3.0, distance=0.45)
    a = scale_deceleration(trajectory, 0.0, 3.0, [0.7], max_rate=5.0)
    # The stretch from tau 1.5 to 1.8 is blocked for the first 3 s of wall time.
    blocked = lambda t, tau: 0.0 if (t < 3.0 and 1.5 <= tau < 1.8) else 1.0  # noqa: E731
    rows = simulate(trajectory, blocked, a)
    t, tau, scale = rows.T
    assert tau[t < 3.0].max() <= 1.5 + 1e-6  # never entered the blocked stretch
    assert scale[(t > 2.0) & (t < 3.0)].max() == pytest.approx(0.0, abs=1e-9)  # waiting, stopped
    assert tau[-1] == pytest.approx(3.0)  # then finished
    # The scale never changed faster than the joints allow, so no surge after the block.
    assert np.max(np.abs(np.diff(scale))) <= a * 0.02 + 1e-9
    assert scale.max() <= 1.0


def test_a_cap_ahead_is_reached_at_or_below_the_cap():
    trajectory = line(duration=3.0, distance=0.45)
    a = 1.5
    capped = lambda t, tau: 0.3 if 1.0 <= tau < 2.0 else 1.0  # noqa: E731
    rows = simulate(trajectory, capped, a)
    t, tau, scale = rows.T
    inside = (tau >= 1.0) & (tau < 2.0)
    assert scale[inside].max() <= 0.3 + a * 0.02  # one step of slack for the discrete loop


def test_scale_deceleration_follows_the_fastest_joint():
    trajectory = Trajectory([0.0, 1.0], [[0.0, 0.0], [0.2, 1.0]], velocities=[[0.2, 1.0], [0.2, 1.0]])
    # Joint 1 moves at 1.0 with deceleration 2 -> the scale can change at 2 per s.
    assert scale_deceleration(trajectory, 0.0, 1.0, [1.0, 2.0], max_rate=5.0) == pytest.approx(2.0)
    assert scale_deceleration(trajectory, 0.0, 1.0, [100.0, 100.0], max_rate=5.0) == pytest.approx(5.0)
