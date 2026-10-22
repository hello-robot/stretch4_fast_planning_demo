"""Tests for following a trajectory at a variable speed (no ROS)."""

import numpy as np
import pytest

from stretch_dynamic_trajectory.timing import (CheckResult, ResumeGate, Trajectory, blocked_anywhere, check_times,
                                               creep_scale, creeping_limits, scale_deceleration, segment_caps, step_scale,
                                               stopping_for_block, target_scale, with_standoff)


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


def simulate(trajectory, cap_at, a, dt=0.02, seconds=6.0, horizon=1.0, step=0.1, gate=None, up_rate=None,
             creep=(0.02, 0.1, [1.0])):
    """Run the scale and tau forward with checks every 0.1 s, as the server does; cap_at(t, tau)
    gives the cap at trajectory time tau at wall time t, and a segment gets the tightest cap
    anywhere in it (as a real check sweeps the whole segment). creep: (speed, acceleration,
    levers[, standoff]) while the gate lets the robot creep. Returns arrays of (t, tau, scale)."""
    tau, scale, rows, result = 0.0, 0.0, [], None
    for k in range(int(seconds / dt)):
        t = k * dt
        if k % 5 == 0:
            starts = np.arange(tau, min(tau + horizon, trajectory.duration), step)
            ends = np.r_[starts[1:], min(tau + horizon, trajectory.duration)]
            caps = [min(cap_at(t, u) for u in np.linspace(s, e, 11)[:-1]) for s, e in zip(starts, ends)]
            result = CheckResult(starts=starts, ends=ends, caps=np.array(caps), stamp=t)
            fresh = True
        else:
            fresh = False
        target = target_scale(tau, result, a, trajectory.duration)
        up = up_rate
        if gate is not None:
            if fresh:
                gate.update(t, stopping_for_block(result, target), blocked_anywhere(result))
            if gate.state == ResumeGate.STOPPED:
                target = 0.0
            elif gate.state == ResumeGate.CREEPING:
                creep_target, creep_rate = creeping_limits(trajectory, tau, min(trajectory.duration, tau + horizon),
                                                           result, *creep)
                target, up = min(target, creep_target), min(up if up is not None else a, creep_rate)
        scale = step_scale(scale, target, a, dt, up_rate=up)
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


def test_speeding_up_is_limited_separately_from_slowing_down():
    assert step_scale(0.0, 1.0, rate=5.0, dt=0.1, up_rate=2.0) == pytest.approx(0.2)
    assert step_scale(1.0, 0.0, rate=5.0, dt=0.1, up_rate=2.0) == pytest.approx(0.5)
    assert step_scale(0.0, 1.0, rate=1.0, dt=0.1, up_rate=2.0) == pytest.approx(0.1)  # still the joints' limit


def test_resume_gate_needs_several_clear_checks_and_some_time():
    gate = ResumeGate(clear_checks=3, delay=0.5)
    FREE, STOPPED = ResumeGate.FREE, ResumeGate.STOPPED
    assert gate.update(0.0, stopping=False, blocked=True) == FREE  # never held without a stop
    assert gate.update(0.1, stopping=True, blocked=True) == STOPPED
    assert gate.update(0.2, stopping=False, blocked=False) == STOPPED  # one clear check isn't enough
    assert gate.update(0.3, stopping=True, blocked=True) == STOPPED
    assert gate.update(0.4, stopping=False, blocked=False) == STOPPED
    assert gate.update(0.5, stopping=False, blocked=False) == STOPPED
    # Three in a row, but only 0.3 s since the last block: 0.5 s since the stop, so creep.
    assert gate.update(0.6, stopping=False, blocked=False) == ResumeGate.CREEPING
    assert gate.update(0.7, stopping=False, blocked=False) == ResumeGate.CREEPING
    assert gate.update(0.8, stopping=False, blocked=False) == FREE  # 0.5 s: go
    assert not gate.held


def test_resume_gate_creeps_while_a_block_remains_ahead():
    gate = ResumeGate(clear_checks=3, delay=0.5)
    STOPPED, CREEPING, FREE = ResumeGate.STOPPED, ResumeGate.CREEPING, ResumeGate.FREE
    assert gate.update(0.1, stopping=True, blocked=True) == STOPPED
    for t in (0.2, 0.3, 0.4, 0.5):  # stopped a while, whether the block is right ahead or not
        assert gate.update(t, stopping=t == 0.3, blocked=True) == STOPPED
    assert gate.update(0.6, stopping=False, blocked=True) == CREEPING  # 0.5 s since the stop
    # Crept up to it: the check itself stops the robot; it stays free to creep when there's room.
    assert gate.update(0.7, stopping=True, blocked=True) == CREEPING
    assert gate.update(1.2, stopping=False, blocked=True) == CREEPING
    for t in (1.3, 1.4, 1.5):  # three clear checks, but only 0.3 s since the last block
        assert gate.update(t, stopping=False, blocked=False) == CREEPING
    assert gate.update(1.6, stopping=False, blocked=True) == CREEPING  # and a block again: start over
    for t in (1.7, 1.8, 1.9, 2.0):
        assert gate.update(t, stopping=False, blocked=False) == CREEPING
    assert gate.update(2.1, stopping=False, blocked=False) == FREE


def test_creep_scale_bounds_the_fastest_point():
    # An arm at 0.1 m/s and a wrist at 0.5 rad/s with a 0.3 m lever: up to 0.25 m/s at the tip.
    trajectory = Trajectory([0.0, 1.0], [[0.0, 0.0], [0.1, 0.5]], velocities=[[0.1, 0.5], [0.1, 0.5]])
    assert creep_scale(trajectory, 0.0, 1.0, 0.02, [1.0, 0.3]) == pytest.approx(0.02 / 0.25)
    assert creep_scale(trajectory, 0.0, 1.0, 0.5, [1.0, 0.3]) == 1.0  # already slower than that


def test_stopping_for_block_only_when_stopping_now():
    blocked_ahead = check([0.0, 0.1, 0.2], [1.0, 1.0, 0.0], checked_until=0.3)
    assert stopping_for_block(blocked_ahead, 0.0)
    assert not stopping_for_block(blocked_ahead, 0.4)  # still slowing for it
    assert not stopping_for_block(check([0.0, 0.1], [1.0, 1.0]), 0.0)  # stopped for another reason


def test_a_flickering_block_holds_the_robot_instead_of_restarting_it():
    # As on the robot (2026-10-20, a hand in front of the arm): the stretch ahead is blocked
    # from 1.0 s to 4.0 s of wall time, but single checks come back clear in between.
    trajectory = line(duration=4.0, distance=0.2)
    a = scale_deceleration(trajectory, 0.0, 4.0, [0.7], max_rate=5.0)
    flicker = (2.0, 2.4)  # these checks see no block

    def cap_at(t, tau):
        clear = any(abs(t - f) < 1e-6 for f in flicker)
        return 0.0 if (1.0 <= t < 4.0 and not clear and tau >= 1.2) else 1.0

    t, tau, scale = simulate(trajectory, cap_at, a, seconds=8.0).T
    assert scale[(t > 2.0) & (t < 2.6)].max() > 0.05  # without the gate, the flicker restarts it

    t, tau, scale = simulate(trajectory, cap_at, a, seconds=8.0, gate=ResumeGate(3, 0.5), up_rate=2.0).T
    stopped = (t > 1.9) & (t < 4.0)
    # Held through the flicker: at most a creep while a clear check lasts, a few mm into what
    # the checks before called blocked.
    assert scale[stopped].max() <= creep_scale(trajectory, 0.0, 4.0, 0.02, [1.0]) + 1e-9
    position = np.array([trajectory.sample(u)[0][0] for u in tau])
    assert position[t < 4.0].max() - trajectory.sample(1.2)[0][0] < 0.005
    assert tau[-1] == pytest.approx(4.0)  # then finished once it stayed clear
    assert np.max(np.diff(scale)) <= 2.0 * 0.02 + 1e-9  # and sped up gently


def test_after_a_stop_the_robot_creeps_up_to_a_block_and_stops_short_of_it():
    # As on the robot (2026-10-22, a hand in front of the arm): the checks first see the block
    # right ahead and stop the robot, then see it a few cm further on, on and off. The robot
    # should creep up to it at the creep speed, not restart at full speed, and stop before it.
    trajectory = line(duration=4.0, distance=0.2)  # 0.05 m/s on average, up to 0.075
    a = scale_deceleration(trajectory, 0.0, 4.0, [0.7], max_rate=5.0)

    def cap_at(t, tau):
        if t < 1.0 or t >= 6.0:
            return 1.0
        near = t < 1.3 or round(t * 10) % 3 == 0  # seen right ahead now and then
        return 0.0 if tau >= (1.1 if near else 1.6) else 1.0

    t, tau, scale = simulate(trajectory, cap_at, a, seconds=10.0, gate=ResumeGate(3, 0.5), up_rate=2.0).T
    stopped_at = t[(t > 1.0) & (scale == 0.0)][0]
    position = np.array([trajectory.sample(u)[0][0] for u in tau])
    speed = np.abs(np.diff(position)) / np.diff(t)
    after_stop = (t[1:] > stopped_at) & (t[1:] < 6.0)
    assert speed[after_stop].max() <= 0.02 + 1e-6  # crept at 2 cm/s, never more
    assert speed[after_stop].max() > 0.01  # but did creep
    assert tau[t < 6.0].max() <= 1.6 + 1e-6  # and never into the block
    assert tau[(t > 5.5) & (t < 6.0)].min() > 1.4  # but got close to it
    assert tau[-1] == pytest.approx(4.0)  # then finished once it stayed clear
    # Smoothly: the speed rose no faster than the creep acceleration (0.1 m/s^2).
    accel = np.diff(speed) / np.diff(t[1:])
    assert accel[after_stop[1:]].max() <= 0.1 + 0.01


def test_creeping_brakes_gently_for_a_block_it_knows_about():
    trajectory = line(duration=4.0, distance=0.2)
    result = check(np.arange(1.0, 2.0, 0.1), [1.0] * 5 + [0.0] * 5, checked_until=2.0)  # blocked from 1.5
    cap = creep_scale(trajectory, 1.0, 2.0, 0.02, [1.0])
    far, _ = creeping_limits(trajectory, 1.0, 2.0, result, 0.02, 0.1, [1.0])
    assert far == pytest.approx(cap)  # room to creep at full creep speed
    near, rate = creeping_limits(trajectory, 1.49, 2.0, result, 0.02, 0.1, [1.0])
    assert 0.0 < near < cap  # already braking, gently, just short of the block
    assert near == pytest.approx(np.sqrt(2 * rate * 0.01))


def test_check_times_are_finer_near_the_robot_when_asked():
    assert check_times(1.0, 1.35, 0.1) == pytest.approx([1.1, 1.2, 1.3, 1.35])
    fine = check_times(1.0, 2.0, 0.1, fine_step=0.02, fine_until=1.1)
    assert fine == pytest.approx([1.02, 1.04, 1.06, 1.08, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0])
    assert check_times(1.0, 1.05, 0.1, fine_step=0.02, fine_until=1.3) == pytest.approx([1.02, 1.04, 1.05])
    assert check_times(2.0, 2.0, 0.1) == [2.0]


def test_the_standoff_brings_the_first_block_nearer_by_that_much_travel():
    trajectory = line(duration=4.0, distance=0.2)
    result = check(np.arange(1.0, 2.0, 0.1), [1.0] * 6 + [0.0] * 4, checked_until=2.0)  # blocked from 1.6
    moved = with_standoff(trajectory, 1.0, result, 0.03, [1.0])
    hold = moved.starts[moved.caps == 0.0][0]
    position = lambda u: trajectory.sample(u)[0][0]  # noqa: E731
    assert position(1.6) - position(hold) == pytest.approx(0.03, abs=0.001)
    assert np.all(moved.caps[moved.starts < hold] == 1.0) and moved.ends[moved.starts < hold].max() <= hold
    assert moved.checked_until == result.checked_until
    # Nearer than the standoff already: hold right here.
    near = with_standoff(trajectory, 1.55, result, 0.03, [1.0])
    assert near.starts[near.caps == 0.0][0] == pytest.approx(1.55)
    assert with_standoff(trajectory, 1.0, None, 0.03, [1.0]) is None
    unblocked = check(np.arange(1.0, 2.0, 0.1), [1.0] * 10, checked_until=2.0)
    assert with_standoff(trajectory, 1.0, unblocked, 0.03, [1.0]) is unblocked


def test_creeping_stops_the_standoff_short_of_a_block():
    # robot_lift_hand_12 (2026-10-22): within the self-filter padding (3 cm) the robot can't see
    # an obstacle, so creeping right up to a block carried on into Elaine's hand. With a 5 cm
    # standoff the robot holds while the block is still in view.
    # The block is first seen just ahead, stopping the robot, then further on (9 cm on, within
    # the checked horizon, so the robot keeps creeping rather than going back to full speed).
    trajectory = line(duration=4.0, distance=0.2)
    a = scale_deceleration(trajectory, 0.0, 4.0, [0.7], max_rate=5.0)

    def cap_at(t, tau):
        if t < 1.0 or t >= 12.0:
            return 1.0
        return 0.0 if tau >= (1.1 if t < 2.0 else 2.6) else 1.0

    t, tau, scale = simulate(trajectory, cap_at, a, seconds=16.0, horizon=2.0, gate=ResumeGate(3, 0.5), up_rate=2.0,
                             creep=(0.02, 0.1, [1.0], 0.05)).T
    position = np.array([trajectory.sample(u)[0][0] for u in tau])
    block = trajectory.sample(2.6)[0][0]
    assert block - position[(t > 1.5) & (t < 2.0)].max() > 0.08  # stopped well short at first
    held = position[(t > 10.0) & (t < 12.0)]
    assert held.max() <= block - 0.05 + 0.002  # never nearer than the standoff
    assert held.min() >= block - 0.05 - 0.01  # but crept up to it
    assert tau[-1] == pytest.approx(4.0)  # and finished once the block went away

