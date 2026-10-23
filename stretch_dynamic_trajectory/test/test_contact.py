"""Tests for contact detection from joint effort (no ROS)."""

import numpy as np
import pytest

from stretch_dynamic_trajectory.contact import CONTACT_SETTINGS, YIELD_SETTINGS, ContactDetector, ContactSettings, Yielding


def run(detector, samples):
    """samples: (t, effort, commanded velocity); returns the first (t, rise) or None."""
    for t, effort, velocity in samples:
        rise = detector.update(t, effort, velocity)
        if rise is not None:
            return t, rise
    return None


def lift_down(contact_at=None, rate=15.0, until=3.0, dt=0.01):
    """Like the lift going down on 2026-10-19: effort +32 at rest, about +22 moving down freely,
    and falling (pushing harder down) by `rate` points/s once a hand takes the load."""
    samples = []
    for t in np.arange(0.0, until, dt):
        effort = 32.0 if t < 0.1 else 22.0
        if contact_at is not None and t >= contact_at:
            effort -= rate * (t - contact_at)
        samples.append((t, effort, 0.0 if t < 0.1 else -0.05))
    return samples


def test_free_motion_doesnt_trip_and_a_contact_does_soon_after():
    assert run(ContactDetector(CONTACT_SETTINGS["lift_joint"]), lift_down()) is None
    t, rise = run(ContactDetector(CONTACT_SETTINGS["lift_joint"]), lift_down(contact_at=1.0))
    assert rise > 12.0 and t == pytest.approx(1.0 + 12.0 / 15.0, abs=0.03)


def test_the_free_level_is_kept_across_stops():
    # Moving, stopped (effort back at the resting +32: pushing less), then moving again: the
    # restart neither trips nor needs re-learning, so a contact right after it is caught.
    detector = ContactDetector(CONTACT_SETTINGS["lift_joint"])
    samples = lift_down(until=1.0)
    samples += [(t, 32.0, 0.0) for t in np.arange(1.0, 1.5, 0.01)]
    samples += [(t, 22.0 - 30.0 * (t - 1.55), -0.05) for t in np.arange(1.5, 2.2, 0.01) if t >= 1.55]
    t, _ = run(detector, samples)
    assert t == pytest.approx(1.55 + 12.0 / 30.0, abs=0.02)  # as soon as it has risen 12


def test_wrists_skip_the_start_transient_each_time():
    detector = ContactDetector(CONTACT_SETTINGS["wrist_yaw_joint"])
    spike = [(t, 26.0 if t < 0.25 else 0.0, 0.2) for t in np.arange(0.0, 1.0, 0.01)]
    assert run(detector, spike) is None
    assert run(detector, [(t, 0.0, 0.0) for t in np.arange(1.0, 1.2, 0.01)]) is None
    again = [(t, 26.0 if t < 1.45 else 0.0, 0.2) for t in np.arange(1.2, 2.0, 0.01)]
    assert run(detector, again) is None


def test_each_direction_has_its_own_free_level_and_threshold():
    settings = ContactSettings(12.0, 8.0, signed=True)
    detector = ContactDetector(settings)
    up = [(t, 10.0, 0.05) for t in np.arange(0.0, 1.0, 0.01)]
    assert run(detector, up) is None
    # Going down, push is -effort: 5 when free, then 14 against something, a rise of 9, over
    # the downward threshold (8) but under the upward one (12).
    down = [(t, -5.0 if t < 1.8 else -14.0, -0.05) for t in np.arange(1.0, 2.5, 0.01)]
    t, rise = run(detector, down)
    assert t == pytest.approx(1.8, abs=0.011) and rise == pytest.approx(9.0)


ARM = YIELD_SETTINGS["arm_joint"]


def test_a_held_joint_yields_only_to_a_push_beyond_its_holding_band():
    yielder = Yielding(ARM, max_speed=0.05)
    lo, hi = ARM.free_effort
    # Anywhere in the band between its free-motion efforts is just holding.
    assert not any(yielder.watch(t, e) for t, e in zip(np.arange(0, 2, 0.01), np.linspace(lo - 2, hi + 2, 200)))
    # Pushed in: the motor pushes out harder to hold, above the band. It yields inward once the
    # push has lasted trigger_time.
    pushed = hi + ARM.threshold + 3.0
    assert not yielder.watch(2.0, pushed)
    assert not yielder.watch(2.05, pushed)
    assert yielder.watch(2.11, pushed) and yielder.direction == -1


def test_yielding_follows_the_push_and_stops_when_it_eases():
    yielder = Yielding(ARM, max_speed=0.05)
    yielder.direction = -1  # yielding inward
    free_in = ARM.free_effort[0]
    # Pushed along hard: the motor needs much less than free-motion effort, so full speed.
    assert yielder.update(0.0, free_in + 20.0) == pytest.approx(-0.05)
    # A lighter push: slower.
    v = yielder.update(0.1, free_in + ARM.release + 4.0)
    assert -0.05 < v < 0.0 and v == pytest.approx(-ARM.gain * 4.0)
    # The push eases (the motor back at its free effort): stop yielding after release_time.
    assert yielder.update(0.2, free_in) == 0.0 and yielder.active
    assert yielder.update(0.71, free_in) == 0.0 and not yielder.active


def test_the_lift_yields_down_when_pushed_down_and_up_when_lifted():
    lift = YIELD_SETTINGS["lift_joint"]
    yielder = Yielding(lift, max_speed=0.05)
    lo, hi = lift.free_effort
    assert yielder.pushed(hi + lift.threshold + 1.0) == -1  # holding up harder: pushed down
    assert yielder.pushed(lo - lift.threshold - 1.0) == 1   # holding up less: lifted
    assert yielder.pushed((lo + hi) / 2) == 0


def test_after_a_yield_another_waits_for_the_push_to_let_go():
    yielder = Yielding(ARM, max_speed=0.05)
    pushed = ARM.free_effort[1] + ARM.threshold + 3.0
    assert yielder.watch(0.0, pushed) is False and yielder.watch(0.2, pushed)
    yielder.stop()
    # Still out of the band (never let go): no new yield.
    assert not any(yielder.watch(t, pushed) for t in np.arange(1.0, 3.0, 0.05))
    # Let go, then pushed again: yields again.
    assert not yielder.watch(3.0, 0.0)
    assert not yielder.watch(3.1, pushed) and yielder.watch(3.25, pushed)


def test_a_yield_that_runs_into_something_stops_at_once():
    yielder = Yielding(ARM, max_speed=0.02)
    yielder.direction = -1  # yielding inward, where the motor's free effort is free_in
    free_in = ARM.free_effort[0]
    assert yielder.update(0.0, free_in + 10.0) < 0.0  # pushed along
    # Something behind: the motor pushes inward harder than free motion needs.
    assert yielder.update(0.05, free_in - ARM.threshold - 1.0) == 0.0 and not yielder.active

