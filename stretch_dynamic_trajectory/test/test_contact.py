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


def held(yielder, t0, t1, effort, dt=0.01):
    """Watch a held joint over [t0, t1) at a constant (or callable) effort; True if it yields."""
    for t in np.arange(t0, t1, dt):
        if yielder.watch(t, effort(t) if callable(effort) else effort):
            return t
    return None


def test_a_held_joint_yields_to_a_push_from_its_own_holding_level():
    # As on the robot (2026-10-23): the held arm at about -2, a firm push in raising it to 10-12.
    yielder = Yielding(ARM, max_speed=0.02)
    # Settling after a move (effort decaying from its motion value) isn't a push.
    assert held(yielder, 0.0, 2.5, lambda t: 11.0 - 13.0 * min(1.0, t / 1.8)) is None
    # Slow drift isn't either.
    assert held(yielder, 2.5, 5.0, lambda t: -2.0 + 0.2 * (t - 2.5)) is None
    t = held(yielder, 5.0, 7.0, lambda t: -1.5 + 12.0 * min(1.0, (t - 5.0) / 0.4))
    assert t is not None and t < 5.0 + 0.4 + 0.11 and yielder.direction == -1


def test_the_hold_level_starts_over_after_moving():
    yielder = Yielding(ARM, max_speed=0.02)
    assert held(yielder, 0.0, 3.0, 0.0) is None
    # It moved (no watch for a while), and now rests at a different level: no push.
    assert held(yielder, 4.0, 7.0, 10.0) is None


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
    assert yielder.update(1.21, free_in) == 0.0 and not yielder.active


def test_the_lift_yields_down_when_pushed_down_and_up_when_lifted():
    lift = YIELD_SETTINGS["lift_joint"]
    yielder = Yielding(lift, max_speed=0.02)
    yielder.hold_level = 30.0
    assert yielder.pushed(30.0 + lift.threshold + 1.0) == -1  # holding up harder: pushed down
    assert yielder.pushed(30.0 - lift.threshold - 1.0) == 1   # holding up less: lifted
    assert yielder.pushed(30.0) == 0


def test_after_a_yield_another_waits_for_the_push_to_let_go():
    yielder = Yielding(ARM, max_speed=0.02)
    pushed = ARM.threshold + 3.0
    assert held(yielder, 0.0, 2.5, 0.0) is None
    assert held(yielder, 2.5, 3.0, pushed) is not None
    yielder.stop()
    # Held again (from the hold level before the push) and never let go: no new yield.
    assert held(yielder, 3.0, 6.0, pushed) is None
    # Let go (back to where it held before): not a push the other way. Then pushed again: yields.
    assert held(yielder, 6.0, 8.0, 0.0) is None
    assert held(yielder, 8.0, 9.0, pushed) is not None


def test_a_yield_that_runs_into_something_stops_at_once():
    yielder = Yielding(ARM, max_speed=0.02)
    yielder.direction = -1  # yielding inward, where the motor's free effort is free_in
    free_in = ARM.free_effort[0]
    assert yielder.update(0.0, free_in + 10.0) < 0.0  # pushed along
    # Something behind: the motor pushes inward harder than free motion needs.
    assert yielder.update(0.05, free_in - ARM.threshold - 1.0) == 0.0 and not yielder.active

