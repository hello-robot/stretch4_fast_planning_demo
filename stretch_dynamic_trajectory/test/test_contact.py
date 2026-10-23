"""Tests for contact detection from joint effort (no ROS)."""

import numpy as np
import pytest

from stretch_dynamic_trajectory.contact import CONTACT_SETTINGS, ContactDetector, ContactSettings


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
