"""Time a joint-space path as a series of straight-line moves.

Each segment between path waypoints is one synchronized move: every joint starts and
stops together on the same trapezoidal speed profile (scaled to its distance), sized so
no joint exceeds its velocity or acceleration limit. The robot stops at each waypoint,
so it follows the straight lines the planner collision-checked. Segments are split into
evenly spaced output waypoints so a trajectory server that tracks waypoints one at a
time still stays near those lines. No ROS imports.
"""

import math

import numpy as np


def trapezoid_duration(distance, max_velocity, max_acceleration):
    """Shortest rest-to-rest time to move `distance`."""
    distance = abs(distance)
    if distance * max_acceleration < max_velocity ** 2:  # never reaches max velocity
        return 2.0 * math.sqrt(distance / max_acceleration)
    return distance / max_velocity + max_velocity / max_acceleration


def segment_profile(delta, max_velocity, max_acceleration):
    """(duration, accel_fraction) of a synchronized move by `delta`.

    The profile's shape (the fraction of the move spent accelerating, and again
    decelerating) comes from the joint that needs the longest on its own. The duration
    is then stretched until every joint, following that shape, stays within its limits.
    """
    durations = [trapezoid_duration(d, v, a) for d, v, a in zip(delta, max_velocity, max_acceleration)]
    slowest = int(np.argmax(durations))
    if durations[slowest] == 0.0:
        return 0.0, 0.5
    v, a = max_velocity[slowest], max_acceleration[slowest]
    accel_time = min(v / a, 0.5 * durations[slowest])
    f = accel_time / durations[slowest]

    # With accel fraction f, a joint moving d peaks at d / (T (1 - f)) and accelerates
    # at d / (T^2 f (1 - f)).
    d = np.abs(delta)
    duration = max(
        np.max(d / (max_velocity * (1.0 - f))),
        np.max(np.sqrt(d / (max_acceleration * f * (1.0 - f)))),
    )
    return float(duration), f


def progress(t, duration, accel_fraction):
    """Fraction (0 to 1) of a move completed at time t."""
    accel_time = accel_fraction * duration
    peak_rate = 1.0 / (duration - accel_time)  # fraction of the move per second at cruise
    if t <= accel_time:
        return 0.5 * peak_rate / accel_time * t ** 2
    if t >= duration - accel_time:
        return 1.0 - 0.5 * peak_rate / accel_time * (duration - t) ** 2
    return peak_rate * (t - 0.5 * accel_time)


def time_at_progress(fraction, duration, accel_fraction):
    """Inverse of progress(), by bisection (progress is monotonic)."""
    lo, hi = 0.0, duration
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if progress(mid, duration, accel_fraction) < fraction:
            lo = mid
        else:
            hi = mid
    return hi


def time_path(path, max_velocity, max_acceleration, waypoint_spacing):
    """Return (positions, times): the path split into waypoints at most
    `waypoint_spacing` apart per joint, and each waypoint's time from start."""
    positions = [np.array(path[0], dtype=float)]
    times = [0.0]

    for a, b in zip(path[:-1], path[1:]):
        delta = b - a
        duration, accel_fraction = segment_profile(delta, max_velocity, max_acceleration)
        if duration == 0.0:
            continue
        segment_start = times[-1]
        steps = max(1, math.ceil(np.max(np.abs(delta) / waypoint_spacing)))
        for i in range(1, steps + 1):
            fraction = i / steps
            positions.append(a + delta * fraction)
            times.append(segment_start + time_at_progress(fraction, duration, accel_fraction))

    return positions, times
