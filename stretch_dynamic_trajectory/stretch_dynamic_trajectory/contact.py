"""Detecting contact from joint effort while following a trajectory (no ROS).

Neither the firmware guards nor the servos' stall flags catch a person's hand reliably: a hand
that gives way never stalls a joint, and the arm's guard trips late and then leaves the arm
limp (contact tests, 2026-10-19, in notes/stretch4_fast_planner/dynamic_obstacles_design.md).
So contact is a rise in each moving joint's effort over its level in free motion:

- push: the effort in the direction of motion. The arm and lift steppers report signed effort
  (the lift's includes holding against gravity), so it's effort times the direction; the
  Feetech wrists report unsigned effort, so it's the effort itself.
- The free level is learned per direction of motion, as the lowest push over the last
  baseline_window s, once the joint has moved that way steadily for settle s. It falls at once
  but rises at most free_level_rate points/s (0.5), far slower than a contact (10-15 points/s and
  up), so a contact that starts before the joint has settled again doesn't become the new free
  level, while slow real changes (friction along the travel) are still followed. It's kept
  across stops: the dynamic server stops and restarts often (slowdowns, creeping), and starting from
  rest shifts the effort by about as much as a contact does (the lift's by about 9 points), so
  the level is not re-learned after each start. Joints with a start transient (the wrists
  overshoot to 2-3x the commanded speed, with effort spikes up to 26, for about 0.25 s) aren't
  checked for settle s after each start.
- Contact: push more than the joint's threshold over the free level.

Thresholds: 8 points for the arm (Elaine, 2026-10-23: at 12 the arm's firmware guard tripped
first, leaving it limp to be shoved back) and 12 for the others, from the contact tests
(2026-10-19) and a free-motion session through the dynamic server (2026-10-23). Wrist pitch at 8 tripped with
nothing touching it: coming up from 0.5 rad down, the gripper's gravity load grows toward
level, and the wrists' effort comes in steps of about 4.33, so two steps beat 8. With the wrist
rolled, yaw carries gravity the same way, so both wrist joints use 12 (Elaine, 2026-10-23).
Pitch going down is firmer than the rest even so, because the unsigned effort misses load that
gravity helps carry. Roll and base rotation aren't checked (Elaine, 2026-10-19).
"""

import collections
import dataclasses

import numpy as np


@dataclasses.dataclass
class ContactSettings:
    """How one joint's contact is detected."""
    threshold: float                     # effort points over the free level, moving the + way
    threshold_negative: float = None     # moving the - way, if different
    signed: bool = True                  # effort is signed (steppers) rather than a magnitude
    start_transient: bool = False        # skip `settle` s after each start


# Wrist pitch: positive is down (stretch4_body and the driver).
CONTACT_SETTINGS = {
    "arm_joint": ContactSettings(8.0),
    "lift_joint": ContactSettings(12.0),
    "wrist_yaw_joint": ContactSettings(12.0, signed=False, start_transient=True),
    "wrist_pitch_joint": ContactSettings(12.0, signed=False, start_transient=True),
}


class ContactDetector:
    """Effort-rise contact detection for one joint; see the module docstring."""

    def __init__(self, settings, settle=0.3, baseline_window=1.0, free_level_rate=0.5, moving_speed=1e-3,
                 threshold_scale=1.0):
        self.settings = settings
        self.threshold_scale = threshold_scale  # multiplies the thresholds: below 1 is gentler
        self.settle = settle
        self.baseline_window = baseline_window
        self.free_level_rate = free_level_rate
        self.moving_speed = moving_speed  # commanded speed (joint units/s) that counts as moving
        self.free_level = {}  # direction (+1 or -1) -> (learned free-motion push, when)
        self._recent = collections.deque()  # (t, push) while moving steadily in self._direction
        self._direction = 0
        self._moving_since = None  # when the joint started moving in self._direction

    def push(self, effort, direction):
        return direction * effort if self.settings.signed else abs(effort)

    def threshold(self, direction):
        s = self.settings
        return self.threshold_scale * (s.threshold if direction > 0 or s.threshold_negative is None
                                       else s.threshold_negative)

    def update(self, t, effort, commanded_velocity):
        """Take one sample (s, effort, commanded velocity); returns the rise over the free
        level if it's contact, else None."""
        direction = 0 if abs(commanded_velocity) < self.moving_speed else int(np.sign(commanded_velocity))
        if direction == 0:
            self._moving_since = None
            self._recent.clear()
            return None
        if direction != self._direction or self._moving_since is None:
            self._direction, self._moving_since = direction, t
            self._recent.clear()
        push = self.push(effort, direction)
        settled = t - self._moving_since >= self.settle
        free = self.free_level.get(direction, (None, None))[0]
        if settled:
            # The lowest push lately: a contact's rise doesn't raise it until it has lasted a
            # whole window, and by then it has tripped.
            self._recent.append((t, push))
            while self._recent and self._recent[0][0] < t - self.baseline_window:
                self._recent.popleft()
            lowest = min(p for _, p in self._recent)
            if free is not None:
                # It may rise only for time spent moving steadily, not stopped or settling.
                steady = t - max(self.free_level[direction][1], self._moving_since + self.settle)
                lowest = min(lowest, free + self.free_level_rate * max(0.0, steady))
            free = lowest
            self.free_level[direction] = (free, t)
        if free is None or (self.settings.start_transient and not settled):
            return None
        rise = push - free
        return rise if rise > self.threshold(direction) else None


# --- Yielding to a push ----------------------------------------------------------------------
#
# Elaine (2026-10-23): a joint that's pushed should move with the push, not hold firm until its
# firmware guard trips and leaves it limp. Holding still, a stepper's effort sits anywhere
# between its free-motion efforts in the two directions (static friction, and whichever way it
# last moved), so a push is effort beyond that band by more than a threshold: pushing the arm
# in makes the motor push out harder, above the band; pushing the lift down makes it hold up
# harder, likewise. While yielding in direction d, the motor needs the free-motion effort for d
# (less, when someone pushes it along), so the push is how far the effort falls short of that;
# the yield speed follows it. If the motor instead has to push harder than free motion, by the
# threshold, the yield has run into something else, and it stops at once.

@dataclasses.dataclass
class YieldSettings:
    """How one stepper joint yields to a push. Efforts are signed, as reported."""
    free_effort: tuple      # effort moving freely (- direction, + direction): medians, 2026-10-23
    threshold: float        # effort beyond the band between them that starts a yield
    release: float          # push (effort points) under which yielding stops
    gain: float             # yield speed (joint units/s) per effort point of push over release


# Free-motion efforts from free_motion_session (2026-10-23, 28 moves): arm in -4.8 (-5.4..-3.9
# p5..p95), out +9.7 (6.4..12.4); lift down +21.6 (19.3..23.6), up +39.5 (37.7..42.2).
YIELD_SETTINGS = {
    "arm_joint": YieldSettings((-4.8, 9.7), 8.0, 3.0, 0.005),
    "lift_joint": YieldSettings((21.6, 39.5), 12.0, 4.0, 0.004),
}


class Yielding:
    """Watches one held stepper joint for a push, and yields to it."""

    def __init__(self, settings, max_speed, threshold=None, trigger_time=0.1, release_time=0.5):
        self.settings = settings
        self.max_speed = max_speed  # joint units/s
        self.threshold = settings.threshold if threshold is None else threshold
        self.trigger_time = trigger_time  # s a push must last to start a yield
        self.release_time = release_time  # s the push must stay under release to end it
        self.direction = 0  # while yielding: which way
        self._pushed = (0, None)  # (direction, since) of a push seen while held
        # After a yield, another starts only once the effort has been back inside the holding
        # band: a person lets go between pushes, while an effort that stays out of the band
        # (miscalibrated, or something leaning on the joint) mustn't walk it away a yield at a time.
        self._armed = True
        self._released_since = None

    @property
    def active(self):
        return self.direction != 0

    def pushed(self, effort):
        """The way a push on the held joint points (+1 or -1), or 0."""
        lo, hi = self.settings.free_effort
        if effort > hi + self.threshold:
            return -1  # the motor pushes + to hold against it
        if effort < lo - self.threshold:
            return 1
        return 0

    def watch(self, t, effort):
        """While held: True (and yielding from now) once a push has lasted trigger_time."""
        direction = self.pushed(effort)
        if direction == 0:
            self._armed = True
        if not self._armed:
            return False
        if direction == 0 or direction != self._pushed[0]:
            self._pushed = (direction, t if direction else None)
            return False
        if t - self._pushed[1] < self.trigger_time:
            return False
        self.direction, self._released_since, self._pushed, self._armed = direction, None, (0, None), False
        return True

    def stop(self):
        """End a yield now (it went as far as it may)."""
        self.direction = 0

    def push(self, effort):
        """While yielding: how hard (effort points) it's being pushed along."""
        free = self.settings.free_effort[1 if self.direction > 0 else 0]
        return -self.direction * (effort - free)

    def update(self, t, effort):
        """While yielding: the velocity to command; 0 once the push has eased for release_time,
        or at once if the yield runs into something (and yielding has ended)."""
        push = self.push(effort)
        if push < -self.threshold:
            self.direction = 0
            return 0.0
        if push < self.settings.release:
            self._released_since = self._released_since if self._released_since is not None else t
            if t - self._released_since >= self.release_time:
                self.direction = 0
                return 0.0
        else:
            self._released_since = None
        speed = float(np.clip(self.settings.gain * (push - self.settings.release), 0.0, self.max_speed))
        return self.direction * speed

