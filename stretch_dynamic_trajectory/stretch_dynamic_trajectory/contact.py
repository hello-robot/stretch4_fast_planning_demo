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
  but rises at most free_level_rate points/s, far slower than a contact (10-15 points/s and
  up), so a contact that starts before the joint has settled again doesn't become the new free
  level, while slow real changes (friction along the travel) are still followed. It's kept
  across stops: the dynamic server stops and restarts often (slowdowns, creeping), and starting from
  rest shifts the effort by about as much as a contact does (the lift's by about 9 points), so
  the level is not re-learned after each start. Joints with a start transient (the wrists
  overshoot to 2-3x the commanded speed, with effort spikes up to 26, for about 0.25 s) aren't
  checked for settle s after each start.
- Contact: push more than the joint's threshold over the free level.

Thresholds from the contact tests: 12 points for the arm, lift and wrist yaw; 8 for wrist
pitch. Pitch going down is firm even at 8, because the unsigned effort misses load that
gravity helps carry. Going up it carries some gravity (free level about 4.3), and the stop
Elaine liked came at 13, a rise of 8.7 over that. The wrists' effort comes in steps of about
4.33, so 8 is two steps. Roll and base rotation aren't checked (Elaine, 2026-10-19).
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
    "arm_joint": ContactSettings(12.0),
    "lift_joint": ContactSettings(12.0),
    "wrist_yaw_joint": ContactSettings(12.0, signed=False, start_transient=True),
    "wrist_pitch_joint": ContactSettings(8.0, signed=False, start_transient=True),
}


class ContactDetector:
    """Effort-rise contact detection for one joint; see the module docstring."""

    def __init__(self, settings, settle=0.3, baseline_window=1.0, free_level_rate=2.0, moving_speed=1e-3,
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
