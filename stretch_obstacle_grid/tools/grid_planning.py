"""The obstacle grid as the motion planner sees it (no ROS).

GridObstacles is the planner's `obstacles` hook (stretch_motion_planning.planner): whether a
straight-line motion is clear of the grid's obstacles, and how long a path would take. It
checks motion exactly as the obstacle monitor checks the trajectory server's plans: the
planner's configurations, cut into steps at the planner's check resolution, posed as the
robot's planning body, through MotionCheck.

- Blocked: a step sweeps an obstacle (occupied or static voxel). Only obstacles are hard
  constraints.
- Time: each step takes at least its nominal time (the slowest joint's change at that joint's
  top speed) and at least MotionCheck's minimum for it: contact speed through unsafe space
  (unknown, contaminated shadow), slower near obstacles. So a path through unknown space is
  allowed but costs what it would take, and clean shadow (the arm's own, below it) costs
  nothing extra (Elaine, 2026-10-23: the arm must still be able to move down).

Configurations are planner order (base rotation relative to base_footprint now, lift, arm,
wrist yaw, pitch, roll), as the monitor's check_motion uses.
"""

import math
import time

import numpy as np

import stretch_obstacle_grid as og
from grid_setup import FastPlanningBody, step_braking


class GridObstacles:
    def __init__(self, grid, robot, resolution, max_velocity, limits=None, age=0.0, theta_slack=0.0, coarse=4.0):
        """grid: an og.ObstacleGrid with current labels and tables; robot: grid_setup.RobotShapes
        with the fingers set as they'll be (posed here through a FastPlanningBody, so not shared
        with another thread); resolution, max_velocity: per joint,
        planner order. age (s) and theta_slack (rad) as for MotionCheck.check. coarse: how many
        times the resolution motion_clear tries first (see there)."""
        self.grid = grid
        self.robot = robot
        self.pose = FastPlanningBody(robot)
        self.resolution = np.asarray(resolution, dtype=float)
        self.max_velocity = np.asarray(max_velocity, dtype=float)
        self.checker = og.MotionCheck(limits or og.MotionLimits())
        self.age = age
        self.theta_slack = theta_slack
        self.coarse = coarse
        self._bodies = {}  # posed planning bodies by configuration (RRT reuses its nodes)
        self.stats = {"edges": 0, "steps": 0, "pose_s": 0.0, "check_s": 0.0}

    def _body(self, q):
        key = tuple(np.round(q, 6))
        body = self._bodies.get(key)
        if body is None:
            body = self._bodies[key] = self.pose(q)
        return body

    def samples(self, a, b, scale=1.0):
        """Configurations from a to b, at most scale times the check resolution apart."""
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        n = max(1, math.ceil(np.max(np.abs(b - a) / (scale * self.resolution))))
        return [a + (b - a) * (k / n) for k in range(n + 1)]

    def steps(self, a, b):
        """MotionCheck's StepLimits along the straight line from a to b, and the samples."""
        qs = self.samples(a, b)
        t0 = time.perf_counter()
        bodies = [self._body(q) for q in qs]
        t1 = time.perf_counter()
        steps = self.checker.check(self.grid, bodies, self.age, self.theta_slack, step_braking(qs))
        t2 = time.perf_counter()
        self.stats["edges"] += 1
        self.stats["steps"] += len(steps)
        self.stats["pose_s"] += t1 - t0
        self.stats["check_s"] += t2 - t1
        return steps, qs

    def motion_clear(self, a, b):
        """True if the straight line from a to b sweeps no obstacle (MotionCheck.first_blocked:
        check()'s blocked test alone, stopping at the first blocked step). Most of the time goes
        to posing the robot at every sample, so it's checked at `coarse` times the resolution
        first (about 4 cm steps): MotionCheck bounds each step's whole sweep, larger for larger
        steps, so clear there is clear; only a coarse block is checked again at the resolution."""
        return self._clear(a, b, self.coarse) or (self.coarse > 1.0 and self._clear(a, b, 1.0))

    def _clear(self, a, b, scale):
        qs = self.samples(a, b, scale)
        t0 = time.perf_counter()
        bodies = [self._body(q) for q in qs]
        t1 = time.perf_counter()
        blocked = self.checker.first_blocked(self.grid, bodies, self.theta_slack)
        self.stats["edges"] += 1
        self.stats["steps"] += len(qs) - 1
        self.stats["pose_s"] += t1 - t0
        self.stats["check_s"] += time.perf_counter() - t1
        return blocked < 0

    def path_time(self, path):
        """Predicted execution time (s) of a path of waypoints: per step, the larger of its
        nominal time and MotionCheck's minimum. Infinite if any step is blocked."""
        total = 0.0
        for a, b in zip(path[:-1], path[1:]):
            steps, qs = self.steps(a, b)
            for step, q0, q1 in zip(steps, qs[:-1], qs[1:]):
                if step.blocked:
                    return math.inf
                nominal = float(np.max(np.abs(q1 - q0) / self.max_velocity))
                total += max(nominal, step.min_duration)
        return total
