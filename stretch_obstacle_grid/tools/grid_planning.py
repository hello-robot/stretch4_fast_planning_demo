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

Off the robot, the planner plans against a snapshot of the monitor's grid (its get_grid
service): encode_snapshot / decode_snapshot carry the three layers the checks use (obstacles,
the moving ones, unsafe space), cropped to the voxels within reach (crop_voxels), about 35 KB
compressed; a grid made with no LiDARs takes them with set_labels.
"""

import math
import time
import zlib

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

    def blocked_at(self, q):
        """True if configuration q is already as close to an obstacle as a step's blocked."""
        return self.checker.blocked_at(self.grid, self._body(q), self.theta_slack)

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


# --- Grid snapshots ---------------------------------------------------------------------------

OBSTACLE_LABELS = (int(og.OCCUPIED), int(og.STATIC))
UNSAFE_LABELS = (int(og.UNKNOWN), int(og.CONTAMINATED_SHADOW))


def crop_voxels(grid, r_max):
    """How many of the grid's voxels (an og.CylindricalGrid; they're ordered by radial bin
    first) lie in radial bins starting within r_max of the axis."""
    config = grid.config
    first_outside = int(np.clip(np.ceil((r_max - config.r_min) / config.cell), 0, grid.radial_bins))
    return grid.size if first_outside >= grid.radial_bins else int(grid.index(first_outside, 0, 0))


def encode_snapshot(labels, voxels):
    """The first `voxels` labels as zlib-compressed bit layers: obstacles, moving obstacles,
    unsafe space."""
    labels = np.asarray(labels)[:voxels]
    layers = (np.isin(labels, OBSTACLE_LABELS), labels == int(og.OCCUPIED), np.isin(labels, UNSAFE_LABELS))
    return zlib.compress(b"".join(np.packbits(layer).tobytes() for layer in layers), 1)


def decode_snapshot(data, voxels, size):
    """Labels for a whole grid of `size` voxels from encode_snapshot's bytes: occupied, static,
    unknown (for all unsafe space) or free; voxels past the crop unknown."""
    bits = np.unpackbits(np.frombuffer(zlib.decompress(data), dtype=np.uint8))
    stride = (voxels + 7) // 8 * 8
    obstacle, moving, unsafe = (bits[k * stride:k * stride + voxels].astype(bool) for k in range(3))
    labels = np.full(size, int(og.UNKNOWN), dtype=np.uint8)
    head = np.full(voxels, int(og.FREE), dtype=np.uint8)
    head[unsafe] = int(og.UNKNOWN)
    head[obstacle] = int(og.STATIC)
    head[moving] = int(og.OCCUPIED)
    labels[:voxels] = head
    return labels

