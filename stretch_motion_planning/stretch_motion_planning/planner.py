"""Joint-space path planner for Stretch 4's base rotation, lift, arm and wrist.

Tries three stages, fastest first, and returns the first path found:
  1. direct:   a straight line in joint space from start to goal
  2. template: the "retract, stow, move, unstow, extend" motions a person would use
  3. rrt:      RRT-Connect, for anything the first two can't do
Paths from the template and RRT stages are shortened by skipping any waypoints that a
straight line can go around. No ROS imports.

Obstacles beyond the robot itself come in through an optional `obstacles` hook, an object
with motion_clear(a, b) (the straight line from a to b is clear of them) and path_time(path)
(predicted execution time of a path, s; the obstacle grid's is
stretch_obstacle_grid/tools/grid_planning.py). plan() returns the first path found;
plan_best() gathers several and returns the one path_time says is fastest.
"""

import math
import time
from dataclasses import dataclass, field

import numpy as np

from stretch_motion_planning.collision_model import JOINT_NAMES, joint_vector

THETA, LIFT, ARM, YAW, PITCH, ROLL = range(len(JOINT_NAMES))
WRIST = [YAW, PITCH, ROLL]


@dataclass
class PlanResult:
    success: bool
    message: str
    method: str = ""
    path: list = field(default_factory=list)  # waypoints (np arrays), start to goal
    planning_time: float = 0.0


class StretchPlanner:
    def __init__(self, collision_model, config, seed=None, obstacles=None):
        self.collision = collision_model
        self.obstacles = obstacles
        planner_config = config["planner"]
        self.resolution = joint_vector(planner_config["check_resolution"])
        self.weights = joint_vector(planner_config["distance_weights"])
        self.rrt_step = planner_config["rrt_step"]
        self.goal_bias = planner_config["rrt_goal_bias"]
        self.default_timeout = planner_config["default_timeout"]
        self.escape_radii = planner_config["escape_radii"]
        self.escape_samples = planner_config["escape_samples"]
        self.stow = config["stow"]
        self.rng = np.random.default_rng(seed)

    # --- collision checks -------------------------------------------------------------

    def motion_valid(self, a, b, padding=None):
        """True if every configuration on the straight line from a to b (inclusive of b,
        not a) is collision-free, checked at the configured per-joint resolution."""
        steps = max(1, math.ceil(np.max(np.abs(b - a) / self.resolution)))
        for i in range(1, steps + 1):
            if self.collision.in_collision(a + (b - a) * (i / steps), padding):
                return False
        return self.obstacles is None or self.obstacles.motion_clear(a, b)

    def _escape_padding(self, start, active):
        """A configuration near `start` that clears the padding, reachable from `start`
        without contact, for when the robot is stopped just inside the padding (e.g. a
        few hundredths of a radian past the stow pose). Tries small moves of the active
        joints, nearest first. Returns None if there is none."""
        for radius in self.escape_radii:
            for _ in range(self.escape_samples):
                direction = np.where(active, self.rng.normal(size=len(start)), 0.0)
                norm = np.linalg.norm(self.weights * direction)
                if norm == 0.0:
                    continue
                q = start + direction * (radius / norm)
                if not self.collision.in_collision(q) and self.motion_valid(start, q, padding=0.0):
                    return q
        return None

    def path_valid(self, path):
        return all(self.motion_valid(a, b) for a, b in zip(path[:-1], path[1:]))

    def _describe_invalid(self, q):
        out_of_limits = self.collision.out_of_limits(q)
        if out_of_limits:
            return f"outside joint limits: {out_of_limits}"
        return f"in collision: {self.collision.colliding_pairs(q)}"

    # --- planning ---------------------------------------------------------------------

    def plan(self, start, goal, active=None, timeout=None):
        """Plan from start to goal (arrays ordered as JOINT_NAMES).

        `active` is a boolean mask of joints allowed to move; inactive joints stay at
        their start value (the goal must agree). By default, every joint may move.
        """
        t0 = time.perf_counter()
        start = np.asarray(start, dtype=float)
        goal = np.asarray(goal, dtype=float)
        active = np.ones(len(JOINT_NAMES), bool) if active is None else np.asarray(active, bool)
        timeout = self.default_timeout if not timeout else timeout

        def result(success, message, method="", path=()):
            return PlanResult(success, message, method, [np.array(q) for q in path],
                              time.perf_counter() - t0)

        if not np.allclose(start[~active], goal[~active]):
            moved = [JOINT_NAMES[i] for i in np.flatnonzero(~active & ~np.isclose(start, goal))]
            return result(False, f"Goal moves joints that are locked: {moved}")
        if self.collision.in_collision(goal):
            return result(False, f"Goal is {self._describe_invalid(goal)}")

        # A start inside the padding but not touching anything gets a short move out first.
        prefix = []
        if self.collision.in_collision(start):
            if self.collision.in_collision(start, padding=0.0):
                return result(False, f"Start is {self._describe_invalid(start)}")
            escape = self._escape_padding(start, active)
            if escape is None:
                return result(False, f"Start is within the padding and no small move clears it: "
                                     f"{self.collision.colliding_pairs(start)}")
            prefix, start = [start], escape

        def found(method, message, path):
            return result(True, message, method, prefix + list(path))

        if self.motion_valid(start, goal):
            return found("direct", "Direct path", [start, goal])

        for path in self._template_paths(start, goal, active):
            if self.path_valid(path):
                return found("template", "Template path", self.shortcut(path))

        path = self._rrt_connect(start, goal, active, deadline=t0 + timeout)
        if path is None:
            return result(False, f"No path found within {timeout:.2f} s")
        return found("rrt", "RRT-Connect path", self.shortcut(path))

    def plan_best(self, start, goal, active=None, timeout=None, rrt_runs=3):
        """Like plan(), but gathers the direct path, every valid template path and up to
        rrt_runs RRT-Connect paths (within timeout), shortcut, and returns the one with the
        least predicted time (obstacles.path_time, or the nominal time at the joints' top
        speeds without obstacles)."""
        first = self.plan(start, goal, active, timeout)
        if not first.success or first.method == "direct":
            return first  # a straight line is as good as it gets
        t0 = time.perf_counter() - first.planning_time
        timeout = self.default_timeout if not timeout else timeout
        active = np.ones(len(JOINT_NAMES), bool) if active is None else np.asarray(active, bool)
        start = first.path[0]
        candidates = [(first.method, first.path)]
        for path in self._template_paths(start, np.asarray(goal, dtype=float), active):
            if self.path_valid(path):
                candidates.append(("template", self.shortcut(path)))
        for _ in range(rrt_runs):
            remaining = t0 + timeout - time.perf_counter()
            if remaining <= 0.0:
                break
            path = self._rrt_connect(start, np.asarray(goal, dtype=float), active, time.perf_counter() + remaining)
            if path is not None:
                candidates.append(("rrt", self.shortcut(path)))
        times = [self.path_time(path) for _, path in candidates]
        k = int(np.argmin(times))
        method, path = candidates[k]
        return PlanResult(True, f"Fastest of {len(candidates)} candidates ({times[k]:.2f} s predicted)", method,
                          [np.array(q) for q in path], time.perf_counter() - t0)

    def path_time(self, path):
        """Predicted time (s): the obstacles hook's, or the slowest joint's change at a nominal
        1 unit/s per weight (the RRT's metric) without one."""
        if self.obstacles is not None:
            return self.obstacles.path_time(path)
        return float(sum(np.max(np.abs(self.weights * (b - a))) for a, b in zip(path[:-1], path[1:])))

    def _template_paths(self, start, goal, active):
        """Candidate paths that move one group of joints at a time through the stow pose:
        retract the arm and stow the wrist (in either order), move the base and lift,
        then unstow the wrist and extend the arm (in either order)."""
        stowed_arm = self.stow["arm_joint"]
        stowed_wrist = [self.stow[JOINT_NAMES[i]] for i in WRIST]

        def with_values(q, indices, values):
            q = q.copy()
            for i, v in zip(indices, values):
                if active[i]:
                    q[i] = v
            return q

        retracted = with_values(start, [ARM], [stowed_arm])
        stowed = with_values(start, WRIST, stowed_wrist)
        tucked_start = with_values(retracted, WRIST, stowed_wrist)
        tucked_goal = with_values(tucked_start, [THETA, LIFT], goal[[THETA, LIFT]])
        goal_retracted = with_values(goal, [ARM], [stowed_arm])
        goal_stowed = with_values(goal, WRIST, stowed_wrist)

        for tuck in (retracted, stowed):
            for untuck in (goal_retracted, goal_stowed):
                yield _dedupe([start, tuck, tucked_start, tucked_goal, untuck, goal])

    def _rrt_connect(self, start, goal, active, deadline):
        """RRT-Connect (Kuffner & LaValle 2000) over the active joints."""
        lower = np.where(active, self.collision.lower, start)
        upper = np.where(active, self.collision.upper, start)
        start_tree = Tree(start)
        trees = (start_tree, Tree(goal))

        while time.perf_counter() < deadline:
            tree_a, tree_b = trees
            target = goal if tree_a is start_tree else start
            sample = target if self.rng.random() < self.goal_bias else self.rng.uniform(lower, upper)
            new = self._extend(tree_a, sample)
            if new is not None and self._connect(tree_b, tree_a.nodes[new]) == "reached":
                # tree_b's newest node is tree_a's new node; join the two root paths there
                path = tree_a.path_to(new) + tree_b.path_to(len(tree_b.nodes) - 1)[::-1][1:]
                return path if tree_a is start_tree else path[::-1]
            trees = (tree_b, tree_a)
        return None

    def _steer(self, near, target):
        """At most one rrt_step (weighted distance) from near toward target."""
        delta = target - near
        distance = np.linalg.norm(self.weights * delta)
        if distance <= self.rrt_step:
            return target, True
        return near + delta * (self.rrt_step / distance), False

    def _extend(self, tree, target):
        """Add one step from the nearest node toward target; return its index or None."""
        near = tree.nearest(target, self.weights)
        q, _ = self._steer(tree.nodes[near], target)
        if not self.motion_valid(tree.nodes[near], q):
            return None
        return tree.add(q, near)

    def _connect(self, tree, target):
        """Step from tree toward target until it's reached ('reached') or blocked."""
        near = tree.nearest(target, self.weights)
        while True:
            q, reached = self._steer(tree.nodes[near], target)
            if not self.motion_valid(tree.nodes[near], q):
                return "blocked"
            near = tree.add(q, near)
            if reached:
                return "reached"

    def shortcut(self, path):
        """Greedily replace runs of waypoints with straight lines where possible."""
        out = [path[0]]
        i = 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1 and not self.motion_valid(path[i], path[j]):
                j -= 1
            out.append(path[j])
            i = j
        return out


class Tree:
    def __init__(self, root):
        self.nodes = [np.array(root)]
        self.parents = [-1]
        self._array = np.array([root])

    def add(self, q, parent):
        self.nodes.append(q)
        self.parents.append(parent)
        self._array = np.vstack([self._array, q])
        return len(self.nodes) - 1

    def nearest(self, q, weights):
        return int(np.argmin(np.sum((weights * (self._array - q)) ** 2, axis=1)))

    def path_to(self, i):
        """Configurations from the root to node i."""
        path = []
        while i != -1:
            path.append(self.nodes[i])
            i = self.parents[i]
        return path[::-1]


def _dedupe(path):
    out = [path[0]]
    for q in path[1:]:
        if not np.allclose(q, out[-1]):
            out.append(q)
    return out
