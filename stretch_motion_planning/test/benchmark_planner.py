#!/usr/bin/env python3
"""Plan between random collision-free start/goal pairs and report which stage solved
each one, how long planning took, and whether every returned path is valid, re-checked
at 1/4 of the planner's collision-check resolution: with the configured padding, and
with no padding (actual contact between the primitives).

    python3 benchmark_planner.py [--queries 200] [--obstacles] [--rotate]
"""

import argparse
import os
import time
from collections import Counter

import numpy as np

from stretch_motion_planning.collision_model import StretchCollisionModel, load_config
from stretch_motion_planning.planner import StretchPlanner, THETA, PITCH

DEFAULT_CONFIG = os.path.join(
    os.path.dirname(__file__), "..", "config", "stretch4_collision_primitives.yaml"
)

# A table in front of the robot (the arm points along +x) and a shelf above it.
OBSTACLES = [
    {"type": "box", "dims": [0.5, 0.8, 0.04], "pose": np.array([
        [1, 0, 0, 0.55], [0, 1, 0, 0.0], [0, 0, 1, 0.72], [0, 0, 0, 1]], dtype=float)},
    {"type": "box", "dims": [0.4, 0.8, 0.03], "pose": np.array([
        [1, 0, 0, 0.6], [0, 1, 0, 0.0], [0, 0, 1, 1.1], [0, 0, 0, 1]], dtype=float)},
]


def random_free_configuration(planner, lower, upper, rng):
    while True:
        q = rng.uniform(lower, upper)
        if not planner.collision.in_collision(q):
            return q


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--obstacles", action="store_true", help="add a table and a shelf")
    parser.add_argument("--rotate", action="store_true", help="let the base rotate")
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = load_config(args.config)
    collision = StretchCollisionModel(config)
    if args.obstacles:
        collision.set_obstacles(OBSTACLES)
    planner = StretchPlanner(collision, config, seed=args.seed)
    fine = StretchPlanner(collision, config)
    fine.resolution = planner.resolution / 4
    unpadded_collision = StretchCollisionModel({**config, "padding": 0.0})
    if args.obstacles:
        unpadded_collision.set_obstacles(OBSTACLES)
    fine_unpadded = StretchPlanner(unpadded_collision, config)
    fine_unpadded.resolution = fine.resolution

    lower, upper = collision.lower.copy(), collision.upper.copy()
    lower[PITCH], upper[PITCH] = -1.57, 0.56  # the URDF's pitch range is a placeholder
    active = np.ones(len(lower), bool)
    if not args.rotate:
        lower[THETA] = upper[THETA] = 0.0
        active[THETA] = False

    rng = np.random.default_rng(args.seed)
    methods = Counter()
    times = {}
    within_padding = 0
    in_contact = 0
    for _ in range(args.queries):
        start = random_free_configuration(planner, lower, upper, rng)
        goal = random_free_configuration(planner, lower, upper, rng)
        result = planner.plan(start, goal, active, timeout=args.timeout)
        method = result.method if result.success else "failed"
        methods[method] += 1
        times.setdefault(method, []).append(result.planning_time)
        if result.success:
            assert np.allclose(result.path[0], start) and np.allclose(result.path[-1], goal)
            if not fine_unpadded.path_valid(result.path):
                in_contact += 1
            elif not fine.path_valid(result.path):
                within_padding += 1

    print(f"{args.queries} queries, obstacles={args.obstacles}, rotate={args.rotate}")
    for method in ("direct", "template", "rrt", "failed"):
        if methods[method]:
            t = np.array(times[method]) * 1000
            print(f"  {method:9s} {methods[method]:4d}  median {np.median(t):7.1f} ms   max {t.max():7.1f} ms")
    print(f"  paths closer than the padding between checks: {within_padding}")
    print(f"  paths in contact between checks: {in_contact}")


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    print(f"total {time.perf_counter() - t0:.1f} s")
