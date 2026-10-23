"""Tests for planning against the obstacle grid off the robot (grid snapshots)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import grid_planning as gp  # noqa: E402
import stretch_obstacle_grid as og  # noqa: E402


def test_crop_is_the_voxels_within_reach():
    grid = og.CylindricalGrid(og.GridConfig())
    n = gp.crop_voxels(grid, 1.15)
    r = grid.config.r_min + (grid.cells()[:, 0] + 0.5) * grid.config.cell
    assert np.all(r[:n] <= 1.15 + grid.config.cell) and np.all(r[n:] > 1.15)
    assert gp.crop_voxels(grid, 10.0) == grid.size


def test_a_snapshot_keeps_what_the_checks_use():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 7, 5000).astype(np.uint8)
    out = gp.decode_snapshot(gp.encode_snapshot(labels, 3001), 3001, 5000)
    head = labels[:3001]
    assert np.array_equal(out[:3001] == og.OCCUPIED, head == og.OCCUPIED)
    assert np.array_equal(out[:3001] == og.STATIC, head == og.STATIC)
    assert np.array_equal(np.isin(out[:3001], gp.UNSAFE_LABELS), np.isin(head, gp.UNSAFE_LABELS))
    assert np.all(out[3001:] == og.UNKNOWN)


def fingertips_blocked_by_static_voxels(q):
    """A grid (no LiDARs) whose only obstacles are static voxels in the outer 5 cm of the
    gripper box at q (furthest from the robot's axis), as if the robot had stopped with its
    fingertips against something, and a GridObstacles on it."""
    from grid_setup import FastPlanningBody, RobotShapes
    from stretch_motion_planning.collision_model import joint_vector

    robot = RobotShapes()
    shape = FastPlanningBody(robot)(q).shapes()[robot.shape_names().index("gripper_box")]
    pose, half = np.asarray(shape.pose, dtype=float), np.asarray(shape.dims, dtype=float)
    grid = og.ObstacleGrid(og.GridConfig(), [])
    centers = grid.grid.centers()
    local = (centers - pose[:3, 3]) @ pose[:3, :3]
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    corners = pose[:3, 3] + (signs * half) @ pose[:3, :3].T
    reach = np.max(np.hypot(corners[:, 0], corners[:, 1]))  # of the gripper box from the axis
    labels = np.full(grid.grid.size, int(og.FREE), dtype=np.uint8)
    inside = np.all(np.abs(local) <= half, axis=1) & (np.hypot(centers[:, 0], centers[:, 1]) > reach - 0.05)
    assert inside.any()
    labels[inside] = int(og.STATIC)
    grid.set_labels(labels)
    config = robot.config
    return gp.GridObstacles(grid, robot, joint_vector(config["planner"]["check_resolution"]),
                            joint_vector(config["timing"]["max_velocity"]), margin=0.05)


def test_the_robot_can_withdraw_slowly_from_what_it_touches():
    from stretch_motion_planning.collision_model import StretchCollisionModel
    from stretch_motion_planning.planner import StretchPlanner

    q = np.array([0.0, 0.6, 0.3, 0.0, 0.0, 0.0])
    obstacles = fingertips_blocked_by_static_voxels(q)
    assert obstacles.blocked_at(q)
    up = q + [0.0, 0.2, 0.0, 0.0, 0.0, 0.0]
    assert obstacles.motion_clear(q, up)
    assert obstacles.motion_clear(q, q + [0.0, 0.0, -0.2, 0.0, 0.0, 0.0])
    # Slow while it's still touching: escape speed at the gripper for at least the first steps.
    nominal = float(np.max(np.abs(up - q) / obstacles.max_velocity))
    assert obstacles.path_time([q, up]) > nominal + 0.01 / obstacles.checker.limits.escape_speed
    # Further into it is still blocked.
    assert not obstacles.motion_clear(q, q + [0.0, 0.0, 0.05, 0.0, 0.0, 0.0])

    planner = StretchPlanner(StretchCollisionModel(obstacles.robot.config), obstacles.robot.config, seed=0,
                             obstacles=obstacles)
    result = planner.plan(q, up)
    assert result.success, result.message

    # Without escapes, a start touching an obstacle can't move.
    limits = og.MotionLimits()
    limits.escape_speed = 0.0
    obstacles.checker = og.MotionCheck(limits)
    assert not obstacles.motion_clear(q, up)
