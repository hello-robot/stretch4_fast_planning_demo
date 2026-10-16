"""Tests for the collision model, planner and trajectory timing (no ROS needed)."""

import os

import numpy as np
import pytest

from stretch_motion_planning.collision_model import (
    JOINT_NAMES, StretchCollisionModel, joint_vector, load_config,
)
from stretch_motion_planning.planner import ARM, LIFT, PITCH, THETA, YAW, StretchPlanner
from stretch_motion_planning.timing import time_path

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "stretch4_collision_primitives.yaml")

# Positive wrist pitch points the gripper down.
#            theta  lift  arm   yaw   pitch roll
STOW = np.array([0.0, 0.15, 0.0, 3.14, 0.0, 0.0])
READY = np.array([0.0, 0.8, 0.1, 0.0, 0.0, 0.0])
GRIPPER_DOWN_AT_BASE = np.array([0.0, 0.0, 0.0, 0.0, 1.5, 0.0])


def pose(x, y, z):
    T = np.eye(4)
    T[:3, 3] = [x, y, z]
    return T


@pytest.fixture(scope="module")
def config():
    return load_config(CONFIG_PATH)


@pytest.fixture
def collision(config):
    return StretchCollisionModel(config)


@pytest.fixture
def planner(collision, config):
    return StretchPlanner(collision, config, seed=0)


def test_joint_order_matches_driver_names(collision):
    assert JOINT_NAMES[ARM] == "arm_joint"
    assert collision.model.nq == len(JOINT_NAMES)


def test_known_configurations(collision):
    assert not collision.in_collision(STOW)
    assert not collision.in_collision(READY)
    assert ("wrist_pitch", "base") in collision.colliding_pairs(GRIPPER_DOWN_AT_BASE)


def test_joint_limits(collision):
    q = READY.copy()
    q[LIFT] = collision.upper[LIFT] + 0.01
    assert collision.in_collision(q)
    assert collision.out_of_limits(q) == ["lift_joint"]


def test_arm_box_extends_with_the_arm(collision):
    # Find where the gripper tip ends up with the arm retracted, then put a small block
    # just past the extended arm's reach along the arm direction.
    retracted = READY.copy()
    extended = READY.copy()
    extended[ARM] = 0.5
    arm_pose = dict((name, T) for name, _, T in collision.shape_poses(retracted))["arm"]
    block = pose(*(arm_pose.translation + arm_pose.rotation[:, 0] * 0.35))
    collision.set_obstacles([{"type": "sphere", "dims": [0.02], "pose": block}])
    assert not collision.in_collision(retracted)
    assert collision.in_collision(extended)


def test_floor(collision):
    # Wrist pointing down, arm out past the base, lift all the way down: hits the floor.
    q = np.array([0.0, 0.0, 0.5, 0.0, 1.5, 0.0])
    assert ("fingers", "floor") in collision.colliding_pairs(q)


def test_direct_path(planner):
    goal = READY.copy()
    goal[LIFT] = 0.5
    result = planner.plan(READY, goal)
    assert result.success and result.method == "direct"
    assert len(result.path) == 2


def test_rejects_start_in_collision(planner):
    result = planner.plan(GRIPPER_DOWN_AT_BASE, READY)
    assert not result.success
    assert "Start is in collision" in result.message


def test_locked_joint_must_not_move(planner):
    goal = READY.copy()
    goal[THETA] = 0.5
    active = np.ones(len(JOINT_NAMES), bool)
    active[THETA] = False
    assert not planner.plan(READY, goal, active).success


def test_plans_around_a_table(planner, collision, config):
    # Reach from above a table to below it: the gripper has to go around the edge.
    collision.set_obstacles([{"type": "box", "dims": [0.5, 1.0, 0.04], "pose": pose(0.55, 0.0, 0.72)}])
    above = np.array([0.0, 0.9, 0.3, 0.0, 1.0, 0.0])
    below = np.array([0.0, 0.3, 0.3, 0.0, 0.0, 0.0])
    active = np.ones(len(JOINT_NAMES), bool)
    active[THETA] = False
    result = planner.plan(above, below, active, timeout=5.0)
    assert result.success, result.message
    assert result.method in ("template", "rrt")
    # Re-check 4x finer with no padding: no contact anywhere along the path.
    unpadded = StretchCollisionModel({**config, "padding": 0.0})
    unpadded.set_obstacles([{"type": "box", "dims": [0.5, 1.0, 0.04], "pose": pose(0.55, 0.0, 0.72)}])
    fine = StretchPlanner(unpadded, config)
    fine.resolution = planner.resolution / 4
    assert fine.path_valid(result.path)
    assert all(q[THETA] == 0.0 for q in result.path)


def test_timing_respects_limits(config):
    max_velocity = joint_vector(config["timing"]["max_velocity"])
    max_acceleration = joint_vector(config["timing"]["max_acceleration"])
    spacing = joint_vector(config["timing"]["waypoint_spacing"])
    path = [STOW, READY, np.array([0.5, 0.6, 0.4, 1.0, -0.5, 0.3])]

    positions, times = time_path(path, max_velocity, max_acceleration, spacing)

    assert np.allclose(positions[0], path[0]) and np.allclose(positions[-1], path[-1])
    assert np.all(np.diff(times) > 0)
    steps = np.abs(np.diff(positions, axis=0))
    assert np.all(steps <= spacing + 1e-9)
    velocities = steps / np.diff(times)[:, None]
    assert np.all(velocities <= max_velocity * 1.001)
    # Every original waypoint is an output waypoint (the robot stops there).
    for q in path:
        assert any(np.allclose(q, p) for p in positions)


def test_wrist_yaw_stow_from_ready(planner):
    """Stowing from the ready pose: the gripper swings back over the arm."""
    result = planner.plan(READY, STOW)
    assert result.success, result.message
    assert result.path[-1][YAW] == pytest.approx(3.14)


def test_pitch_down_is_positive(collision):
    down = READY.copy()
    down[PITCH] = 1.0
    poses = {name: T for name, _, T in collision.shape_poses(down)}
    assert poses["fingers"].translation[2] < poses["wrist"].translation[2]


def test_start_just_inside_the_padding(planner, collision):
    """A robot stopped slightly past the stow pose is inside the padding but not touching:
    the plan starts with a short move out of the padding."""
    past_stow = STOW.copy()
    past_stow[YAW] = 3.18
    assert collision.in_collision(past_stow) and not collision.in_collision(past_stow, padding=0.0)
    result = planner.plan(past_stow, READY)
    assert result.success, result.message
    assert np.allclose(result.path[0], past_stow)
    assert not collision.in_collision(result.path[1])
    assert planner.motion_valid(result.path[0], result.path[1], padding=0.0)
