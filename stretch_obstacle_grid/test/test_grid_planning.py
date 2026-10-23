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
