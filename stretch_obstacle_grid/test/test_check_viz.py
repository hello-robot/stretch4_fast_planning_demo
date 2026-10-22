"""Tests for check_viz: what the check markers say, how stale blocks and replaced plans fade,
and that markers recorded at one heading stay put in the world as the base turns.

    python3 -m pytest -q test/test_check_viz.py   (from the package, with the workspace sourced)
"""

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from std_msgs.msg import Header
from visualization_msgs.msg import Marker

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import check_viz as cv  # noqa: E402
import stretch_obstacle_grid as og  # noqa: E402

NAMES = ["wrist_yaw", "gripper_box"]


def box(center, half=(0.05, 0.05, 0.05)):
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 3] = center
    return og.SelfShape.box(pose, np.array(half, dtype=np.float32))


def body(x):
    """A wrist at (x, 0, 0.5) with the gripper box reaching 0.2 m further out along +x."""
    return og.SelfModel([box((x, 0.0, 0.5)), box((x + 0.1, 0.0, 0.5), (0.1, 0.04, 0.03))])


def step(blocked=False, unsafe=False, min_duration=0.0):
    return SimpleNamespace(blocked=blocked, unsafe=unsafe, min_duration=min_duration, shape=1,
                           clearance_moving=0.3, clearance_occupied=0.2)


def hit(voxel, in_both=True):
    return SimpleNamespace(voxel=voxel, shape=1, in_both=in_both)


def blocked_record(stamp, voxels, heading=0.0, centers=None):
    bodies = [body(0.3), body(0.4)]
    centers = np.array(centers if centers is not None else [[0.7, 0.0, 0.5 + 0.02 * v] for v in voxels])
    focus = cv.Focus(step=0, status=cv.BLOCKED, shapes=bodies[1].shapes(), text="BLOCKED",
                     voxels=np.array(voxels), centers=centers, colors=[(1, 0, 0, 0.9)] * len(voxels))
    wrist, tip = cv.path_points(bodies, NAMES)
    return cv.CheckRecord(stamp, heading, wrist, tip, [cv.BLOCKED], focus)


def clear_record(stamp, heading=0.0):
    bodies = [body(0.3), body(0.4)]
    wrist, tip = cv.path_points(bodies, NAMES)
    return cv.CheckRecord(stamp, heading, wrist, tip, [cv.CLEAR], None)


def by_ns(markers, prefix):
    return [m for m in markers if m.ns.startswith(prefix) and m.action == Marker.ADD]


def test_fingertip_is_the_gripper_box_end_away_from_the_wrist():
    wrist, tip = cv.wrist_and_fingertip(body(0.3), NAMES)
    np.testing.assert_allclose(wrist, [0.3, 0.0, 0.5], atol=1e-6)
    np.testing.assert_allclose(tip, [0.5, 0.0, 0.5], atol=1e-6)


def test_statuses_and_focus():
    steps = [step(min_duration=0.1), step(min_duration=0.3), step(unsafe=True), step(blocked=True)]
    durations = [0.1] * 4
    statuses = cv.step_statuses(steps, durations)
    assert statuses == [cv.CLEAR, cv.SLOWED, cv.UNSAFE, cv.BLOCKED]
    assert cv.choose_focus(statuses, steps, durations) == (3, cv.BLOCKED)
    assert cv.choose_focus(statuses[:3], steps[:3], durations[:3]) == (2, cv.UNSAFE)
    assert cv.choose_focus(statuses[:2], steps[:2], durations[:2]) == (1, cv.SLOWED)
    assert cv.choose_focus(statuses[:1], steps[:1], durations[:1]) is None


def test_block_text_counts_voxels_by_lidar():
    text = cv.focus_text(cv.BLOCKED, step(blocked=True), 0.4, "gripper_box", [hit(1), hit(2), hit(3, in_both=False)],
                         seen_by=np.array([2, 2, 3]))
    assert text == "BLOCKED 0.40 s ahead: gripper_box\n3 voxels (right 2, left + right 1, 1 at the edge)"


def test_voxel_size_is_between_one_and_two_cells():
    grid = og.CylindricalGrid(og.GridConfig())
    assert 0.02 < cv.mean_voxel_size(grid) < 0.02 * 2 ** (1 / 3)


def test_a_cleared_block_fades_to_grey_and_goes():
    display = cv.CheckDisplay(voxel_size=0.025, fade_time=2.0)
    header = Header(frame_id="base_footprint")
    display.add_check(blocked_record(0.0, [1, 2]))
    markers = display.markers(header, 0.05, 0.0)
    assert markers[0].action == Marker.DELETEALL
    assert len(by_ns(markers, "check_voxels")[0].points) == 2
    assert by_ns(markers, "check_text")

    display.add_check(clear_record(0.1))
    soon = display.markers(header, 0.2, 0.0)
    assert not by_ns(soon, "check_voxels")
    stale = by_ns(soon, "stale_voxels")[0]
    assert len(stale.points) == 2
    assert stale.colors[0].r == pytest.approx(cv.STALE_COLOR[0])
    later = display.markers(header, 1.5, 0.0)
    assert by_ns(later, "stale_voxels")[0].colors[0].a < stale.colors[0].a
    assert by_ns(later, "stale_ghost")
    gone = display.markers(header, 2.2, 0.0)
    assert not by_ns(gone, "stale_")


def test_a_block_that_moves_leaves_only_the_voxels_it_left_behind():
    display = cv.CheckDisplay(voxel_size=0.025)
    header = Header(frame_id="base_footprint")
    display.add_check(blocked_record(0.0, [1, 2]))
    display.add_check(blocked_record(0.1, [2, 3]))
    markers = display.markers(header, 0.15, 0.0)
    assert len(by_ns(markers, "stale_voxels")[0].points) == 1  # voxel 1
    assert not by_ns(markers, "stale_ghost")  # still blocked: no stale ghost
    assert len(by_ns(markers, "check_voxels")[0].points) == 2


def test_checks_that_stop_coming_go_stale():
    display = cv.CheckDisplay(voxel_size=0.025, check_timeout=0.5)
    header = Header(frame_id="base_footprint")
    display.add_check(blocked_record(0.0, [1]))
    markers = display.markers(header, 1.0, 0.0)
    assert not by_ns(markers, "check_")
    assert by_ns(markers, "stale_voxels")


def test_markers_stay_put_in_the_world_as_the_base_turns():
    display = cv.CheckDisplay(voxel_size=0.025)
    header = Header(frame_id="base_footprint")
    display.add_check(blocked_record(0.0, [1], centers=[[1.0, 0.0, 0.5]]))
    # The base has turned +90 degrees since: the voxel, straight ahead then, is to its right now.
    p = by_ns(display.markers(header, 0.1, np.pi / 2), "check_voxels")[0].points[0]
    assert (p.x, p.y, p.z) == pytest.approx((0.0, -1.0, 0.5), abs=1e-6)


def test_a_replaced_plan_fades_while_the_new_one_shows():
    display = cv.CheckDisplay(voxel_size=0.025, plan_fade_time=4.0)
    header = Header(frame_id="base_footprint")
    bodies = [body(0.3), body(0.5)]
    wrist, tip = cv.path_points(bodies, NAMES)
    display.add_plan(cv.PlanRecord(0.0, wrist, tip, bodies[-1].shapes()), now=0.0)
    display.add_plan(cv.PlanRecord(0.0, wrist, tip, bodies[-1].shapes()), now=1.0)
    lines = by_ns(display.markers(header, 2.0, 0.0), "plan")
    lines = [m for m in lines if m.type == Marker.LINE_LIST]
    assert len(lines) == 2
    old, new = lines
    assert old.colors[0].r == pytest.approx(cv.STALE_COLOR[0]) and old.colors[0].a < new.colors[0].a
    lines = [m for m in by_ns(display.markers(header, 5.5, 0.0), "plan") if m.type == Marker.LINE_LIST]
    assert len(lines) == 1


def test_the_ghost_leaves_out_shapes_that_stay_put():
    still = box((0.0, 0.0, 1.0))
    bodies = [og.SelfModel([still, box((x, 0.0, 0.5))]) for x in (0.3, 0.35, 0.4)]
    ghost = cv.moved_shapes(bodies, 1)
    assert len(ghost) == 1
    np.testing.assert_allclose(np.asarray(ghost[0].pose)[:3, 3], [0.4, 0.0, 0.5], atol=1e-6)
