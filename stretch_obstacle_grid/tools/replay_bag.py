#!/usr/bin/env python3
"""Run the obstacle grid over a recorded LiDAR bag: timing, label counts, and pictures.

The bag needs /lidar_points_left and /lidar_points_right (Hesai PointCloud2 with ring and
timestamp fields), /tf_static and /joint_states. The robot's body comes from the planner's
collision shapes at each frame's joint state.

    ros2 run stretch_obstacle_grid replay_bag.py BAG [--start S] [--end S]
        [--images DIR --image-times T1 T2 ...] [--csv FILE]

Times are seconds from the start of the bag.
"""

import argparse
import csv
import os
import time

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import JointState, PointCloud2
from tf2_msgs.msg import TFMessage

import stretch_obstacle_grid as og
from grid_setup import (GRID_FRAME, LABEL_NAMES, LIDARS, RobotShapes, cloud_fields, load_calibration, make_grid,
                        transform_matrix)

LABEL_COLORS = {og.OCCUPIED: "#b2182b", og.SELF: "#2166ac", og.SHADOW: "#7fbc41",
                og.CONTAMINATED_SHADOW: "#f4a582", og.UNKNOWN: "#bababa"}


# --- Reading the bag ---------------------------------------------------------------------

def open_bag(path, topics):
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=path, storage_id="mcap"), rosbag2_py.ConverterOptions("cdr", "cdr"))
    reader.set_filter(rosbag2_py.StorageFilter(topics=topics))
    return reader


def static_transforms(path):
    """{child: (parent, 4x4 parent_from_child)} from /tf_static."""
    reader = open_bag(path, ["/tf_static"])
    tree = {}
    while reader.has_next():
        _, data, _ = reader.read_next()
        for t in deserialize_message(data, TFMessage).transforms:
            tree[t.child_frame_id] = (t.header.frame_id, transform_matrix(t.transform))
    return tree


def frame_in(tree, frame, root):
    """root_from_frame by chaining static transforms."""
    m = np.eye(4)
    while frame != root:
        parent, parent_from_frame = tree[frame]
        m = parent_from_frame @ m
        frame = parent
    return m


def first_frames(path, count):
    """The first `count` clouds of each LiDAR, as {topic: [(xyz, ring), ...]}."""
    reader = open_bag(path, list(LIDARS))
    frames = {topic: [] for topic in LIDARS}
    while reader.has_next() and min(len(f) for f in frames.values()) < count:
        topic, data, _ = reader.read_next()
        if len(frames[topic]) < count:
            frames[topic].append(cloud_fields(deserialize_message(data, PointCloud2)))
    return frames


# --- Pictures ------------------------------------------------------------------------------

def save_image(path, grid, centers, cells, labels, arm_height, title):
    """Three views: a horizontal slice at arm height, a vertical slice along the arm, and a
    top-down view of everything occupied or unsafe at any height."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cell = grid.config.cell
    fig, axes = plt.subplots(1, 3, figsize=(21, 7.5))

    def draw(ax, mask, u, v, size):
        for label, color in LABEL_COLORS.items():
            pick = mask & (labels == int(label))
            ax.scatter(u[pick], v[pick], s=size, c=color, marker="s", linewidths=0, label=LABEL_NAMES[int(label)])

    x, y, z = centers[:, 0], centers[:, 1], centers[:, 2]
    level = np.abs(z - arm_height) < 0.5 * cell
    draw(axes[0], level, x, y, 3)
    axes[0].set_title(f"Slice at the arm's height (z = {arm_height:.2f} m)")

    # The first theta bin of each band starts at +x and the middle one at -x.
    theta_bins = np.zeros(grid.radial_bins, dtype=np.int64)
    for band in grid.bands:
        theta_bins[band.r_begin:band.r_end] = band.theta_bins
    n = theta_bins[cells[:, 0]]
    ahead, behind = cells[:, 1] == 0, cells[:, 1] == n // 2
    r = np.hypot(x, y)
    draw(axes[1], ahead | behind, np.where(ahead, r, -r), z, 6)
    axes[1].set_title("Vertical slice along the arm (+x right, -x left)")
    axes[1].set_xlabel("x (m)")
    axes[1].set_ylabel("z (m)")

    # Top-down: per (radial, theta) column, the most important label at any height.
    priority = np.array([0, 5, 1, 2, 4, 3])  # free < self < shadow < unknown < contaminated < occupied
    columns = cells[:, 0].astype(np.int64) * 1024 + cells[:, 1]
    rank = priority[labels]
    order = np.lexsort((rank, columns))
    last = np.r_[columns[order][1:] != columns[order][:-1], True]
    top = order[last]
    draw(axes[2], np.isin(np.arange(len(labels)), top) & (labels != int(og.FREE)), x, y, 2)
    axes[2].set_title("Top-down: worst label in each column")

    for ax in (axes[0], axes[2]):
        ax.set_xlim(-2.05, 2.05)
        ax.set_ylim(-2.05, 2.05)
        ax.set_xlabel("x (m, arm points +x)")
        ax.set_ylabel("y (m)")
    for ax in axes:
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
    axes[2].legend(loc="upper right", markerscale=4, fontsize=9)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


# --- Main ------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bag")
    parser.add_argument("--start", type=float, default=0.0, help="seconds into the bag")
    parser.add_argument("--end", type=float, default=float("inf"), help="seconds into the bag")
    parser.add_argument("--images", help="folder for label pictures")
    parser.add_argument("--image-times", type=float, nargs="*", default=[], help="seconds into the bag")
    parser.add_argument("--csv", help="per-frame label counts and timing")
    parser.add_argument("--contamination-reach", type=float, default=og.ObstacleGridParams().contamination_reach,
                        help="m into a shadow from where it touches unknown space or an obstacle; inf for all of it")
    parser.add_argument("--calibration", help="stretch_calibration_values.yaml with LiDAR joint origins to use "
                        "instead of /tf_static's (use with --no-leveling)")
    parser.add_argument("--no-leveling", action="store_true",
                        help="use the LiDAR mounts from /tf_static as they are, without fitting the floor")
    args = parser.parse_args()

    tree = static_transforms(args.bag)
    frames = first_frames(args.bag, 10)
    topics = list(LIDARS)
    t0 = time.monotonic()
    if args.calibration:
        for child, (parent, parent_from_child) in load_calibration(args.calibration).items():
            tree[child] = (parent, parent_from_child)
    grid, _ = make_grid([frame_in(tree, LIDARS[t], GRID_FRAME) for t in topics], [frames[t] for t in topics],
                     level=not args.no_leveling, contamination_reach=args.contamination_reach)
    print(f"Grid: {grid.grid.size:,} voxels, bands {grid.grid.bands}; set up in {time.monotonic() - t0:.1f} s")
    centers = grid.grid.centers() if args.images else None
    cells = grid.grid.cells() if args.images else None
    robot = RobotShapes()
    image_times = sorted(args.image_times)
    if args.images:
        os.makedirs(args.images, exist_ok=True)

    reader = open_bag(args.bag, topics + ["/joint_states"])
    joint_state = None
    bag_start = None
    rows = []
    while reader.has_next():
        topic, data, receive_ns = reader.read_next()
        bag_start = receive_ns if bag_start is None else bag_start
        t_bag = (receive_ns - bag_start) * 1e-9
        if topic == "/joint_states":
            joint_state = deserialize_message(data, JointState)
            continue
        if joint_state is None or t_bag < args.start:
            continue
        if t_bag > args.end:
            break

        msg = deserialize_message(data, PointCloud2)
        xyz, ring = cloud_fields(msg)
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self_model, q, arm_height = robot.moving_model(joint_state)
        grid.add_frame(topics.index(topic), xyz, ring, stamp, self_model)
        bin_ms = grid.timing()["bin_points"]
        grid.update(self_model)
        timing = grid.timing()
        counts = np.bincount(grid.labels(), minlength=len(LABEL_NAMES))
        rows.append({"t": round(t_bag, 3), "lidar": topic.rsplit("_", 1)[-1], **{k: round(v, 2) for k, v in timing.items()},
                     "bin_points": round(bin_ms, 2), **{n: int(c) for n, c in zip(LABEL_NAMES, counts)}})

        while image_times and t_bag >= image_times[0]:
            name = os.path.join(args.images, f"labels_{image_times[0]:06.2f}s.png")
            save_image(name, grid.grid, centers, cells, grid.labels(), arm_height,
                       f"{os.path.basename(args.bag)} at {t_bag:.2f} s (lift {q[1]:.2f} m, arm {q[2]:.2f} m)")
            print(f"Saved {name}")
            image_times.pop(0)

    if not rows:
        print("No frames in range.")
        return
    keys = ["bin_points", "self_voxels", "label", "contaminate", "tables"]
    print(f"\n{len(rows)} frames, {rows[0]['t']:.1f}-{rows[-1]['t']:.1f} s. Milliseconds per frame (median / p95 / max):")
    for k in keys:
        v = np.array([r[k] for r in rows])
        print(f"  {k:12s} {np.median(v):7.2f} {np.percentile(v, 95):7.2f} {v.max():7.2f}")
    total = np.array([sum(r[k] for k in keys) for r in rows])
    print(f"  {'total':12s} {np.median(total):7.2f} {np.percentile(total, 95):7.2f} {total.max():7.2f}")
    print("Voxels per label (median over frames):")
    for n in LABEL_NAMES:
        print(f"  {n:13s} {int(np.median([r[n] for r in rows])):>9,}")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()
