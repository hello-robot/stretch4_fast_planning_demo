#!/usr/bin/env python3
"""Calibrate the two Hesai LiDARs' mounts from a bag, without a target.

Fits a 6-DOF correction to each LiDAR's mount (head_link -> lidar_*_link) so that:
  1. the floor each LiDAR sees is the plane z = 0 of base_footprint (tilt and height);
  2. returns from the robot's own body lie on the URDF's visual meshes at the recorded
     joint states (rotation about vertical and horizontal position, which the floor can't
     give);
  3. the two LiDARs agree about the static room.

Only frames where the robot is still are used (base and arm not moving), and only
geometry that stays put: points are binned in 3 cm cells in each LiDAR's own frame, and a
cell counts only if it holds returns in most frames. People moving around (and a hand near
the arm) drop out. A robust loss keeps what's left of them from pulling the fit.

    ros2 run stretch_obstacle_grid calibrate_lidars.py BAG [--start S] [--end S] [--yaml OUT]

The bag needs /lidar_points_left and /lidar_points_right, /tf_static, /joint_states and
/wheel_odom, with the robot still for a while (tens of seconds) and the floor visible. The
result is printed, and with --yaml written as stretch_calibration_values.yaml entries for
the two LiDAR joints (absolute joint origins; see stretch4_urdf's calibration.md). Nothing is
written to the robot.
"""

import argparse
import datetime
import io
import math

import numpy as np
import trimesh
import yaml
import yourdfpy
from rclpy.serialization import deserialize_message
from nav_msgs.msg import Odometry
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import JointState, PointCloud2
from stretch4_urdf import get_urdf

from grid_setup import GRID_FRAME, LIDARS, cloud_fields
from replay_bag import frame_in, open_bag, static_transforms

LIDAR_JOINTS = {"/lidar_points_left": "lidar_left_joint", "/lidar_points_right": "lidar_right_joint"}
STABLE_CELL = 0.03         # m: cells for deciding what's static, in each LiDAR's frame
STABLE_FRACTION = 0.8      # of the still frames a cell must hold returns in
MIN_RANGE = 0.35           # m: nearer returns are unreliable
FLOOR_BAND = 0.05          # m: |z| of returns used as floor
BODY_BAND = 0.04           # m: returns this close to the robot's meshes are the robot
ROOM_MATCH = 0.05          # m: room returns matched between LiDARs within this
ROBOT_CLEARANCE = 0.10     # m: room returns must be this far from the robot
MESH_SPACING = 0.004       # m: spacing of the samples on the robot's meshes
SAMPLES_PER_TERM = 4000    # residuals per LiDAR per term
# Links whose meshes the LiDARs can see. The head holds the LiDARs (too near), wheels and the
# base's underside aren't visible.
BODY_LINKS = ("base_link", "mast_link", "lift_link", "arm_l0_link", "arm_l1_link", "arm_l2_link",
              "arm_l3_link", "arm_l4_link", "wrist_link", "wrist_yaw_link", "wrist_pitch_link",
              "wrist_roll_link", "gripper_camera_link", "quick_connect_interface_link")


def transform_points(T, p):
    return p @ T[:3, :3].T + T[:3, 3]


def correction(params):
    """4x4 from (rx, ry, rz, tx, ty, tz): a rotation vector (rad) and translation (m), applied
    in the LiDAR's own frame (base_from_lidar = nominal @ correction)."""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(params[:3]).as_matrix()
    T[:3, 3] = params[3:]
    return T


def normals_by_pca(points, k=12):
    tree = cKDTree(points)
    _, idx = tree.query(points, k=k)
    neighbors = points[idx] - points[idx].mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", neighbors, neighbors)
    _, vecs = np.linalg.eigh(cov)
    return vecs[:, :, 0]  # smallest eigenvalue's vector


# --- Data from the bag ----------------------------------------------------------------------

def still_times(bag, max_turn=0.02, max_arm=0.01):
    """Function t -> True where the base isn't turning (rad/s) and the arm isn't moving (m/s),
    from /wheel_odom and the arm's joint states (by position change)."""
    reader = open_bag(bag, ["/wheel_odom", "/joint_states"])
    start, turn, arm = None, [], []
    while reader.has_next():
        topic, data, ns = reader.read_next()
        start = start or ns
        t = (ns - start) * 1e-9
        if topic == "/wheel_odom":
            turn.append((t, deserialize_message(data, Odometry).twist.twist.angular.z))
        else:
            js = deserialize_message(data, JointState)
            p = dict(zip(js.name, js.position))
            arm.append((t, sum(p[k] for k in ("arm_l1_joint", "arm_l2_joint", "arm_l3_joint", "arm_l4_joint")),
                        p["lift_joint"]))
    turn, arm = np.array(turn), np.array(arm)
    speed = np.abs(np.gradient(arm[:, 1], arm[:, 0])) + np.abs(np.gradient(arm[:, 2], arm[:, 0]))
    speed = np.convolve(speed, np.ones(20) / 20, mode="same")  # joint states are 100 Hz and noisy

    def still(t):
        return (abs(np.interp(t, turn[:, 0], turn[:, 1])) < max_turn and
                np.interp(t, arm[:, 0], speed) < max_arm)
    return still


def read_still_frames(bag, still, start, end):
    """Per LiDAR, the still frames' points (LiDAR frame, one return per dual pair, past
    MIN_RANGE), and the joint state of each frame."""
    reader = open_bag(bag, list(LIDARS) + ["/joint_states"])
    frames = {topic: [] for topic in LIDARS}
    t0, joint_state = None, None
    while reader.has_next():
        topic, data, ns = reader.read_next()
        t0 = t0 or ns
        t = (ns - t0) * 1e-9
        if topic == "/joint_states":
            joint_state = deserialize_message(data, JointState)
            continue
        if t < start or t > end or joint_state is None or not still(t):
            continue
        xyz, _ = cloud_fields(deserialize_message(data, PointCloud2))
        xyz = xyz[0::2] if len(xyz) % 2 == 0 else xyz
        keep = np.isfinite(xyz).all(axis=1) & (np.linalg.norm(xyz, axis=1) > MIN_RANGE)
        frames[topic].append((xyz[keep].astype(np.float64), joint_state))
    return frames


def stable_points(clouds):
    """The returns in 3 cm cells (LiDAR frame) that hold returns in at least STABLE_FRACTION
    of the frames, one point per cell (the mean)."""
    counts, sums, n = {}, {}, {}
    for points, _ in clouds:
        cells = np.floor(points / STABLE_CELL).astype(np.int64)
        keys, inverse = np.unique(cells, axis=0, return_inverse=True)
        cell_sum = np.zeros((len(keys), 3))
        np.add.at(cell_sum, inverse.ravel(), points)
        cell_n = np.bincount(inverse.ravel())
        for k, s_, c in zip(map(tuple, keys), cell_sum, cell_n):
            counts[k] = counts.get(k, 0) + 1
            sums[k] = sums.get(k, 0) + s_
            n[k] = n.get(k, 0) + c
    needed = STABLE_FRACTION * len(clouds)
    return np.array([sums[k] / n[k] for k, c in counts.items() if c >= needed])


class RobotSurface:
    """Points and normals sampled on the robot's visual meshes, posed for a joint state."""

    def __init__(self):
        urdf_text = get_urdf(model_name="SE4", batch_name="francis", tool_name="eoa_wrist_dw4_tool_sg4")
        self.urdf = yourdfpy.URDF.load(io.BytesIO(urdf_text.encode()), load_meshes=False, build_scene_graph=True)
        self.samples = {}
        for name in BODY_LINKS:
            meshes = []
            for visual in self.urdf.link_map[name].visuals:
                if visual.geometry.mesh is None:
                    continue
                mesh = trimesh.load(visual.geometry.mesh.filename.replace("file://", ""), force="mesh")
                if visual.geometry.mesh.scale is not None:
                    mesh.apply_scale(visual.geometry.mesh.scale)
                mesh.apply_transform(visual.origin if visual.origin is not None else np.eye(4))
                meshes.append(mesh)
            if meshes:
                mesh = trimesh.util.concatenate(meshes)
                count = max(100, int(mesh.area / MESH_SPACING ** 2))
                points, faces = trimesh.sample.sample_surface_even(mesh, count)
                self.samples[name] = (points, mesh.face_normals[faces])

    def posed(self, joint_state):
        positions = dict(zip(joint_state.name, joint_state.position))
        self.urdf.update_cfg({j: positions[j] for j in self.urdf.actuated_joint_names if j in positions})
        points, normals = [], []
        for name, (p, n) in self.samples.items():
            T = self.urdf.get_transform(name, GRID_FRAME)
            points.append(transform_points(T, p))
            normals.append(n @ T[:3, :3].T)
        return np.concatenate(points), np.concatenate(normals)


# --- The fit -------------------------------------------------------------------------------

def fit(nominal, room, body, surface, rng):
    """Corrections (12 params) for both LiDARs. nominal: base_from_lidar per LiDAR; room and
    body: stable points per LiDAR (LiDAR frame); surface: (points, normals) of the robot."""
    surface_tree = cKDTree(surface[0])
    params = np.zeros(12)

    def poses(x):
        return [nominal[i] @ correction(x[6 * i:6 * i + 6]) for i in range(2)]

    def pick(points, mask):
        idx = np.nonzero(mask)[0]
        return points[rng.choice(idx, size=min(SAMPLES_PER_TERM, len(idx)), replace=False)] if len(idx) else points[:0]

    terms = {}
    for outer in range(8):
        T = poses(params)
        # Correspondences for this round, from the current estimate.
        floor, on_body, room_pairs = [], [], []
        room_base = [transform_points(T[i], room[i]) for i in range(2)]
        room_normals = [normals_by_pca(r) for r in room_base]
        room_trees = [cKDTree(r) for r in room_base]
        for i in range(2):
            pts = room_base[i]
            near_robot = surface_tree.query(pts, distance_upper_bound=ROBOT_CLEARANCE)[0] < ROBOT_CLEARANCE
            floor.append(pick(room[i], (np.abs(pts[:, 2]) < FLOOR_BAND) & (np.hypot(pts[:, 0], pts[:, 1]) > 0.4)))
            b = transform_points(T[i], body[i])
            d, j = surface_tree.query(b, distance_upper_bound=BODY_BAND)
            sel = np.nonzero(d < BODY_BAND)[0]
            sel = rng.choice(sel, size=min(SAMPLES_PER_TERM, len(sel)), replace=False) if len(sel) else sel
            on_body.append((body[i][sel], j[sel]))
            other = 1 - i
            d, j = room_trees[other].query(pts, distance_upper_bound=ROOM_MATCH)
            sel = np.nonzero((d < ROOM_MATCH) & ~near_robot & (pts[:, 2] > FLOOR_BAND))[0]
            sel = rng.choice(sel, size=min(SAMPLES_PER_TERM, len(sel)), replace=False) if len(sel) else sel
            room_pairs.append((room[i][sel], j[sel]))

        def residuals(x, split=False):
            Ts = poses(x)
            out = {"floor": [], "body": [], "room": []}
            for i in range(2):
                out["floor"].append(transform_points(Ts[i], floor[i])[:, 2])
                p, j = on_body[i]
                out["body"].append(np.einsum("ij,ij->i", transform_points(Ts[i], p) - surface[0][j], surface[1][j]))
                p, j = room_pairs[i]
                target = transform_points(Ts[1 - i], room[1 - i][j])
                # The other LiDAR's normals, rotated with its current pose (from this round's estimate)
                out["room"].append(np.einsum("ij,ij->i", transform_points(Ts[i], p) - target, room_normals[1 - i][j]))
            if split:
                return {k: [np.asarray(v) for v in vs] for k, vs in out.items()}
            return np.concatenate([np.concatenate(v) for v in out.values()])

        if outer == 0:
            terms["before"] = residuals(params, split=True)
        result = least_squares(residuals, params, loss="huber", f_scale=0.01, x_scale=[0.01] * 3 + [0.01] * 3 + [0.01] * 3 + [0.01] * 3)
        step = np.abs(result.x - params).max()
        params = result.x
        if step < 1e-5:
            break
    terms["after"] = residuals(params, split=True)
    # Rough 1-sigma uncertainty of each parameter from the final Jacobian.
    J = result.jac
    sigma = np.sqrt(np.mean(result.fun ** 2))
    try:
        covariance = np.linalg.inv(J.T @ J) * sigma ** 2
        uncertainty = np.sqrt(np.diag(covariance))
    except np.linalg.LinAlgError:
        uncertainty = np.full(12, np.nan)
    return params, terms, uncertainty


def joint_origin(head_from_lidar):
    """URDF origin (xyz, rpy) strings for a 4x4."""
    xyz = head_from_lidar[:3, 3]
    rpy = Rotation.from_matrix(head_from_lidar[:3, :3]).as_euler("xyz")
    return " ".join(f"{v:.6f}" for v in xyz), " ".join(f"{v:.6f}" for v in rpy)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bag")
    parser.add_argument("--start", type=float, default=0.0, help="seconds into the bag")
    parser.add_argument("--end", type=float, default=float("inf"), help="seconds into the bag")
    parser.add_argument("--yaml", help="write stretch_calibration_values.yaml entries here")
    parser.add_argument("--robot-id", default="unknown")
    args = parser.parse_args()
    rng = np.random.default_rng(0)

    tree = static_transforms(args.bag)
    topics = list(LIDARS)
    nominal = [frame_in(tree, LIDARS[t], GRID_FRAME) for t in topics]
    still = still_times(args.bag)
    frames = read_still_frames(args.bag, still, args.start, args.end)
    print("Still frames:", {t: len(f) for t, f in frames.items()})
    if min(len(f) for f in frames.values()) < 10:
        raise SystemExit("Need at least 10 still frames per LiDAR.")

    surface_model = RobotSurface()
    # The robot is still, so one joint state (the middle frame's) poses it for every frame.
    joint_state = frames[topics[0]][len(frames[topics[0]]) // 2][1]
    surface = surface_model.posed(joint_state)
    stable = [stable_points(frames[t]) for t in topics]
    print("Stable points per LiDAR (3 cm cells in >= 80% of still frames):", [len(s) for s in stable])

    params, terms, uncertainty = fit(nominal, stable, stable, surface, rng)

    print("\nResiduals (cm): median |r| / RMS, per LiDAR (left, right)")
    for name in ("floor", "body", "room"):
        line = []
        for when in ("before", "after"):
            r = terms[when][name]
            line.append(f"{when}: " + ", ".join(f"{100 * np.median(np.abs(x)):.2f}/{100 * np.sqrt(np.mean(x ** 2)):.2f} (n={len(x)})" for x in r))
        print(f"  {name:6s} " + "   ".join(line))

    results = {}
    for i, topic in enumerate(topics):
        p, u = params[6 * i:6 * i + 6], uncertainty[6 * i:6 * i + 6]
        angle = np.degrees(np.linalg.norm(p[:3]))
        print(f"\n{topic}: correction {angle:.2f} deg about {np.round(p[:3] / max(np.linalg.norm(p[:3]), 1e-12), 2)}, "
              f"shift {np.round(100 * p[3:], 2)} cm (LiDAR frame)")
        print(f"  1-sigma: rotation {np.round(np.degrees(u[:3]), 3)} deg, shift {np.round(100 * u[3:], 2)} cm")
        base_from_lidar = nominal[i] @ correction(p)
        head_from_base = np.linalg.inv(frame_in(tree, "head_link", GRID_FRAME))
        xyz, rpy = joint_origin(head_from_base @ base_from_lidar)
        print(f"  {LIDAR_JOINTS[topic]} origin: xyz=\"{xyz}\" rpy=\"{rpy}\"")
        results[LIDAR_JOINTS[topic]] = {"xyz": xyz, "rpy": rpy, "parent": "head_link", "child": LIDARS[topic]}

    if args.yaml:
        stamp = datetime.datetime.now().isoformat()
        entries = {name: {"data": data, "robot_id": args.robot_id, "timestamp": stamp, "fit_method": "floor+body+room",
                          "source_bag": args.bag} for name, data in results.items()}
        with open(args.yaml, "w") as f:
            yaml.safe_dump({"version": "2.0", "joint_calibration": entries}, f, sort_keys=False)
        print(f"\nWrote {args.yaml} (not applied anywhere)")


if __name__ == "__main__":
    main()
