"""Setting up an obstacle grid for Stretch 4's Hesai LiDARs: reading their clouds, leveling
their mounts to the floor, and the robot's body. Shared by replay_bag.py and grid_viz_node.py.
"""

import os

import numpy as np
from ament_index_python.packages import get_package_share_directory

import stretch_obstacle_grid as og
from stretch_motion_planning.collision_model import StretchCollisionModel, load_config

# LiDAR cloud topic -> its frame.
LIDARS = {"/lidar_points_left": "lidar_left_link", "/lidar_points_right": "lidar_right_link"}
GRID_FRAME = "base_footprint"
LABEL_NAMES = ["free", "occupied", "self", "shadow", "contaminated", "unknown"]

ARM_SEGMENTS = ["arm_l1_joint", "arm_l2_joint", "arm_l3_joint", "arm_l4_joint"]
POINT_FIELD_TYPES = {1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4", 7: "f4", 8: "f8"}

# Planner shapes that never move relative to the base.
STATIC_SHAPES = ("base", "mast", "head")
# The base cylinder raised (m) for the grid: its dark, glossy top returns noise a few cm
# above it.
BASE_TOP_MARGIN = 0.05
# The lift box grown (m, every side) for the grid: the carriage's side returns sit about
# 4 cm outside the planner's box.
LIFT_MARGIN = 0.03
# Planner shapes replaced by one box around the gripper, aligned with the gripper: the
# LiDARs' returns around it are a mess of reflections, and one box is simple to check
# motion against.
GRIPPER_SHAPES = ("wrist_roll", "wrist_camera", "gripper", "fingers")
GRIPPER_BOX_MARGIN = 0.02  # m past the shapes on every side
# Returns this close to the robot's shapes are the robot (m).
SELF_POINT_PADDING = 0.04


# --- Clouds ----------------------------------------------------------------------------------

def cloud_fields(msg):
    """(xyz float32 (n, 3), ring uint16 (n,)) from a Hesai PointCloud2."""
    dtype = np.dtype({"names": [f.name for f in msg.fields],
                      "formats": [POINT_FIELD_TYPES[f.datatype] for f in msg.fields],
                      "offsets": [f.offset for f in msg.fields],
                      "itemsize": msg.point_step})
    points = np.frombuffer(bytes(msg.data), dtype=dtype, count=msg.width * msg.height)
    xyz = np.stack([points["x"], points["y"], points["z"]], axis=1).astype(np.float32)
    return xyz, points["ring"].astype(np.uint16)


def transform_matrix(t):
    """4x4 from a geometry_msgs Transform."""
    x, y, z, w = (t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w)
    m = np.eye(4)
    m[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                 [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                 [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    m[:3, 3] = (t.translation.x, t.translation.y, t.translation.z)
    return m


def ring_elevations(xyz, ring):
    """Each ring's elevation (rad): Hesai points are range x (ring direction), so every
    return on a ring has exactly the ring's elevation."""
    rng = np.linalg.norm(xyz, axis=1)
    valid = np.isfinite(rng) & (rng > 0)
    elevation = np.arcsin(xyz[valid, 2] / rng[valid])
    per_ring = np.array([np.median(elevation[ring[valid] == r]) for r in range(int(ring.max()) + 1)])
    if np.isnan(per_ring).any():
        raise RuntimeError("some rings have no returns in the first frame")
    return per_ring.astype(np.float32)


def level_to_floor(base_from_lidar, clouds, r_range=(0.4, 1.8)):
    """Correct a LiDAR's mount so the floor it sees is the plane z = 0 of base_footprint.

    Fits z = a x + b y + c to the returns near the floor (iteratively dropping outliers),
    then rotates the plane's normal onto +z about the origin and shifts it down to 0.
    Returns the corrected base_from_lidar and the tilt (deg) and height (m) it removed."""
    points = []
    for xyz, _ in clouds:
        ok = np.isfinite(xyz).all(axis=1) & (np.abs(xyz).sum(axis=1) > 0)
        points.append((base_from_lidar[:3, :3] @ xyz[ok].T).T + base_from_lidar[:3, 3])
    p = np.concatenate(points)
    r = np.hypot(p[:, 0], p[:, 1])
    near_floor = (r > r_range[0]) & (r < r_range[1]) & (np.abs(p[:, 2]) < 0.1)
    coef = np.zeros(3)
    for tolerance in (0.05, 0.03, 0.02):
        A = np.c_[p[near_floor, 0], p[near_floor, 1], np.ones(near_floor.sum())]
        coef, *_ = np.linalg.lstsq(A, p[near_floor, 2], rcond=None)
        residual = p[:, 2] - (coef[0] * p[:, 0] + coef[1] * p[:, 1] + coef[2])
        near_floor = (r > r_range[0]) & (r < r_range[1]) & (np.abs(residual) < tolerance)

    normal = np.array([-coef[0], -coef[1], 1.0])
    height = coef[2] / np.linalg.norm(normal)
    normal /= np.linalg.norm(normal)
    # Smallest rotation taking the normal to +z (Rodrigues).
    axis = np.cross(normal, [0.0, 0.0, 1.0])
    s, c = np.linalg.norm(axis), normal[2]
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]) / max(s, 1e-12)
    correction = np.eye(4)
    correction[:3, :3] = np.eye(3) + s * k + (1 - c) * (k @ k)
    correction[2, 3] = -height
    return correction @ base_from_lidar, np.degrees(np.arcsin(s)), height


def load_calibration(path):
    """{child link: 4x4 parent_from_child} from a stretch_calibration_values.yaml (absolute
    joint origins, e.g. calibrate_lidars.py's output)."""
    import yaml
    from scipy.spatial.transform import Rotation
    with open(path) as f:
        entries = (yaml.safe_load(f) or {}).get("joint_calibration", {})
    origins = {}
    for entry in entries.values():
        data = entry["data"]
        T = np.eye(4)
        T[:3, :3] = Rotation.from_euler("xyz", [float(v) for v in str(data["rpy"]).split()]).as_matrix()
        T[:3, 3] = [float(v) for v in str(data["xyz"]).split()]
        origins[data["child"]] = (data["parent"], T)
    return origins


# --- The grid --------------------------------------------------------------------------------

def make_grid(base_from_lidars, first_clouds, level=True, contamination_reach=None, log=print):
    """(an ObstacleGrid for the LiDARs with the robot's static body set, the LiDAR mounts it
    uses after leveling).

    base_from_lidars: one 4x4 per LiDAR (from TF). first_clouds: per LiDAR, a few
    [(xyz, ring), ...] from the start, for ring elevations and leveling."""
    models, mounts = [], []
    for i, (base_from_lidar, clouds) in enumerate(zip(base_from_lidars, first_clouds)):
        if level:
            base_from_lidar, tilt, height = level_to_floor(base_from_lidar, clouds)
            log(f"LiDAR {i}: leveled to the floor, removing {tilt:.2f} deg of tilt and {100 * height:+.1f} cm of height")
        xyz, ring = clouds[0]
        models.append(og.LidarModel(base_from_lidar.astype(np.float32), ring_elevations(xyz, ring)))
        mounts.append(base_from_lidar)

    params = og.ObstacleGridParams()
    params.self_point_padding = SELF_POINT_PADDING
    params.duplicate_stride = len(models[0].ring_elevation)  # the Hesais run in dual-return mode
    if contamination_reach is not None:
        params.contamination_reach = contamination_reach
    grid = og.ObstacleGrid(og.GridConfig(), models, params)
    grid.set_static_self(RobotShapes().static_model())
    return grid, mounts


class RobotShapes:
    """The robot's body for the grid, from the planner's collision shapes.

    The base, mast and head never move relative to the base
    (static_model, set once). The lift, arm and wrist follow the joint state
    (moving_model), with the wrist roll, camera, gripper and fingers replaced by one box."""

    def __init__(self):
        config = load_config(os.path.join(get_package_share_directory("stretch_motion_planning"),
                                          "config", "stretch4_collision_primitives.yaml"))
        self.model = StretchCollisionModel(config)

    def static_model(self):
        shapes = []
        for name, shape, placement in self.model.shape_poses(np.zeros(6)):
            if name not in STATIC_SHAPES:
                continue
            if name == "base":
                # Raise the top, keeping the bottom where it is.
                pose = placement.homogeneous.astype(np.float32)
                pose[:3, 3] += pose[:3, 2] * (0.5 * BASE_TOP_MARGIN)
                shapes.append(og.SelfShape.cylinder(pose, shape.radius, shape.halfLength + 0.5 * BASE_TOP_MARGIN))
            else:
                shapes.append(to_self_shape(shape, placement))
        return og.SelfModel(shapes)

    def moving_model(self, joint_state):
        """(og.SelfModel of the moving parts, planner q, the arm's center height)."""
        positions = dict(zip(joint_state.name, joint_state.position))
        q = np.array([0.0, positions["lift_joint"], sum(positions[s] for s in ARM_SEGMENTS),
                      positions["wrist_yaw_joint"], positions["wrist_pitch_joint"], positions["wrist_roll_joint"]])
        shapes = []
        arm_height = None
        gripper_corners = []
        gripper_frame = None
        for name, shape, placement in self.model.shape_poses(q):
            if name == "arm":
                arm_height = float(placement.translation[2])
            if name == "gripper":
                gripper_frame = placement
            if name in GRIPPER_SHAPES:
                gripper_corners.append(box_corners(shape, placement))
            elif name == "lift":
                half = np.asarray(shape.halfSide, dtype=np.float32) + LIFT_MARGIN
                shapes.append(og.SelfShape.box(placement.homogeneous.astype(np.float32), half))
            elif name not in STATIC_SHAPES:
                shape = to_self_shape(shape, placement)
                if shape is not None:
                    shapes.append(shape)
        shapes.append(gripper_box(np.concatenate(gripper_corners), gripper_frame))
        return og.SelfModel(shapes), q, arm_height


def box_corners(shape, placement):
    """The 8 corners (base_footprint) of a box shape."""
    signs = np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).T.reshape(-1, 3)
    local = signs * np.asarray(shape.halfSide)
    return (placement.rotation @ local.T).T + placement.translation


def gripper_box(corners, frame):
    """One box around the gripper's shapes (their corners in base_footprint), aligned with
    `frame` (the gripper box's pin.SE3), plus a margin. The shapes move rigidly with the
    gripper, so the box's size never changes."""
    local = (frame.rotation.T @ (corners - frame.translation).T).T
    lo, hi = local.min(axis=0), local.max(axis=0)
    pose = np.eye(4)
    pose[:3, :3] = frame.rotation
    pose[:3, 3] = frame.translation + frame.rotation @ (0.5 * (lo + hi))
    half = 0.5 * (hi - lo) + GRIPPER_BOX_MARGIN
    return og.SelfShape.box(pose.astype(np.float32), half.astype(np.float32))


def to_self_shape(shape, placement):
    """An og.SelfShape from a coal shape and its pin.SE3 pose; None for the floor half-space."""
    pose = placement.homogeneous.astype(np.float32)
    kind = type(shape).__name__
    if kind == "Box":
        return og.SelfShape.box(pose, np.asarray(shape.halfSide, dtype=np.float32))
    if kind == "Cylinder":
        return og.SelfShape.cylinder(pose, shape.radius, shape.halfLength)
    if kind == "Capsule":
        return og.SelfShape.capsule(pose, shape.radius, shape.halfLength)
    if kind == "Sphere":
        return og.SelfShape.sphere(pose, shape.radius)
    return None
