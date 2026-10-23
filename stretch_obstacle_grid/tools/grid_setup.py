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
LABEL_NAMES = ["free", "occupied", "self", "shadow", "contaminated", "unknown", "static"]

ARM_SEGMENTS = ["arm_l1_joint", "arm_l2_joint", "arm_l3_joint", "arm_l4_joint"]
POINT_FIELD_TYPES = {1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4", 7: "f4", 8: "f8"}

# Planner shapes that never move relative to the base.
STATIC_SHAPES = ("base", "mast", "head")
# The base cylinder's top raised (m) for the grid: on the robot its dark, glossy top returns
# noise a few cm above it (5 cm was used for that); 2 cm fits the sim (Elaine, 2026-10-20).
# Check on the robot.
BASE_TOP_MARGIN = 0.02
# The lift box grown (m, every side) for the grid. 3 cm covered the real carriage's side
# returns (about 4 cm outside the planner's box) but was too big in the sim (2026-10-20,
# Elaine); check on the robot.
LIFT_MARGIN = 0.01
# The arm box's top raised and the mast box's front (toward the arm, +x) moved out (m): both
# planner boxes sit a little inside the robot (Elaine, in the sim, then against the robot's
# returns in the first LiDAR bag, 2026-10-20).
ARM_TOP_RAISE = 0.025
MAST_FRONT_EXTEND = 0.01
WRIST_TOP_RAISE = 0.03  # the box at the arm's tip (wrist_link), likewise
# A box for the arm's first segment (arm_l0), which is thicker on the side by the robot: the
# extending arm box as it is fully retracted, fixed to arm_l0, with its top raised this much
# (m) over the planner's (1.5 cm over the extending box).
ARM_BASE_TOP_RAISE = 0.04
# Planner shapes replaced by two boxes for the grid and motion checks (the LiDARs' returns
# around the gripper are a mess of reflections, and boxes are simple to check motion against):
# one around the wrist roll link and its camera, aligned with the wrist roll link, and one
# around the gripper's body and its fingers as they're open now, aligned with the gripper.
CAMERA_SHAPES = ("wrist_roll", "wrist_camera")
GRIPPER_SHAPES = ("gripper", "fingers")
GRIPPER_BOX_MARGIN = 0.01  # m past the shapes on every side, both boxes
FINGER_LINKS = ("gripper_finger_left_link", "gripper_fingertip_left_link",
                "gripper_finger_right_link", "gripper_fingertip_right_link")
FINGER_JOINTS = ("gripper_finger_left_joint", "gripper_finger_right_joint")
# Returns this close to the robot's shapes are the robot (m).
SELF_POINT_PADDING = 0.04

# How hard each joint can brake (m/s^2 or rad/s^2), in planner joint order (base rotation, lift,
# arm, wrist yaw, pitch, roll), measured with braking_test.py on 2026-10-20: lift 1.15-1.38 at
# 1.0 commanded, arm 0.68 at 0.7, base 5.1-5.4 rad/s^2 at 4.0. The wrists slowing to a crawl
# braked at 7.8-11.5 rad/s^2 or more (a rough fit to 50 Hz positions) with 7 or 12 commanded;
# 7 is the lower bound. (A wrist commanded exactly 0 quick-stops within ~20 ms.)
JOINT_DECELERATION = np.array([4.0, 1.0, 0.7, 7.0, 7.0, 7.0])
# stretch4_body's sentry lowers the base's rotational acceleration as the lift rises, from 4.0
# at 0.2 m or lower to 1.0 at the top of the lift (m).
BASE_DECEL_FULL_BELOW, LIFT_TOP, BASE_DECEL_AT_TOP = 0.2, 1.2, 1.0


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


def planar_pose(x, y, yaw):
    """4x4 of a pose in the floor plane, e.g. base_footprint in odom."""
    m = np.eye(4)
    m[:2, :2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
    m[:2, 3] = (x, y)
    return m


def odom_pose(msg):
    """4x4 base pose from a nav_msgs Odometry."""
    p, q = msg.pose.pose.position, msg.pose.pose.orientation
    return planar_pose(p.x, p.y, np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)))


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


def load_ring_elevations(path):
    """Each ring's elevation (rad) from a LiDAR calibration CSV with an Elevation column (deg),
    one row per ring in ring order (as stretch4_mujoco's lidar_calibration.csv)."""
    import csv
    with open(path) as f:
        return np.deg2rad([float(row["Elevation"]) for row in csv.DictReader(f)]).astype(np.float32)


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

def make_grid(base_from_lidars, first_clouds, level=True, contamination_reach=None, log=print, overrides=None,
              ring_elevations_table=None, azimuth_bins=720):
    """(an ObstacleGrid for the LiDARs with the robot's static body set, the LiDAR mounts it
    uses after leveling).

    base_from_lidars: one 4x4 per LiDAR (from TF). first_clouds: per LiDAR, a few
    [(xyz, ring), ...] from the start, for ring elevations and leveling. overrides: {name:
    value} for other ObstacleGridParams, for experiments. ring_elevations_table: each ring's
    elevation (rad), for every LiDAR, instead of measuring them from the first clouds (the
    sim's upward rings return nothing in an open room). azimuth_bins: range image columns; at
    most the LiDAR's firings per turn, so every column gets returns."""
    models, mounts = [], []
    for i, (base_from_lidar, clouds) in enumerate(zip(base_from_lidars, first_clouds)):
        if level:
            base_from_lidar, tilt, height = level_to_floor(base_from_lidar, clouds)
            log(f"LiDAR {i}: leveled to the floor, removing {tilt:.2f} deg of tilt and {100 * height:+.1f} cm of height")
        xyz, ring = clouds[0]
        elevations = (np.asarray(ring_elevations_table, dtype=np.float32) if ring_elevations_table is not None
                      else ring_elevations(xyz, ring))
        models.append(og.LidarModel(base_from_lidar.astype(np.float32), elevations, azimuth_bins))
        mounts.append(base_from_lidar)

    params = og.ObstacleGridParams()
    params.self_point_padding = SELF_POINT_PADDING
    params.duplicate_stride = len(models[0].ring_elevation)  # the Hesais run in dual-return mode
    if contamination_reach is not None:
        params.contamination_reach = contamination_reach
    for name, value in (overrides or {}).items():
        setattr(params, name, value)
    grid = og.ObstacleGrid(og.GridConfig(), models, params)
    grid.set_static_self(RobotShapes().static_model())
    return grid, mounts


class Fingers:
    """The gripper's finger collision meshes (sg4 tool), posed in the quick-connect frame for
    given finger joint angles, so the gripper box can fit the fingers as they're open."""

    def __init__(self):
        import io
        import trimesh
        import yourdfpy
        from stretch4_kinematics.kinematic_models.utils import get_urdf
        urdf = get_urdf(model_name="SE4", batch_name="francis", tool_name="eoa_wrist_dw4_tool_sg4")
        self.urdf = yourdfpy.URDF.load(io.StringIO(urdf), load_meshes=False, build_scene_graph=True)
        self.vertices = {}  # link -> hull vertices in the link's frame
        for link in FINGER_LINKS:
            meshes = [trimesh.load(c.geometry.mesh.filename.replace("file://", ""))
                      for c in self.urdf.link_map[link].collisions]
            self.vertices[link] = trimesh.util.concatenate(meshes).convex_hull.vertices
        self._cache = {}

    def points(self, left, right):
        """Finger hull vertices in the quick_connect_interface_link frame."""
        key = (round(float(left), 2), round(float(right), 2))
        if key not in self._cache:
            self.urdf.update_cfg({FINGER_JOINTS[0]: key[0], FINGER_JOINTS[1]: key[1]})
            points = []
            for link, vertices in self.vertices.items():
                t = self.urdf.get_transform(frame_to=link, frame_from="quick_connect_interface_link")
                points.append(vertices @ t[:3, :3].T + t[:3, 3])
            self._cache[key] = np.concatenate(points)
        return self._cache[key]


class RobotShapes:
    """The robot's body for the grid, from the planner's collision shapes.

    The base, mast and head never move relative to the base
    (static_model, set once). The lift, arm and wrist follow the joint state
    (moving_model), with the wrist roll and camera, and the gripper and fingers, replaced by
    two boxes. planning_body is the whole body that can sweep through anything, for checking
    motion. The gripper box fits the fingers as they were in the last joint state given to
    moving_model (or set_fingers); full open until then, or without the finger meshes."""

    def __init__(self):
        self.config = load_config(os.path.join(get_package_share_directory("stretch_motion_planning"),
                                               "config", "stretch4_collision_primitives.yaml"))
        self.model = StretchCollisionModel(self.config)
        try:
            self.fingers = Fingers()
        except Exception as e:  # another tool, or no meshes: use the planner's full-open box
            print(f"RobotShapes: no finger meshes ({e}); the gripper box covers the fingers fully open")
            self.fingers = None
        self.finger_angles = None
        self.fingers_center = np.asarray(self.config["shapes"]["fingers"]["center"], dtype=float)

    def set_fingers(self, joint_state):
        """Take the finger angles from a JointState, if it has them."""
        positions = dict(zip(joint_state.name, joint_state.position))
        if all(j in positions for j in FINGER_JOINTS):
            self.finger_angles = tuple(positions[j] for j in FINGER_JOINTS)

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
            elif name == "mast":
                pose, half = grow_face(placement.homogeneous, shape.halfSide, [1.0, 0.0, 0.0], MAST_FRONT_EXTEND)
                shapes.append(og.SelfShape.box(pose.astype(np.float32), half.astype(np.float32)))
            else:
                shapes.append(to_self_shape(shape, placement))
        return og.SelfModel(shapes)

    def moving_model(self, joint_state):
        """(og.SelfModel of the moving parts, planner q, the arm's center height)."""
        q = joint_vector(joint_state)
        self.set_fingers(joint_state)
        shapes, arm_height = self._shapes(q, set(STATIC_SHAPES))
        return og.SelfModel(shapes), q, arm_height

    def planning_body(self, q):
        """og.SelfModel of everything but the base at planner configuration q (base rotation
        first, relative to base_footprint now), in a fixed order, for MotionCheck. The base
        is round and turns in place, so it never sweeps anything new."""
        shapes, _ = self._shapes(q, {"base"})
        return og.SelfModel(shapes)

    def shape_names(self):
        """Names of planning_body's shapes, in order."""
        names = []
        for name, shape, _ in self.model.shape_poses(np.zeros(6)):
            if name != "base" and name not in CAMERA_SHAPES + GRIPPER_SHAPES and type(shape).__name__ != "Halfspace":
                names += [name, "arm_base"] if name == "arm" else [name]
        return names + ["camera_box", "gripper_box"]

    def _arm_base(self, arm_placement, extension):
        """og.SelfShape of the arm's first segment, raised by ARM_BASE_TOP_RAISE."""
        spec = self.config["shapes"]["arm"]
        pose, half_x = _arm_base_box(arm_placement, extension, spec["size"][0])
        half = np.array([half_x, 0.5 * spec["size"][1], 0.5 * spec["size"][2]])
        pose, half = grow_face(pose, half, [0.0, 0.0, 1.0], ARM_BASE_TOP_RAISE)
        return og.SelfShape.box(pose.astype(np.float32), half.astype(np.float32))

    def _shapes(self, q, skip):
        """(og.SelfShapes of every shape not in `skip` at q, the arm's center height)."""
        shapes = []
        arm_height = None
        camera_corners, gripper_corners = [], []
        frames = {}
        for name, shape, placement in self.model.shape_poses(q):
            frames[name] = placement
            if name == "arm":
                arm_height = float(placement.translation[2])
            if name in CAMERA_SHAPES:
                camera_corners.append(box_corners(shape, placement))
            elif name == "fingers":
                if self.fingers is not None and self.finger_angles is not None:
                    # The fingers' meshes as they're open, from the quick-connect frame (the
                    # fingers box is posed at an offset from it, with no rotation).
                    link = placement.homogeneous.copy()
                    link[:3, 3] -= placement.rotation @ self.fingers_center
                    points = self.fingers.points(*self.finger_angles)
                    gripper_corners.append(points @ link[:3, :3].T + link[:3, 3])
                else:
                    gripper_corners.append(box_corners(shape, placement))
            elif name in GRIPPER_SHAPES:
                gripper_corners.append(box_corners(shape, placement))
            elif name in skip:
                continue
            elif name == "lift":
                half = np.asarray(shape.halfSide, dtype=np.float32) + LIFT_MARGIN
                shapes.append(og.SelfShape.box(placement.homogeneous.astype(np.float32), half))
            elif name in ("arm", "wrist"):
                raise_by = ARM_TOP_RAISE if name == "arm" else WRIST_TOP_RAISE
                pose, half = grow_face(placement.homogeneous, shape.halfSide, [0.0, 0.0, 1.0], raise_by)
                shapes.append(og.SelfShape.box(pose.astype(np.float32), half.astype(np.float32)))
                if name == "arm":
                    shapes.append(self._arm_base(placement, q[2]))
            else:
                shape = to_self_shape(shape, placement)
                if shape is not None:
                    shapes.append(shape)
        shapes.append(fitted_box(np.concatenate(camera_corners), frames["wrist_roll"]))
        shapes.append(fitted_box(np.concatenate(gripper_corners), frames["gripper"]))
        return shapes, arm_height


class FastPlanningBody:
    """RobotShapes.planning_body for many configurations, about 10x faster, for planning.

    Every shape of the planning body is a box that moves rigidly with one of the model's joints
    (geometry object parentJoint), except that the arm's two boxes stretch with the arm's
    extension: their sizes and offsets are affine in it. So the body is posed once at two
    extensions, each box's transform from its joint and its half sizes are kept as affine in the
    extension, and a configuration then takes one forward kinematics call and a matrix product
    per box. The gripper box fits the fingers as they were when this was made (set_fingers)."""

    GEOMETRY_OF = {"arm_base": "arm", "camera_box": "wrist_roll", "gripper_box": "gripper"}

    def __init__(self, robot, extensions=(0.0, 0.5)):
        import pinocchio as pin
        self._pin = pin
        model = robot.model
        self.model, self.data = model.model, model.model.createData()
        parents = {obj.name: obj.parentJoint for obj in model.geometry.geometryObjects}
        names = robot.shape_names()
        self.joints = [parents[self.GEOMETRY_OF.get(n, n)] for n in names]
        e0, e1 = extensions
        q0, q1 = np.zeros(self.model.nq), np.zeros(self.model.nq)
        q0[2], q1[2] = e0, e1
        rel0, half0 = self._relative(robot, q0)
        rel1, half1 = self._relative(robot, q1)
        if not all(np.allclose(a[:3, :3], b[:3, :3], atol=1e-6) for a, b in zip(rel0, rel1)):
            raise ValueError("FastPlanningBody: a box turns with the arm's extension")
        self.e0 = e0
        self.rel0, self.half0 = np.array(rel0), np.array(half0)
        self.d_offset = (np.array(rel1)[:, :3, 3] - self.rel0[:, :3, 3]) / (e1 - e0)
        self.d_half = (np.array(half1) - self.half0) / (e1 - e0)

    def _relative(self, robot, q):
        """Each box's transform from its joint, and its half sizes, at q."""
        self._pin.forwardKinematics(self.model, self.data, q)
        rel, half = [], []
        for shape, joint in zip(robot.planning_body(q).shapes(), self.joints):
            rel.append(np.linalg.inv(self.data.oMi[joint].homogeneous) @ np.asarray(shape.pose, dtype=float))
            half.append(np.asarray(shape.dims, dtype=float))
        return rel, half

    def __call__(self, q):
        """og.SelfModel of the planning body at q (planner order, as planning_body)."""
        q = np.asarray(q, dtype=float)
        self._pin.forwardKinematics(self.model, self.data, q)
        de = q[2] - self.e0
        shapes = []
        for k, joint in enumerate(self.joints):
            rel = self.rel0[k].copy()
            rel[:3, 3] += self.d_offset[k] * de
            pose = self.data.oMi[joint].homogeneous @ rel
            shapes.append(og.SelfShape.box(pose.astype(np.float32), (self.half0[k] + self.d_half[k] * de).astype(np.float32)))
        return og.SelfModel(shapes)


def _arm_base_box(arm_placement, extension, arm_length):
    """The arm box as it is fully retracted (arm_l0, the first segment): the extending box is
    stretched along its +x by the extension, keeping its back face."""
    pose = arm_placement.homogeneous.copy()
    pose[:3, 3] -= arm_placement.rotation[:, 0] * 0.5 * extension
    return pose, 0.5 * arm_length


def joint_deceleration(lift):
    """JOINT_DECELERATION with the base's limit for this lift height (m)."""
    k = np.clip(1.0 - (lift - BASE_DECEL_FULL_BELOW) / (LIFT_TOP - BASE_DECEL_FULL_BELOW), 0.0, 1.0)
    decel = JOINT_DECELERATION.copy()
    decel[0] = BASE_DECEL_AT_TOP + (JOINT_DECELERATION[0] - BASE_DECEL_AT_TOP) * k
    return decel


def step_braking(qs):
    """MotionCheck's per-step braking for planner configurations qs (n, 6): for each step, the
    largest |joint change| / joint deceleration (s^2), the base's at the higher lift of the two."""
    qs = np.asarray(qs, dtype=float)
    braking = []
    for a, b in zip(qs[:-1], qs[1:]):
        braking.append(float(np.max(np.abs(b - a) / joint_deceleration(max(a[1], b[1])))))
    return braking


def joint_vector(joint_state, base_rotation=0.0):
    """Planner q (base rotation, lift, arm, wrist yaw, pitch, roll) from a JointState."""
    positions = dict(zip(joint_state.name, joint_state.position))
    return np.array([base_rotation, positions["lift_joint"], sum(positions[s] for s in ARM_SEGMENTS),
                     positions["wrist_yaw_joint"], positions["wrist_pitch_joint"], positions["wrist_roll_joint"]])


def box_corners(shape, placement):
    """The 8 corners (base_footprint) of a box shape."""
    signs = np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).T.reshape(-1, 3)
    local = signs * np.asarray(shape.halfSide)
    return (placement.rotation @ local.T).T + placement.translation


def grow_face(pose, half_sizes, direction, amount):
    """(pose, half sizes) of a box (4x4 pose) with the face whose outward normal is nearest
    `direction` (base_footprint) moved out by `amount` (m), the opposite face staying put."""
    pose = np.array(pose, dtype=float)
    half = np.array(half_sizes, dtype=float)
    along = pose[:3, :3].T @ np.asarray(direction, dtype=float)
    axis = int(np.argmax(np.abs(along)))
    half[axis] += 0.5 * amount
    pose[:3, 3] += pose[:3, axis] * np.sign(along[axis]) * 0.5 * amount
    return pose, half


def fitted_box(corners, frame):
    """One box around points (base_footprint), aligned with `frame` (a pin.SE3), plus
    GRIPPER_BOX_MARGIN."""
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
