"""Fast collision checking for Stretch 4 from a few primitive shapes.

The robot is the stretch4_kinematics BASE_ROTATE pinocchio model (base rotation, lift,
merged arm, wrist yaw/pitch/roll). Each link that matters gets one primitive (box,
cylinder, capsule or sphere) from a YAML config, and coal checks them through
pinocchio's computeCollisions. No ROS imports, so this can be used outside ROS.
"""

import numpy as np
import pinocchio as pin
import coal
import yaml

from stretch4_kinematics.kinematic_models.utils import _load_stretch4_urdf
from stretch4_kinematics.state.modes import Stretch4IKModes

# Planning joints, in the order of the configuration vector q. These are the driver's
# command names; URDF_JOINT_NAMES are the matching joints in the pinocchio model.
JOINT_NAMES = (
    "mobile_base_rotation_joint",
    "lift_joint",
    "arm_joint",
    "wrist_yaw_joint",
    "wrist_pitch_joint",
    "wrist_roll_joint",
)
URDF_JOINT_NAMES = (
    "mobile_base_rotation_joint",
    "lift_joint",
    "arm_l4_joint",
    "wrist_yaw_joint",
    "wrist_pitch_joint",
    "wrist_roll_joint",
)

FLOOR_NAME = "floor"


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def joint_vector(config_section, default=None):
    """A per-joint config section ({joint_name: value}) as an array in q order."""
    return np.array([config_section.get(name, default) for name in JOINT_NAMES], dtype=float)


def make_shape(shape_type, size=None, radius=None, length=None):
    """A coal shape. Cylinders and capsules have their axis along z."""
    if shape_type == "box":
        return coal.Box(*size)
    if shape_type == "cylinder":
        return coal.Cylinder(radius, length)
    if shape_type == "capsule":
        return coal.Capsule(radius, length)
    if shape_type == "sphere":
        return coal.Sphere(radius)
    if shape_type == "cone":
        return coal.Cone(radius, length)
    raise ValueError(f"Unknown shape type '{shape_type}'")


def obstacle_shape(obstacle_type, dims):
    """A coal shape from shape_msgs/SolidPrimitive-style dimensions:
    box [x, y, z]; sphere [radius]; cylinder, cone and capsule [height, radius]."""
    if obstacle_type == "box":
        return make_shape("box", size=dims[:3])
    if obstacle_type == "sphere":
        return make_shape("sphere", radius=dims[0])
    if obstacle_type in ("cylinder", "cone", "capsule"):
        return make_shape(obstacle_type, length=dims[0], radius=dims[1])
    raise ValueError(f"Unknown obstacle type '{obstacle_type}'")


def _pose_from_spec(spec):
    rotation = pin.rpy.rpyToMatrix(*spec.get("rpy", [0.0, 0.0, 0.0]))
    return pin.SE3(rotation, np.array(spec.get("center", [0.0, 0.0, 0.0]), dtype=float))


class StretchCollisionModel:
    """Checks a configuration q (ordered as JOINT_NAMES) for self-collision and for
    collision with obstacles, all in the frame of base_footprint at the start of the
    plan (the base rotates about its origin)."""

    def __init__(self, config):
        self.model = _load_stretch4_urdf(Stretch4IKModes.BASE_ROTATE)
        model_joints = tuple(self.model.names[1:])
        if model_joints != URDF_JOINT_NAMES:
            raise RuntimeError(f"Unexpected kinematic model joints {model_joints}")
        self.data = self.model.createData()

        self.padding = float(config.get("padding", 0.0))
        self.lower = self.model.lowerPositionLimit.copy()
        self.upper = self.model.upperPositionLimit.copy()
        self.set_joint_limits(config.get("joint_limits", {}))

        self._build_robot_shapes(config["shapes"])
        self._self_pairs = [
            (self._shape_ids[a], self._shape_ids[b]) for a, b in config["self_collision_pairs"]
        ]
        self.set_obstacles([])

    def set_joint_limits(self, limits):
        """Override joint limits with {joint_name: [lower, upper]}."""
        for name, (lower, upper) in limits.items():
            i = JOINT_NAMES.index(name)
            self.lower[i], self.upper[i] = lower, upper

    def _build_robot_shapes(self, shape_specs):
        self.robot_objects = []  # GeometryObjects, in geometry id order
        self._shape_ids = {}
        self._checks_obstacles = []
        self._extending = []  # (geometry id, joint index, base length, base center)

        for name, spec in shape_specs.items():
            frame = self.model.frames[self.model.getFrameId(spec["link"])]
            shape = make_shape(spec["type"], spec.get("size"), spec.get("radius"), spec.get("length"))
            offset = _pose_from_spec(spec)
            geometry_id = len(self.robot_objects)
            self.robot_objects.append(
                pin.GeometryObject(name, frame.parentJoint, frame.placement * offset, shape)
            )
            self._shape_ids[name] = geometry_id
            if spec.get("check_obstacles", True):
                self._checks_obstacles.append(geometry_id)

            if "extend_with" in spec:
                if spec["type"] != "box" or "rpy" in spec:
                    raise ValueError(f"Shape '{name}': only unrotated boxes can extend")
                self._extending.append(
                    (geometry_id, JOINT_NAMES.index(spec["extend_with"]),
                     spec["size"][0], np.array(spec["center"], dtype=float))
                )

    def set_obstacles(self, obstacles):
        """Replace the obstacles. Each is a dict with 'type' (box, sphere, cylinder, cone
        or capsule), 'dims' (shape_msgs/SolidPrimitive convention) and 'pose' (pin.SE3 or
        4x4 matrix in the planning frame). A floor at z = 0 is always added."""
        self.geometry = pin.GeometryModel()
        for geometry_object in self.robot_objects:
            self.geometry.addGeometryObject(geometry_object)

        # Points with z <= 0 are inside the floor.
        floor = coal.Halfspace(np.array([0.0, 0.0, 1.0]), 0.0)
        obstacle_ids = [
            self.geometry.addGeometryObject(pin.GeometryObject(FLOOR_NAME, 0, pin.SE3.Identity(), floor))
        ]
        for i, obstacle in enumerate(obstacles):
            pose = obstacle["pose"]
            pose = pose if isinstance(pose, pin.SE3) else pin.SE3(np.asarray(pose))
            shape = obstacle_shape(obstacle["type"], obstacle["dims"])
            obstacle_ids.append(
                self.geometry.addGeometryObject(pin.GeometryObject(f"obstacle_{i}", 0, pose, shape))
            )

        for a, b in self._self_pairs:
            self.geometry.addCollisionPair(pin.CollisionPair(a, b))
        for a in self._checks_obstacles:
            for b in obstacle_ids:
                self.geometry.addCollisionPair(pin.CollisionPair(a, b))

        self.geometry_data = pin.GeometryData(self.geometry)
        self._set_padding(self.padding)

        # The model's copies of the robot objects are the ones collision checks use.
        self._extending_objects = [
            (self.geometry.geometryObjects[gid], self.robot_objects[gid].placement, *rest)
            for gid, *rest in self._extending
        ]

    def _stretch_extending_shapes(self, q):
        """Lengthen each extending box by its joint's position along the link's +x,
        keeping its back face fixed."""
        for geometry_object, placement, joint_i, length, center in self._extending_objects:
            extension = q[joint_i]
            box = geometry_object.geometry
            box.halfSide = np.array([0.5 * (length + extension), box.halfSide[1], box.halfSide[2]])
            box.computeLocalAABB()
            shift = pin.SE3(np.eye(3), np.array([0.5 * extension, 0.0, 0.0]))
            geometry_object.placement = placement * shift

    def in_collision(self, q, padding=None):
        """True if q is in collision (closer than `padding`, by default the configured
        padding) or outside the joint limits."""
        if np.any(q < self.lower) or np.any(q > self.upper):
            return True
        self._stretch_extending_shapes(q)
        if padding is None:
            return pin.computeCollisions(self.model, self.data, self.geometry, self.geometry_data, q, True)
        self._set_padding(padding)
        try:
            return pin.computeCollisions(self.model, self.data, self.geometry, self.geometry_data, q, True)
        finally:
            self._set_padding(self.padding)

    def _set_padding(self, padding):
        for request in self.geometry_data.collisionRequests:
            request.security_margin = padding

    def colliding_pairs(self, q):
        """Names of the shape pairs in collision at q, for debugging."""
        self._stretch_extending_shapes(q)
        pin.computeCollisions(self.model, self.data, self.geometry, self.geometry_data, q, False)
        names = [obj.name for obj in self.geometry.geometryObjects]
        return [
            (names[pair.first], names[pair.second])
            for pair, result in zip(self.geometry.collisionPairs, self.geometry_data.collisionResults)
            if result.isCollision()
        ]

    def out_of_limits(self, q):
        """Names of the joints of q that are outside their limits."""
        return [
            name for name, value, lower, upper in zip(JOINT_NAMES, q, self.lower, self.upper)
            if value < lower or value > upper
        ]

    def shape_poses(self, q):
        """(name, coal shape, world pin.SE3) of every geometry object at q, for
        visualization."""
        self._stretch_extending_shapes(q)
        pin.framesForwardKinematics(self.model, self.data, q)
        pin.updateGeometryPlacements(self.model, self.data, self.geometry, self.geometry_data)
        return [
            (obj.name, obj.geometry, self.geometry_data.oMg[i])
            for i, obj in enumerate(self.geometry.geometryObjects)
        ]
