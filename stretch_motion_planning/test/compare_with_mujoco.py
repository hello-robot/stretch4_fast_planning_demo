#!/usr/bin/env python3
"""Compare the planner's primitive self-collision model with stretch4_body's mesh-based
MuJoCo self-collision checker (the one the robot runs onboard).

Samples random configurations and counts, for each mesh link pair that MuJoCo reports
in contact, how often the primitive model missed it. The primitives are meant to be
conservative, so misses should be ~0; false alarms (primitives collide, meshes don't)
are expected and reported as a rate.

    python3 compare_with_mujoco.py [--samples 5000] [--pitch-range -1.57 0.56]
"""

import argparse
import os
from collections import Counter

import numpy as np

from stretch4_body.behavior.sentries.self_collision.self_collision_mujoco import SelfCollisionMujoco
from stretch4_body.core.mujoco_urdf import MujocoJointStates

from stretch_motion_planning.collision_model import StretchCollisionModel, load_config

DEFAULT_CONFIG = os.path.join(
    os.path.dirname(__file__), "..", "config", "stretch4_collision_primitives.yaml"
)

# Which primitive covers each URDF link. Links not listed aren't modeled.
# EXTRA_SHAPES lists links that are also covered by a second, tighter primitive.
EXTRA_SHAPES = {"gripper_camera_link": "wrist_camera"}
LINK_TO_SHAPE = {
    "base_link": "base", "wheel_0_link": "base", "wheel_1_link": "base", "wheel_2_link": "base",
    "mast_link": "mast",
    "head_link": "head",
    "lift_link": "lift",
    "arm_l0_link": "arm", "arm_l1_link": "arm", "arm_l2_link": "arm", "arm_l3_link": "arm",
    "arm_l4_link": "arm",
    "wrist_link": "wrist", "wrist_aruco_link": "wrist", "wrist_reflector_link": "wrist",
    "wrist_yaw_link": "wrist_yaw",
    "wrist_pitch_link": "wrist_pitch",
    "wrist_roll_link": "wrist_roll", "gripper_camera_link": "wrist_roll",
    "tool_attachment_site_link": "wrist_roll",
    "quick_connect_interface_link": "gripper",
    "gripper_finger_left_link": "fingers", "gripper_finger_right_link": "fingers",
    "gripper_fingertip_left_link": "fingers", "gripper_fingertip_right_link": "fingers",
    "aruco_fingertip_left_link": "fingers", "aruco_fingertip_right_link": "fingers",
}


def covering_shapes(link):
    return [shape for shape in (LINK_TO_SHAPE.get(link), EXTRA_SHAPES.get(link)) if shape]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--pitch-range", type=float, nargs=2, default=[-1.57, 0.56],
                        help="wrist pitch range to sample (the URDF's is a placeholder)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = load_config(args.config)
    primitives = StretchCollisionModel(config)
    checked_pairs = {frozenset(pair) for pair in config["self_collision_pairs"]}

    meshes = SelfCollisionMujoco()
    if not meshes.startup():
        raise SystemExit("Could not load the MuJoCo self-collision model")

    lower, upper = primitives.lower.copy(), primitives.upper.copy()
    lower[4], upper[4] = args.pitch_range
    rng = np.random.default_rng(args.seed)

    mesh_hits = Counter()      # mesh link pair -> configurations in contact
    missed = Counter()         # ... of which the primitives called collision-free
    unchecked = Counter()      # shape pair the primitives don't check -> configurations
    unchecked_missed = Counter()  # ... of which the primitives called collision-free
    false_alarms = 0
    for _ in range(args.samples):
        q = rng.uniform(lower, upper)
        theta, lift, arm, yaw, pitch, roll = q
        contacts = meshes.get_collisions(MujocoJointStates(
            lift_joint=lift, arm_l1_joint=arm / 4, arm_l2_joint=arm / 4, arm_l3_joint=arm / 4,
            arm_l4_joint=arm / 4, wrist_yaw_joint=yaw, wrist_pitch_joint=pitch, wrist_roll_joint=roll,
        ))
        mesh_pairs = set()
        unchecked_pairs = set()
        for link_a, others in contacts.items():
            for link_b in others:
                shape_pairs = {
                    frozenset((a, b))
                    for a in covering_shapes(link_a) for b in covering_shapes(link_b) if a != b
                }
                if shape_pairs & checked_pairs:
                    mesh_pairs.add(frozenset((link_a, link_b)))
                elif shape_pairs:
                    unchecked_pairs.add(frozenset((LINK_TO_SHAPE[link_a], LINK_TO_SHAPE[link_b])))

        primitive_collision = primitives.in_collision(q)
        for pair in unchecked_pairs:
            unchecked[pair] += 1
            if not primitive_collision:
                unchecked_missed[pair] += 1
        for pair in mesh_pairs:
            mesh_hits[pair] += 1
            if not primitive_collision:
                missed[pair] += 1
        if primitive_collision and not mesh_pairs:
            false_alarms += 1

    print(f"{args.samples} samples; pitch in {args.pitch_range}")
    print(f"Primitive model collided where meshes didn't: {false_alarms / args.samples:.1%} of samples")
    print("Mesh contacts on checked pairs (missed by primitives / total):")
    for pair, count in mesh_hits.most_common():
        print(f"  {missed[pair]:5d} / {count:5d}  {' - '.join(sorted(pair))}")
    print("Mesh contacts on shape pairs the primitives don't check (missed by primitives / total):")
    for pair, count in unchecked.most_common():
        print(f"  {unchecked_missed[pair]:5d} / {count:5d}  {' - '.join(sorted(pair))}")
    total_missed = sum(missed.values()) + sum(unchecked_missed.values())
    print("OK: primitives caught every mesh contact" if total_missed == 0
          else f"WARNING: primitives missed {total_missed} mesh contacts")


if __name__ == "__main__":
    main()
