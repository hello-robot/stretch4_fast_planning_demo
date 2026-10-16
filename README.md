# stretch4_fast_planning_demo
Fast planner + obstacle avoidance for real-world reaching.

## Packages

- **`stretch_motion_planning`** plans collision-free joint trajectories for Stretch 4's base rotation, lift, arm and wrist (yaw, pitch, roll).
  - The robot is modeled as a few primitive shapes (`config/stretch4_collision_primitives.yaml`), checked with pinocchio + coal on the `stretch4_kinematics` model.
  - The planner tries a straight joint-space line first, then a retract/stow/move/unstow template, then RRT-Connect.
  - The core (`collision_model.py`, `planner.py`, `timing.py`) has no ROS imports.
  - `planner_node.py` offers it as a ROS service.
- **`stretch_motion_planning_interfaces`** holds `PlanJointTrajectory.srv`: start and goal `JointState`s, obstacles as `shape_msgs/SolidPrimitive` with poses, an optional MoveIt planning scene, and a timeout. It returns a `trajectory_msgs/JointTrajectory`.

## Requirements

- **`stretch4_kinematics`, pinocchio and coal**: pip packages, already installed alongside stretch4_kinematics.
- **`stretch4_urdf` with the `merge_arm` fix** (branch `bugfix/ik_merge_arm`, until it's merged). Without it, the merged-arm model puts the wrist 4 cm too far out.
- **`stretch4_ros2` branch `feature/trajectory-server-base-rotation`**, to run trajectories that include `mobile_base_rotation_joint` (set `trajectory_server.allow_rotation:=true`). Lift, arm and wrist trajectories run on any driver. On the real robot, that branch also makes the driver publish its `joint_limit.*` parameters, which the planner node reads.
- **`trajectory_server.mode:=adaptive_velocity`**, with the joints in velocity mode, follows the planned timing smoothly. `target_priority` also works, but the wrist stops and starts at every waypoint.

## Running

```bash
colcon build --symlink-install --packages-select stretch_motion_planning_interfaces stretch_motion_planning
ros2 launch stretch_motion_planning trajectory_planner.launch.py driver_node:=stretch_driver   # or stretch_mujoco_driver
```

The planner only plans. Send the returned trajectory to the driver's `follow_joint_trajectory` action to run it.

## Tests

All of these run from `stretch_motion_planning/`:

- `python3 -m pytest test/test_planner.py`: unit tests, no ROS.
- `python3 test/benchmark_planner.py [--obstacles] [--rotate]`: planning speed and path validity on random queries.
- `python3 test/compare_with_mujoco.py`: checks the primitives against stretch4_body's mesh-based self-collision model. The primitives should never miss a contact the meshes find.
- `python3 test/sim_plan_and_execute.py [--driver stretch_driver] [--mode adaptive_velocity] [--only ...]`: plans, executes, records and checks the result. **This moves the robot.** Check `ros2 node list` first, because sim commands also reach a real robot on the same graph.
