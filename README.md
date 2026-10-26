# stretch4_fast_planning_demo

Fast motion planning and LiDAR-based obstacle avoidance for Stretch 4's base rotation, lift, arm and wrist.

The system has three parts:
- a **planner** that finds collision-free paths in milliseconds;
- an **obstacle monitor** that turns the two head LiDARs into a voxel grid of what's around the robot;
- a **trajectory follower** that runs inside the driver. It follows a path while slowing, stopping, creeping and detouring around what the grid sees, and backs off when something touches the robot.

Use them together (plan around what the LiDARs see, then follow with avoidance) or separately.

## Packages

| Package | What it is |
|---|---|
| `stretch_motion_planning` | The planner (`planner.py`, `collision_model.py`, `timing.py`, no ROS) and its node `planner_node.py`, which serves `plan_joint_trajectory` and `replan_around`. |
| `stretch_obstacle_grid` | The obstacle grid (C++ with Python bindings) and `obstacle_monitor_node.py`, which serves `check_motion` and `get_grid`. Also bag replay, visualization and LiDAR calibration tools. |
| `stretch_dynamic_trajectory` | `DynamicTrajectoryActionServer`, loaded into the Stretch 4 driver. It serves `follow_joint_trajectory_dynamic` (`control_msgs/FollowJointTrajectory`). |
| `stretch_motion_planning_interfaces` | `PlanJointTrajectory.srv`, `CheckMotion.srv`, `GetGrid.srv`, `ReplanAround.srv`. |

```
 LiDARs ──► obstacle monitor (robot) ──get_grid──► planner node (robot or desktop)
                   ▲                                   ▲
              check_motion                       replan_around
                   │                                   │
 your code ──plan_joint_trajectory──► planner node     │
 your code ──follow_joint_trajectory_dynamic──► trajectory follower (in the driver)
```

## Requirements

- **`stretch4_ros2` on branch `feature/dynamic-trajectory-server`.** It adds the driver's `extra_trajectory_servers` launch argument, base rotation in trajectories, and the driver's `joint_limit.*` parameters, which the planner reads.
- **`stretch4_kinematics`, pinocchio and coal** (pip packages, installed alongside stretch4_kinematics).
- **`stretch4_urdf` with the `merge_arm` fix** (branch `bugfix/ik_merge_arm`, until it's merged). Without it, the merged-arm model puts the wrist 4 cm too far out.
- **pybind11 ≥ 2.12** for `stretch_obstacle_grid`. numpy 2 needs it, and Ubuntu 24.04 ships 2.11: `pip install --user "pybind11>=2.12"`. Eigen and OpenMP are also needed.
- **Hesai LiDARs** publishing `/lidar_points_left` and `/lidar_points_right` with `ring` fields (`stretch_core dual_hesai.launch.py`), and TF for their mounts.

## Build

```bash
cd ~/ament_ws
colcon build --symlink-install --packages-select stretch_motion_planning_interfaces stretch_motion_planning \
    stretch_obstacle_grid stretch_dynamic_trajectory
```

With `--symlink-install`, Python changes apply on a node restart. Changes to `stretch_obstacle_grid`'s C++ need a rebuild.

## Running

On the robot:

```bash
ros2 launch stretch_core dual_hesai.launch.py
ros2 launch stretch_core stretch_driver.launch.py \
    extra_trajectory_servers:=stretch_dynamic_trajectory.server:DynamicTrajectoryActionServer broadcast_odom_tf:=True
ros2 run stretch_obstacle_grid obstacle_monitor_node.py
```

Then the planner node, on the robot or on a desktop on the robot's ROS graph. It only needs a grid snapshot of about 35 KB per plan, so running it off the robot works well over wifi.

```bash
ros2 launch stretch_motion_planning trajectory_planner.launch.py driver_node:=stretch_driver
# or: ros2 run stretch_motion_planning planner_node.py
```

The trajectory follower works without the planner node, but then it can only slow, stop and creep, not detour.

## Planning a motion: `plan_joint_trajectory`

**Joints.** The driver's command names: `mobile_base_rotation_joint`, `lift_joint`, `arm_joint`, `wrist_yaw_joint`, `wrist_pitch_joint`, `wrist_roll_joint`.
- `mobile_base_rotation_joint` is relative to the base's heading at the start, so it always starts at 0.
- The base only turns if the goal names it.

| Request field | Meaning |
|---|---|
| `start` | Empty: plan from the robot's current state. Joints left out are filled in from `/joint_states`. |
| `goal` | Joints left out keep their start position. |
| `move_only_goal_joints` | Normally the other joints may move on the way and return by the end. Set this to keep them still. |
| `obstacles`, `obstacle_poses`, `obstacle_frame` | Known obstacles as `shape_msgs/SolidPrimitive` (box, sphere, cylinder, cone). The frame defaults to `base_footprint` at the start. A floor at z = 0 is always included. |
| `use_planning_scene` | Also avoid the collision objects in MoveIt's planning scene. |
| `use_obstacle_grid` | Also avoid what the LiDARs see (the monitor's grid). Obstacles block. Unknown space and contaminated shadow only count as slow, so paths that avoid them are preferred. |
| `timeout` | Seconds to search; 0 for the node's default. |

The response has `success`, `message`, `method`, `planning_time`, and a `trajectory_msgs/JointTrajectory` (positions and `time_from_start`). Base rotation is included only if the base turns.

**How it plans.** It tries, in order, a straight joint-space line, then a retract/stow/move/unstow template, then RRT-Connect; `method` says which found the path. Typical plans take a few ms.

**The robot model.** The robot is a handful of primitive shapes (`stretch_motion_planning/config/stretch4_collision_primitives.yaml`), checked with pinocchio and coal. The obstacle grid and the trajectory follower use the same shapes.

**Timing.** Trajectories are timed at the joints' top speeds from the same yaml:
- base 0.5 rad/s;
- lift and arm 0.15 m/s;
- wrist joints 1.0 rad/s.

**Without ROS.** `StretchPlanner` (`planner.py`) offers `plan()` and `plan_best()`. `plan_best()` gathers several paths and returns the fastest by predicted time.

## Following a trajectory: `follow_joint_trajectory_dynamic`

The follower takes any `FollowJointTrajectory` goal: planner output, or waypoints you write yourself. It uses the same joint names and base-rotation convention as the planner (base rotation relative to the heading when the goal starts).

**Before sending a goal:**
- Put each joint in the goal except the base in velocity mode: `joint_mode.<joint>:=velocity` on `/stretch_driver`. The follower commands velocities.
- For base rotation, set `dynamic_trajectory_server.allow_rotation:=true` on `/stretch_driver`.
- Start the trajectory where the robot is, or leave time to get there.

**While it runs:**
- **Checking ahead.** About 10 times a second, the follower asks the monitor's `check_motion` about the next 1 s of the path. It keeps a speed scale (0–1) low enough to be at each segment's allowed speed when it gets there. If no fresh answer arrives within 0.3 s, it slows to a stop.
- **Near obstacles.** The allowed speed falls with distance, so the robot can always stop in time.
  - Moving obstacles (people) are assumed to approach at up to 1.5 m/s.
  - Obstacles that stay put for 3 s count as static.
  - Braking distances and latencies are measured values.
- **Space it can't see.** When the robot moves into it, its fastest point is held to 0.05 m/s, and the driver's guarded-contact sensitivity is raised until it's clear again. Two kinds of space count:
  - *unknown*: never seen, or hidden behind an obstacle;
  - *contaminated shadow*: space hidden behind the robot's own body with an obstacle within 20 cm of it.
- **Blocked.** The robot stops short. After 0.5 s it creeps toward the block at 0.02 m/s while the path just ahead is clear, stopping 5 cm short. It returns to full speed once the path clears.
- **Detours** (needs the planner node). While the path is blocked, the follower asks `replan_around` for detours (at most every 0.5 s) and splices the best one in. It's checked against the live grid before switching. How it picks depends on `replan_mode`:
  - `match_path` (default): keep to the given path. Take the detour that strays least, and rejoin the path a little past where the robot is.
  - `reach_goal`: only the end matters. Take the fastest detour, which may go straight to the goal.

  Detours move only the goal's joints unless `replan_joints:=all`.
- **Contact.** A rise in a moving joint's effort over its free-motion level counts as contact. The robot stops and the joint backs off 2 cm. Then `contact_action` decides: `abort` (default) ends the goal, and `resume` carries on.
- **Pushes.** A held arm or lift that's pushed gives way: it moves with the push at up to 2 cm/s, for up to 10 cm, and stops if it runs into something. It never goes limp.
- **Wrist overspeed.** A wrist joint running well over its commanded speed is stopped and the goal aborted.

**Result.** `error_code` 0 means it finished; otherwise `error_string` says why.

### Example: plan around the grid, then follow

```python
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.parameter_client import AsyncParameterClient
from control_msgs.action import FollowJointTrajectory
from sensor_msgs.msg import JointState
from stretch_motion_planning_interfaces.srv import PlanJointTrajectory

rclpy.init()
node = Node("plan_and_go")
plan_client = node.create_client(PlanJointTrajectory, "plan_joint_trajectory")
follow = ActionClient(node, FollowJointTrajectory, "follow_joint_trajectory_dynamic")
driver = AsyncParameterClient(node, "stretch_driver")

def wait(future, timeout=10.0):
    rclpy.spin_until_future_complete(node, future, timeout_sec=timeout)
    return future.result()

plan_client.wait_for_service()
request = PlanJointTrajectory.Request(use_obstacle_grid=True, timeout=2.0)
request.goal = JointState(name=["lift_joint", "arm_joint"], position=[0.8, 0.3])  # empty start: from here
plan = wait(plan_client.call_async(request))
assert plan.success, plan.message
trajectory = plan.trajectory

# The follower commands velocities: put the trajectory's joints in velocity mode while it runs.
# (Read the old modes first with get_parameters and set them back afterwards.)
modes = [Parameter(f"joint_mode.{j}", value="velocity") for j in trajectory.joint_names
         if j != "mobile_base_rotation_joint"]
rotates = "mobile_base_rotation_joint" in trajectory.joint_names
wait(driver.set_parameters(modes + [Parameter("dynamic_trajectory_server.allow_rotation", value=rotates)]))

follow.wait_for_server()
handle = wait(follow.send_goal_async(FollowJointTrajectory.Goal(trajectory=trajectory)))
result = wait(handle.get_result_async(), timeout=300.0).result  # MOVES THE ROBOT
print(result.error_code, result.error_string)
```

The planner's trajectories always include the three wrist joints, even when they don't move, so this puts the wrists in velocity mode too.

### Parameters worth knowing

All are on `/stretch_driver` under `dynamic_trajectory_server.`, and can be changed live with `ros2 param set`.

| Parameter | Default | Use |
|---|---|---|
| `allow_rotation` | false | Needed for goals with base rotation. |
| `contact_threshold_scale` | 1.0 | Scales all contact thresholds. Raise it if fast or loaded motions false-trigger contact; effort readings are noisy. |
| `contact_threshold.<joint>` | arm 8, lift 12, wrist yaw and pitch 12 | Per-joint effort rise that counts as contact. |
| `contact_action` | `abort` | `resume`: carry on after backing off (up to `contact_resume_limit`, 3). |
| `yield_to_pushes` | true | Pushed arm or lift gives way. |
| `yield_threshold.arm_joint` / `.lift_joint` | 3 / 5 | Raise them if the arm or lift yields during fast motion. |
| `replan` | true | Off: stop and creep only, no detours. |
| `replan_mode` | `match_path` | `reach_goal` when only the end pose matters. |
| `rejoin_margin` | 0.5 s | How far past the robot's place on the path a detour may rejoin it. |
| `replan_joints` | `goal` | `all`: detours may also move joints not in the goal (all must be in velocity mode). Six-joint RRT can wander, so take care near people. |
| `blocked_timeout` | 0 (never) | End a goal that stays blocked this many seconds, so task code can try something else. |
| `creep_speed`, `creep_standoff` | 0.02 m/s, 0.05 m | Approach when blocked. |
| `switch_sensitivity`, `unsafe_sensitivity`, `safe_sensitivity` | true, `high_sensitivity_manipulation`, `default` | Guarded-contact sensitivity in and out of unseen space. |

The full list, with explanations, is in the docstring of `stretch_dynamic_trajectory/server.py`.

## The obstacle monitor and grid

**The grid.** A cylinder around the robot's axis, out to 1.6 m and from 4 cm to 2 m up. Voxels are 2 cm radially and vertically; the angular bins widen with distance. Every LiDAR frame goes in as it arrives (two LiDARs at 10 Hz), and the grid is ready about 50 ms after each sweep.

**Labels:** free, occupied, static (occupied and still for 3 s), self, shadow (hidden behind the robot's own body), contaminated shadow (shadow within 20 cm of an obstacle, which a person could be reaching into), and unknown.

**What it filters out:**
- the robot's own body, using the planner's shapes posed from `/joint_states`;
- the floor (below 4 cm);
- clusters of 2–4 points (noise);
- multipath ghosts off the shiny gripper;
- veil points: the false returns a beam gives when it clips an edge, the robot's own edges included.

**Monitor parameters** (set at start):

| Parameter | Default | Meaning |
|---|---|---|
| `contamination_reach` | 0.2 m | How close an obstacle has to be to make shadow unsafe. |
| `static_after` | 3 s | How long an obstacle must stay put to count as static. |
| `obstacle_speed` | 1.5 m/s | Assumed approach speed of moving obstacles. |
| `latency` | 0.15 s | From a check's answer to the robot starting to slow. |
| `stale_after` | 0.5 s | Checks fail if a LiDAR frame is older than this. |
| `publish_labels` | false | Also publish the grid's labels as clouds (costs a few ms per frame). |
| `publish_checks` | true | Publish markers for each check (see RViz below). |

The full list is in the docstring of `stretch_obstacle_grid/tools/obstacle_monitor_node.py`.

**In the MuJoCo sim**, run the monitor with `ring_elevations_file:=<stretch4_mujoco>/models/stretch_4/lidar_calibration.csv` and `azimuth_bins:=250`. In the sim, the gripper always sits in "unsafe" space, so motions there run slowly.

**Blind spots:**
- straight behind the robot above about 0.5–1 m is outside both LiDARs' view, so it's unknown;
- under an extended arm is the arm's shadow;
- returns closer than 0.35 m to a LiDAR are ignored.

### RViz

```bash
rviz2 -d $(ros2 pkg prefix stretch_obstacle_grid)/share/stretch_obstacle_grid/rviz/stretch4_check_view.rviz
```

**`/obstacle_grid/check_markers`** shows the motion the follower is checking (the next second):
- lines through the wrist and fingertip, colored green (clear), yellow (slowed by an obstacle), orange (sweeping unseen space, contact speed) or red (blocked);
- a ghost of the shapes that hit something;
- a label such as "BLOCKED: 0.4 s";
- the voxels responsible. For orange steps these are the obstacles within 20 cm that contaminated the shadow, not the shadow itself.

The plan being followed is drawn in blue, and replaced plans fade out. With `publish_labels:=true`, the `occupied`, `static`, `shadow` and `contaminated` voxels are shown too.

### Diagnostics topics

| Topic | Contents |
|---|---|
| `dynamic_trajectory_server_diagnostics/check` | Each check: limiting shape, clearances, blocked and unsafe segments. |
| `dynamic_trajectory_server_diagnostics/plan` | The plan being followed (latched). |
| `dynamic_trajectory_server_diagnostics/timing` | Speed scale and trajectory time. |
| `obstacle_grid/timing` | The grid's per-step timing. |

### Tools

These are installed with `stretch_obstacle_grid` and run with `ros2 run stretch_obstacle_grid ...`:

| Tool | What it does |
|---|---|
| `replay_bag.py BAG` | Runs the grid over a recorded bag: timing, label counts, images. |
| `grid_viz_node.py` | A Python debug view of the grid's labels, for checking by eye. Not for the control path. |
| `calibrate_lidars.py BAG` | Fits the two LiDARs' mounts from a bag of the robot standing still, using the floor, the robot's own body and the static room. Writes `stretch_calibration_values.yaml` entries. Nothing is written to the robot. |

## Known limitations

- **Slow near unseen space.** Motion that sweeps unknown or contaminated space runs at 0.05 m/s at its fastest point. Low under the gripper and near people, that's a lot of the time. In a test circling the arm past a person's arm, a 60 s trajectory took 95 s.
- **Phantoms under the gripper.** While the lift is low, a few false obstacles still appear under the gripper from mixed LiDAR returns. They can make the shadow there unsafe.
- **Effort-based contact sensing is noisy.** It's a fallback, behind slowing down and the firmware's guarded contact, which stays in force.
- **Held-still obstacles.** An obstacle held still for 3 s counts as static: the robot still avoids it, but no longer assumes it will move toward the robot.
- **Detours** are shortcut but not smoothed, and the follower may switch between similar detours.
- **Not yet supported:** dodging while holding a position (only while following a goal), and using wheel effort or the IMU for contact.

## Tests

| Package | Command | Notes |
|---|---|---|
| `stretch_motion_planning` | `python3 -m pytest test/test_planner.py` | Unit tests, no ROS. |
| | `python3 test/benchmark_planner.py [--obstacles] [--rotate]` | Speed and validity on random queries. |
| | `python3 test/compare_with_mujoco.py` | The primitives against stretch4_body's mesh self-collision model. The primitives should never miss a contact the meshes find. |
| | `python3 test/sim_plan_and_execute.py [--driver stretch_driver] [--only ...]` | Plans, executes and checks. **This moves the robot.** |
| `stretch_obstacle_grid` | `colcon test --packages-select stretch_obstacle_grid` | gtests: grid, labels, filters, motion checks. |
| `stretch_dynamic_trajectory` | `python3 -m pytest test/` | Timing, replanning, contact and messages, no ROS. |

**Before anything that moves the robot**, check `ros2 node list`. Commands sent to the sim reach a real robot on the same ROS graph.
