# EDULITE A3 Gazebo Classic validation spike

This isolated experiment runs the vendor EDULITE A3 description in Gazebo
Classic 11 on ROS 2 Humble, without changing `third_party/EDULITE_A3`.

`gazebo_classic_sim` renders the vendor xacro at launch time and replaces its
hardware-specific `ros2_control` element with a `gazebo_ros2_control`
`GazeboSystem`.  It declares position commands plus position/velocity state
for `L1_joint` through `L7_joint`, assigns zero initial values, and fixes
`base_link` to `world`.  The Gazebo plugin loads the controller configuration
with the robot, then the launch file activates the joint-state broadcaster,
arm trajectory controller, and gripper trajectory controller as one group.

The visual gripper jaws have a position offset that Gazebo Classic's built-in
mimic support cannot represent.  `gazebo_classic_gripper_mimic` is a local
model plugin that applies the vendor mapping
`jaw = clamp(0.05 - 0.031831 * L7, 0, 0.05)` to both visible jaw joints each
simulation update.  The jaws are exported as read-only state interfaces, so
`/joint_states` reports their actual Gazebo positions while the existing
gripper controller continues to command only `L7_joint`.

The inline world intentionally contains its own ground plane, light, and
workbench.  This avoids `model://` downloads and lets the controller plugin
come up before the spawner timeout.
The launch also prepends the local plugin paths and the EDULITE description
share path to `GAZEBO_PLUGIN_PATH` and `GAZEBO_MODEL_PATH`; no manual export is
needed.

## Build and start

```bash
cd <spike-checkout>/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --base-paths src ../third_party/EDULITE_A3/el_a3_ros/el_a3_description \
  --packages-select el_a3_description gazebo_classic_gripper_mimic gazebo_classic_sim --symlink-install
source install/setup.bash
ros2 launch gazebo_classic_sim classic_el_a3.launch.py
```

The last command starts the Gazebo Classic GUI.  For automated/headless
diagnostics only, append `gui:=false`.

## Safe controller check

After automatic startup, send a small L4 movement:

```bash
ros2 action send_goal /arm_controller/follow_joint_trajectory \
  control_msgs/action/FollowJointTrajectory \
  "{trajectory: {joint_names: [L1_joint, L2_joint, L3_joint, L4_joint, L5_joint, L6_joint], points: [{positions: [0.0, 0.0, 0.0, 0.30, 0.0, 0.0], time_from_start: {sec: 2}}]}}"
```

## Verified locally (2026-07-14)

The GUI launch created both `gzserver` and `gzclient`, loaded the GazeboSystem
with zero initial joint state/commands, and activated all three controllers.
In that GUI-backed simulation, the initial pose changed by at most
`8.94e-05 rad` over 11.988 simulation seconds; no gravity drop occurred.
The safe trajectory completed with `Goal successfully reached!`; L4 measured
`0.300005 rad` afterward and changed by only `1e-8 rad` over a further
11.986 simulation seconds.

### Gripper verification (2026-07-14)

In the Gazebo Classic GUI, all gripper actions returned `SUCCEEDED` and the
actual left/right jaw positions published by Gazebo were:

| L7 target | left jaw | right jaw | observed state |
| --- | --- | --- | --- |
| 0.0 | 0.050000 | 0.050000 | closed |
| 0.8 | 0.024535 | 0.024535 | visibly open |
| 1.5708 | 0.000000 | 0.000000 | fully open |
| 0.0 | 0.050000 | 0.050000 | closed again |
