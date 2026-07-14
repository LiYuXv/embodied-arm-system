# Gazebo and camera perception routes

All Gazebo routes start paused. The launch sequence creates the robot, loads
the joint-state, arm trajectory, and gripper trajectory controllers inactive,
resumes the world, then activates all three as one group. This prevents the
robot from falling while its controllers are coming online.

Build the required overlay first:

```bash
cd ~/embodied-arm-system/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select embodied_interfaces embodied_perception embodied_simulation --symlink-install
source install/setup.bash
```

Start the controller-only workcell:

```bash
ros2 launch embodied_simulation spawn_el_a3.launch.py
```

The active controllers are `joint_state_broadcaster`, `arm_controller`, and
`gripper_controller`. The arm and gripper accept standard
`FollowJointTrajectory` goals at `/arm_controller/follow_joint_trajectory` and
`/gripper_controller/follow_joint_trajectory`.

## RGB-D simulation route

```bash
ros2 launch embodied_simulation rgbd_sim.launch.py
```

This route mounts an RGB-D camera above the wrist, looking down at the grasp
area. It publishes only this RGB-D route (not the dual RGB route):

- `/camera/color/image_raw`
- `/camera/aligned_depth_to_color/image_raw`
- `/camera/color/camera_info`

## Dual RGB simulation route

```bash
ros2 launch embodied_simulation dual_rgb_sim.launch.py
```

`camera_main` is front-left and looks across the work mat; `camera_aux` is
rear-right to reduce arm occlusion. This route publishes only the two RGB
cameras:

- `/camera_main/image_raw`, `/camera_main/camera_info`
- `/camera_aux/image_raw`, `/camera_aux/camera_info`

Gazebo Sim 6 registers, but does not emit, camera-info messages for these
sensors. `camera_info_from_image` derives standard pinhole calibration from
each 60-degree-FOV image stream so the documented `CameraInfo` topics are
always usable. Both routes run the existing `perception_node` and publish the
common `/detected_objects` output.

## Dual physical USB cameras

Install the runtime driver once on the target machine:

```bash
sudo apt-get install ros-humble-v4l2-camera
```

Then connect the cameras at the fixed USB ports and launch:

```bash
ros2 launch embodied_simulation dual_usb_cameras.launch.py
```

The launch intentionally uses these stable paths rather than volatile
`/dev/videoN` names:

- `/dev/v4l/by-path/pci-0000:00:14.0-usb-0:6.1:1.0-video-index0` → main
- `/dev/v4l/by-path/pci-0000:00:14.0-usb-0:6.2:1.0-video-index0` → auxiliary

It publishes exactly the dual-RGB topic contract and the shared
`/detected_objects` output.
