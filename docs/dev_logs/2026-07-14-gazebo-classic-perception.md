# 2026-07-14 Gazebo Classic control and perception acceptance

## Scope

This log records the Classic 11 route only.  `feat/gazebo-perception` PR #6
remains the separate Gazebo Fortress implementation and was not changed or
merged.

## Control acceptance already completed by the user

- EDULITE A3 L1–L6 trajectory control works in Gazebo Classic.
- L7 closes and opens the visible left/right jaws.
- The robot remains stable under gravity and does not fall.
- The Classic launch configures `GAZEBO_MODEL_PATH` automatically.
- The Classic world now mirrors the prior `sorting_workcell` map, including
  the mount, work mat, target zones, and three workpieces. The robot uses the
  map's `yaw=pi` spawn orientation, starts with `L7=1.5708` (visible jaws
  open), and has explicit yellow-arm/dark-gripper Gazebo materials.

The Classic branch implements offset-aware jaw mimic behavior because the
vendor mapping is `jaw = 0.05 - 0.031831 * L7`; it leaves
`third_party/EDULITE_A3` unchanged.

## Perception implementation

Three mutually exclusive routes share `embodied_perception/perception_node`
and publish `/detected_objects`:

| Route | Launch | Inputs |
| --- | --- | --- |
| Wrist RGB-D | `classic_rgbd_sim.launch.py` | `/camera/color/image_raw`, `/camera/aligned_depth_to_color/image_raw`, `/camera/color/camera_info` |
| Dual RGB simulation | `classic_dual_rgb_sim.launch.py` | `/camera_main/image_raw`, `/camera_main/camera_info`, `/camera_aux/image_raw`, `/camera_aux/camera_info` |
| Dual USB | `classic_dual_usb_cameras.launch.py` | Same dual-RGB topic contract; fixed by-path devices |

The RGB-D camera is added by the Classic adapter above the wrist.  The dual
RGB cameras are independent static cameras in the Classic workbench world:
main is front-left and auxiliary is rear-right.  The physical route uses the
exact devices requested:

```text
/dev/v4l/by-path/pci-0000:00:14.0-usb-0:6.1:1.0-video-index0
/dev/v4l/by-path/pci-0000:00:14.0-usb-0:6.2:1.0-video-index0
```

## Verification performed

1. Built `embodied_interfaces`, `embodied_perception`, `el_a3_description`,
   `gazebo_classic_gripper_mimic`, and `gazebo_classic_sim` successfully.
2. Ran `colcon test --packages-select embodied_perception`: **3 passed, 1 skipped**.
3. RGB-D: started a GUI-backed Classic instance. `gzserver` and `gzclient`
   were both present; the sensor published the required color, aligned-depth,
   and CameraInfo topics. The perception node reported its configured inputs
   ready and `/detected_objects` was observed.
4. Dual RGB: started a separate GUI-backed Classic instance. Both 640×480 RGB
   cameras and both CameraInfo streams published; the perception node reported
   main RGB, auxiliary RGB, and CameraInfo ready; `/detected_objects` was
   observed.
5. Dual USB: both requested by-path devices were present and opened. The first
   attempt exposed USB bandwidth pressure with uncompressed capture; the local
   publisher now requests MJPEG before 640×480/30 fps. Both streams then
   published, the perception node reported ready, and `/detected_objects` was
   observed with `camera_main_optical_frame`.

## Start commands

```bash
cd <spike-checkout>/ros2_ws
source install/setup.bash

ros2 launch gazebo_classic_sim classic_rgbd_sim.launch.py
# or
ros2 launch gazebo_classic_sim classic_dual_rgb_sim.launch.py
# or
ros2 launch gazebo_classic_sim classic_dual_usb_cameras.launch.py
```
