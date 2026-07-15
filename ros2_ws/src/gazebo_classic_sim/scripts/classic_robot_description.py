#!/usr/bin/env python3
"""Render the vendor A3 xacro with Gazebo Classic control hardware.

The vendor xacro remains untouched.  This adapter reuses its visual, collision,
inertial and joint definitions, replacing only the hardware-specific
ros2_control block with gazebo_ros2_control's GazeboSystem.
"""

import argparse
import subprocess
import sys
import xml.etree.ElementTree as etree

from ament_index_python.packages import get_package_share_directory


ARM_JOINTS = ("L1_joint", "L2_joint", "L3_joint", "L4_joint", "L5_joint", "L6_joint")
JAW_JOINTS = ("left_jaw_joint", "right_jaw_joint")
INITIAL_POSITIONS = {
    "L1_joint": 0.0,
    "L2_joint": 0.0,
    "L3_joint": 0.0,
    "L4_joint": 0.0,
    "L5_joint": 0.0,
    "L6_joint": 0.0,
    "L7_joint": 1.5708,
}
JAW_OPEN_POSITION = 0.0
YELLOW_ARM_LINKS = (
    "base_link", "l1_link_urdf_asm", "l1_urdf_urdf_asm", "l2_l3_urdf_asm",
    "l3_lnik_urdf_asm", "l4_l5_urdf_asm", "l5_l6_urdf_asm", "end_effector",
)
DARK_LINKS = ("gripper_base_link", "gripper_driver_link", "left_jaw_link", "right_jaw_link")


def interface(name, initial_value=None):
    element = etree.Element("state_interface" if initial_value is not None else "command_interface")
    element.set("name", name)
    if initial_value is not None:
        param = etree.SubElement(element, "param", name="initial_value")
        param.text = str(initial_value)
    return element


def add_rgbd_camera(robot):
    """Attach the RGB-D sensor above the wrist without changing vendor files."""
    camera_link = etree.SubElement(robot, "link", name="rgbd_camera_link")
    visual = etree.SubElement(camera_link, "visual")
    etree.SubElement(visual, "origin", xyz="0 0 0", rpy="0 0 0")
    geometry = etree.SubElement(visual, "geometry")
    etree.SubElement(geometry, "box", size="0.045 0.030 0.020")
    material = etree.SubElement(visual, "material", name="matte_black")
    etree.SubElement(material, "color", rgba="0.04 0.04 0.04 1")
    camera_joint = etree.SubElement(robot, "joint", name="rgbd_camera_fixed_joint", type="fixed")
    etree.SubElement(camera_joint, "parent", link="l5_l6_urdf_asm")
    etree.SubElement(camera_joint, "child", link="rgbd_camera_link")
    # Above the end-effector and rotated to observe the work surface.
    etree.SubElement(camera_joint, "origin", xyz="0.055 0 0.105", rpy="0 1.5708 0")

    gazebo = etree.SubElement(robot, "gazebo", reference="rgbd_camera_link")
    sensor = etree.SubElement(gazebo, "sensor", name="wrist_rgbd_sensor", type="depth")
    etree.SubElement(sensor, "always_on").text = "true"
    etree.SubElement(sensor, "update_rate").text = "30"
    etree.SubElement(sensor, "visualize").text = "false"
    camera = etree.SubElement(sensor, "camera")
    etree.SubElement(camera, "horizontal_fov").text = "1.047"
    image = etree.SubElement(camera, "image")
    etree.SubElement(image, "width").text = "640"
    etree.SubElement(image, "height").text = "480"
    etree.SubElement(image, "format").text = "R8G8B8"
    etree.SubElement(camera, "depth_camera")
    clip = etree.SubElement(camera, "clip")
    etree.SubElement(clip, "near").text = "0.10"
    etree.SubElement(clip, "far").text = "4.0"
    plugin = etree.SubElement(sensor, "plugin", name="wrist_rgbd_ros", filename="libgazebo_ros_camera.so")
    ros = etree.SubElement(plugin, "ros")
    etree.SubElement(ros, "namespace").text = "/camera"
    etree.SubElement(ros, "remapping").text = "color/depth/image_raw:=aligned_depth_to_color/image_raw"
    etree.SubElement(plugin, "camera_name").text = "color"
    etree.SubElement(plugin, "frame_name").text = "rgbd_camera_link"
    etree.SubElement(plugin, "min_depth").text = "0.10"
    etree.SubElement(plugin, "max_depth").text = "4.0"


def add_aux_rgb_camera(robot):
    """Attach the auxiliary RGB sensor to the gripper base for dual RGB."""
    camera_link = etree.SubElement(robot, "link", name="camera_aux_link")
    visual = etree.SubElement(camera_link, "visual")
    geometry = etree.SubElement(visual, "geometry")
    etree.SubElement(geometry, "box", size="0.040 0.030 0.020")
    material = etree.SubElement(visual, "material", name="matte_black")
    etree.SubElement(material, "color", rgba="0.04 0.04 0.04 1")
    camera_joint = etree.SubElement(
        robot,
        "joint",
        name="camera_aux_fixed_joint",
        type="fixed",
    )
    etree.SubElement(camera_joint, "parent", link="gripper_base_link")
    etree.SubElement(camera_joint, "child", link="camera_aux_link")
    etree.SubElement(
        camera_joint,
        "origin",
        xyz="0.055 0.0 0.045",
        rpy="0 1.5708 0",
    )

    gazebo = etree.SubElement(robot, "gazebo", reference="camera_aux_link")
    sensor = etree.SubElement(gazebo, "sensor", name="camera_aux_sensor", type="camera")
    etree.SubElement(sensor, "always_on").text = "true"
    etree.SubElement(sensor, "update_rate").text = "30"
    etree.SubElement(sensor, "visualize").text = "false"
    camera = etree.SubElement(sensor, "camera")
    etree.SubElement(camera, "horizontal_fov").text = "1.047"
    image = etree.SubElement(camera, "image")
    etree.SubElement(image, "width").text = "640"
    etree.SubElement(image, "height").text = "480"
    etree.SubElement(image, "format").text = "R8G8B8"
    clip = etree.SubElement(camera, "clip")
    etree.SubElement(clip, "near").text = "0.10"
    etree.SubElement(clip, "far").text = "4.0"
    plugin = etree.SubElement(
        sensor,
        "plugin",
        name="camera_aux_ros",
        filename="libgazebo_ros_camera.so",
    )
    ros = etree.SubElement(plugin, "ros")
    etree.SubElement(ros, "namespace").text = "/"
    etree.SubElement(plugin, "camera_name").text = "camera_aux"
    etree.SubElement(plugin, "frame_name").text = "camera_aux_link"


def add_main_rgb_camera(robot):
    """Add a world-fixed (base-mounted) overhead RGB camera.

    ``base_link`` is rigidly mounted in the Classic workcell, so this link is
    global while still being part of the spawned robot model.  Loading it from
    the robot avoids the Classic world-plugin startup race that can leave a
    standalone camera without a ROS publisher.
    """
    camera_link = etree.SubElement(robot, "link", name="camera_main_link")
    visual = etree.SubElement(camera_link, "visual")
    geometry = etree.SubElement(visual, "geometry")
    etree.SubElement(geometry, "box", size="0.090 0.060 0.050")
    material = etree.SubElement(visual, "material", name="matte_black")
    etree.SubElement(material, "color", rgba="0.03 0.03 0.03 1")
    camera_joint = etree.SubElement(robot, "joint", name="camera_main_fixed_joint", type="fixed")
    etree.SubElement(camera_joint, "parent", link="base_link")
    etree.SubElement(camera_joint, "child", link="camera_main_link")
    # Compensate the pi spawn yaw: this places the camera at the front-side
    # overhead corner of the workbench and points it toward the operation area.
    etree.SubElement(camera_joint, "origin", xyz="-0.73 1.05 0.93", rpy="0 0.55 -1.57")
    gazebo = etree.SubElement(robot, "gazebo", reference="camera_main_link")
    sensor = etree.SubElement(gazebo, "sensor", name="camera_main_sensor", type="camera")
    etree.SubElement(sensor, "always_on").text = "true"
    etree.SubElement(sensor, "update_rate").text = "30"
    etree.SubElement(sensor, "visualize").text = "false"
    camera = etree.SubElement(sensor, "camera")
    etree.SubElement(camera, "horizontal_fov").text = "1.047"
    image = etree.SubElement(camera, "image")
    etree.SubElement(image, "width").text = "640"
    etree.SubElement(image, "height").text = "480"
    etree.SubElement(image, "format").text = "R8G8B8"
    clip = etree.SubElement(camera, "clip")
    etree.SubElement(clip, "near").text = "0.10"
    etree.SubElement(clip, "far").text = "5.0"
    plugin = etree.SubElement(sensor, "plugin", name="camera_main_ros", filename="libgazebo_ros_camera.so")
    ros = etree.SubElement(plugin, "ros")
    etree.SubElement(ros, "namespace").text = "/"
    etree.SubElement(plugin, "camera_name").text = "camera_main"
    etree.SubElement(plugin, "frame_name").text = "camera_main_link"


def add_classic_materials(robot):
    """Restore the yellow arm and dark gripper appearance in Classic."""
    for link_name in YELLOW_ARM_LINKS:
        gazebo = etree.SubElement(robot, "gazebo", reference=link_name)
        etree.SubElement(gazebo, "material").text = "Gazebo/Yellow"
    for link_name in DARK_LINKS:
        gazebo = etree.SubElement(robot, "gazebo", reference=link_name)
        etree.SubElement(gazebo, "material").text = "Gazebo/DarkGrey"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--controllers", required=True)
    parser.add_argument(
        "--camera-mode",
        choices=("none", "rgbd", "aux_rgb"),
        default="none",
    )
    args = parser.parse_args()

    description_share = get_package_share_directory("el_a3_description")
    source_xacro = f"{description_share}/urdf/el_a3.urdf.xacro"
    rendered = subprocess.run(
        ["xacro", source_xacro, "use_mock_hardware:=true"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    robot = etree.fromstring(rendered.stdout)

    for control in list(robot.findall("ros2_control")):
        robot.remove(control)

    add_classic_materials(robot)

    if args.camera_mode == "rgbd":
        add_rgbd_camera(robot)
    elif args.camera_mode == "aux_rgb":
        add_aux_rgb_camera(robot)
        add_main_rgb_camera(robot)

    world_link = etree.Element("link", name="world")
    robot.insert(0, world_link)
    world_joint = etree.Element("joint", name="world_to_base", type="fixed")
    etree.SubElement(world_joint, "parent", link="world")
    etree.SubElement(world_joint, "child", link="base_link")
    robot.append(world_joint)

    control = etree.SubElement(robot, "ros2_control", name="GazeboSystem", type="system")
    hardware = etree.SubElement(control, "hardware")
    etree.SubElement(hardware, "plugin").text = "gazebo_ros2_control/GazeboSystem"
    for joint_name in (*ARM_JOINTS, "L7_joint"):
        joint = etree.SubElement(control, "joint", name=joint_name)
        joint.append(interface("position"))
        joint.append(interface("position", INITIAL_POSITIONS[joint_name]))
        joint.append(interface("velocity", 0.0))
    # Gazebo Classic does not apply URDF mimic tags to simulated joints.  The
    # companion model plugin owns the two visual jaw positions; declaring them
    # as state-only interfaces makes their actual positions observable through
    # /joint_states without exposing them to the L7 controller.
    for joint_name in JAW_JOINTS:
        joint = etree.SubElement(control, "joint", name=joint_name)
        joint.append(interface("position", JAW_OPEN_POSITION))
        joint.append(interface("velocity", 0.0))

    gazebo = etree.SubElement(robot, "gazebo")
    plugin = etree.SubElement(
        gazebo,
        "plugin",
        filename="libgazebo_ros2_control.so",
        name="gazebo_ros2_control",
    )
    etree.SubElement(plugin, "robot_param").text = "robot_description"
    etree.SubElement(plugin, "robot_param_node").text = "robot_state_publisher"
    etree.SubElement(plugin, "parameters").text = args.controllers

    mimic_plugin = etree.SubElement(
        gazebo,
        "plugin",
        filename="libgazebo_classic_gripper_mimic.so",
        name="gazebo_classic_gripper_mimic",
    )
    etree.SubElement(mimic_plugin, "driver_joint").text = "L7_joint"
    etree.SubElement(mimic_plugin, "left_joint").text = "left_jaw_joint"
    etree.SubElement(mimic_plugin, "right_joint").text = "right_jaw_joint"
    etree.SubElement(mimic_plugin, "multiplier").text = "-0.031831"
    etree.SubElement(mimic_plugin, "offset").text = "0.05"
    etree.SubElement(mimic_plugin, "lower_limit").text = "0.0"
    etree.SubElement(mimic_plugin, "upper_limit").text = "0.05"

    sys.stdout.write(etree.tostring(robot, encoding="unicode"))


if __name__ == "__main__":
    main()
