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
    "L7_joint": 0.0,
}
JAW_CLOSED_POSITION = 0.05


def interface(name, initial_value=None):
    element = etree.Element("state_interface" if initial_value is not None else "command_interface")
    element.set("name", name)
    if initial_value is not None:
        param = etree.SubElement(element, "param", name="initial_value")
        param.text = str(initial_value)
    return element


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--controllers", required=True)
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
        joint.append(interface("position", JAW_CLOSED_POSITION))
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
