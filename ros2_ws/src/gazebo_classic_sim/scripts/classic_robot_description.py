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
JAW_PAD_CENTER_Z = -0.2361
JAW_PAD_SIZE_M = (0.010, 0.042, 0.038)
INITIAL_POSITIONS = {
    "L1_joint": 0.0,
    # Start in the public, collision-free ``home`` posture.  The old startup
    # value silently used ``ready``, so Gazebo appeared to begin in a task
    # posture before the user had supplied any instruction.
    "L2_joint": 0.785,
    "L3_joint": -0.785,
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


def state_interface(name, initial_value=None):
    """Create a state interface, optionally seeded at spawn time."""
    element = etree.Element("state_interface")
    element.set("name", name)
    if initial_value is not None:
        param = etree.SubElement(element, "param", name="initial_value")
        param.text = str(initial_value)
    return element


def command_interface(name, initial_value=None):
    """Create a command interface with the same initial hold target.

    GazeboSystem does not copy a state-interface initial value into its command
    buffer.  Omitting this parameter makes the robot drive from the configured
    ready posture toward zero before the first FollowJointTrajectory goal
    arrives.
    """
    element = etree.Element("command_interface")
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
    """Attach the auxiliary RGB sensor above the fixed final wrist link."""
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
    etree.SubElement(camera_joint, "parent", link="l5_l6_urdf_asm")
    etree.SubElement(camera_joint, "child", link="camera_aux_link")
    # It is on the jaw centre line, behind and above the finger tips.  The
    # rotation is the look-at result from (0,.16,-.04) to (0,0,-.166), so the
    # wrist body is behind the image plane rather than blocking the gap.  Roll
    # pi makes the image upright while keeping its optical axis unchanged.
    etree.SubElement(
        camera_joint,
        "origin",
        xyz="0.0 0.16 -0.04",
        rpy="3.141593 0.50 -1.570796",
    )

    gazebo = etree.SubElement(robot, "gazebo", reference="camera_aux_link")
    etree.SubElement(gazebo, "material").text = "Gazebo/DarkGrey"
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
    # Spawn yaw is pi.  The translation is Rz(-pi) * (camera_world - base),
    # with camera_world=(0.65,-1.05,1.75).  The rpy is the explicit look-at
    # rotation for camera +X toward deck centre=(0.08,0,1.16), transformed
    # from world into base_link: Rz(-pi) * R_look_at.
    etree.SubElement(
        camera_joint,
        "origin",
        xyz="-0.73 1.05 0.93",
        rpy="0 0.458701 -1.073454",
    )
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
    """Restore appearance and configure high-friction physical finger pads."""
    for link_name in YELLOW_ARM_LINKS:
        gazebo = etree.SubElement(robot, "gazebo", reference=link_name)
        etree.SubElement(gazebo, "material").text = "Gazebo/Yellow"
        # The arm is position-servoed by GazeboSystem.  Compensate gravity on
        # its own rigid links so its controller does not have to fight the
        # incomplete CAD inertias; collision geometry remains enabled.
        etree.SubElement(gazebo, "gravity").text = "false"
    for link_name in DARK_LINKS:
        gazebo = etree.SubElement(robot, "gazebo", reference=link_name)
        etree.SubElement(gazebo, "material").text = "Gazebo/DarkGrey"
        etree.SubElement(gazebo, "gravity").text = "false"
        # Gazebo Classic merges URDF extension values only from one
        # <gazebo reference="link"> element.  Put visual/gravity and ODE
        # contact values in this same element; emitting a second reference
        # silently left the jaw collision pads at Gazebo's default friction.
        if link_name in ("left_jaw_link", "right_jaw_link"):
            # The adapted collision is a thin rubber side pad.  Select the
            # pad-local vertical direction explicitly as ODE's primary
            # friction axis so a two-sided pinch can transmit a lift force.
            etree.SubElement(gazebo, "mu1").text = "8.0"
            etree.SubElement(gazebo, "mu2").text = "8.0"
            etree.SubElement(gazebo, "fdir1").text = "0 0 1"
            etree.SubElement(gazebo, "kp").text = "30000"
            etree.SubElement(gazebo, "kd").text = "100"
            etree.SubElement(gazebo, "minDepth").text = "0.002"
            etree.SubElement(gazebo, "maxVel").text = "0"


def configure_gripper_side_pads(robot):
    """Configure deck-clear, side-facing physical pads on vendor jaw links."""
    # Keep each pad's inner face at the same closing-plane location as the
    # vendor 32 mm box: reducing the box thickness by 22 mm requires moving
    # its centre 11 mm toward the gripper centre.
    for link_name, pad_x in (
        ("left_jaw_link", "0.0567"),
        ("right_jaw_link", "-0.0567"),
    ):
        link = robot.find(f"link[@name='{link_name}']")
        collision = link.find("collision") if link is not None else None
        origin = collision.find("origin") if collision is not None else None
        geometry = collision.find("geometry") if collision is not None else None
        box = geometry.find("box") if geometry is not None else None
        if origin is None or box is None:
            raise RuntimeError(f"missing box collision on {link_name}")
        origin.set("xyz", f"{pad_x} 0.0003 {JAW_PAD_CENTER_Z}")
        box.set("size", " ".join(f"{value:.3f}" for value in JAW_PAD_SIZE_M))


def add_gripper_contact_sensors(robot):
    """Expose the existing jaw collision contacts for physical-task checks.

    This adds no collision geometry and does not command a joint.  It only
    publishes contacts from the vendor jaw collision boxes so a failed grasp
    can be distinguished from insufficient friction without inspecting or
    changing object state.
    """
    for link_name, collision_name, topic_name in (
        ("left_jaw_link", "left_jaw_link_collision", "gripper_contacts/left"),
        ("right_jaw_link", "right_jaw_link_collision", "gripper_contacts/right"),
    ):
        gazebo = etree.SubElement(robot, "gazebo", reference=link_name)
        sensor = etree.SubElement(
            gazebo, "sensor", name=f"{link_name}_contact_sensor", type="contact"
        )
        etree.SubElement(sensor, "always_on").text = "true"
        etree.SubElement(sensor, "update_rate").text = "100"
        contact = etree.SubElement(sensor, "contact")
        etree.SubElement(contact, "collision").text = collision_name
        plugin = etree.SubElement(
            sensor,
            "plugin",
            name=f"{link_name}_contact_publisher",
            filename="libgazebo_ros_bumper.so",
        )
        ros = etree.SubElement(plugin, "ros")
        etree.SubElement(ros, "namespace").text = "/"
        etree.SubElement(ros, "remapping").text = f"bumper_states:={topic_name}"
        etree.SubElement(plugin, "frame_name").text = link_name


def add_grasp_center_frame(robot):
    """Add a massless semantic frame at the midpoint of the two jaw pads.

    This is an adapter-only fixed frame: it does not alter vendor meshes,
    joints, collisions or the physical gripper.  It lets task code obtain the
    TCP-to-contact-centre transform from TF instead of maintaining a
    direction-sensitive hand-written offset.
    """
    etree.SubElement(robot, "link", name="grasp_center")
    joint = etree.SubElement(robot, "joint", name="grasp_center_joint", type="fixed")
    etree.SubElement(joint, "parent", link="gripper_base_link")
    etree.SubElement(joint, "child", link="grasp_center")
    # Midpoint of the existing left/right jaw collision-pad centres.
    etree.SubElement(
        joint, "origin", xyz=f"0 0.0003 {JAW_PAD_CENTER_Z}", rpy="0 0 0"
    )


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

    configure_gripper_side_pads(robot)
    add_classic_materials(robot)
    add_gripper_contact_sensors(robot)
    add_grasp_center_frame(robot)

    if args.camera_mode == "rgbd":
        add_rgbd_camera(robot)
    elif args.camera_mode == "aux_rgb":
        add_aux_rgb_camera(robot)

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
        joint.append(command_interface("position", INITIAL_POSITIONS[joint_name]))
        joint.append(state_interface("position", INITIAL_POSITIONS[joint_name]))
        joint.append(state_interface("velocity", 0.0))
    # Gazebo Classic does not apply URDF mimic tags to simulated joints.  The
    # companion model plugin owns the two visual jaw positions; declaring them
    # as state-only interfaces makes their actual positions observable through
    # /joint_states without exposing them to the L7 controller.
    for joint_name in JAW_JOINTS:
        joint = etree.SubElement(control, "joint", name=joint_name)
        joint.append(state_interface("position", JAW_OPEN_POSITION))
        joint.append(state_interface("velocity", 0.0))

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
    # Preserve the vendor's actual jaw kinematic mapping.  Reducing this
    # offset to 0.038 made a ``close`` command stop with a roughly 61 mm
    # physical finger gap around a 40 mm cube: the pads could report contact
    # on an edge yet could not preload the two opposite faces for a frictional
    # lift.  The effort controller below, rather than a changed geometry or
    # a kinematic SetPosition call, bounds the contact load safely.
    etree.SubElement(mimic_plugin, "offset").text = "0.05"
    etree.SubElement(mimic_plugin, "lower_limit").text = "0.0"
    etree.SubElement(mimic_plugin, "upper_limit").text = "0.05"
    # The mimic plugin applies bounded physical effort, never a kinematic
    # attachment.  A 30 g cube needs only about 0.05 N normal force per pad
    # at mu=3 to resist gravity.  Each vendor prismatic jaw also has 1 N of
    # static slide friction, so the earlier 0.6 N cap could not close either
    # carriage at all.  A 1.5 N cap is just above that physical threshold:
    # after slide friction it leaves roughly 0.5 N per pad, with ample
    # friction margin and far below the former 8 N ejection impulse.
    etree.SubElement(mimic_plugin, "position_kp").text = "250.0"
    etree.SubElement(mimic_plugin, "velocity_kd").text = "20.0"
    etree.SubElement(mimic_plugin, "max_force").text = "1.5"

    sys.stdout.write(etree.tostring(robot, encoding="unicode"))


if __name__ == "__main__":
    main()
