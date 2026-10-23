#!/usr/bin/env python3
"""Generate a position-control MuJoCo XML for OpenArm from the original motor XML.

The original XML uses <motor> actuators for arm joints. Dragging Control sliders
in the simulate viewer only writes torque commands, and without a controller the
arm falls under gravity. This script converts the 7 arm joints on each side to
<position> actuators whose ctrlrange matches the joint range, so the Control
sliders directly command desired joint angles.

Usage:
    conda activate yolo
    python generate_position_control_xml.py
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path


ARM_JOINT_RE = "openarm_(left|right)_joint[1-7]"


def build_position_control_xml(src_xml: Path, dst_xml: Path) -> None:
    tree = ET.parse(src_xml)
    root = tree.getroot()

    # Give collision-class geoms a 10 mm margin so the collision engine reports
    # near-miss contact pairs (used by the proximity/collision safety guard)
    for default in root.iter("default"):
        if default.get("class") == "collision":
            for geom in default.iter("geom"):
                geom.set("margin", "0.01")

    # Exclude gripper finger-vs-finger contacts (they overlap by design at the
    # closed pose and would constantly trigger the collision guard)
    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    for b1, b2 in (
        ("openarm_left_right_finger", "openarm_left_left_finger"),
        ("openarm_right_right_finger", "openarm_right_left_finger"),
    ):
        contact.append(ET.Element("exclude", body1=b1, body2=b2))

    # Build a map: joint name -> range string
    joint_ranges: dict[str, str] = {}
    for joint in root.iter("joint"):
        name = joint.get("name")
        rng = joint.get("range")
        if name and rng:
            joint_ranges[name] = rng

    # Modify the actuator section: replace <motor> for arm joints with <position>
    for actuator in root.iter("actuator"):
        for motor in list(actuator.findall("motor")):
            joint_name = motor.get("joint")
            if joint_name and (joint_name.startswith(("openarm_left_joint", "openarm_right_joint")) or "_finger_joint" in joint_name):
                rng = joint_ranges.get(joint_name)
                if rng is None:
                    raise ValueError(f"Joint range not found for {joint_name}")
                # Replace motor with position actuator
                actuator.remove(motor)
                pos = ET.Element("position")
                pos.set("name", joint_name + "_ctrl")
                pos.set("joint", joint_name)
                pos.set("ctrlrange", rng)
                pos.set("kp", "1000")
                pos.set("kv", "100")
                # Preserve class attributes if present
                cls = motor.get("class")
                if cls:
                    pos.set("class", cls)
                actuator.append(pos)

    # Pretty print
    ET.indent(tree, space="  ", level=0)
    tree.write(dst_xml, encoding="utf-8", xml_declaration=True)
    print(f"Wrote position-control XML: {dst_xml}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src",
        default="src/openarm_description/mujoco/openarm_bimanual.xml",
        help="Original MuJoCo XML",
    )
    parser.add_argument(
        "--dst",
        default="src/openarm_description/mujoco/openarm_bimanual_position_control.xml",
        help="Output position-control XML",
    )
    args = parser.parse_args()
    build_position_control_xml(Path(args.src), Path(args.dst))


if __name__ == "__main__":
    main()
