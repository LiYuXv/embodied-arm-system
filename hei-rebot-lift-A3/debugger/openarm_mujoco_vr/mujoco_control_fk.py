"""Interactive MuJoCo + Pinocchio FK demo with Control sliders.

This script loads the position-control version of the OpenArm MuJoCo model.
Dragging the **Control** sliders in the viewer directly commands desired joint
angles for the 7-DOF arm joints. The simulation is stepped so the arm follows
the sliders, and the end-effector pose is printed in real time.

Usage:
    conda activate yolo
    python mujoco_control_fk.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import pinocchio as pin

from generate_position_control_xml import build_position_control_xml


URDF_PATH = "src/openarm_moveit_config/model/table_scene_full.urdf"
MJCF_PATH = "src/openarm_description/mujoco/scene_position_control.xml"

ARM_JOINTS = {
    "left": [f"openarm_left_joint{i}" for i in range(1, 8)],
    "right": [f"openarm_right_joint{i}" for i in range(1, 8)],
}
EE_FRAMES = {
    "left": "openarm_left_hand_tcp",
    "right": "openarm_right_hand_tcp",
}


def build_pinocchio_model(urdf_path: str) -> tuple[pin.Model, pin.Data]:
    path = Path(urdf_path)
    if not path.is_absolute():
        path = Path(__file__).parent / path
    model = pin.buildModelFromUrdf(str(path))
    data = model.createData()
    return model, data


def qpos_to_pin_q(model: pin.Model, mj_data, joint_names: list[str]) -> np.ndarray:
    """Build a full Pinocchio q vector from the current MuJoCo qpos by joint name."""
    q = pin.neutral(model)
    for name in joint_names:
        pin_joint_id = model.getJointId(name)
        if pin_joint_id >= len(model.names):
            raise ValueError(f"Joint '{name}' not found in Pinocchio model")
        q_offset = int(model.joints[pin_joint_id].idx_q)

        mj_joint_id = mujoco.mj_name2id(mj_data.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if mj_joint_id < 0:
            raise ValueError(f"Joint '{name}' not found in MuJoCo model")
        qpos_adr = mj_data.model.jnt_qposadr[mj_joint_id]
        q[q_offset] = mj_data.qpos[qpos_adr]
    return q


def compute_ee_pose(
    model: pin.Model, data: pin.Data, q: np.ndarray, ee_frame_id: int
) -> tuple[np.ndarray, np.ndarray]:
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    pose = data.oMf[ee_frame_id]
    translation = pose.translation.copy()
    quat = pin.Quaternion(pose.rotation).coeffs()  # [x, y, z, w]
    return translation, quat


def sync_control_to_qpos(mj_data, joint_names: list[str]) -> None:
    """Directly copy position-control actuator commands to qpos.

    The generated position-control XML sets each actuator's ctrlrange equal to
    the joint range, so the Control slider value is already the desired joint
    angle. Bypassing mj_step avoids arm swing / gravity effects and gives an
    instantaneous response to the sliders.
    """
    model = mj_data.model
    for name in joint_names:
        act_name = name + "_ctrl"
        act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
        if act_id < 0:
            continue
        jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jnt_id < 0:
            continue
        mj_data.qpos[model.jnt_qposadr[jnt_id]] = mj_data.ctrl[act_id]


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive MuJoCo FK via Control sliders")
    parser.add_argument("--urdf", default=URDF_PATH, help="Path to URDF")
    parser.add_argument("--mjcf", default=MJCF_PATH, help="Path to position-control MuJoCo XML")
    parser.add_argument("--rate", type=float, default=10.0, help="Print rate in Hz (default: 10)")
    args = parser.parse_args()

    # Auto-generate the position-control XML if it is missing
    mjcf_path = Path(args.mjcf)
    if not mjcf_path.is_absolute():
        mjcf_path = Path(__file__).parent / mjcf_path
    if not mjcf_path.exists():
        print(f"Position-control XML not found: {mjcf_path}")
        src_xml = Path("src/openarm_description/mujoco/openarm_bimanual.xml")
        if not src_xml.is_absolute():
            src_xml = Path(__file__).parent / src_xml
        print(f"Generating it from {src_xml} ...")
        build_position_control_xml(src_xml, mjcf_path)
        # Also create the scene wrapper if needed
        scene_path = Path("src/openarm_description/mujoco/scene_position_control.xml")
        if not scene_path.is_absolute():
            scene_path = Path(__file__).parent / scene_path
        if not scene_path.exists():
            scene_path.write_text(
                f'<mujoco model="openarm_bimanual scene (position control)">\n'
                f'  <include file="{mjcf_path.name}"/>\n'
                f'  <statistic center="0 0 0.43" extent=".85" meansize="0.05"/>\n'
                f'  <visual>\n'
                f'    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>\n'
                f'    <rgba haze="0.15 0.25 0.35 1"/>\n'
                f'    <global azimuth="160" elevation="-10"/>\n'
                f'    <quality shadowsize="8192"/>\n'
                f'  </visual>\n'
                f'  <asset>\n'
                f'    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>\n'
                f'    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"\n'
                f'      markrgb="0.8 0.8 0.8" width="300" height="300"/>\n'
                f'    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>\n'
                f'  </asset>\n'
                f'  <worldbody>\n'
                f'    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>\n'
                f'    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>\n'
                f'  </worldbody>\n'
                f'</mujoco>\n',
                encoding="utf-8",
            )
            print(f"Wrote scene wrapper: {scene_path}")

    pin_model, pin_data = build_pinocchio_model(args.urdf)
    ee_frame_ids = {
        side: pin_model.getFrameId(name) for side, name in EE_FRAMES.items()
    }
    print(f"Loaded Pinocchio model: {pin_model.nq} DOF")
    for side, fid in ee_frame_ids.items():
        print(f"  {side} EE frame '{EE_FRAMES[side]}' -> id {fid}")

    mj_model = mujoco.MjModel.from_xml_path(args.mjcf)
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, mj_data)

    # Verify that the actuators for arm joints are position actuators
    for side, joints in ARM_JOINTS.items():
        for name in joints:
            act_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, name + "_ctrl")
            if act_id < 0:
                print(f"Warning: position actuator for {name} not found")

    print("\nMuJoCo viewer opened with position-control actuators.")
    print("Drag the **Control** sliders to command each arm's joint angles.")
    print("The joint angles are copied directly from the slider values, so the arm")
    print("responds instantly and does not fall under gravity.\n")

    dt_print = 1.0 / args.rate
    last_print = 0.0

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        while viewer.is_running():
            # Position actuators: Control slider value == desired joint angle.
            # Copy directly to qpos for instantaneous response.
            sync_control_to_qpos(mj_data, ARM_JOINTS["left"])
            sync_control_to_qpos(mj_data, ARM_JOINTS["right"])
            mujoco.mj_forward(mj_model, mj_data)

            # Read current MuJoCo state and compute FK
            left_q = qpos_to_pin_q(pin_model, mj_data, ARM_JOINTS["left"])
            right_q = qpos_to_pin_q(pin_model, mj_data, ARM_JOINTS["right"])

            left_pos, left_quat = compute_ee_pose(
                pin_model, pin_data, left_q, ee_frame_ids["left"]
            )
            right_pos, right_quat = compute_ee_pose(
                pin_model, pin_data, right_q, ee_frame_ids["right"]
            )

            now = time.time()
            if now - last_print >= dt_print:
                last_print = now
                print(
                    f"\rL: pos=({left_pos[0]:+.3f}, {left_pos[1]:+.3f}, {left_pos[2]:+.3f})  "
                    f"quat=({left_quat[0]:+.3f}, {left_quat[1]:+.3f}, {left_quat[2]:+.3f}, {left_quat[3]:+.3f}) | "
                    f"R: pos=({right_pos[0]:+.3f}, {right_pos[1]:+.3f}, {right_pos[2]:+.3f})  "
                    f"quat=({right_quat[0]:+.3f}, {right_quat[1]:+.3f}, {right_quat[2]:+.3f}, {right_quat[3]:+.3f})",
                    end="",
                    flush=True,
                )

            viewer.sync()
            # ~60 FPS refresh
            time.sleep(1.0 / 60.0)

    print("\nViewer closed.")


if __name__ == "__main__":
    main()
