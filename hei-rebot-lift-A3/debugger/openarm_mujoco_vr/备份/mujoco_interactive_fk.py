"""Interactive MuJoCo + Pinocchio FK demo.

Open the MuJoCo viewer, drag the **Joint** sliders on the left panel, and this
script prints the end-effector position + orientation (as quaternion) of the
left and right arms in real time.

Usage:
    conda activate yolo
    python mujoco_interactive_fk.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import pinocchio as pin


URDF_PATH = "src/openarm_moveit_config/model/table_scene_full.urdf"
MJCF_PATH = "src/openarm_description/mujoco/scene.xml"

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
    """Build a full Pinocchio q vector from the current MuJoCo qpos.

    We map by joint name so that the order of joints in the MuJoCo model does
    not have to match the Pinocchio model exactly.
    """
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive MuJoCo FK demo")
    parser.add_argument("--urdf", default=URDF_PATH, help="Path to URDF")
    parser.add_argument("--mjcf", default=MJCF_PATH, help="Path to MuJoCo XML")
    parser.add_argument(
        "--rate", type=float, default=10.0, help="Print rate in Hz (default: 10)"
    )
    args = parser.parse_args()

    # Load Pinocchio model
    pin_model, pin_data = build_pinocchio_model(args.urdf)
    ee_frame_ids = {
        side: pin_model.getFrameId(name) for side, name in EE_FRAMES.items()
    }
    print(f"Loaded Pinocchio model: {pin_model.nq} DOF")
    for side, fid in ee_frame_ids.items():
        print(f"  {side} EE frame '{EE_FRAMES[side]}' -> id {fid}")

    # Load MuJoCo model
    mj_model = mujoco.MjModel.from_xml_path(args.mjcf)
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, mj_data)

    print("\nMuJoCo viewer opened.")
    print(
        "Use the **Joint** sliders in the left panel to move each arm's joints."
    )
    print("The script prints FK results below.\n")

    dt_print = 1.0 / args.rate
    last_print = 0.0

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        while viewer.is_running():
            # Read current MuJoCo state and compute FK
            left_q = qpos_to_pin_q(pin_model, mj_data, ARM_JOINTS["left"])
            right_q = qpos_to_pin_q(pin_model, mj_data, ARM_JOINTS["right"])

            left_pos, left_quat = compute_ee_pose(
                pin_model, pin_data, left_q, ee_frame_ids["left"]
            )
            right_pos, right_quat = compute_ee_pose(
                pin_model, pin_data, right_q, ee_frame_ids["right"]
            )

            # Throttle console output to the requested rate
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
            time.sleep(1.0 / 60.0)

    print("\nViewer closed.")


if __name__ == "__main__":
    main()
