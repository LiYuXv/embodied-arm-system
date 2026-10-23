#!/usr/bin/env python3
"""Extract Modified Denavit-Hartenberg (MDH) parameters from the OpenArm URDF.

The script loads the URDF with Pinocchio, computes the transform between
consecutive link frames at q=0, and decomposes each transform into the MDH
form:

    T = Rot_x(alpha) * Trans_x(a) * Rot_z(theta) * Trans_z(d)

Output is saved to a YAML file.

Usage:
    conda activate yolo
    python extract_mdh.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pinocchio as pin
import yaml


URDF_PATH = "src/openarm_moveit_config/model/table_scene_full.urdf"
OUTPUT_PATH = "openarm_mdh.yaml"

ARM_CHAINS = {
    "right": {
        "links": [f"openarm_right_link{i}" for i in range(0, 8)],
        "joints": [f"openarm_right_joint{i}" for i in range(1, 8)],
        "tcp": "openarm_right_hand_tcp",
    },
    "left": {
        "links": [f"openarm_left_link{i}" for i in range(0, 8)],
        "joints": [f"openarm_left_joint{i}" for i in range(1, 8)],
        "tcp": "openarm_left_hand_tcp",
    },
}


def rot_x(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c],
    ])


def rot_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1],
    ])


def decompose_to_mdh(T: pin.SE3) -> dict:
    """Decompose T into MDH parameters assuming T = Rx(alpha) Tx(a) Rz(theta) Tz(d).

    For a revolute joint, theta is the joint variable (or fixed offset). At q=0,
    we decompose the constant part and report the fixed theta included in the
    URDF joint origin.
    """
    R = T.rotation
    p = T.translation

    # R should be of the form R_x(alpha) @ R_z(theta)
    # R_x(alpha) @ R_z(theta) =
    # [[cos(t), -sin(t), 0],
    #  [sin(t)c, cos(t)c, -s],
    #  [sin(t)s, cos(t)s, c]]
    # where c = cos(alpha), s = sin(alpha), t = theta

    # Check if R[0, 2] is close to zero (required for exact MDH form)
    r02 = R[0, 2]
    if not np.isclose(r02, 0.0, atol=1e-6):
        return {
            "error": f"Rotation not in MDH form: R[0,2]={r02:.6e}",
            "a": float(np.nan),
            "alpha": float(np.nan),
            "d": float(np.nan),
            "theta": float(np.nan),
            "rotation": R.tolist(),
            "translation": p.tolist(),
        }

    alpha = float(np.arctan2(-R[1, 2], R[2, 2]))
    theta = float(np.arctan2(-R[0, 1], R[0, 0]))

    # p_local = R_x(-alpha) @ p  => [a, 0, d]
    p_local = rot_x(-alpha) @ p
    a = float(p_local[0])
    d = float(p_local[2])

    # Normalize angles to [-pi, pi]
    alpha = float((alpha + np.pi) % (2 * np.pi) - np.pi)
    theta = float((theta + np.pi) % (2 * np.pi) - np.pi)

    return {
        "a": a,
        "alpha": alpha,
        "d": d,
        "theta": theta,
        "alpha_deg": float(np.degrees(alpha)),
        "theta_deg": float(np.degrees(theta)),
    }


def extract_arm_mdh(model: pin.Model, data: pin.Data, chain: dict) -> list[dict]:
    """Extract MDH parameters for one arm chain."""
    link_names = chain["links"]
    joint_names = chain["joints"]

    # Compute world transforms of all link frames at q=0
    q0 = pin.neutral(model)
    pin.forwardKinematics(model, data, q0)
    pin.updateFramePlacements(model, data)

    world_poses = {}
    for name in link_names:
        fid = model.getFrameId(name)
        if fid >= len(model.frames):
            raise ValueError(f"Frame '{name}' not found in model")
        world_poses[name] = data.oMf[fid]

    mdh_params = []
    for i in range(1, len(link_names)):
        prev_link = link_names[i - 1]
        curr_link = link_names[i]
        joint_name = joint_names[i - 1]

        # T_{i-1}^{i} = T_world_{i-1}^{-1} * T_world_i
        T = world_poses[prev_link].actInv(world_poses[curr_link])
        params = decompose_to_mdh(T)
        params["joint"] = joint_name
        params["from_link"] = prev_link
        params["to_link"] = curr_link
        mdh_params.append(params)

    return mdh_params


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract MDH parameters from OpenArm URDF")
    parser.add_argument("--urdf", default=URDF_PATH, help="Path to URDF")
    parser.add_argument("--output", default=OUTPUT_PATH, help="Output YAML path")
    args = parser.parse_args()

    urdf_path = Path(args.urdf)
    if not urdf_path.is_absolute():
        urdf_path = Path(__file__).parent / urdf_path

    model = pin.buildModelFromUrdf(str(urdf_path))
    data = model.createData()

    result = {}
    for side, chain in ARM_CHAINS.items():
        print(f"\n=== {side.capitalize()} arm ===")
        mdh = extract_arm_mdh(model, data, chain)
        for p in mdh:
            if "error" in p:
                print(f"{p['joint']}: {p['error']}")
            else:
                print(
                    f"{p['joint']}: a={p['a']:.6f}, alpha={p['alpha_deg']:.3f}°, "
                    f"d={p['d']:.6f}, theta={p['theta_deg']:.3f}°"
                )
        result[side] = mdh

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path(__file__).parent / output_path
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(result, f, sort_keys=False, allow_unicode=True)
    print(f"\nSaved MDH parameters to: {output_path}")


if __name__ == "__main__":
    main()
