"""Standalone Pinocchio FK/IK demo for OpenArm (no ROS, no MoveIt).

This script loads the URDF with Pinocchio, computes forward kinematics for
one arm, then solves inverse kinematics numerically with a Jacobian-based
Levenberg-Marquardt solver. The result can optionally be visualized in MuJoCo.

Run in the yolo environment:
    conda activate yolo
    python pinocchio_demo.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pinocchio as pin


# ---------------------------------------------------------------------------
# Arm configuration
# ---------------------------------------------------------------------------
ARM_JOINTS = {
    "left": [f"openarm_left_joint{i}" for i in range(1, 8)],
    "right": [f"openarm_right_joint{i}" for i in range(1, 8)],
}

# Use the TCP (tool-center-point) frame as the end-effector.
# These frames exist in the MoveIt-ready URDF.
EE_FRAMES = {
    "left": "openarm_left_hand_tcp",
    "right": "openarm_right_hand_tcp",
}

# For the root URDF (openarm_bimanual.urdf) the <robot> tag has no name.
# We use the MoveIt package URDF which already has a valid name and no
# package:// mesh references.
DEFAULT_URDF = "src/openarm_moveit_config/model/table_scene_full.urdf"


class OpenArmKinematics:
    """Encapsulates Pinocchio model, data, and FK/IK helpers for one arm."""

    def __init__(self, urdf_path: str, side: str = "right") -> None:
        self.side = side
        self.urdf_path = Path(urdf_path)
        if not self.urdf_path.is_absolute():
            self.urdf_path = Path(__file__).parent / self.urdf_path

        # Load model and create data
        self.model = pin.buildModelFromUrdf(str(self.urdf_path))
        self.data = self.model.createData()

        # Resolve joint indices and joint position slice for this arm
        self.joint_names = ARM_JOINTS[side]
        self.joint_ids = [self.model.getJointId(name) for name in self.joint_names]
        for name, jid in zip(self.joint_names, self.joint_ids):
            if jid >= len(self.model.names):
                raise ValueError(f"Joint '{name}' not found in URDF")

        # Build a list of (joint_id, qpos_offset) for this arm.  Each joint's
        # configuration in the full q vector starts at model.joints[jid].idx_q.
        self.joint_q_offsets = [int(self.model.joints[jid].idx_q) for jid in self.joint_ids]
        self.joint_v_offsets = [int(self.model.joints[jid].idx_v) for jid in self.joint_ids]

        # End-effector frame
        self.ee_frame_name = EE_FRAMES[side]
        self.ee_frame_id = self.model.getFrameId(self.ee_frame_name)
        if self.ee_frame_id >= len(self.model.frames):
            raise ValueError(f"Frame '{self.ee_frame_name}' not found in URDF")

        # Joint limits (lower/upper) for the 7 arm joints
        self.lower_limits = np.array([
            self.model.lowerPositionLimit[q] for q in self.joint_q_offsets
        ])
        self.upper_limits = np.array([
            self.model.upperPositionLimit[q] for q in self.joint_q_offsets
        ])

    # ------------------------------------------------------------------
    # Forward kinematics
    # ------------------------------------------------------------------
    def fk(self, q_arm: np.ndarray) -> pin.SE3:
        """Return the end-effector pose (SE3) for the given 7-DOF arm config."""
        q_full = self._full_q(q_arm)
        pin.forwardKinematics(self.model, self.data, q_full)
        pin.updateFramePlacements(self.model, self.data)
        return self.data.oMf[self.ee_frame_id]

    def fk_translation(self, q_arm: np.ndarray) -> np.ndarray:
        """Return only the end-effector position [x, y, z]."""
        return self.fk(q_arm).translation.copy()

    def fk_quaternion(self, q_arm: np.ndarray) -> np.ndarray:
        """Return the end-effector orientation as [x, y, z, w] quaternion."""
        return pin.Quaternion(self.fk(q_arm).rotation).coeffs()

    # ------------------------------------------------------------------
    # Inverse kinematics (numerical, Jacobian-based Levenberg-Marquardt)
    # ------------------------------------------------------------------
    def ik(
        self,
        target: pin.SE3,
        q_init: np.ndarray | None = None,
        max_iters: int = 100,
        tol: float = 1e-5,
        dt: float = 0.1,
        damping: float = 1e-3,
        clamp_to_limits: bool = True,
    ) -> np.ndarray:
        """Solve IK for a desired SE3 target pose.

        Returns the 7-DOF arm configuration that reaches the target pose.
        """
        if q_init is None:
            q_arm = np.zeros(7)
        else:
            q_arm = np.asarray(q_init, dtype=float).copy()

        q_full = self._full_q(q_arm)
        pin.forwardKinematics(self.model, self.data, q_full)
        pin.updateFramePlacements(self.model, self.data)
        pin.computeJointJacobians(self.model, self.data, q_full)

        for i in range(max_iters):
            # Current EE pose and error (6D twist in local frame)
            current = self.data.oMf[self.ee_frame_id]
            err = pin.log(target.inverse() * current)
            err_norm = np.linalg.norm(err)
            if err_norm < tol:
                break

            # Geometric Jacobian of the EE frame in LOCAL frame
            J = pin.computeFrameJacobian(
                self.model, self.data, q_full, self.ee_frame_id
            )
            # J is 6 x model.nv. Select columns corresponding to our 7 joints.
            J_arm = J[:, self._v_indices()]

            # Levenberg-Marquardt update: dq = (J^T J + lambda I)^{-1} J^T err
            H = J_arm.T @ J_arm + damping * np.eye(7)
            dq = np.linalg.solve(H, J_arm.T @ (-err))

            q_arm += dt * dq

            if clamp_to_limits:
                q_arm = np.clip(q_arm, self.lower_limits, self.upper_limits)

            q_full = self._full_q(q_arm)
            pin.forwardKinematics(self.model, self.data, q_full)
            pin.updateFramePlacements(self.model, self.data)
            pin.computeJointJacobians(self.model, self.data, q_full)
        else:
            print(f"IK did not converge after {max_iters} iterations (error={err_norm:.2e})")

        return q_arm

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _full_q(self, q_arm: np.ndarray) -> np.ndarray:
        """Expand the 7-DOF arm config into the full model configuration vector."""
        q_full = pin.neutral(self.model)
        q_arm = np.asarray(q_arm, dtype=float).flatten()
        if q_arm.size != 7:
            raise ValueError(f"Expected 7-DOF arm config, got {q_arm.size}")
        for offset, value in zip(self.joint_q_offsets, q_arm):
            q_full[offset] = value
        return q_full

    def _v_indices(self) -> list[int]:
        """Return the model.v indices for the 7 arm joints."""
        return self.joint_v_offsets


# ---------------------------------------------------------------------------
# Optional MuJoCo visualization helpers
# ---------------------------------------------------------------------------
def visualize_in_mujoco(q_left: np.ndarray, q_right: np.ndarray, xml_path: str) -> None:
    """Open MuJoCo viewer and set the arm positions to the given configs."""
    import mujoco
    import mujoco.viewer

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # Map Pinocchio joint names to MuJoCo joint names. The MuJoCo XML uses the
    # same joint names as the URDF in this project.
    joint_name_to_qpos_id = {}
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        joint_name_to_qpos_id[name] = model.jnt_qposadr[i]

    def set_arm(side: str, q_arm: np.ndarray) -> None:
        for jname, val in zip(ARM_JOINTS[side], q_arm):
            if jname in joint_name_to_qpos_id:
                data.qpos[joint_name_to_qpos_id[jname]] = val

    set_arm("left", q_left)
    set_arm("right", q_right)
    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("MuJoCo viewer open. Press Ctrl+C in terminal to close.")
        while viewer.is_running():
            viewer.sync()
            time.sleep(1.0 / 60.0)


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Pinocchio FK/IK demo for OpenArm")
    parser.add_argument("--urdf", default=DEFAULT_URDF, help="Path to URDF")
    parser.add_argument("--side", default="right", choices=["left", "right"])
    parser.add_argument("--mujoco-xml", default=None, help="Optional MuJoCo XML to visualize")
    args = parser.parse_args()

    kin = OpenArmKinematics(args.urdf, side=args.side)
    print(f"Loaded model from: {kin.urdf_path}")
    print(f"Model DOF: {kin.model.nq}, selected arm: {args.side}")
    print(f"EE frame: {kin.ee_frame_name} (id={kin.ee_frame_id})")
    print(f"Arm joints: {kin.joint_names}")

    # --- FK demo: compute pose at a "ready" configuration ------------------
    q_ready = np.array([0.0, 1.5, 0.0, 1.5, 0.0, 0.0, 0.0])
    if args.side == "left":
        q_ready[1] = -1.5
    pose_ready = kin.fk(q_ready)
    print("\n--- Forward Kinematics ---")
    print(f"q_ready ({args.side}): {q_ready}")
    print(f"EE position:    {pose_ready.translation}")
    print(f"EE rotation:\n{pose_ready.rotation}")

    # --- IK demo: move the EE 10 cm along world x and solve from perturbed start --
    target = pose_ready.copy()
    target.translation[0] += 0.10  # move 10 cm along world x

    q_init = q_ready + np.array([0.2, -0.3, 0.1, -0.2, 0.1, 0.0, 0.0])
    q_init = np.clip(q_init, kin.lower_limits, kin.upper_limits)

    print("\n--- Inverse Kinematics ---")
    print(f"Target position: {target.translation}")
    print(f"q_init: {q_init}")
    t0 = time.time()
    q_ik = kin.ik(target, q_init=q_init, max_iters=200, tol=1e-6, dt=0.5, damping=1e-4)
    t1 = time.time()
    print(f"IK solved in {t1 - t0:.3f}s")
    print(f"q_ik:   {q_ik}")

    # Verify
    pose_ik = kin.fk(q_ik)
    pos_err = np.linalg.norm(pose_ik.translation - target.translation)
    rot_err = np.linalg.norm(pin.log(target.inverse() * pose_ik))
    print(f"Position error: {pos_err:.6e}")
    print(f"Rotation error: {rot_err:.6e}")

    # --- Optional: visualize in MuJoCo ------------------------------------
    if args.mujoco_xml:
        q_left = q_ik if args.side == "left" else np.zeros(7)
        q_right = q_ik if args.side == "right" else np.zeros(7)
        visualize_in_mujoco(q_left, q_right, args.mujoco_xml)


if __name__ == "__main__":
    main()
