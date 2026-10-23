#!/usr/bin/env python3
"""MuJoCo + Pinocchio dual-arm Cartesian control with gravity torque display.

Features:
- Independent X/Y/Z translation and Roll/Pitch/Yaw rotation buttons for right and left arms.
- Reset button resets both arms to their neutral ready pose.
- Gripper sliders for both arms.
- Real-time FK printing of both end-effector poses.
- Real-time gravity-compensation joint torques and whole-robot CoM display.

Usage:
    conda activate yolo
    python mujoco_cartesian_control_fk.py
"""

from __future__ import annotations

import argparse
import time
import tkinter as tk
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, ClassVar

import mujoco
import mujoco.viewer
import numpy as np
import pinocchio as pin

from generate_position_control_xml import build_position_control_xml
from generate_mobile_base_xml import (
    build_mobile_base_xml,
    build_mobile3_xml,
    prepare_mobile3_assets,
)

# VR support (optional)
try:
    from vr_receiver import VRControllerReceiver, VRControllerState, map_trigger_to_gripper, get_local_ips
    VR_AVAILABLE = True
except ImportError:
    VR_AVAILABLE = False
    print("[VR] vr_receiver module not found. VR control disabled.")


URDF_PATH = "openarm_bimanual.urdf"
MJCF_PATH = "src/openarm_description/mujoco/scene_position_control.xml"
MJCF_MOBILE_PATH = "src/openarm_description/mujoco/scene_mobile.xml"
MJCF_MOBILE2_PATH = "src/openarm_description/mujoco/scene_mobile2.xml"
MJCF_MOBILE3_PATH = "src/openarm_description/mujoco/scene_mobile3.xml"

# Mobile base (mecanum/omni + lead-screw lift) parameters.
# Wheel layout geometry is read from the model at runtime.
BASE_JOINTS = {"x": "base_x", "y": "base_y", "yaw": "base_yaw", "lift": "lift_z"}
WHEEL_JOINTS = ["wheel_fl", "wheel_fr", "wheel_rl", "wheel_rr"]
WAIST_JOINTS = {"pitch": "waist_pitch", "yaw": "waist_yaw"}  # mobile3 only
# Hierarchical Cartesian IK (mobile4) waist-direction policy:
# before every Cartesian motion the waist pitch/yaw joints are checked. If
# they are at zero (within WAIST_ZERO_TOL), the normal hierarchy applies:
# arm only -> arm+waist -> arm+waist+base/lift. If the waist is NOT at zero,
# the first solve ("腰回零") clamps each waist joint's limits to its
# toward-zero side: steps achievable while homing the waist are done by the
# waist moving toward zero (gradually, only as much as the step needs); steps
# that would need the waist to move away from zero leave it pinned and the
# arm works alone; only when the arm saturates is the free waist
# (away-from-zero motion) allowed, with the base + lift as the last resort.
WAIST_ZERO_TOL = np.radians(2.0)  # waist counts as "at zero" within this band
WHEEL_RADIUS = 0.076  # 152 mm wheel diameter
LIFT_MAX = 0.8  # 800 mm stroke
LIFT_RATE = 0.2  # m/s lift speed

# HMD (headset) -> base/lift control mapping gains and deadzones
HMD_BASE_LIN_GAIN = 1.2   # m/s per meter of head translation beyond deadzone
HMD_BASE_YAW_GAIN = 1.2   # rad/s per rad of head turn beyond deadzone
HMD_LIFT_GAIN = 1.5       # lift meters per meter of crouch/stand beyond deadzone
HMD_DEADZONE_XY = 0.03    # m
HMD_DEADZONE_YAW = 5.0    # deg
HMD_DEADZONE_Z = 0.03     # m
HMD_BASE_MAX_LIN = 0.4    # m/s
HMD_BASE_MAX_YAW = 1.0    # rad/s
HMD_YAW_SIGN = +1.0       # flip if head-turn direction is mirrored
HMD_STRAFE_SIGN = -1.0    # head-right must map to base-right (base vy+ is LEFT)
HMD_ABS_POS_GAIN = 1.0    # absolute mode: base meters per meter of head translation
HMD_ABS_LIFT_GAIN = 1.0   # absolute mode: lift meters per meter of head height change

ARM_JOINTS = {
    "left": [f"openarm_left_joint{i}" for i in range(1, 8)],
    "right": [f"openarm_right_joint{i}" for i in range(1, 8)],
}
FINGER_JOINTS = {
    "left": ["openarm_left_finger_joint1", "openarm_left_finger_joint2"],
    "right": ["openarm_right_finger_joint1", "openarm_right_finger_joint2"],
}
EE_FRAMES = {
    "left": "openarm_left_hand_tcp",
    "right": "openarm_right_hand_tcp",
}


@dataclass
class ArmState:
    target_translation: np.ndarray = field(default_factory=lambda: np.zeros(3))
    target_rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    q_target: np.ndarray | None = None
    cartesian_dirty: bool = False


def fix_urdf_for_pinocchio(urdf_path: str) -> Path:
    """Create a copy of the URDF usable by Pinocchio.

    Adds a robot name if missing and replaces package:// mesh URIs with
    relative filesystem paths. The fixed file is written next to the original
    with a `.fixed.urdf` suffix.
    """
    path = Path(urdf_path)
    if not path.is_absolute():
        path = Path(__file__).parent / path
    text = path.read_text(encoding="utf-8")
    needs_fix = False

    if "<robot>" in text and "<robot name=" not in text:
        text = text.replace("<robot>", '<robot name="openarm_bimanual">')
        needs_fix = True

    if "package://openarm_description/" in text:
        text = text.replace("package://openarm_description/", "src/openarm_description/")
        needs_fix = True

    if not needs_fix:
        return path

    fixed_path = path.with_suffix(".fixed.urdf")
    fixed_path.write_text(text, encoding="utf-8")
    print(f"Created fixed URDF for Pinocchio: {fixed_path}")
    return fixed_path


class RobotDynamics:
    """Pinocchio FK/IK/dynamics for the full dual-arm robot."""

    def __init__(self, urdf_path: str) -> None:
        path = fix_urdf_for_pinocchio(urdf_path)
        self.model = pin.buildModelFromUrdf(str(path))
        self.data = self.model.createData()

        self.arm_info = {}
        for side in ("right", "left"):
            joint_names = ARM_JOINTS[side]
            joint_ids = [self.model.getJointId(n) for n in joint_names]
            q_offsets = [int(self.model.joints[jid].idx_q) for jid in joint_ids]
            v_offsets = [int(self.model.joints[jid].idx_v) for jid in joint_ids]
            lower_limits = np.array([self.model.lowerPositionLimit[q] for q in q_offsets])
            upper_limits = np.array([self.model.upperPositionLimit[q] for q in q_offsets])
            ee_frame_id = self.model.getFrameId(EE_FRAMES[side])
            self.arm_info[side] = {
                "joint_names": joint_names,
                "joint_ids": joint_ids,
                "q_offsets": q_offsets,
                "v_offsets": v_offsets,
                "lower_limits": lower_limits,
                "upper_limits": upper_limits,
                "ee_frame_id": ee_frame_id,
            }

    def _full_q(self, q_right: np.ndarray, q_left: np.ndarray) -> np.ndarray:
        q_full = pin.neutral(self.model)
        for side, q in (("right", q_right), ("left", q_left)):
            info = self.arm_info[side]
            for offset, value in zip(info["q_offsets"], q):
                q_full[offset] = value
        return q_full

    def _single_q(self, side: str, q_arm: np.ndarray) -> np.ndarray:
        q_full = pin.neutral(self.model)
        info = self.arm_info[side]
        for offset, value in zip(info["q_offsets"], q_arm):
            q_full[offset] = value
        return q_full

    def fk(self, side: str, q_arm: np.ndarray) -> pin.SE3:
        q_full = self._single_q(side, q_arm)
        pin.forwardKinematics(self.model, self.data, q_full)
        pin.updateFramePlacements(self.model, self.data)
        return self.data.oMf[self.arm_info[side]["ee_frame_id"]]

    def ik(
        self,
        side: str,
        target: pin.SE3,
        q_init: np.ndarray,
        max_iters: int = 200,
        tol: float = 1e-6,
        dt: float = 0.1,
        damping: float = 1e-3,
    ) -> np.ndarray | None:
        info = self.arm_info[side]
        q_arm = np.asarray(q_init, dtype=float).copy()
        q_full = self._single_q(side, q_arm)
        pin.forwardKinematics(self.model, self.data, q_full)
        pin.updateFramePlacements(self.model, self.data)
        pin.computeJointJacobians(self.model, self.data, q_full)

        best_err = float("inf")
        best_q = q_arm.copy()
        for _ in range(max_iters):
            current = self.data.oMf[info["ee_frame_id"]]
            err = pin.log(target.inverse() * current)
            err_norm = np.linalg.norm(err)
            if err_norm < best_err:
                best_err = err_norm
                best_q = q_arm.copy()
            if err_norm < tol:
                break
            J = pin.computeFrameJacobian(self.model, self.data, q_full, info["ee_frame_id"])
            J_arm = J[:, info["v_offsets"]]
            H = J_arm.T @ J_arm + damping * np.eye(7)
            dq = np.linalg.solve(H, J_arm.T @ (-err))
            q_arm += dt * dq
            q_arm = np.clip(q_arm, info["lower_limits"], info["upper_limits"])
            q_full = self._single_q(side, q_arm)
            pin.forwardKinematics(self.model, self.data, q_full)
            pin.updateFramePlacements(self.model, self.data)
            pin.computeJointJacobians(self.model, self.data, q_full)
        else:
            # Max iterations reached. Return best solution if error is small enough.
            if best_err < 1e-2:
                return best_q
            return None
        return q_arm


class MuJoCoDynamics:
    """MuJoCo inverse-dynamics based gravity torque and per-arm CoM computation."""

    MARKER_RADIUS: ClassVar[float] = 0.03
    LEFT_COM_RGBA: ClassVar[np.ndarray] = np.array([1.0, 0.0, 0.0, 0.9])
    RIGHT_COM_RGBA: ClassVar[np.ndarray] = np.array([1.0, 0.2, 0.0, 0.9])
    FRAME_AXIS_LENGTH: ClassVar[float] = 0.05
    BASE_FRAME_AXIS_LENGTH: ClassVar[float] = 0.15
    JOINT_FRAME_AXIS_LENGTH: ClassVar[float] = 0.09
    JOINT_FRAME_AXIS_WIDTH: ClassVar[float] = 2.0
    FRAME_AXIS_WIDTH: ClassVar[float] = 3.0  # pixels for line

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.model = model
        self.data = data

        self.arm_dof = {
            "right": self._arm_dof_indices("openarm_right_"),
            "left": self._arm_dof_indices("openarm_left_"),
        }
        self.arm_limits = {
            "right": self._arm_joint_limits("openarm_right_"),
            "left": self._arm_joint_limits("openarm_left_"),
        }
        self.arm_body_indices = {
            "right": self._collect_body_indices("openarm_right_"),
            "left": self._collect_body_indices("openarm_left_"),
        }
        self.arm_joint_ids = {
            "right": self._collect_joint_ids("openarm_right_"),
            "left": self._collect_joint_ids("openarm_left_"),
        }
        self.last_ik_err = {"right": float("inf"), "left": float("inf")}

    def _arm_dof_indices(self, prefix: str) -> np.ndarray:
        indices = []
        for i in range(1, 8):
            name = f"{prefix}joint{i}"
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(f"Joint '{name}' not found in MuJoCo model")
            indices.append(int(self.model.jnt_dofadr[jid]))
        return np.array(indices, dtype=np.intp)

    def _arm_joint_limits(self, prefix: str) -> dict[str, np.ndarray]:
        lower = []
        upper = []
        for i in range(1, 8):
            name = f"{prefix}joint{i}"
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(f"Joint '{name}' not found in MuJoCo model")
            lower.append(float(self.model.jnt_range[jid][0]))
            upper.append(float(self.model.jnt_range[jid][1]))
        return {"lower": np.array(lower), "upper": np.array(upper)}

    def _collect_body_indices(self, prefix: str) -> list[int]:
        indices = []
        for i in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name and name.startswith(prefix):
                indices.append(i)
        return indices

    def _collect_joint_ids(self, prefix: str) -> list[int]:
        indices = []
        for i in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name and name.startswith(prefix):
                indices.append(i)
        return indices

    def _rotation_from_z(self, z_axis: np.ndarray) -> np.ndarray:
        """Build a rotation matrix whose z-axis is aligned with the given vector."""
        z = np.asarray(z_axis, dtype=float)
        norm = np.linalg.norm(z)
        if norm < 1e-12:
            return np.eye(3)
        z = z / norm
        if abs(z[0]) < 0.9:
            a = np.array([1.0, 0.0, 0.0])
        else:
            a = np.array([0.0, 1.0, 0.0])
        x = np.cross(a, z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        return np.column_stack([x, y, z])

    def gravity_torques(self) -> dict[str, np.ndarray]:
        """Compute gravity-compensation torques for both arms using mj_inverse."""
        saved_qvel = self.data.qvel.copy()
        saved_qacc = self.data.qacc.copy()

        self.data.qvel[:] = 0.0
        self.data.qacc[:] = 0.0
        mujoco.mj_inverse(self.model, self.data)
        qfrc_gravity = self.data.qfrc_inverse.copy()

        self.data.qvel[:] = saved_qvel
        self.data.qacc[:] = saved_qacc

        return {
            "right": qfrc_gravity[self.arm_dof["right"]],
            "left": qfrc_gravity[self.arm_dof["left"]],
        }

    def arm_com(self, side: str) -> tuple[np.ndarray, float]:
        """Compute center of mass and total mass for one arm from body data."""
        total_mass = 0.0
        weighted_com = np.zeros(3)
        for i in self.arm_body_indices[side]:
            mass = self.model.body_mass[i]
            if mass <= 0:
                continue
            weighted_com += mass * self.data.xipos[i]
            total_mass += mass
        if total_mass > 0:
            weighted_com /= total_mass
        return weighted_com, total_mass

    def ee_frame_pose(self, side: str) -> tuple[np.ndarray, np.ndarray]:
        """Return TCP position and orientation (3x3 rotation matrix) from MuJoCo data."""
        body_name = f"openarm_{side}_hand_tcp"
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise ValueError(f"Body '{body_name}' not found in MuJoCo model")
        pos = self.data.xpos[body_id].copy()
        rot = np.asarray(self.data.xmat[body_id]).reshape(3, 3).copy()
        return pos, rot

    def ik(
        self,
        side: str,
        target_pos: np.ndarray,
        target_rot: np.ndarray,
        q_init: np.ndarray,
        max_iters: int = 200,
        tol: float = 1e-6,
        dt: float = 0.1,
        damping: float = 1e-3,
        rot_weight: float = 1.0,
        pos_weight: float = 1.0,
    ) -> np.ndarray | None:
        """MuJoCo-based DLS IK for the end-effector TCP frame.

        The position and orientation errors are computed in the world frame and
        mapped to joint updates via the body Jacobian (mj_jacBody) using the
        damped least-squares (DLS) formula: dq = (J'J + λI)^{-1} J' e.

        rot_weight scales the orientation error/Jacobian rows and pos_weight
        scales the position rows; setting one to 0.0 gives a position-only or
        orientation-only solve (used by the waist gripper-hold compensation to
        hold position and orientation independently). The best error is
        stored in self.last_ik_err[side].
        """
        q_arm = np.asarray(q_init, dtype=float).copy()
        body_name = f"openarm_{side}_hand_tcp"
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise ValueError(f"Body '{body_name}' not found in MuJoCo model")

        dof_indices = self.arm_dof[side]
        lower = self.arm_limits[side]["lower"]
        upper = self.arm_limits[side]["upper"]

        best_err = float("inf")
        best_q = q_arm.copy()

        saved_qpos = self.data.qpos.copy()
        try:
            for _ in range(max_iters):
                # Set current arm joint angles
                for i, name in enumerate(ARM_JOINTS[side]):
                    jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                    self.data.qpos[self.model.jnt_qposadr[jnt_id]] = q_arm[i]
                mujoco.mj_forward(self.model, self.data)

                # Current pose
                current_pos = self.data.xpos[body_id].copy()
                current_rot = np.asarray(self.data.xmat[body_id]).reshape(3, 3).copy()

                # Error in world frame
                err_pos = target_pos - current_pos
                err_rot = pin.log3(target_rot @ current_rot.T)
                err = np.concatenate([err_pos * pos_weight, err_rot * rot_weight])
                err_norm = np.linalg.norm(err)

                if err_norm < best_err:
                    best_err = err_norm
                    best_q = q_arm.copy()
                    self.last_ik_err[side] = best_err
                if err_norm < tol:
                    return q_arm

                # Jacobian of TCP body w.r.t. arm joints
                jacp = np.zeros((3, self.model.nv))
                jacr = np.zeros((3, self.model.nv))
                mujoco.mj_jacBody(self.model, self.data, jacp, jacr, body_id)
                J = np.vstack([jacp[:, dof_indices] * pos_weight, jacr[:, dof_indices] * rot_weight])

                # DLS update
                H = J.T @ J + damping * np.eye(7)
                dq = np.linalg.solve(H, J.T @ err)
                q_arm += dt * dq
                q_arm = np.clip(q_arm, lower, upper)

            if best_err < 1e-2:
                return best_q
            return None
        finally:
            self.data.qpos[:] = saved_qpos

    def _ik_dls_general(
        self,
        body_id: int,
        target_pos: np.ndarray,
        target_rot: np.ndarray,
        entries: list[tuple[int, int, float, float, float]],
        max_iters: int,
        tol: float,
        dt: float,
        damping: float,
        rot_weight: float,
    ) -> tuple[np.ndarray, float]:
        """DLS IK over an arbitrary dof set.

        entries: list of (dof_adr, qpos_adr, lower, upper, q0). The per-joint
        lower/upper limits are enforced by clipping every iteration, so
        callers can pass tightened limits (e.g. the toward-zero side of the
        current waist angle) to constrain the solution direction. Returns
        (best joint values, best 6D error norm). qpos is restored afterwards.
        """
        n = len(entries)
        dofs = np.array([e[0] for e in entries], dtype=np.intp)
        qadrs = np.array([e[1] for e in entries], dtype=np.intp)
        lower = np.array([e[2] for e in entries])
        upper = np.array([e[3] for e in entries])
        q = np.array([e[4] for e in entries], dtype=float)

        best_err = float("inf")
        best_q = q.copy()
        stall = 0
        saved_qpos = self.data.qpos.copy()
        try:
            for _ in range(max_iters):
                self.data.qpos[qadrs] = q
                mujoco.mj_forward(self.model, self.data)
                cur_pos = self.data.xpos[body_id].copy()
                cur_rot = np.asarray(self.data.xmat[body_id]).reshape(3, 3).copy()
                err_pos = target_pos - cur_pos
                err_rot = pin.log3(target_rot @ cur_rot.T)
                err = np.concatenate([err_pos, err_rot * rot_weight])
                err_norm = np.linalg.norm(err)
                if err_norm < best_err - 1e-7:
                    best_err = err_norm
                    best_q = q.copy()
                    stall = 0
                else:
                    stall += 1
                    if stall >= 80:
                        break  # saturated / stuck at this level -> escalate fast
                if err_norm < tol:
                    break
                jacp = np.zeros((3, self.model.nv))
                jacr = np.zeros((3, self.model.nv))
                mujoco.mj_jacBody(self.model, self.data, jacp, jacr, body_id)
                J = np.vstack([jacp[:, dofs], jacr[:, dofs] * rot_weight])
                H = J.T @ J + damping * np.eye(n)
                dq = np.linalg.solve(H, J.T @ err)
                q = np.clip(q + dt * dq, lower, upper)
        finally:
            self.data.qpos[:] = saved_qpos
        return best_q, best_err

    def _joint_entry(self, name: str, lo: float | None = None, hi: float | None = None):
        """Build a (dof_adr, qpos_adr, lower, upper, q0) entry for one joint."""
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            return None
        qadr = int(self.model.jnt_qposadr[jid])
        lo_m, hi_m = float(self.model.jnt_range[jid][0]), float(self.model.jnt_range[jid][1])
        if lo is not None:
            lo_m = lo
        if hi is not None:
            hi_m = hi
        return (int(self.model.jnt_dofadr[jid]), qadr, lo_m, hi_m, float(self.data.qpos[qadr]))

    def ik_whole_body(
        self,
        side: str,
        target_pos: np.ndarray,
        target_rot: np.ndarray,
        q_arm_init: np.ndarray,
        max_iters: int = 800,
        tol: float = 1e-6,
        dt: float = 0.15,
        damping: float = 1e-3,
        rot_weight: float = 1.0,
    ) -> dict:
        """One-shot WHOLE-BODY DLS IK for the selected arm's TCP target.

        Solves the 7 arm joints + waist pitch/yaw + lift_z (10 DOF, the
        full upper-body chain) simultaneously — no hierarchical escalation —
        so lift, waist and the arm coordinate to reach the target. The mobile
        base planar pose (x/y/yaw) is NOT moved. Falls back gracefully when
        waist/lift joints are absent (arm only).

        Returns dict: err, arm (np7), waist (np2 or None), lift (float or None).
        """
        body_name = f"openarm_{side}_hand_tcp"
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise ValueError(f"Body '{body_name}' not found in MuJoCo model")

        arm_jids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"openarm_{side}_joint{i}")
            for i in range(1, 8)
        ]
        arm_lo = self.arm_limits[side]["lower"]
        arm_hi = self.arm_limits[side]["upper"]
        arm_entries = [
            (
                int(self.model.jnt_dofadr[jid]),
                int(self.model.jnt_qposadr[jid]),
                float(arm_lo[i]),
                float(arm_hi[i]),
                float(q_arm_init[i]),
            )
            for i, jid in enumerate(arm_jids)
        ]

        waist_entries = [self._joint_entry(name) for name in WAIST_JOINTS.values()]
        lift_entry = self._joint_entry(BASE_JOINTS["lift"])

        entries = list(arm_entries)
        has_waist = all(e is not None for e in waist_entries)
        if has_waist:
            entries += [e for e in waist_entries]
        has_lift = lift_entry is not None
        if has_lift:
            entries.append(lift_entry)

        q_sol, err = self._ik_dls_general(
            body_id, target_pos, target_rot, entries,
            max_iters=max_iters, tol=tol, dt=dt, damping=damping, rot_weight=rot_weight,
        )
        out: dict = {"err": err, "arm": None, "waist": None, "lift": None}
        if q_sol is None:
            return out
        out["arm"] = q_sol[:7].copy()
        idx = 7
        if has_waist:
            out["waist"] = q_sol[idx:idx + 2].copy()
            idx += 2
        if has_lift:
            out["lift"] = float(q_sol[idx])
        self.last_ik_err[side] = err
        return out

    def ik_mobile(
        self,
        side: str,
        target_pos: np.ndarray,
        target_rot: np.ndarray,
        q_arm_init: np.ndarray,
        max_iters: int = 600,
        tol: float = 1e-6,
        dt: float = 0.15,
        damping: float = 1e-3,
        rot_weight: float = 1.0,
        level_tol: float = 5e-3,
        min_level: int = 1,
        waist_zero_tol: float = WAIST_ZERO_TOL,
    ) -> dict:
        """Hierarchical DLS IK for the mobile manipulator (waist-direction policy).

        The waist pitch/yaw joints are checked BEFORE every solve:

        - Waist at zero (within ``waist_zero_tol``): normal hierarchy --
          level 1 = 7 arm joints only; when the DLS residual stays above
          ``level_tol`` (near the arm workspace limit / joint saturation),
          escalate to level 2 (arm + waist), then level 3 (arm + waist +
          base x/y/yaw + lift).
        - Waist NOT at zero: (1) "腰回零" level -- each waist joint is
          clamped to its toward-zero side, so Cartesian steps achievable
          while homing the waist are done BY the waist moving toward zero,
          gradually, only as much as the step needs (no one-shot snap to
          zero); a boxed solution that makes NO homing progress is reported
          as the classic arm-only level 1 (waist untouched); (2) steps that
          would need the waist to move AWAY from zero leave it pinned at the
          clamp boundary -- the arm alone tries next; (3) only when the arm
          saturates, the free waist (away-from-zero motion allowed); (4) last
          resort: base x/y/yaw + lift.

        ``min_level`` forces the solver to start deeper in the chain (used
        for collision-triggered escalation; indexed by chain position).

        Returns dict: level (1=arm / 2=+waist / 3=+base/lift), err,
        tag ("腰回零"/"L1"/"L2+腰"/"L3+底盘"), max_level,
        waist_home (True when the waist was off zero at solve time),
        waist_mode (None/"boxed"/"free"), chain_pos/chain_len, arm (np7),
        waist (np2 or None), base (np4 or None: x, y, yaw, lift).
        """
        body_name = f"openarm_{side}_hand_tcp"
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise ValueError(f"Body '{body_name}' not found in MuJoCo model")

        # Level-1 entries: the 7 arm joints (initial values from the current target)
        arm_jids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"openarm_{side}_joint{i}")
            for i in range(1, 8)
        ]
        arm_lo = self.arm_limits[side]["lower"]
        arm_hi = self.arm_limits[side]["upper"]
        arm_entries = [
            (
                int(self.model.jnt_dofadr[jid]),
                int(self.model.jnt_qposadr[jid]),
                float(arm_lo[i]),
                float(arm_hi[i]),
                float(q_arm_init[i]),
            )
            for i, jid in enumerate(arm_jids)
        ]

        waist_entries = [self._joint_entry(name) for name in WAIST_JOINTS.values()]
        base_entries = [
            self._joint_entry(BASE_JOINTS["x"], lo=-10.0, hi=10.0),
            self._joint_entry(BASE_JOINTS["y"], lo=-10.0, hi=10.0),
            self._joint_entry(BASE_JOINTS["yaw"], lo=-np.pi, hi=np.pi),
            self._joint_entry(BASE_JOINTS["lift"]),
        ]

        # Waist-direction policy: check the CURRENT waist configuration (the
        # entries' initial values are read from the live qpos) and pick the
        # escalation chain accordingly. Each level is (entries, waist_mode)
        # with waist_mode in (None, "boxed", "free").
        waist_ok = all(e is not None for e in waist_entries)
        base_ok = all(e is not None for e in base_entries)
        waist_at_zero = waist_ok and all(abs(e[4]) <= waist_zero_tol for e in waist_entries)
        levels: list[tuple[list, str | None]] = []
        if waist_at_zero or not waist_ok:
            # Waist at zero (or no waist): normal hierarchy -- arm first, the
            # waist joins when the arm saturates, base + lift last.
            levels.append((arm_entries, None))
            if waist_ok:
                levels.append((arm_entries + waist_entries, "free"))
                if base_ok:
                    levels.append((arm_entries + waist_entries + base_entries, "free"))
        else:
            # Waist off zero. First level ("腰回零"): clamp each waist joint
            # to its toward-zero side, so Cartesian steps that can be
            # achieved while homing the waist are done BY the waist moving
            # toward zero -- gradually, only as much as the step needs (no
            # one-shot snap to zero). Steps that would need the waist to move
            # AWAY from zero leave it pinned at the clamp boundary, i.e. the
            # arm alone performs the motion. If the arm alone cannot reach
            # the target (residual above threshold / saturation), escalate to
            # the free waist (away-from-zero motion allowed), then to base +
            # lift.
            boxed_waist = []
            for e in waist_entries:
                dof, qadr, lo, hi, q0 = e
                if q0 > waist_zero_tol:
                    lo, hi = 0.0, q0    # positive waist: may only decrease
                elif q0 < -waist_zero_tol:
                    lo, hi = q0, 0.0    # negative waist: may only increase
                else:
                    lo = hi = q0        # within the zero band: freeze
                boxed_waist.append((dof, qadr, lo, hi, q0))
            levels.append((arm_entries + boxed_waist, "boxed"))
            levels.append((arm_entries, None))
            levels.append((arm_entries + waist_entries, "free"))
            if base_ok:
                levels.append((arm_entries + waist_entries + base_entries, "free"))
        def _level_of(entries: list) -> int:
            # Report by actuated dof groups: 1 = arm only, 2 = +waist,
            # 3 = +base x/y/yaw + lift (with or without the waist).
            n = len(entries)
            return 1 if n <= 7 else (2 if n <= 9 else 3)

        chain_len = len(levels)
        max_level = _level_of(levels[-1][0])  # deepest level the chain can reach
        min_level = max(1, min(int(min_level), chain_len))
        levels = levels[min_level - 1:]

        # best = (err, level, q_sol, waist_mode, chain_pos)
        best = (float("inf"), _level_of(levels[0][0]), None, None, min_level - 1)
        prev_best_q: np.ndarray | None = None
        for offset, (entries, waist_mode) in enumerate(levels):
            chain_pos = min_level - 1 + offset
            level_idx = _level_of(entries)
            if prev_best_q is not None:
                # warm start: the arm dofs are always a shared prefix
                for i in range(min(len(prev_best_q), len(entries))):
                    entries[i] = (*entries[i][:4], float(prev_best_q[i]))
            q_sol, err = self._ik_dls_general(
                body_id, target_pos, target_rot, entries,
                max_iters=max_iters, tol=tol, dt=dt, damping=damping, rot_weight=rot_weight,
            )
            if err < best[0]:
                best = (err, level_idx, q_sol, waist_mode, chain_pos)
            prev_best_q = q_sol
            if err < level_tol:
                break  # converged at this level; no need to escalate

        err, level, q_sol, waist_mode, chain_pos = best
        waist_out: np.ndarray | None = None
        base_out: np.ndarray | None = None
        if q_sol is not None:
            n = len(q_sol)
            if n == 9:         # arm + waist
                waist_out = q_sol[7:9].copy()
            elif n == 11:      # arm + base (waist kept home)
                base_out = q_sol[7:11].copy()
            elif n >= 13:      # arm + waist + base
                waist_out = q_sol[7:9].copy()
                base_out = q_sol[9:13].copy()
        # A boxed solution that made NO homing progress is equivalent to the
        # classic arm-only level: report it as L1 and leave the waist out.
        if waist_mode == "boxed" and waist_out is not None:
            progressed = any(
                abs(waist_out[k]) < abs(waist_entries[k][4]) - 1e-4
                for k in range(len(waist_entries))
            )
            if not progressed:
                level, waist_mode, waist_out = 1, None, None
        tag = (
            "腰回零" if waist_mode == "boxed"
            else "L1" if level == 1
            else "L2+腰" if level == 2
            else "L3+底盘"
        )
        out = {
            "level": level, "err": err, "tag": tag, "max_level": max_level,
            "waist_home": not waist_at_zero, "waist_mode": waist_mode,
            "chain_pos": chain_pos, "chain_len": chain_len,
            "arm": None, "waist": waist_out, "base": base_out,
        }
        if q_sol is None:
            return out
        out["arm"] = q_sol[:7].copy()
        self.last_ik_err[side] = err
        return out

    def _add_axis_line(
        self,
        scene,
        idx: int,
        origin: np.ndarray,
        direction: np.ndarray,
        length: float,
        width: float,
        rgba: np.ndarray,
    ) -> int:
        geom = scene.geoms[idx]
        end = origin + length * direction
        mujoco.mjv_initGeom(
            geom,
            type=mujoco.mjtGeom.mjGEOM_LINE,
            size=np.array([width, 0.0, 0.0]),
            pos=np.zeros(3),
            mat=np.eye(3).flatten(),
            rgba=rgba.astype(np.float32),
        )
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, width, origin, end)
        return idx + 1

    def _add_frame(
        self,
        scene,
        idx: int,
        pos: np.ndarray,
        rot: np.ndarray,
        axis_length: float,
        width: float,
    ) -> int:
        axes = [
            (rot[:, 0], np.array([1.0, 0.0, 0.0, 0.9])),  # x: red
            (rot[:, 1], np.array([0.0, 1.0, 0.0, 0.9])),  # y: green
            (rot[:, 2], np.array([0.0, 0.0, 1.0, 0.9])),  # z: blue
        ]
        for direction, rgba in axes:
            idx = self._add_axis_line(scene, idx, pos, direction, axis_length, width, rgba)
        return idx

    def draw_markers(
        self,
        viewer,
        left_com: np.ndarray,
        show_left_com: bool,
        right_com: np.ndarray,
        show_right_com: bool,
        left_pos: np.ndarray | None = None,
        left_rot: np.ndarray | None = None,
        right_pos: np.ndarray | None = None,
        right_rot: np.ndarray | None = None,
        show_ee_frames: bool = False,
        show_base_frame: bool = False,
        show_joint_frames: bool = False,
        extra_markers: list[tuple[np.ndarray, np.ndarray, float]] | None = None,
    ) -> None:
        """Draw CoM spheres, base frame, EE frames, joint frames, and extra markers."""
        if not hasattr(viewer, "user_scn"):
            return

        marker_count = int(show_left_com) + int(show_right_com)
        frame_count = 0
        if show_ee_frames:
            frame_count += 3 * int(left_pos is not None and left_rot is not None)
            frame_count += 3 * int(right_pos is not None and right_rot is not None)
        if show_base_frame:
            frame_count += 3
        joint_frame_count = 0
        if show_joint_frames:
            for side in ("right", "left"):
                joint_frame_count += 3 * len(self.arm_joint_ids[side])
        extra_count = len(extra_markers) if extra_markers else 0
        total_count = marker_count + frame_count + joint_frame_count + extra_count

        with viewer.lock():
            scene = viewer.user_scn
            scene.ngeom = total_count
            if total_count == 0 or total_count > scene.maxgeom:
                return

            identity = np.eye(3).flatten()
            idx = 0
            if show_left_com:
                geom = scene.geoms[idx]
                mujoco.mjv_initGeom(
                    geom,
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=np.array([self.MARKER_RADIUS, 0.0, 0.0]),
                    pos=left_com,
                    mat=identity,
                    rgba=self.LEFT_COM_RGBA.astype(np.float32),
                )
                geom.label = b"left_com"
                idx += 1

            if show_right_com:
                geom = scene.geoms[idx]
                mujoco.mjv_initGeom(
                    geom,
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=np.array([self.MARKER_RADIUS, 0.0, 0.0]),
                    pos=right_com,
                    mat=identity,
                    rgba=self.RIGHT_COM_RGBA.astype(np.float32),
                )
                geom.label = b"right_com"
                idx += 1

            if show_base_frame:
                idx = self._add_frame(scene, idx, np.zeros(3), np.eye(3), self.BASE_FRAME_AXIS_LENGTH, self.FRAME_AXIS_WIDTH)

            if show_ee_frames and left_pos is not None and left_rot is not None:
                idx = self._add_frame(scene, idx, left_pos, left_rot, self.FRAME_AXIS_LENGTH, self.FRAME_AXIS_WIDTH)
            if show_ee_frames and right_pos is not None and right_rot is not None:
                idx = self._add_frame(scene, idx, right_pos, right_rot, self.FRAME_AXIS_LENGTH, self.FRAME_AXIS_WIDTH)

            if show_joint_frames:
                for side in ("right", "left"):
                    for joint_id in self.arm_joint_ids[side]:
                        body_id = self.model.jnt_bodyid[joint_id]
                        pos = self.data.xpos[body_id].copy()
                        axis = self.data.xaxis[joint_id].copy()
                        rot = self._rotation_from_z(axis)
                        idx = self._add_frame(scene, idx, pos, rot, self.JOINT_FRAME_AXIS_LENGTH, self.JOINT_FRAME_AXIS_WIDTH)

            if extra_markers:
                for mpos, mrgba, msize in extra_markers:
                    geom = scene.geoms[idx]
                    mujoco.mjv_initGeom(
                        geom,
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=np.array([msize, 0.0, 0.0]),
                        pos=np.asarray(mpos, dtype=float),
                        mat=identity,
                        rgba=np.asarray(mrgba, dtype=np.float32),
                    )
                    geom.label = b"collision_marker"
                    idx += 1

            scene.ngeom = idx


def _beyond_deadzone(x: float, dz: float) -> float:
    """Zero inside the deadzone, otherwise shifted so response starts at zero."""
    return 0.0 if abs(x) < dz else (x - np.sign(x) * dz)


class CollisionGuard:
    """Proximity / collision safety guard built on MuJoCo's contact pipeline.

    MuJoCo computes contacts every mj_forward (broadphase sweep-and-prune +
    convex narrowphase with GJK/EPA), and collision geoms carry a 10 mm
    margin, so near-miss body pairs show up as contacts with
    ``0 <= dist < margin`` — zero extra simulation cost for monitoring.

    Features:
    - runtime-adjustable distance threshold (1..10 mm)
    - violation list: body pair names, distance, contact position
    - safety clamp: reverts monitored joints (arms + waist + lift) to the
      last safe configuration whenever a violation appears — this clamps
      *every* control path (GUI sliders, Cartesian IK, VR, and future VLA
      policy outputs)
    - would_violate(): hypothetical-configuration check used to pre-emptively
      block Cartesian IK solutions that would end in collision
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.model = model
        self.data = data
        self.enabled = True
        self.threshold = 0.005  # 5 mm
        self.last_safe_qpos: np.ndarray | None = None
        self.violations: list[dict] = []

        # joints restored to the last safe pose on violation
        names = [f"openarm_{s}_joint{i}" for s in ("left", "right") for i in range(1, 8)]
        names += [WAIST_JOINTS["pitch"], WAIST_JOINTS["yaw"], BASE_JOINTS["lift"]]
        self._restore_qadr: list[int] = []
        self._restore_act: list[int] = []
        for name in names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                continue
            self._restore_qadr.append(int(model.jnt_qposadr[jid]))
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name + "_ctrl")
            self._restore_act.append(aid)  # -1 if no actuator (waist/lift)

    def _pair_names(self, geom1: int, geom2: int) -> tuple[str, str]:
        b1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, self.model.geom_bodyid[geom1]) or "world"
        b2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, self.model.geom_bodyid[geom2]) or "world"
        return b1, b2

    @staticmethod
    def _skip_pair(b1: str, b2: str) -> bool:
        # wheels rest on the floor by design
        if b1 == "world" and b2.startswith("wheel_"):
            return True
        if b2 == "world" and b1.startswith("wheel_"):
            return True
        return False

    def check(self) -> list[dict]:
        """Collect monitored contacts with dist < threshold (call after mj_forward)."""
        self.violations = []
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            if con.dist >= self.threshold:
                continue
            b1, b2 = self._pair_names(con.geom1, con.geom2)
            if self._skip_pair(b1, b2):
                continue
            self.violations.append({
                "b1": b1,
                "b2": b2,
                "dist_mm": float(con.dist * 1000.0),
                "pos": np.asarray(con.pos).copy(),
            })
        return self.violations

    def save_safe(self) -> None:
        self.last_safe_qpos = self.data.qpos.copy()

    def restore_safe(self) -> None:
        """Revert monitored joints (qpos + actuator ctrl) to the last safe pose."""
        if self.last_safe_qpos is None:
            return
        for qadr, aid in zip(self._restore_qadr, self._restore_act):
            self.data.qpos[qadr] = self.last_safe_qpos[qadr]
            if aid >= 0:
                self.data.ctrl[aid] = self.last_safe_qpos[qadr]

    def would_violate(self, overrides: dict[str, float] | None = None) -> list[dict]:
        """Check collisions at a hypothetical configuration (joint name -> value).

        Sim state is fully restored afterwards. Used to pre-check Cartesian IK
        solutions, and safe to call from a future VLA policy loop.
        """
        saved = self.data.qpos.copy()
        try:
            if overrides:
                for name, val in overrides.items():
                    jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                    if jid >= 0:
                        self.data.qpos[self.model.jnt_qposadr[jid]] = val
            mujoco.mj_forward(self.model, self.data)
            return self.check()
        finally:
            self.data.qpos[:] = saved
            mujoco.mj_forward(self.model, self.data)


def qpos_to_qarm(mj_data, robot: RobotDynamics, side: str) -> np.ndarray:
    """Read current arm joint angles from MuJoCo qpos."""
    info = robot.arm_info[side]
    q_arm = np.zeros(7)
    for i, name in enumerate(info["joint_names"]):
        jnt_id = mujoco.mj_name2id(mj_data.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        q_arm[i] = mj_data.qpos[mj_data.model.jnt_qposadr[jnt_id]]
    return q_arm


def set_arm_qpos(mj_data, robot: RobotDynamics, side: str, q_arm: np.ndarray) -> None:
    """Write arm joint angles to both qpos and ctrl."""
    info = robot.arm_info[side]
    for name, val in zip(info["joint_names"], q_arm):
        jnt_id = mujoco.mj_name2id(mj_data.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        mj_data.qpos[mj_data.model.jnt_qposadr[jnt_id]] = val
        act_id = mujoco.mj_name2id(mj_data.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name + "_ctrl")
        if act_id >= 0:
            mj_data.ctrl[act_id] = val


def set_finger_qpos(mj_data, side: str, value: float) -> None:
    """Write gripper opening (0=closed, 0.044=open) to finger joints."""
    for name in FINGER_JOINTS[side]:
        jnt_id = mujoco.mj_name2id(mj_data.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jnt_id >= 0:
            mj_data.qpos[mj_data.model.jnt_qposadr[jnt_id]] = value
        act_id = mujoco.mj_name2id(mj_data.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name + "_ctrl")
        if act_id >= 0:
            mj_data.ctrl[act_id] = value


def sync_sliders_to_qpos(mj_data, robot: RobotDynamics, side: str) -> None:
    """Copy position-control actuator slider values to qpos."""
    info = robot.arm_info[side]
    for name in info["joint_names"]:
        act_id = mujoco.mj_name2id(mj_data.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name + "_ctrl")
        if act_id < 0:
            continue
        jnt_id = mujoco.mj_name2id(mj_data.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jnt_id < 0:
            continue
        mj_data.qpos[mj_data.model.jnt_qposadr[jnt_id]] = mj_data.ctrl[act_id]


class DualArmCartesianUI:
    """Tkinter window with independent Cartesian controls and dynamics display."""

    def __init__(
        self,
        on_right_translate: Callable[[str, float], None],
        on_right_rotate: Callable[[str, float], None],
        on_left_translate: Callable[[str, float], None],
        on_left_rotate: Callable[[str, float], None],
        on_gripper: Callable[[str, float], None],
        on_reset: Callable[[], None],
        on_step_change: Callable[[float], None],
        on_get_current_pose: Callable[[str], tuple[np.ndarray, np.ndarray]],
        on_solve_ik: Callable[[str, np.ndarray, np.ndarray], bool],
        on_get_whole_body: Callable[[], dict] | None = None,
        on_set_whole_body: Callable[[dict], bool] | None = None,
        on_whole_body_solve_ik: Callable[[str, np.ndarray, np.ndarray], bool] | None = None,
        on_vr_toggle: Callable[[bool], None] | None = None,
        on_vr_calibrate: Callable[[], None] | None = None,
        on_vr_port_change: Callable[[str], None] | None = None,
        on_hmd_base_toggle: Callable[[bool], None] | None = None,
        vr_net_info: str = "",
        on_base_dir: Callable[[float, float, float], None] | None = None,
        on_base_reset: Callable[[], None] | None = None,
        on_lift_change: Callable[[float], None] | None = None,
        on_base_home: Callable[[], None] | None = None,
        on_waist_change: Callable[[float, float], None] | None = None,
        waist_ranges: dict[str, tuple[float, float]] | None = None,
        on_collision_toggle: Callable[[bool], None] | None = None,
        on_collision_threshold: Callable[[float], None] | None = None,
    ) -> None:
        self.root = tk.Tk()
        self.root.title("OpenArm Dual-Arm Cartesian Control")
        self.root.geometry("1180x1100")

        self.step = tk.DoubleVar(value=0.02)
        on_step_change(self.step.get())

        # Step size + gripper sliders combined in a single row
        sliders_frame = tk.LabelFrame(self.root, text="Step size & Grippers (joint8 / finger)", padx=10, pady=10)
        sliders_frame.pack(padx=10, pady=(10, 0), fill="x")

        step_f = tk.Frame(sliders_frame)
        step_f.pack(side=tk.LEFT, padx=10)
        tk.Label(step_f, text="Step size (m / rad):").pack()
        tk.Scale(
            step_f,
            variable=self.step,
            from_=0.001,
            to_=0.1,
            resolution=0.001,
            orient=tk.HORIZONTAL,
            length=250,
            command=lambda v: on_step_change(float(v)),
        ).pack()

        for side in ("right", "left"):
            f = tk.Frame(sliders_frame)
            f.pack(side=tk.LEFT, padx=20)
            tk.Label(f, text=f"{side.capitalize()} gripper (0=closed, 0.044=open)").pack()
            scale = tk.Scale(
                f,
                from_=0.0,
                to_=0.044,
                resolution=0.001,
                orient=tk.HORIZONTAL,
                length=250,
                command=lambda v, s=side: on_gripper(s, float(v)),
            )
            scale.pack()
            setattr(self, f"gripper_{side}", scale)

        # Two columns for right and left arms (+ collision guard on the right)
        arms_frame = tk.Frame(self.root)
        arms_frame.pack(padx=10, pady=10, fill="both")

        self._build_arm_panel(arms_frame, "Right Arm", 0, on_right_translate, on_right_rotate)
        self._build_arm_panel(arms_frame, "Left Arm", 1, on_left_translate, on_left_rotate)

        # Collision guard panel sits to the right of the Cartesian motion buttons
        if on_collision_toggle is not None:
            self._build_collision_panel(arms_frame, on_collision_toggle, on_collision_threshold)

        # Mobile base panel (only when the MJCF contains base joints).
        # The base panel and the IK target panel share a horizontal container
        # (base on the left, IK target on the right) to save vertical space.
        self._base_ik_container: tk.Frame | None = None
        if on_base_dir is not None:
            self._base_ik_container = tk.Frame(self.root)
            self._base_ik_container.pack(padx=10, pady=10, fill="x")
            self._build_base_panel(
                self._base_ik_container,
                on_base_dir, on_base_reset, on_lift_change, on_base_home, on_waist_change, waist_ranges,
            )

        # VR Control Panel
        self.vr_enabled = tk.BooleanVar(value=False)
        self.vr_connected = False
        self._on_vr_toggle_cb = on_vr_toggle
        self._on_vr_calibrate_cb = on_vr_calibrate
        self._on_hmd_base_toggle_cb = on_hmd_base_toggle

        vr_frame = tk.LabelFrame(self.root, text="VR Control (Meta Quest 3)", padx=10, pady=10)
        vr_frame.pack(padx=10, pady=10, fill="x")

        # Network info row: local IPs + editable listen port
        if on_vr_port_change is not None or vr_net_info:
            net_frame = tk.Frame(vr_frame)
            net_frame.pack(fill="x", pady=2)
            tk.Label(net_frame, text=f"本机IP: {vr_net_info}", fg="blue").pack(side=tk.LEFT, padx=5)
            if on_vr_port_change is not None:
                tk.Label(net_frame, text="监听端口:").pack(side=tk.LEFT, padx=5)
                self.vr_port_entry = tk.Entry(net_frame, width=7)
                self.vr_port_entry.insert(0, "5006")
                self.vr_port_entry.pack(side=tk.LEFT)
                tk.Button(
                    net_frame, text="应用端口",
                    command=lambda: on_vr_port_change(self.vr_port_entry.get()),
                ).pack(side=tk.LEFT, padx=5)
            tk.Label(
                net_frame, text="(Quest APK 中应填写以上IP和端口)", fg="gray", font=("Arial", 8)
            ).pack(side=tk.LEFT, padx=8)

        vr_btn_frame = tk.Frame(vr_frame)
        vr_btn_frame.pack(fill="x", pady=5)

        self.vr_status_var = tk.StringVar(value="VR: Disabled")
        tk.Label(vr_btn_frame, textvariable=self.vr_status_var, font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)

        self.vr_toggle_btn = tk.Checkbutton(
            vr_btn_frame,
            text="Enable VR Control",
            variable=self.vr_enabled,
            command=self._on_vr_toggle,
            font=("Arial", 10),
        )
        self.vr_toggle_btn.pack(side=tk.LEFT, padx=10)

        tk.Button(
            vr_btn_frame,
            text="Re-anchor VR Origin",
            command=self._on_vr_calibrate,
            bg="lightblue",
        ).pack(side=tk.LEFT, padx=10)

        tk.Button(
            vr_btn_frame,
            text="Reset VR Smoothing",
            command=self._on_vr_reset_smoothing,
            bg="lightyellow",
        ).pack(side=tk.LEFT, padx=10)

        # HMD drives the mobile base + lift (only for mobile scenes)
        if on_hmd_base_toggle is not None:
            self.vr_hmd_base = tk.BooleanVar(value=False)
            tk.Checkbutton(
                vr_btn_frame,
                text="HMD 头显控制底盘+升降",
                variable=self.vr_hmd_base,
                command=self._on_hmd_base_toggle,
                font=("Arial", 10),
                bg="lightgreen",
            ).pack(side=tk.LEFT, padx=10)
            tk.Label(
                vr_btn_frame, text="B键=VR归0, 按住B=暂停底盘/升降", fg="gray", font=("Arial", 9)
            ).pack(side=tk.LEFT, padx=5)
            # HMD control mode: relative (velocity) or absolute (1:1 pose)
            hmd_mode_frame = tk.Frame(vr_frame)
            hmd_mode_frame.pack(fill="x", pady=2)
            tk.Label(hmd_mode_frame, text="HMD底盘模式:").pack(side=tk.LEFT, padx=5)
            self.vr_hmd_mode = tk.StringVar(value="relative")
            tk.Radiobutton(
                hmd_mode_frame, text="相对 (头部位移→底盘速度)", variable=self.vr_hmd_mode, value="relative"
            ).pack(side=tk.LEFT, padx=5)
            tk.Radiobutton(
                hmd_mode_frame, text="绝对 (头部位姿→底盘位姿1:1)", variable=self.vr_hmd_mode, value="absolute"
            ).pack(side=tk.LEFT, padx=5)
            # re-anchor on mode switch so the base does not jump
            self.vr_hmd_mode.trace_add("write", lambda *_: self._on_hmd_base_toggle())

        # VR motion scale slider
        scale_frame = tk.Frame(vr_frame)
        scale_frame.pack(fill="x", pady=2)
        tk.Label(scale_frame, text="Motion scale (robot = VR × scale):").pack(side=tk.LEFT, padx=5)
        self.vr_scale = tk.DoubleVar(value=1.0)
        tk.Scale(
            scale_frame,
            variable=self.vr_scale,
            from_=0.1,
            to=3.0,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            length=250,
        ).pack(side=tk.LEFT, padx=5)
        self.vr_scale_label = tk.Label(scale_frame, text="1.0x", width=5)
        self.vr_scale_label.pack(side=tk.LEFT)
        self.vr_scale.trace_add("write", lambda *_: self.vr_scale_label.config(text=f"{self.vr_scale.get():.1f}x"))

        # VR control mode selector (single switch for arm pos+rot)
        mode_frame = tk.Frame(vr_frame)
        mode_frame.pack(fill="x", pady=2)
        tk.Label(mode_frame, text="手臂控制模式:").pack(side=tk.LEFT, padx=5)
        self.vr_mode = tk.StringVar(value="relative")
        tk.Radiobutton(
            mode_frame, text="相对 (Relative, 锚点增量)", variable=self.vr_mode, value="relative"
        ).pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(
            mode_frame, text="绝对 (Absolute, 1:1 首次握持标定)", variable=self.vr_mode, value="absolute"
        ).pack(side=tk.LEFT, padx=5)

        # VR status display
        self.vr_right_status = tk.StringVar(value="R: ---")
        self.vr_left_status = tk.StringVar(value="L: ---")
        self.vr_hmd_status = tk.StringVar(value="HMD: ---")
        tk.Label(vr_frame, textvariable=self.vr_right_status, font=("Consolas", 9), anchor=tk.W).pack(fill="x", pady=1)
        tk.Label(vr_frame, textvariable=self.vr_left_status, font=("Consolas", 9), anchor=tk.W).pack(fill="x", pady=1)
        tk.Label(vr_frame, textvariable=self.vr_hmd_status, font=("Consolas", 9), anchor=tk.W, fg="blue").pack(fill="x", pady=1)

        # IK target input
        self._get_current_pose_cb = on_get_current_pose
        self._solve_ik_cb = on_solve_ik
        self._get_whole_body_cb = on_get_whole_body
        self._set_whole_body_cb = on_set_whole_body
        self._whole_body_solve_cb = on_whole_body_solve_ik
        self._build_ik_frame()

        # Reset
        tk.Button(self.root, text="Reset Both Arms", command=on_reset, bg="orange", font=("Arial", 12)).pack(pady=10)

        # Forward-kinematics display (terminal line reproduced in UI)
        fk_frame = tk.LabelFrame(self.root, text="End-effector pose (FK)", padx=10, pady=5)
        fk_frame.pack(side=tk.BOTTOM, fill="x", padx=10, pady=(0, 5))
        self.fk_right_var = tk.StringVar(
            value="R: pos=(+0.000,-0.000,+0.000) q=(+1.000,+0.000,+0.000,+0.000)"
        )
        self.fk_left_var = tk.StringVar(
            value="L: pos=(+0.000,+0.000,+0.000) q=(+1.000,+0.000,+0.000,+0.000)"
        )
        tk.Label(fk_frame, textvariable=self.fk_right_var, font=("Consolas", 9), anchor=tk.W).pack(
            fill="x", pady=1
        )
        tk.Label(fk_frame, textvariable=self.fk_left_var, font=("Consolas", 9), anchor=tk.W).pack(
            fill="x", pady=1
        )

        # Status
        self.status = tk.StringVar(value="Ready")
        tk.Label(self.root, textvariable=self.status, relief=tk.SUNKEN, anchor=tk.W).pack(side=tk.BOTTOM, fill="x")

        # Dynamics display (above status + FK output)
        dyn_frame = tk.LabelFrame(
            self.root, text="Dynamics (MuJoCo inverse dynamics: gravity torques + per-arm CoM)", padx=10, pady=5
        )
        dyn_frame.pack(side=tk.BOTTOM, fill="x", padx=10, pady=(0, 5))

        cb_frame = tk.Frame(dyn_frame)
        cb_frame.pack(fill="x", pady=2)
        self.show_left_com = tk.BooleanVar(value=False)
        self.show_right_com = tk.BooleanVar(value=False)
        self.show_ee_frames = tk.BooleanVar(value=False)
        self.show_base_frame = tk.BooleanVar(value=False)
        self.show_joint_frames = tk.BooleanVar(value=False)
        tk.Checkbutton(
            cb_frame, text="Show left CoM", variable=self.show_left_com
        ).pack(side=tk.LEFT, padx=(0, 10))
        tk.Checkbutton(
            cb_frame, text="Show right CoM", variable=self.show_right_com
        ).pack(side=tk.LEFT, padx=(0, 10))
        tk.Checkbutton(
            cb_frame, text="Show EE frames (xyz)", variable=self.show_ee_frames
        ).pack(side=tk.LEFT, padx=(0, 10))
        tk.Checkbutton(
            cb_frame, text="Show base frame", variable=self.show_base_frame
        ).pack(side=tk.LEFT, padx=(0, 10))
        tk.Checkbutton(
            cb_frame, text="Show joint frames", variable=self.show_joint_frames
        ).pack(side=tk.LEFT)

        self.right_torque_var = tk.StringVar(value="Right tau: [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00] Nm")
        tk.Label(dyn_frame, textvariable=self.right_torque_var, font=("Consolas", 10), anchor=tk.W).pack(
            fill="x", pady=1
        )

        self.left_torque_var = tk.StringVar(value="Left  tau: [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00] Nm")
        tk.Label(dyn_frame, textvariable=self.left_torque_var, font=("Consolas", 10), anchor=tk.W).pack(
            fill="x", pady=1
        )

        self.left_com_var = tk.StringVar(value="Left  CoM: [0.000, 0.000, 0.000] m  mass 0.000 kg")
        tk.Label(dyn_frame, textvariable=self.left_com_var, font=("Consolas", 10), anchor=tk.W).pack(fill="x", pady=1)

        self.right_com_var = tk.StringVar(value="Right CoM: [0.000, 0.000, 0.000] m  mass 0.000 kg")
        tk.Label(dyn_frame, textvariable=self.right_com_var, font=("Consolas", 10), anchor=tk.W).pack(fill="x", pady=1)

    def _build_arm_panel(
        self,
        parent: tk.Widget,
        title: str,
        column: int,
        on_translate: Callable[[str, float], None],
        on_rotate: Callable[[str, float], None],
    ) -> None:
        frame = tk.LabelFrame(parent, text=title, padx=10, pady=10)
        frame.grid(row=0, column=column, padx=10, pady=10, sticky="n")

        trans_frame = tk.LabelFrame(frame, text="Translation (m)", padx=5, pady=5)
        trans_frame.pack(fill="x", pady=5)
        for axis in ("X", "Y", "Z"):
            f = tk.Frame(trans_frame)
            f.pack(fill="x", pady=2)
            tk.Label(f, text=f"{axis}:", width=4).pack(side=tk.LEFT)
            tk.Button(f, text="-", width=3, command=lambda a=axis: on_translate(a, -1)).pack(side=tk.LEFT, padx=2)
            tk.Button(f, text="+", width=3, command=lambda a=axis: on_translate(a, +1)).pack(side=tk.LEFT, padx=2)

        rot_frame = tk.LabelFrame(frame, text="Rotation (rad)", padx=5, pady=5)
        rot_frame.pack(fill="x", pady=5)
        for axis in ("Roll", "Pitch", "Yaw"):
            f = tk.Frame(rot_frame)
            f.pack(fill="x", pady=2)
            tk.Label(f, text=f"{axis}:", width=6).pack(side=tk.LEFT)
            tk.Button(f, text="-", width=3, command=lambda a=axis: on_rotate(a, -1)).pack(side=tk.LEFT, padx=2)
            tk.Button(f, text="+", width=3, command=lambda a=axis: on_rotate(a, +1)).pack(side=tk.LEFT, padx=2)

    def _build_base_panel(
        self,
        parent: tk.Widget,
        on_base_dir: Callable[[float, float, float], None],
        on_base_reset: Callable[[], None] | None,
        on_lift_change: Callable[[float], None] | None,
        on_base_home: Callable[[], None] | None = None,
        on_waist_change: Callable[[float, float], None] | None = None,
        waist_ranges: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        """Mecanum mobile base + lift control panel (GUI only)."""
        base_frame = tk.LabelFrame(parent, text="Mobile Base 移动底盘 (全向轮/麦轮 + 丝杆升降)", padx=10, pady=10)
        base_frame.pack(side=tk.LEFT, padx=(0, 5), fill="both", expand=True)

        # Normalized direction state: (dx, dy, d_omega); 0 = stopped
        self._base_dir = (0.0, 0.0, 0.0)

        def apply_base() -> None:
            dx, dy, do = self._base_dir
            norm = np.hypot(dx, dy)
            if norm > 1.0:
                dx, dy = dx / norm, dy / norm
            on_base_dir(
                dx * self.base_lin_speed.get(),
                dy * self.base_lin_speed.get(),
                do * self.base_ang_speed.get(),
            )

        def set_dir(dx: float, dy: float, do: float) -> None:
            self._base_dir = (dx, dy, do)
            apply_base()

        # Speed sliders
        spd = tk.Frame(base_frame)
        spd.pack(fill="x", pady=2)
        tk.Label(spd, text="线速度 (m/s):").pack(side=tk.LEFT, padx=5)
        self.base_lin_speed = tk.DoubleVar(value=0.2)
        tk.Scale(
            spd, variable=self.base_lin_speed, from_=0.05, to=0.6, resolution=0.05,
            orient=tk.HORIZONTAL, length=200, command=lambda *_: apply_base(),
        ).pack(side=tk.LEFT, padx=5)
        tk.Label(spd, text="角速度 (rad/s):").pack(side=tk.LEFT, padx=5)
        self.base_ang_speed = tk.DoubleVar(value=0.8)
        tk.Scale(
            spd, variable=self.base_ang_speed, from_=0.2, to=1.5, resolution=0.1,
            orient=tk.HORIZONTAL, length=160, command=lambda *_: apply_base(),
        ).pack(side=tk.LEFT, padx=5)

        # Direction pad: vx forward (+x), vy left (+y)
        pad = tk.Frame(base_frame)
        pad.pack(pady=5)
        pad_grid = [
            ("↖", +1, +1), ("↑", +1, 0), ("↗", +1, -1),
            ("←", 0, +1), ("■", 0, 0), ("→", 0, -1),
            ("↙", -1, +1), ("↓", -1, 0), ("↘", -1, -1),
        ]
        for idx, (label, dx, dy) in enumerate(pad_grid):
            row, col = divmod(idx, 3)
            tk.Button(
                pad, text=label, width=4, font=("Arial", 12),
                command=lambda a=dx, b=dy: set_dir(a, b, 0),
            ).grid(row=row, column=col, padx=2, pady=2)

        # Rotation + reset + lift
        ctrl = tk.Frame(base_frame)
        ctrl.pack(pady=2)
        tk.Button(ctrl, text="⟲ 左旋", width=7, command=lambda: set_dir(0, 0, +1)).pack(side=tk.LEFT, padx=4)
        tk.Button(
            ctrl, text="停止", width=7, bg="red", fg="white", command=lambda: set_dir(0, 0, 0)
        ).pack(side=tk.LEFT, padx=4)
        tk.Button(ctrl, text="⟳ 右旋", width=7, command=lambda: set_dir(0, 0, -1)).pack(side=tk.LEFT, padx=4)
        if on_base_reset is not None:
            tk.Button(ctrl, text="底盘回零", width=8, command=on_base_reset, bg="lightyellow").pack(side=tk.LEFT, padx=4)
        if on_base_home is not None:
            tk.Button(ctrl, text="一键回位", width=8, command=on_base_home, bg="lightgreen").pack(side=tk.LEFT, padx=4)

        if on_lift_change is not None:
            lift_f = tk.Frame(base_frame)
            lift_f.pack(fill="x", pady=2)
            tk.Label(lift_f, text="升降高度 (mm, 丝杆 0~800):").pack(side=tk.LEFT, padx=5)
            self.lift_slider = tk.Scale(
                lift_f, from_=0, to=800, resolution=5, orient=tk.HORIZONTAL, length=380,
                command=lambda v: on_lift_change(float(v)),
            )
            self.lift_slider.pack(side=tk.LEFT, padx=5)

        # Waist joints (mobile3): pitch (lean fwd/back) + yaw (horizontal rotation)
        if on_waist_change is not None and waist_ranges is not None:
            waist_f = tk.Frame(base_frame)
            waist_f.pack(fill="x", pady=2)
            p_lo, p_hi = waist_ranges["pitch"]
            y_lo, y_hi = waist_ranges["yaw"]
            tk.Label(waist_f, text="腰部俯仰 (°):").pack(side=tk.LEFT, padx=5)
            self.waist_pitch_slider = tk.Scale(
                waist_f, from_=p_lo, to=p_hi, resolution=1, orient=tk.HORIZONTAL, length=170,
                command=lambda v: on_waist_change(float(v), self.waist_yaw_slider.get()),
            )
            self.waist_pitch_slider.pack(side=tk.LEFT, padx=5)
            tk.Label(waist_f, text="腰部旋转 (°):").pack(side=tk.LEFT, padx=5)
            self.waist_yaw_slider = tk.Scale(
                waist_f, from_=y_lo, to=y_hi, resolution=1, orient=tk.HORIZONTAL, length=170,
                command=lambda v: on_waist_change(self.waist_pitch_slider.get(), float(v)),
            )
            self.waist_yaw_slider.pack(side=tk.LEFT, padx=5)

        # Gripper hold switches (all mobile versions): while the waist joints
        # (sliders / one-click homing), the chassis (direction pad / HMD / IK
        # servo) or the lift move, the arms compensate (arm-only IK) so the
        # grippers keep their Cartesian position and/or orientation — two
        # independent switches. Base/lift motion that would break the hold is
        # restricted (reverted).
        hold_f = tk.Frame(base_frame)
        hold_f.pack(fill="x", pady=2)
        self.waist_hold_pos = tk.BooleanVar(value=False)
        tk.Checkbutton(
            hold_f, text="腰/底盘/升降动时保持夹爪位置", variable=self.waist_hold_pos,
        ).pack(side=tk.LEFT, padx=5)
        self.waist_hold_rot = tk.BooleanVar(value=False)
        tk.Checkbutton(
            hold_f, text="腰/底盘/升降动时保持夹爪姿态", variable=self.waist_hold_rot,
        ).pack(side=tk.LEFT, padx=5)
        tk.Label(
            hold_f, text="(不能锁定时底盘/升降自动限位)", fg="gray", font=("Arial", 8),
        ).pack(side=tk.LEFT, padx=5)

        self.base_cmd_var = tk.StringVar(value="底盘指令: vx=+0.00 m/s vy=+0.00 m/s ω=+0.00 rad/s")
        tk.Label(base_frame, textvariable=self.base_cmd_var, font=("Consolas", 9), anchor=tk.W).pack(fill="x", pady=1)
        self.base_pose_var = tk.StringVar(value="底盘位姿: x=+0.00 m y=+0.00 m yaw=+0.0° 升降=0 mm")
        tk.Label(base_frame, textvariable=self.base_pose_var, font=("Consolas", 9), anchor=tk.W).pack(fill="x", pady=1)

    def set_base_status(
        self,
        x: float,
        y: float,
        yaw_deg: float,
        lift_mm: float,
        vx: float,
        vy: float,
        omega: float,
        waist_pitch_deg: float | None = None,
        waist_yaw_deg: float | None = None,
    ) -> None:
        """Update mobile base pose/command readouts (no-op if panel absent)."""
        if not hasattr(self, "base_pose_var"):
            return
        pose = f"底盘位姿: x={x:+.2f} m y={y:+.2f} m yaw={yaw_deg:+.1f}° 升降={lift_mm:.0f} mm"
        if waist_pitch_deg is not None and waist_yaw_deg is not None:
            pose += f" 腰俯仰={waist_pitch_deg:+.0f}° 腰旋转={waist_yaw_deg:+.0f}°"
        self.base_pose_var.set(pose)
        self.base_cmd_var.set(f"底盘指令: vx={vx:+.2f} m/s vy={vy:+.2f} m/s ω={omega:+.2f} rad/s")

    def _build_collision_panel(
        self,
        parent: tk.Widget,
        on_collision_toggle: Callable[[bool], None],
        on_collision_threshold: Callable[[float], None] | None,
    ) -> None:
        """Collision guard panel (right column of the Cartesian motion panels).

        Enable switch, marker display, threshold slider; stacked vertically to
        keep the column narrow.
        """
        cg = tk.LabelFrame(parent, text="碰撞安全保护 Collision Guard", padx=10, pady=10)
        cg.grid(row=0, column=2, padx=10, pady=10, sticky="n")

        self.collision_enable = tk.BooleanVar(value=True)
        tk.Checkbutton(
            cg, text="启用碰撞安全限制", variable=self.collision_enable,
            command=lambda: on_collision_toggle(self.collision_enable.get()),
        ).pack(anchor="w", padx=5, pady=2)
        self.collision_show = tk.BooleanVar(value=True)
        tk.Checkbutton(cg, text="显示碰撞标记", variable=self.collision_show).pack(anchor="w", padx=5, pady=2)
        tk.Label(cg, text="距离阈值 (mm):").pack(anchor="w", padx=5)
        self.collision_thr = tk.DoubleVar(value=5.0)
        tk.Scale(
            cg, variable=self.collision_thr, from_=1, to=10, resolution=1,
            orient=tk.HORIZONTAL, length=150,
            command=lambda v: on_collision_threshold(float(v)) if on_collision_threshold else None,
        ).pack(padx=5)

        self.collision_status_var = tk.StringVar(value="碰撞保护: 正常\n(阈值 5 mm)")
        self.collision_status_label = tk.Label(
            cg, textvariable=self.collision_status_var, font=("Consolas", 9), anchor=tk.W,
            fg="green", justify=tk.LEFT, wraplength=200,
        )
        self.collision_status_label.pack(fill="x", padx=5, pady=(5, 0))

    def set_collision_status(self, violations: list | None, threshold_mm: float, enabled: bool) -> None:
        """Update collision guard status line (no-op if panel absent)."""
        if not hasattr(self, "collision_status_var"):
            return
        if not enabled:
            self.collision_status_var.set("碰撞保护: 已关闭")
            self.collision_status_label.config(fg="gray")
        elif violations:
            v = violations[0]
            self.collision_status_var.set(
                f"!! 碰撞保护触发 ({len(violations)} 处): {v['b1']} <-> {v['b2']} @ {v['dist_mm']:.1f} mm，已回退到安全位形"
            )
            self.collision_status_label.config(fg="red")
        else:
            self.collision_status_var.set(f"碰撞保护: 正常 (阈值 {threshold_mm:.0f} mm)")
            self.collision_status_label.config(fg="green")

    def _build_ik_frame(self) -> None:
        """Build the IK target input frame with position/quaternion entries and action buttons."""
        # When a mobile base panel exists, sit to its right inside the shared
        # container; otherwise occupy a full-width row of the main window.
        container = self._base_ik_container
        ik_frame = tk.LabelFrame(
            container if container is not None else self.root,
            text="IK Target (position + quaternion)", padx=10, pady=10,
        )
        if container is not None:
            ik_frame.pack(side=tk.LEFT, padx=(5, 0), fill="both", expand=True)
        else:
            ik_frame.pack(padx=10, pady=10, fill="x")

        # Arm selection
        tk.Label(ik_frame, text="Arm:").grid(row=0, column=0, sticky="w", padx=2)
        self.ik_side = tk.StringVar(value="right")
        tk.Radiobutton(ik_frame, text="Right", variable=self.ik_side, value="right").grid(row=0, column=1, sticky="w", padx=2)
        tk.Radiobutton(ik_frame, text="Left", variable=self.ik_side, value="left").grid(row=0, column=2, sticky="w", padx=2)

        # Position inputs
        tk.Label(ik_frame, text="X").grid(row=1, column=0, padx=2)
        self.ik_x = tk.Entry(ik_frame, width=10)
        self.ik_x.grid(row=1, column=1, padx=2)
        self.ik_x.insert(0, "0.0000")

        tk.Label(ik_frame, text="Y").grid(row=1, column=2, padx=2)
        self.ik_y = tk.Entry(ik_frame, width=10)
        self.ik_y.grid(row=1, column=3, padx=2)
        self.ik_y.insert(0, "0.0000")

        tk.Label(ik_frame, text="Z").grid(row=1, column=4, padx=2)
        self.ik_z = tk.Entry(ik_frame, width=10)
        self.ik_z.grid(row=1, column=5, padx=2)
        self.ik_z.insert(0, "0.0000")

        # Quaternion inputs (x, y, z, w order to match FK display)
        tk.Label(ik_frame, text="qx").grid(row=2, column=0, padx=2)
        self.ik_qx = tk.Entry(ik_frame, width=10)
        self.ik_qx.grid(row=2, column=1, padx=2)
        self.ik_qx.insert(0, "0.0000")

        tk.Label(ik_frame, text="qy").grid(row=2, column=2, padx=2)
        self.ik_qy = tk.Entry(ik_frame, width=10)
        self.ik_qy.grid(row=2, column=3, padx=2)
        self.ik_qy.insert(0, "0.0000")

        tk.Label(ik_frame, text="qz").grid(row=2, column=4, padx=2)
        self.ik_qz = tk.Entry(ik_frame, width=10)
        self.ik_qz.grid(row=2, column=5, padx=2)
        self.ik_qz.insert(0, "0.0000")

        tk.Label(ik_frame, text="qw").grid(row=2, column=6, padx=2)
        self.ik_qw = tk.Entry(ik_frame, width=10)
        self.ik_qw.grid(row=2, column=7, padx=2)
        self.ik_qw.insert(0, "1.0000")

        # Buttons
        btn_frame = tk.Frame(ik_frame)
        btn_frame.grid(row=3, column=0, columnspan=8, pady=(10, 0))
        tk.Button(btn_frame, text="Get current pose", command=self._on_get_current_pose).pack(side=tk.LEFT, padx=10)
        tk.Button(
            btn_frame, text="Solve IK", command=self._on_solve_ik, bg="green", fg="white"
        ).pack(side=tk.LEFT, padx=10)

        # ---- Second part: whole-body config (lift + waist + both arms) ----
        wb_frame = tk.LabelFrame(
            ik_frame, text="全身 (升降+腰部2关节+双臂) 配置 & 全身 IK", padx=5, pady=5,
        )
        wb_frame.grid(row=4, column=0, columnspan=8, sticky="ew", pady=(10, 0))

        r0 = tk.Frame(wb_frame)
        r0.pack(fill="x", pady=2)
        tk.Label(r0, text="升降(mm):").pack(side=tk.LEFT)
        self.wb_lift = tk.Entry(r0, width=7)
        self.wb_lift.pack(side=tk.LEFT, padx=3)
        self.wb_lift.insert(0, "400")
        tk.Label(r0, text="腰俯仰(°):").pack(side=tk.LEFT, padx=(8, 0))
        self.wb_pitch = tk.Entry(r0, width=7)
        self.wb_pitch.pack(side=tk.LEFT, padx=3)
        self.wb_pitch.insert(0, "0")
        tk.Label(r0, text="腰旋转(°):").pack(side=tk.LEFT, padx=(8, 0))
        self.wb_yaw = tk.Entry(r0, width=7)
        self.wb_yaw.pack(side=tk.LEFT, padx=3)
        self.wb_yaw.insert(0, "0")

        self.wb_arms: dict[str, list[tk.Entry]] = {}
        for side in ("right", "left"):
            rf = tk.Frame(wb_frame)
            rf.pack(fill="x", pady=2)
            tk.Label(rf, text=f"{side.capitalize()} 臂 J1..J7(°):", width=18, anchor="w").pack(side=tk.LEFT)
            entries: list[tk.Entry] = []
            for _ in range(7):
                e = tk.Entry(rf, width=5)
                e.pack(side=tk.LEFT, padx=1)
                e.insert(0, "0.0")
                entries.append(e)
            self.wb_arms[side] = entries

        wb_btn = tk.Frame(wb_frame)
        wb_btn.pack(fill="x", pady=2)
        tk.Button(
            wb_btn, text="获取全身配置", command=self._on_get_whole_body,
        ).pack(side=tk.LEFT, padx=4)
        tk.Button(
            wb_btn, text="设置全身配置", command=self._on_set_whole_body,
        ).pack(side=tk.LEFT, padx=4)
        tk.Button(
            wb_btn, text="全身 Solve IK (升降+腰+臂同时)", command=self._on_whole_body_solve_ik,
            bg="green", fg="white",
        ).pack(side=tk.LEFT, padx=4)

    def _on_get_current_pose(self) -> None:
        side = self.ik_side.get()
        pos, quat = self._get_current_pose_cb(side)
        entries = [self.ik_x, self.ik_y, self.ik_z, self.ik_qx, self.ik_qy, self.ik_qz, self.ik_qw]
        values = [pos[0], pos[1], pos[2], quat[0], quat[1], quat[2], quat[3]]
        for entry, value in zip(entries, values):
            entry.delete(0, tk.END)
            entry.insert(0, f"{value:.4f}")
        self.set_status(f"Got current {side} pose")

    def _on_solve_ik(self) -> None:
        side = self.ik_side.get()
        entries = [self.ik_x, self.ik_y, self.ik_z, self.ik_qx, self.ik_qy, self.ik_qz, self.ik_qw]
        try:
            values = [float(e.get()) for e in entries]
        except ValueError:
            self.set_status("Invalid IK target input")
            return
        pos = np.array(values[:3])
        quat = np.array(values[3:])
        success = self._solve_ik_cb(side, pos, quat)
        if not success:
            self.set_status(f"{side.capitalize()} IK solve failed")

    def _on_get_whole_body(self) -> None:
        if self._get_whole_body_cb is None:
            self.set_status("全身配置获取不可用")
            return
        cfg = self._get_whole_body_cb()
        self.wb_lift.delete(0, tk.END)
        self.wb_lift.insert(0, f"{cfg['lift'] * 1000.0:.1f}")
        self.wb_pitch.delete(0, tk.END)
        self.wb_pitch.insert(0, f"{np.degrees(cfg['waist_pitch']):.2f}")
        self.wb_yaw.delete(0, tk.END)
        self.wb_yaw.insert(0, f"{np.degrees(cfg['waist_yaw']):.2f}")
        for side in ("right", "left"):
            for e, v in zip(self.wb_arms[side], cfg[side]):
                e.delete(0, tk.END)
                e.insert(0, f"{np.degrees(v):.2f}")
        self.set_status("已获取全身配置 (升降+腰部+双臂)")

    def _on_set_whole_body(self) -> None:
        if self._set_whole_body_cb is None:
            self.set_status("设置全身配置不可用")
            return
        try:
            cfg = {
                "lift": float(self.wb_lift.get()) / 1000.0,
                "waist_pitch": np.radians(float(self.wb_pitch.get())),
                "waist_yaw": np.radians(float(self.wb_yaw.get())),
                "right": np.array([np.radians(float(e.get())) for e in self.wb_arms["right"]]),
                "left": np.array([np.radians(float(e.get())) for e in self.wb_arms["left"]]),
            }
        except ValueError:
            self.set_status("Invalid whole-body config input")
            return
        if not self._set_whole_body_cb(cfg):
            self.set_status("设置全身配置失败 (碰撞拦截或超限)")

    def _on_whole_body_solve_ik(self) -> None:
        if self._whole_body_solve_cb is None:
            self.set_status("全身 Solve IK 不可用")
            return
        side = self.ik_side.get()
        entries = [self.ik_x, self.ik_y, self.ik_z, self.ik_qx, self.ik_qy, self.ik_qz, self.ik_qw]
        try:
            values = [float(e.get()) for e in entries]
        except ValueError:
            self.set_status("Invalid IK target input")
            return
        pos = np.array(values[:3])
        quat = np.array(values[3:])
        if not self._whole_body_solve_cb(side, pos, quat):
            self.set_status(f"{side.capitalize()} 全身 Solve IK failed")

    def _on_vr_toggle(self) -> None:
        enabled = self.vr_enabled.get()
        if self._on_vr_toggle_cb:
            self._on_vr_toggle_cb(enabled)
        self.vr_status_var.set(f"VR: {'Enabled' if enabled else 'Disabled'}")

    def _on_vr_calibrate(self) -> None:
        if self._on_vr_calibrate_cb:
            self._on_vr_calibrate_cb()

    def _on_vr_reset_smoothing(self) -> None:
        # This will be connected to the VR receiver
        pass

    def _on_hmd_base_toggle(self) -> None:
        if self._on_hmd_base_toggle_cb:
            self._on_hmd_base_toggle_cb(self.vr_hmd_base.get())

    def set_vr_status(self, connected: bool, right_status: str, left_status: str, hmd_status: str = "") -> None:
        """Update VR status display."""
        self.vr_connected = connected
        status = "Connected" if connected else "Disconnected"
        enabled = self.vr_enabled.get()
        self.vr_status_var.set(f"VR: {'Enabled' if enabled else 'Disabled'} | {status}")
        self.vr_right_status.set(f"R: {right_status}")
        self.vr_left_status.set(f"L: {left_status}")
        if hmd_status:
            self.vr_hmd_status.set(hmd_status)

    def set_status(self, text: str) -> None:
        self.status.set(text)

    def set_dynamics(
        self,
        right_tau: np.ndarray,
        left_tau: np.ndarray,
        right_com: np.ndarray,
        left_com: np.ndarray,
        right_mass: float,
        left_mass: float,
    ) -> None:
        self.right_torque_var.set(
            "Right tau: [" + ", ".join(f"{v:+.2f}" for v in right_tau) + "] Nm"
        )
        self.left_torque_var.set(
            "Left  tau: [" + ", ".join(f"{v:+.2f}" for v in left_tau) + "] Nm"
        )
        self.right_com_var.set(
            f"Right CoM: [{right_com[0]:+.3f}, {right_com[1]:+.3f}, {right_com[2]:+.3f}] m  "
            f"mass {right_mass:.3f} kg"
        )
        self.left_com_var.set(
            f"Left  CoM: [{left_com[0]:+.3f}, {left_com[1]:+.3f}, {left_com[2]:+.3f}] m  "
            f"mass {left_mass:.3f} kg"
        )

    def set_fk(self, right_pos: np.ndarray, right_quat: np.ndarray, left_pos: np.ndarray, left_quat: np.ndarray) -> None:
        self.fk_right_var.set(
            f"R: pos=({right_pos[0]:+.3f},{right_pos[1]:+.3f},{right_pos[2]:+.3f}) "
            f"q=({right_quat[0]:+.3f},{right_quat[1]:+.3f},{right_quat[2]:+.3f},{right_quat[3]:+.3f})"
        )
        self.fk_left_var.set(
            f"L: pos=({left_pos[0]:+.3f},{left_pos[1]:+.3f},{left_pos[2]:+.3f}) "
            f"q=({left_quat[0]:+.3f},{left_quat[1]:+.3f},{left_quat[2]:+.3f},{left_quat[3]:+.3f})"
        )

    def update(self) -> None:
        self.root.update()

    def is_open(self) -> bool:
        try:
            return self.root.winfo_exists()
        except tk.TclError:
            return False

    def close(self) -> None:
        try:
            self.root.destroy()
        except tk.TclError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", default=URDF_PATH)
    parser.add_argument("--mjcf", default=MJCF_PATH)
    parser.add_argument("--mobile", action="store_true", help="Use the mecanum mobile-base + lift scene (auto-generated if missing)")
    parser.add_argument("--mobile2", action="store_true", help="Use the omni-wheel (X-layout, 90-deg rollers) mobile-base + lift scene")
    parser.add_argument("--mobile3", action="store_true", help="Omni base + chamfered chassis + 900mm lift housing + waist pitch/yaw + cut torso")
    parser.add_argument("--mobile4", action="store_true", help="mobile3 model + hierarchical Cartesian IK with waist-direction policy (腰回零/L1/L2+腰/L3+底盘) + manual-waist gripper hold switches")
    parser.add_argument("--ik-level-tol", type=float, default=5e-3, help="DLS residual threshold for escalating IK level (default 0.005)")
    parser.add_argument("--rate", type=float, default=10.0)
    parser.add_argument("--gravity", nargs=3, type=float, default=[0.0, 0.0, -9.81], help="Gravity vector (m/s^2)")
    args = parser.parse_args()

    if args.mobile3 and args.mjcf == MJCF_PATH:
        args.mjcf = MJCF_MOBILE3_PATH
    elif args.mobile4 and args.mjcf == MJCF_PATH:
        args.mjcf = MJCF_MOBILE3_PATH  # mobile4 = mobile3 model + hierarchical IK
    elif args.mobile2 and args.mjcf == MJCF_PATH:
        args.mjcf = MJCF_MOBILE2_PATH
    elif args.mobile and args.mjcf == MJCF_PATH:
        args.mjcf = MJCF_MOBILE_PATH

    # Auto-generate position-control XML if missing
    mjcf_path = Path(args.mjcf)
    if not mjcf_path.is_absolute():
        mjcf_path = Path(__file__).parent / mjcf_path
    if not mjcf_path.exists():
        if mjcf_path.name in ("scene_mobile.xml", "scene_mobile2.xml", "scene_mobile3.xml"):
            robot_pc = mjcf_path.parent / "openarm_bimanual_position_control.xml"
            if not robot_pc.exists():
                build_position_control_xml(mjcf_path.parent / "openarm_bimanual.xml", robot_pc)
            style_map = {"scene_mobile.xml": "mecanum", "scene_mobile2.xml": "omni", "scene_mobile3.xml": "mobile3"}
            suffix_map = {"scene_mobile.xml": "mobile", "scene_mobile2.xml": "mobile2", "scene_mobile3.xml": "mobile3"}
            wheel_style = style_map[mjcf_path.name]
            suffix = suffix_map[mjcf_path.name]
            print(f"Mobile scene not found: {mjcf_path}; generating {wheel_style} mobile-base XML")
            if wheel_style == "mobile3":
                prepare_mobile3_assets(mjcf_path.parent / "meshes")
                build_mobile3_xml(robot_pc, mjcf_path.parent / f"openarm_bimanual_{suffix}.xml")
            else:
                build_mobile_base_xml(robot_pc, mjcf_path.parent / f"openarm_bimanual_{suffix}.xml", wheel_style=wheel_style)
            mjcf_path.write_text(
                f'<mujoco model="openarm mobile {wheel_style} scene (position control)">\n'
                f'  <include file="openarm_bimanual_{suffix}.xml"/>\n'
                '  <statistic center="0 0 0.9" extent="2.0" meansize="0.05"/>\n'
                '  <visual>\n'
                '    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>\n'
                '    <rgba haze="0.15 0.25 0.35 1"/>\n'
                '    <global azimuth="160" elevation="-10"/>\n'
                '    <quality shadowsize="8192"/>\n'
                '  </visual>\n'
                '  <asset>\n'
                '    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>\n'
                '    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"\n'
                '      markrgb="0.8 0.8 0.8" width="300" height="300"/>\n'
                '    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>\n'
                '  </asset>\n'
                '  <worldbody>\n'
                '    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>\n'
                '    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>\n'
                '  </worldbody>\n'
                '</mujoco>\n',
                encoding="utf-8",
            )
        else:
            print(f"Position-control XML not found: {mjcf_path}")
            src_xml = Path("src/openarm_description/mujoco/openarm_bimanual.xml")
            if not src_xml.is_absolute():
                src_xml = Path(__file__).parent / src_xml
            build_position_control_xml(src_xml, mjcf_path)
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

    # Load full-robot dynamics
    robot = RobotDynamics(args.urdf)
    print(f"Loaded Pinocchio model: nq={robot.model.nq}, nv={robot.model.nv}")

    mj_model = mujoco.MjModel.from_xml_path(args.mjcf)
    mj_data = mujoco.MjData(mj_model)
    mj_model.opt.gravity[:] = np.array(args.gravity)
    mujoco.mj_forward(mj_model, mj_data)

    # MuJoCo-based gravity torque and per-arm CoM computation
    mj_dyn = MuJoCoDynamics(mj_model, mj_data)

    # Detect the optional mecanum mobile base (base_x/base_y/base_yaw/lift_z + wheels)
    base_qadr: dict[str, int | None] = {}
    for key, name in BASE_JOINTS.items():
        jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
        base_qadr[key] = int(mj_model.jnt_qposadr[jid]) if jid >= 0 else None
    base_enabled = all(v is not None for v in base_qadr.values())
    wheel_qadr: list[int] = []
    if base_enabled:
        for name in WHEEL_JOINTS:
            jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
            wheel_qadr.append(int(mj_model.jnt_qposadr[jid]) if jid >= 0 else -1)
        if any(a < 0 for a in wheel_qadr):
            wheel_qadr = []
    if base_enabled:
        print("Mobile base detected: omnidirectional chassis + lead-screw lift enabled")

    # Read wheel geometry from the model (works for mecanum and omni layouts)
    omni_wheels = False
    wheel_geom: list[tuple[float, float, float, float]] = []  # (d_ix, d_iy, x_i, y_i)
    mecanum_lw = 0.0
    if wheel_qadr:
        for name in WHEEL_JOINTS:
            jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
            a = np.asarray(mj_data.xaxis[jid])[:2]  # world axle direction
            a = a / np.linalg.norm(a)
            d_i = np.array([a[1], -a[0]])  # rolling direction for positive spin
            r_i = np.asarray(mj_data.xpos[mj_model.jnt_bodyid[jid]])[:2]
            wheel_geom.append((d_i[0], d_i[1], r_i[0], r_i[1]))
        first_axle = np.array([-wheel_geom[0][1], wheel_geom[0][0]])  # axle a from d=(a_y,-a_x)
        omni_wheels = abs(abs(first_axle[0]) - abs(first_axle[1])) < 0.2
        mecanum_lw = float(np.mean([abs(g[2]) + abs(g[3]) for g in wheel_geom]))
        if omni_wheels:
            print("Wheel mode: omni (X-layout, dual-row 90-deg rollers)")
        else:
            print("Wheel mode: mecanum (45-deg rollers)")

    # Detect optional waist joints (mobile3: pitch between lift and torso, then yaw)
    waist_qadr: dict[str, int | None] = {}
    waist_range: dict[str, tuple[float, float]] = {}
    for key, name in WAIST_JOINTS.items():
        jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0:
            waist_qadr[key] = int(mj_model.jnt_qposadr[jid])
            waist_range[key] = (float(mj_model.jnt_range[jid][0]), float(mj_model.jnt_range[jid][1]))
        else:
            waist_qadr[key] = None
    waist_enabled = base_enabled and all(v is not None for v in waist_qadr.values())
    if waist_enabled:
        print("Waist joints detected: pitch (on lift) + yaw (on torso bracket)")
        # mobile3: start with the lift raised - at lift=0 the zero-pose arms
        # would reach below the chassis top
        mj_data.qpos[base_qadr["lift"]] = 0.4
        mujoco.mj_forward(mj_model, mj_data)

    base_cmd = {"vx": 0.0, "vy": 0.0, "omega": 0.0}
    lift_init = float(mj_data.qpos[base_qadr["lift"]]) if base_enabled else 0.0
    lift_target = [lift_init]
    # mobile4: hierarchical Cartesian IK with waist-direction policy
    # (腰回零 -> L1 arm -> L2 +waist -> L3 +base/lift; see ik_mobile).
    hierarchical_ik = bool(args.mobile4) and base_enabled
    if args.mobile4 and not hierarchical_ik:
        print("WARNING: --mobile4 requires base joints; falling back to arm-only IK")
    if hierarchical_ik:
        print(f"Hierarchical Cartesian IK enabled (level tol = {args.ik_level_tol})")
    # level-3 IK output: absolute base pose servo target (active=False = velocity mode)
    base_pose_target = {"active": False, "x": 0.0, "y": 0.0, "yaw": 0.0}

    # Collision / proximity safety guard (uses MuJoCo contact pipeline, 10 mm margin)
    collision_guard = CollisionGuard(mj_model, mj_data)
    collision_guard.save_safe()
    collision_violations: list[dict] = []

    def ik_solution_collides(side: str, q_arm: np.ndarray, res: dict | None = None) -> list[dict]:
        """Pre-check whether an IK solution ends in collision (hypothetical config)."""
        if not collision_guard.enabled:
            return []
        overrides: dict[str, float] = {}
        for name, val in zip(robot.arm_info[side]["joint_names"], q_arm):
            overrides[name] = float(val)
        if res is not None:
            if res["waist"] is not None and waist_enabled:
                overrides[WAIST_JOINTS["pitch"]] = float(res["waist"][0])
                overrides[WAIST_JOINTS["yaw"]] = float(res["waist"][1])
            if res["base"] is not None and base_enabled:
                overrides[BASE_JOINTS["x"]] = float(res["base"][0])
                overrides[BASE_JOINTS["y"]] = float(res["base"][1])
                overrides[BASE_JOINTS["yaw"]] = float(res["base"][2])
                overrides[BASE_JOINTS["lift"]] = float(np.clip(res["base"][3], 0.0, LIFT_MAX))
        return collision_guard.would_violate(overrides)

    def collision_blocked_status(viol: list[dict]) -> str:
        v = viol[0]
        return f"碰撞保护: 目标已拦截 ({v['b1']} ↔ {v['b2']} @ {v['dist_mm']:.1f} mm)"

    def solve_mobile_ik_safe(
        side: str,
        target_pos: np.ndarray,
        target_rot: np.ndarray,
        q_init: np.ndarray,
    ) -> tuple[dict, list[dict], bool]:
        """Hierarchical IK with collision-triggered escalation.

        First solve with the normal DLS-residual hierarchy. If the resulting
        configuration collides, re-solve forcing the next chain position
        (chain_pos + 1). Returns (res, violations, collision_escalated).
        """
        res = mj_dyn.ik_mobile(side, target_pos, target_rot, q_init, level_tol=args.ik_level_tol)
        viol = ik_solution_collides(side, res["arm"], res) if res["arm"] is not None else []
        escalated = False
        while viol and res["arm"] is not None and res["chain_pos"] + 1 < res["chain_len"]:
            res = mj_dyn.ik_mobile(
                side, target_pos, target_rot, q_init,
                level_tol=args.ik_level_tol, min_level=res["chain_pos"] + 2,
            )
            if res["arm"] is None:
                break
            viol = ik_solution_collides(side, res["arm"], res)
            escalated = True
        return res, viol, escalated

    # Initialize state for both arms using MuJoCo frame (matches the visual model)
    states = {}
    latest_left_pos = np.zeros(3)
    latest_left_rot = np.eye(3)
    latest_right_pos = np.zeros(3)
    latest_right_rot = np.eye(3)
    for side in ("right", "left"):
        q_arm = qpos_to_qarm(mj_data, robot, side)
        pos, rot = mj_dyn.ee_frame_pose(side)
        states[side] = ArmState(
            target_translation=pos.copy(),
            target_rotation=rot.copy(),
            q_target=q_arm.copy(),
        )
        if side == "left":
            latest_left_pos = pos
            latest_left_rot = rot
        else:
            latest_right_pos = pos
            latest_right_rot = rot

    gripper_values = {"right": 0.0, "left": 0.0}
    gripper_dirty = {"right": True, "left": True}
    step_size = 0.02

    # Waist gripper-hold anchors (per arm). While a hold switch is on, the arm
    # compensates toward this FIXED anchor instead of the previous step's
    # achieved pose: IK residuals cannot accumulate, and a step that fails
    # (arm at its reach limit / collision-blocked) keeps the anchor ("lost")
    # so the NEXT waist motion recovers the original held pose instead of
    # permanently slipping it (the "sudden upward jump" bug: at negative
    # waist angles the arm hits its reach limit, position IK fails step by
    # step, and re-anchoring to the drifted pose moved the lock point).
    # Re-engagement happens only when the gripper was clearly moved by an
    # external action (Cartesian buttons / IK target / VR / joint sliders).
    hold_anchor = {
        "right": {"pos": None, "rot": None, "lost": False},
        "left": {"pos": None, "rot": None, "lost": False},
    }
    HOLD_POS_EPS = 5e-3  # m; external TCP motion above this re-engages the anchor
    HOLD_ROT_EPS = np.radians(3.0)  # rad; same for orientation

    def release_hold_anchor(side: str) -> None:
        """Drop the recovery anchor after a deliberate Cartesian arm motion."""
        anchor = hold_anchor[side]
        if anchor["lost"]:
            anchor["lost"] = False
            anchor["pos"] = None
            anchor["rot"] = None

    def hold_switches() -> tuple[bool, bool]:
        """Current (hold_pos, hold_rot) switch states (False when panel absent)."""
        hp = bool(getattr(ui, "waist_hold_pos", None) is not None and ui.waist_hold_pos.get())
        hr = bool(getattr(ui, "waist_hold_rot", None) is not None and ui.waist_hold_rot.get())
        return hp, hr

    def hold_engage(side: str, hold_pos: bool, hold_rot: bool,
                    cur_pos: np.ndarray, cur_rot: np.ndarray) -> None:
        """(Re-)engage one arm's hold anchor from its current TCP pose.

        First use anchors to the current pose; later motions re-engage only
        when the gripper was clearly moved externally (never while "lost").
        A switch turned off clears its anchor (and the lost flag once both
        switches are off).
        """
        anchor = hold_anchor[side]
        if hold_pos:
            if anchor["pos"] is None or (
                not anchor["lost"] and np.linalg.norm(cur_pos - anchor["pos"]) > HOLD_POS_EPS
            ):
                anchor["pos"] = cur_pos.copy()
        else:
            anchor["pos"] = None
            if not hold_rot:
                anchor["lost"] = False
        if hold_rot:
            if anchor["rot"] is None or (
                not anchor["lost"]
                and np.linalg.norm(pin.log3(cur_rot @ anchor["rot"].T)) > HOLD_ROT_EPS
            ):
                anchor["rot"] = cur_rot.copy()
        else:
            anchor["rot"] = None
            if not hold_pos:
                anchor["lost"] = False

    def hold_compensate_side(side: str, hold_pos: bool, hold_rot: bool) -> tuple[str, list]:
        """Arm-only IK compensation toward the side's FIXED hold anchor.

        Returns (status, violations) with status "ok" (held), "poor"
        (applied but off the anchor beyond the eps — near the reach limit),
        "ik" (no solution) or "collision" (solution blocked). The sim state
        is modified only for "ok"/"poor".
        """
        anchor = hold_anchor[side]
        post_pos, post_rot = mj_dyn.ee_frame_pose(side)
        tgt_pos = anchor["pos"] if hold_pos else post_pos
        tgt_rot = anchor["rot"] if hold_rot else post_rot
        sol = mj_dyn.ik(
            side, tgt_pos, tgt_rot, qpos_to_qarm(mj_data, robot, side),
            max_iters=200, dt=0.1, damping=1e-3,
            pos_weight=1.0 if hold_pos else 0.0,
            rot_weight=1.0 if hold_rot else 0.0,
        )
        if sol is None:
            return "ik", []
        viol = ik_solution_collides(side, sol)
        if viol:
            return "collision", viol
        set_arm_qpos(mj_data, robot, side, sol)
        states[side].q_target = sol
        states[side].cartesian_dirty = True
        mujoco.mj_forward(mj_model, mj_data)
        pos_now, rot_now = mj_dyn.ee_frame_pose(side)
        states[side].target_translation = pos_now.copy()
        states[side].target_rotation = rot_now.copy()
        if (hold_pos and np.linalg.norm(pos_now - anchor["pos"]) > HOLD_POS_EPS) or (
            hold_rot and np.linalg.norm(pin.log3(rot_now @ anchor["rot"].T)) > HOLD_ROT_EPS
        ):
            return "poor", []
        return "ok", []

    def apply_mobile_ik_result(side: str, res: dict) -> None:
        """Apply a hierarchical-IK solution: arm joints, waist joints, base/lift servo."""
        state = states[side]
        if res["arm"] is not None:
            state.q_target = res["arm"]
            state.cartesian_dirty = True
        if res["waist"] is not None and waist_enabled:
            mj_data.qpos[waist_qadr["pitch"]] = float(np.clip(res["waist"][0], *waist_range["pitch"]))
            mj_data.qpos[waist_qadr["yaw"]] = float(np.clip(res["waist"][1], *waist_range["yaw"]))
            if hasattr(ui, "waist_pitch_slider"):
                ui.waist_pitch_slider.set(float(np.degrees(mj_data.qpos[waist_qadr["pitch"]])))
                ui.waist_yaw_slider.set(float(np.degrees(mj_data.qpos[waist_qadr["yaw"]])))
        if res["base"] is not None and base_enabled:
            base_pose_target.update(
                active=True,
                x=float(res["base"][0]),
                y=float(res["base"][1]),
                yaw=float(res["base"][2]),
            )
            lift_target[0] = float(np.clip(res["base"][3], 0.0, LIFT_MAX))
            if hasattr(ui, "lift_slider"):
                ui.lift_slider.set(lift_target[0] * 1000.0)

    def recompute_ik(side: str) -> bool:
        """Run IK for the current target. Returns False if blocked/failed.

        On collision-block the caller reverts the target increment, so the
        target pose never drifts into the forbidden region (previously this
        drift made repeated clicks appear stuck).
        """
        state = states[side]
        if hierarchical_ik:
            res, viol, escalated = solve_mobile_ik_safe(
                side, state.target_translation, state.target_rotation, state.q_target
            )
            if res["arm"] is None or viol:
                if viol:
                    ui.set_status(collision_blocked_status(viol))
                else:
                    ui.set_status(f"{side.capitalize()} IK FAILED")
                return False
            apply_mobile_ik_result(side, res)
            release_hold_anchor(side)  # deliberate Cartesian motion: drop recovery anchor
            tag = res.get("tag", f"L{res['level']}")
            if escalated:
                tag += "(碰撞升级)"
            ui.set_status(f"{side.capitalize()} IK {tag} err={res['err']:.4f}")
            return True
        sol = mj_dyn.ik(
            side,
            state.target_translation,
            state.target_rotation,
            state.q_target,
            max_iters=200,
            dt=0.1,
            damping=1e-3,
        )
        if sol is not None:
            viol = ik_solution_collides(side, sol)
            if viol:
                ui.set_status(collision_blocked_status(viol))
                return False
            state.q_target = sol
            state.cartesian_dirty = True
            release_hold_anchor(side)  # deliberate Cartesian motion
            ui.set_status(f"{side.capitalize()} IK OK")
            return True
        ui.set_status(f"{side.capitalize()} IK FAILED")
        return False

    def translate(side: str, axis: str, sign: int) -> None:
        idx = {"X": 0, "Y": 1, "Z": 2}[axis]
        old_val = states[side].target_translation[idx]
        states[side].target_translation[idx] += sign * step_size
        if not recompute_ik(side):
            states[side].target_translation[idx] = old_val

    def rotate(side: str, axis: str, sign: int) -> None:
        angle = sign * step_size
        axis_vec = {
            "Roll": np.array([1.0, 0.0, 0.0]),
            "Pitch": np.array([0.0, 1.0, 0.0]),
            "Yaw": np.array([0.0, 0.0, 1.0]),
        }[axis]
        # Local-axis rotation in end-effector frame
        delta_rot = pin.exp3(angle * axis_vec)
        old_rot = states[side].target_rotation.copy()
        states[side].target_rotation = states[side].target_rotation @ delta_rot
        if not recompute_ik(side):
            states[side].target_rotation = old_rot

    def reset() -> None:
        for side in ("right", "left"):
            q = np.zeros(7)
            q[3] = np.pi / 2  # elbow at exactly 90 degrees (1.571 rad)
            for name, val in zip(robot.arm_info[side]["joint_names"], q):
                jnt_id = mujoco.mj_name2id(mj_data.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                mj_data.qpos[mj_data.model.jnt_qposadr[jnt_id]] = val
            mujoco.mj_forward(mj_model, mj_data)
            pos, rot = mj_dyn.ee_frame_pose(side)
            states[side].target_translation = pos.copy()
            states[side].target_rotation = rot.copy()
            states[side].q_target = q.copy()
            states[side].cartesian_dirty = True
            # home pose: any previous waist-hold anchor is obsolete
            hold_anchor[side]["pos"] = None
            hold_anchor[side]["rot"] = None
            hold_anchor[side]["lost"] = False
        ui.set_status("Reset both arms")

    def set_step(value: float) -> None:
        nonlocal step_size
        step_size = value

    def set_gripper(side: str, value: float) -> None:
        gripper_values[side] = value
        gripper_dirty[side] = True

    def base_set_cmd(vx: float, vy: float, omega: float) -> None:
        """Set commanded base velocity in the chassis frame (vx fwd, vy left, omega CCW)."""
        base_pose_target["active"] = False  # manual driving overrides IK pose servo
        base_cmd["vx"] = float(vx)
        base_cmd["vy"] = float(vy)
        base_cmd["omega"] = float(omega)

    def base_reset() -> None:
        """Stop the base and reset its planar pose to the origin (lift unchanged)."""
        base_set_cmd(0.0, 0.0, 0.0)
        for key in ("x", "y", "yaw"):
            mj_data.qpos[base_qadr[key]] = 0.0
        ui.set_status("Base pose reset to origin")

    def set_lift(mm: float) -> None:
        """Command the lead-screw lift to an absolute height in millimeters."""
        lift_target[0] = float(np.clip(mm / 1000.0, 0.0, LIFT_MAX))

    def set_waist(pitch_deg: float, yaw_deg: float) -> None:
        """Set waist joint angles in degrees (pitch: lean forward, yaw: horizontal).

        While the mobile-base panel's hold switches are on, each arm
        compensates (arm-only IK) toward a FIXED anchor pose so the gripper
        keeps its Cartesian position and/or orientation — the two switches
        are independent. The anchor is recorded when the hold engages and is
        NOT updated on success (no residual creep); a step that fails (arm at
        its reach limit, typical at negative waist angles) or is collision-
        blocked marks the anchor "lost" and keeps it, so the next reachable
        waist motion recovers the original held pose instead of re-anchoring
        to the drifted pose (the permanent "sudden jump" of the lock point).
        Independent from the Cartesian IK hierarchy: only manual waist
        changes (sliders / one-click homing) trigger this compensation.
        """
        if not waist_enabled:
            return
        hold_pos, hold_rot = hold_switches()

        # Anchor (re-)engagement (only needed when a hold switch is on).
        # mj_forward first: IK solves restore qpos but leave xpos stale.
        if hold_pos or hold_rot:
            mujoco.mj_forward(mj_model, mj_data)
            for side in ("right", "left"):
                cur_pos, cur_rot = mj_dyn.ee_frame_pose(side)
                hold_engage(side, hold_pos, hold_rot, cur_pos, cur_rot)

        mj_data.qpos[waist_qadr["pitch"]] = float(
            np.clip(np.radians(pitch_deg), *waist_range["pitch"])
        )
        mj_data.qpos[waist_qadr["yaw"]] = float(
            np.clip(np.radians(yaw_deg), *waist_range["yaw"])
        )
        if not (hold_pos or hold_rot):
            return

        # Gripper pose-hold compensation: arm-only IK back to the FIXED
        # anchor (whichever hold switch is off leaves that component free,
        # starting from the post-waist pose).
        mujoco.mj_forward(mj_model, mj_data)
        compensated = 0
        failed_msg = ""
        for side in ("right", "left"):
            anchor = hold_anchor[side]
            status, viol = hold_compensate_side(side, hold_pos, hold_rot)
            if status == "ik":
                anchor["lost"] = True  # keep the anchor: recover on a later step
                failed_msg = f"{side.capitalize()} 腰部补偿失败: 臂级IK无解（已接近臂展极限，恢复后继续锁定）"
                continue
            if status == "collision":
                anchor["lost"] = True  # keep the anchor: recover on a later step
                failed_msg = f"{side.capitalize()} 腰部补偿已拦截: {collision_blocked_status(viol)}"
                continue
            compensated += 1
            anchor["lost"] = status == "poor"
            if status == "poor":
                failed_msg = (
                    f"{side.capitalize()} 腰部补偿接近臂展极限: 夹爪暂时偏离锁定点，"
                    f"腰部回到可达范围后自动恢复锁定"
                )
        if failed_msg:
            ui.set_status(failed_msg)
        elif compensated:
            held = "+".join(
                part for part, on in (("位置", hold_pos), ("姿态", hold_rot)) if on
            )
            ui.set_status(f"腰部补偿: {compensated} 臂夹爪{held}已保持")

    def base_home() -> None:
        """One-click homing: lift to middle (400 mm) + waist joints to zero."""
        set_lift(400.0)
        if waist_enabled:
            set_waist(0.0, 0.0)
        if hasattr(ui, "lift_slider"):
            ui.lift_slider.set(400.0)
        if waist_enabled and hasattr(ui, "waist_pitch_slider"):
            ui.waist_pitch_slider.set(0.0)
            ui.waist_yaw_slider.set(0.0)
        ui.set_status("Homed: lift -> 400 mm, waist pitch/yaw -> 0")

    def collision_toggle(enabled: bool) -> None:
        collision_guard.enabled = bool(enabled)
        if collision_guard.enabled:
            collision_guard.save_safe()
        ui.set_status(f"碰撞保护: {'开启' if enabled else '关闭'}")

    def collision_threshold(mm: float) -> None:
        collision_guard.threshold = float(np.clip(mm, 1.0, 10.0)) / 1000.0

    def get_current_pose(side: str) -> tuple[np.ndarray, np.ndarray]:
        """Return current TCP position and quaternion (x,y,z,w) for the selected arm."""
        if side == "right":
            pos = latest_right_pos.copy()
            quat = pin.Quaternion(latest_right_rot).coeffs()
        else:
            pos = latest_left_pos.copy()
            quat = pin.Quaternion(latest_left_rot).coeffs()
        return pos, quat

    def get_whole_body_config() -> dict:
        """Return the current whole-body configuration as a dict:
        lift (m), waist pitch/yaw (rad), both arms' 7 joint angles (rad)."""
        return {
            "lift": float(mj_data.qpos[base_qadr["lift"]]) if base_enabled else 0.0,
            "waist_pitch": float(mj_data.qpos[waist_qadr["pitch"]]) if waist_enabled else 0.0,
            "waist_yaw": float(mj_data.qpos[waist_qadr["yaw"]]) if waist_enabled else 0.0,
            "right": qpos_to_qarm(mj_data, robot, "right"),
            "left": qpos_to_qarm(mj_data, robot, "left"),
        }

    def set_whole_body_config(cfg: dict) -> bool:
        """Apply a whole-body configuration (lift + waist + both arms).

        Values are clipped to their limits, collision-pre-checked, and applied
        directly (joint-space set). Returns True on success.
        """
        # clip + collect overrides for the collision pre-check
        overrides: dict[str, float] = {}
        if base_enabled:
            overrides[BASE_JOINTS["lift"]] = float(np.clip(float(cfg["lift"]), 0.0, LIFT_MAX))
        if waist_enabled:
            overrides[WAIST_JOINTS["pitch"]] = float(np.clip(float(cfg["waist_pitch"]), *waist_range["pitch"]))
            overrides[WAIST_JOINTS["yaw"]] = float(np.clip(float(cfg["waist_yaw"]), *waist_range["yaw"]))
        clipped_q: dict[str, np.ndarray] = {}
        for side in ("right", "left"):
            lo = mj_dyn.arm_limits[side]["lower"]
            hi = mj_dyn.arm_limits[side]["upper"]
            q = np.clip(np.asarray(cfg[side], dtype=float), lo, hi)
            clipped_q[side] = q
            for name, val in zip(robot.arm_info[side]["joint_names"], q):
                overrides[name] = float(val)
        if collision_guard.enabled:
            viol = collision_guard.would_violate(overrides)
            if viol:
                ui.set_status(collision_blocked_status(viol))
                return False
        # apply
        if base_enabled:
            lift_target[0] = overrides[BASE_JOINTS["lift"]]
            if hasattr(ui, "lift_slider"):
                ui.lift_slider.set(lift_target[0] * 1000.0)
        if waist_enabled:
            mj_data.qpos[waist_qadr["pitch"]] = overrides[WAIST_JOINTS["pitch"]]
            mj_data.qpos[waist_qadr["yaw"]] = overrides[WAIST_JOINTS["yaw"]]
            if hasattr(ui, "waist_pitch_slider"):
                ui.waist_pitch_slider.set(float(np.degrees(mj_data.qpos[waist_qadr["pitch"]])))
                ui.waist_yaw_slider.set(float(np.degrees(mj_data.qpos[waist_qadr["yaw"]])))
        for side in ("right", "left"):
            q = clipped_q[side]
            set_arm_qpos(mj_data, robot, side, q)
            states[side].q_target = q.copy()
            states[side].cartesian_dirty = True
            mujoco.mj_forward(mj_model, mj_data)
            pos, rot = mj_dyn.ee_frame_pose(side)
            states[side].target_translation = pos.copy()
            states[side].target_rotation = rot.copy()
            release_hold_anchor(side)
        ui.set_status("已设置全身位形 (升降+腰部+双臂)")
        return True

    def apply_whole_body_result(side: str, res: dict) -> None:
        """Apply a whole-body IK solution: arm joints + waist + lift."""
        state = states[side]
        if res["arm"] is not None:
            state.q_target = res["arm"]
            state.cartesian_dirty = True
        if res["waist"] is not None and waist_enabled:
            mj_data.qpos[waist_qadr["pitch"]] = float(np.clip(res["waist"][0], *waist_range["pitch"]))
            mj_data.qpos[waist_qadr["yaw"]] = float(np.clip(res["waist"][1], *waist_range["yaw"]))
            if hasattr(ui, "waist_pitch_slider"):
                ui.waist_pitch_slider.set(float(np.degrees(mj_data.qpos[waist_qadr["pitch"]])))
                ui.waist_yaw_slider.set(float(np.degrees(mj_data.qpos[waist_qadr["yaw"]])))
        if res["lift"] is not None and base_enabled:
            lift_target[0] = float(np.clip(res["lift"], 0.0, LIFT_MAX))
            if hasattr(ui, "lift_slider"):
                ui.lift_slider.set(lift_target[0] * 1000.0)

    def solve_whole_body_ik(side: str, pos: np.ndarray, quat: np.ndarray) -> bool:
        """Whole-body solve IK: lift + waist + arm simultaneously for the target."""
        quat_norm = np.linalg.norm(quat)
        if quat_norm < 1e-12:
            ui.set_status("Invalid quaternion (zero norm)")
            return False
        R = pin.Quaternion(quat / quat_norm).matrix()
        q_init = qpos_to_qarm(mj_data, robot, side)
        res = mj_dyn.ik_whole_body(side, pos, R, q_init)
        if res["arm"] is None:
            ui.set_status(f"{side.capitalize()} 全身 IK 无解 (err={res['err']:.4f})")
            return False
        # collision pre-check across arm + waist + lift
        viol = ik_solution_collides(side, res["arm"], {
            "waist": res["waist"],
            "base": (
                [float(mj_data.qpos[base_qadr["x"]]), float(mj_data.qpos[base_qadr["y"]]),
                 float(mj_data.qpos[base_qadr["yaw"]]), float(res["lift"])]
                if (res["lift"] is not None and base_enabled) else None
            ),
        })
        if viol:
            ui.set_status(collision_blocked_status(viol))
            return False
        states[side].target_translation = pos.copy()
        states[side].target_rotation = R.copy()
        apply_whole_body_result(side, res)
        release_hold_anchor(side)  # deliberate whole-body motion
        ui.set_status(f"{side.capitalize()} 全身 IK OK err={res['err']:.4f}")
        return res["err"] < 1e-2

    def _hierarchical_solve(side: str, pos: np.ndarray, R: np.ndarray, q_init: np.ndarray) -> bool:
        """Hierarchical IK (arm -> +waist -> +base/lift) + apply. Returns success."""
        res, viol, escalated = solve_mobile_ik_safe(side, pos, R, q_init)
        if res["arm"] is None or viol:
            if viol:
                ui.set_status(collision_blocked_status(viol))
            else:
                ui.set_status(f"{side.capitalize()} IK failed")
            return False
        states[side].target_translation = pos.copy()
        states[side].target_rotation = R.copy()
        apply_mobile_ik_result(side, res)
        release_hold_anchor(side)  # deliberate Cartesian motion
        tag = res.get("tag", f"L{res['level']}")
        if escalated:
            tag += "(碰撞升级)"
        ui.set_status(f"{side.capitalize()} IK {tag} err={res['err']:.4f}")
        return res["err"] < args.ik_level_tol

    def solve_ik(side: str, pos: np.ndarray, quat: np.ndarray) -> bool:
        """Solve IK for a target pose given by position and quaternion (x,y,z,w)."""
        quat_norm = np.linalg.norm(quat)
        if quat_norm < 1e-12:
            ui.set_status("Invalid quaternion (zero norm)")
            return False
        R = pin.Quaternion(quat / quat_norm).matrix()
        q_init = qpos_to_qarm(mj_data, robot, side)
        if hierarchical_ik:
            return _hierarchical_solve(side, pos, R, q_init)
        sol = mj_dyn.ik(side, pos, R, q_init, max_iters=500, dt=0.1, damping=1e-3)
        if sol is not None:
            viol = ik_solution_collides(side, sol)
            if viol:
                ui.set_status(collision_blocked_status(viol))
                return False
            states[side].q_target = sol
            states[side].target_translation = pos.copy()
            states[side].target_rotation = R.copy()
            states[side].cartesian_dirty = True
            release_hold_anchor(side)  # deliberate Cartesian motion
            ui.set_status(f"{side.capitalize()} IK solved")
            return True
        else:
            ui.set_status(f"{side.capitalize()} IK failed")
            return False

    # ── VR Control Integration ──────────────────────────────────────────────
    # Relative-motion teleoperation with Grip-button clutch:
    #   - HOLD the Grip (side button) to engage: the arm follows the controller.
    #   - RELEASE the Grip to freeze the arm; pressing again re-anchors to the
    #     current controller pose and EE pose (like lifting a mouse).
    #   While clutched in:
    #     pos_target = ee_anchor_pos + (vr_pos - vr_anchor_pos) * scale
    #     rot_target = ee_anchor_rot @ (vr_anchor_rot^-1 @ vr_rot)
    vr_receiver: VRControllerReceiver | None = None
    vr_state = {
        "enabled": False,
        "clutched": {"right": False, "left": False},  # grip currently held per side
        # absolute-mode calibration flags: anchor persists across clutch releases
        "abs_calibrated": {"right": False, "left": False},
        "anchor": {
            "right": {"vr_pos": None, "vr_rot": None, "ee_pos": None, "ee_rot": None},
            "left": {"vr_pos": None, "vr_rot": None, "ee_pos": None, "ee_rot": None},
        },
    }
    GRIP_CLUTCH_THRESHOLD = 0.5  # grip analog value above this = clutch engaged
    vr_ik_info = {"right": "", "left": ""}  # per-side IK diagnostics for the VR panel
    vr_state["hmd_base"] = False  # HMD pose drives chassis + lift
    vr_state["prev_b"] = False    # right-controller B button edge detection
    vr_state["hmd_suspended"] = False  # True while B held: HMD base/lift control clutched
    vr_state["hmd_anchor"] = {
        "pos": None, "fwd": np.zeros(2), "yaw": 0.0, "lift": 0.0,
        "base_x": 0.0, "base_y": 0.0, "base_yaw": 0.0,
    }

    if VR_AVAILABLE:
        vr_receiver = VRControllerReceiver(port=5006, smoothing_enabled=True)
        vr_net_info = ", ".join(get_local_ips()) or "未检测到"
    else:
        vr_net_info = ""

    def vr_capture_anchor() -> bool:
        """Capture current controller poses and EE poses as relative-motion anchors.

        Also marks absolute mode as calibrated for each captured side, so the
        Re-anchor button doubles as the absolute-mode calibration step.
        """
        right = vr_receiver.get_right_state()
        left = vr_receiver.get_left_state()
        captured = False
        for side, ctrl in (("right", right), ("left", left)):
            if not ctrl.valid:
                continue
            anchor = vr_state["anchor"][side]
            anchor["vr_pos"] = ctrl.position.copy()
            anchor["vr_rot"] = ctrl.rotation.copy()
            if side == "right":
                anchor["ee_pos"] = latest_right_pos.copy()
                anchor["ee_rot"] = latest_right_rot.copy()
            else:
                anchor["ee_pos"] = latest_left_pos.copy()
                anchor["ee_rot"] = latest_left_rot.copy()
            vr_state["abs_calibrated"][side] = True
            captured = True
        return captured

    def vr_toggle(enabled: bool) -> None:
        vr_state["enabled"] = enabled
        vr_state["clutched"]["right"] = False
        vr_state["clutched"]["left"] = False
        vr_state["abs_calibrated"]["right"] = False
        vr_state["abs_calibrated"]["left"] = False
        if enabled and vr_receiver:
            vr_receiver.start()
            ui.set_status("VR enabled - HOLD the Grip (side button) to move the arm")
        elif vr_receiver:
            vr_receiver.stop()
            if vr_state.get("hmd_base"):
                hmd_base_toggle(False)
                ui.vr_hmd_base.set(False)
            ui.set_status("VR control disabled")

    def vr_calibrate() -> None:
        """Manual re-anchor: snap origins to the current poses."""
        if not vr_receiver or not vr_receiver.is_connected():
            ui.set_status("VR not connected - keep headset AWAKE and check Quest app is sending")
            return
        if vr_capture_anchor():
            vr_receiver.reset_smoothing()
            hmd_anchor_reset()  # also re-anchor the HMD base reference
            ui.set_status("VR re-anchored (归0) - hold Grip to control arms")
        else:
            ui.set_status("No valid controller pose - check tracking")

    def vr_reset_smoothing() -> None:
        if vr_receiver:
            vr_receiver.reset_smoothing()
            ui.set_status("VR smoothing reset")

    def vr_apply_port(port_text: str) -> None:
        """Apply a new UDP listen port (recreates the receiver)."""
        nonlocal vr_receiver
        try:
            port = int(port_text)
            if not (1024 <= port <= 65535):
                raise ValueError
        except ValueError:
            ui.set_status("无效端口号 (1024~65535)")
            return
        if vr_receiver is not None:
            vr_receiver.stop()
        vr_receiver = VRControllerReceiver(port=port, smoothing_enabled=True)
        if vr_state["enabled"]:
            vr_receiver.start()
        ui.set_status(f"VR 监听端口已切换为 {port}")

    def hmd_anchor_reset() -> None:
        """Capture the current HMD pose and lift height as the neutral reference."""
        if not vr_receiver:
            return
        hmd = vr_receiver.get_headset_state()
        if not hmd.valid:
            return
        q = hmd.quaternion  # w,x,y,z
        rot = pin.Quaternion(q[0], q[1], q[2], q[3]).toRotationMatrix()
        gaze = rot @ np.array([0.0, 0.0, -1.0])  # HMD forward (VR room frame, Y-up)
        fwd2 = np.array([gaze[0], gaze[2]])
        n = np.linalg.norm(fwd2)
        if n > 1e-9:
            fwd2 /= n
        anchor = vr_state["hmd_anchor"]
        anchor["pos"] = hmd.position.copy()
        anchor["fwd"] = fwd2
        anchor["yaw"] = hmd.yaw
        if base_enabled:
            anchor["lift"] = float(mj_data.qpos[base_qadr["lift"]])
            anchor["base_x"] = float(mj_data.qpos[base_qadr["x"]])
            anchor["base_y"] = float(mj_data.qpos[base_qadr["y"]])
            anchor["base_yaw"] = float(mj_data.qpos[base_qadr["yaw"]])
        else:
            anchor["lift"] = 0.0

    def hmd_base_toggle(enabled: bool) -> None:
        """Toggle HMD-driven base motion (head translate=strafe, turn=rotate, crouch=lift)."""
        if not base_enabled:
            return
        if enabled:
            hmd = vr_receiver.get_headset_state() if vr_receiver else None
            if hmd is None or not hmd.valid:
                ui.set_status("HMD 无信号 - 无法开启头显底盘控制")
                ui.vr_hmd_base.set(False)
                vr_state["hmd_base"] = False
                return
            hmd_anchor_reset()
            vr_state["hmd_base"] = True
            ui.set_status("HMD 底盘控制开启: 头部平移=底盘平移, 转头=旋转, 站起/蹲下=升降")
        else:
            vr_state["hmd_base"] = False
            vr_state["hmd_suspended"] = False
            base_set_cmd(0.0, 0.0, 0.0)
            ui.set_status("HMD 底盘控制关闭，底盘已停止")

    def vr_update() -> None:
        """Update VR status display and robot targets from VR controller poses."""
        if not vr_state["enabled"] or not vr_receiver:
            return

        right = vr_receiver.get_right_state()
        left = vr_receiver.get_left_state()
        hmd = vr_receiver.get_headset_state()

        ee_current = {
            "right": (latest_right_pos, latest_right_rot),
            "left": (latest_left_pos, latest_left_rot),
        }

        # Scale factor from UI slider (default 1.0)
        scale = float(ui.vr_scale.get())
        absolute_mode = ui.vr_mode.get() == "absolute"

        for side, ctrl in (("right", right), ("left", left)):
            anchor = vr_state["anchor"][side]
            gripped = ctrl.valid and ctrl.grip > GRIP_CLUTCH_THRESHOLD

            if not gripped:
                # Clutch released (or tracking lost): freeze this arm.
                # Next grip press re-anchors from current poses.
                vr_state["clutched"][side] = False
                vr_ik_info[side] = ""
                continue

            if absolute_mode:
                # Absolute (calibrated 1:1): the anchor is captured ONCE per
                # side — on the first grip press after enabling, or via the
                # Re-anchor button — and persists across clutch releases.
                # Pressing the grip with hands at the pose matching the robot
                # causes no jump; motion then tracks 1:1 in the world frame.
                if not vr_state["abs_calibrated"][side]:
                    anchor["vr_pos"] = ctrl.position.copy()
                    anchor["vr_rot"] = ctrl.rotation.copy()
                    ee_pos, ee_rot = ee_current[side]
                    anchor["ee_pos"] = ee_pos.copy()
                    anchor["ee_rot"] = ee_rot.copy()
                    vr_state["abs_calibrated"][side] = True

                target_pos = anchor["ee_pos"] + (ctrl.position - anchor["vr_pos"])
                target_rot = anchor["ee_rot"] @ (anchor["vr_rot"].T @ ctrl.rotation)
                states[side].target_translation = target_pos
                states[side].target_rotation = target_rot
                vr_state["clutched"][side] = True

                sol = mj_dyn.ik(
                    side, target_pos, target_rot, states[side].q_target,
                    max_iters=200, dt=0.1, damping=1e-3,
                )
                ik_mode = "6D"
                if sol is None:
                    # Fallback: position-only IK so the arm still tracks the
                    # position when the full-pose target is unreachable.
                    sol = mj_dyn.ik(
                        side, target_pos, target_rot, states[side].q_target,
                        max_iters=200, dt=0.1, damping=1e-3, rot_weight=0.0,
                    )
                    ik_mode = "POS"
                if sol is not None:
                    states[side].q_target = sol
                    states[side].cartesian_dirty = True
                vr_ik_info[side] = (
                    f"tgt=({target_pos[0]:+.2f},{target_pos[1]:+.2f},{target_pos[2]:+.2f}) "
                    f"ik={ik_mode if sol is not None else 'FAIL'} err={mj_dyn.last_ik_err[side]:.3f}"
                )
            else:
                # Relative: delta motion from the anchor captured at grip press
                if not vr_state["clutched"][side]:
                    # Grip just engaged: anchor to current controller & EE pose
                    anchor["vr_pos"] = ctrl.position.copy()
                    anchor["vr_rot"] = ctrl.rotation.copy()
                    ee_pos, ee_rot = ee_current[side]
                    anchor["ee_pos"] = ee_pos.copy()
                    anchor["ee_rot"] = ee_rot.copy()
                    vr_state["clutched"][side] = True

                # Relative translation: ee_anchor + (vr_current - vr_anchor) * scale
                delta_pos = (ctrl.position - anchor["vr_pos"]) * scale
                states[side].target_translation = anchor["ee_pos"] + delta_pos

                # Relative rotation: ee_anchor_rot @ (vr_anchor_rot^-1 @ vr_current_rot)
                delta_rot = anchor["vr_rot"].T @ ctrl.rotation
                states[side].target_rotation = anchor["ee_rot"] @ delta_rot

                recompute_ik(side)
                vr_ik_info[side] = f"err={mj_dyn.last_ik_err[side]:.3f}"

            # Trigger controls gripper (only while clutched in)
            gripper_values[side] = map_trigger_to_gripper(ctrl.trigger, side)
            gripper_dirty[side] = True

        # B button (right controller, rising edge): reset VR origins to zero
        # (same as "Re-anchor VR Origin": re-anchors controllers and the HMD
        # base reference to the current poses)
        if right.button_b and not vr_state["prev_b"]:
            vr_calibrate()
        vr_state["prev_b"] = right.button_b

        # HOLD B to clutch the HMD base/lift control: while held, the chassis
        # is stopped and the lift frozen so the user can re-adjust posture
        # freely; on release the HMD reference is re-anchored to the new
        # posture and control resumes seamlessly (mouse lift-and-reposition).
        if vr_state["hmd_base"]:
            if right.button_b:
                vr_state["hmd_suspended"] = True
            elif vr_state["hmd_suspended"]:
                vr_state["hmd_suspended"] = False
                hmd_anchor_reset()  # new posture becomes the neutral reference

        # HMD pose -> chassis translation/rotation + lift height
        if vr_state["hmd_base"] and base_enabled and hmd.valid:
            hmd_suspended = vr_state["hmd_suspended"]
            if hmd_suspended:
                # B held: stop the chassis (relative mode) and freeze the lift.
                # Absolute mode keeps the last pose target, so the base servo
                # simply holds its current pose.
                if ui.vr_hmd_mode.get() != "absolute":
                    base_set_cmd(0.0, 0.0, 0.0)
            anchor = vr_state["hmd_anchor"]
            if not hmd_suspended and anchor["pos"] is not None:
                delta = hmd.position - anchor["pos"]
                fwd = anchor["fwd"]  # 2D unit (x,z) in the VR room frame
                right2 = np.array([-fwd[1], fwd[0]])
                d_fwd = float(delta[0] * fwd[0] + delta[2] * fwd[1])
                d_lat = float(delta[0] * right2[0] + delta[2] * right2[1])
                dyaw = np.radians(hmd.yaw - anchor["yaw"])
                dyaw = (dyaw + np.pi) % (2.0 * np.pi) - np.pi
                dz = float(hmd.position[1] - anchor["pos"][1])

                if ui.vr_hmd_mode.get() == "absolute":
                    # absolute: head pose maps 1:1 to base pose + lift height
                    # (servo drives the base to the mapped pose; head back to
                    # the anchor returns the base to the anchored pose)
                    base_pose_target.update(
                        active=True,
                        x=float(anchor["base_x"] + _beyond_deadzone(d_fwd, 0.02) * HMD_ABS_POS_GAIN),
                        y=float(anchor["base_y"] + HMD_STRAFE_SIGN * _beyond_deadzone(d_lat, 0.02) * HMD_ABS_POS_GAIN),
                        yaw=float(anchor["base_yaw"] + HMD_YAW_SIGN * _beyond_deadzone(dyaw, np.radians(3.0))),
                    )
                    lift_target[0] = float(np.clip(
                        anchor["lift"] + _beyond_deadzone(dz, HMD_DEADZONE_Z) * HMD_ABS_LIFT_GAIN,
                        0.0, LIFT_MAX,
                    ))
                else:
                    # relative: head displacement beyond deadzone -> velocities
                    # (head back to center stops the base)
                    vx = _beyond_deadzone(d_fwd, HMD_DEADZONE_XY) * HMD_BASE_LIN_GAIN
                    vy = HMD_STRAFE_SIGN * _beyond_deadzone(d_lat, HMD_DEADZONE_XY) * HMD_BASE_LIN_GAIN
                    om = HMD_YAW_SIGN * _beyond_deadzone(dyaw, np.radians(HMD_DEADZONE_YAW)) * HMD_BASE_YAW_GAIN
                    base_set_cmd(
                        float(np.clip(vx, -HMD_BASE_MAX_LIN, HMD_BASE_MAX_LIN)),
                        float(np.clip(vy, -HMD_BASE_MAX_LIN, HMD_BASE_MAX_LIN)),
                        float(np.clip(om, -HMD_BASE_MAX_YAW, HMD_BASE_MAX_YAW)),
                    )
                    # head height rate -> lift velocity
                    vz = _beyond_deadzone(dz, HMD_DEADZONE_Z) * HMD_LIFT_GAIN  # m/s
                    lift_target[0] = float(np.clip(lift_target[0] + vz * (1.0 / 60.0), 0.0, LIFT_MAX))
                # sync lift slider occasionally (cheap UI feedback)
                ui.lift_slider.set(float(np.clip(lift_target[0] * 1000.0, 0.0, 800.0)))

        # Update VR status display (after control so IK diagnostics are current)
        def _status_text(ctrl: VRControllerState, side: str) -> str:
            if not ctrl.valid:
                return "[no signal]"
            grip_txt = "HELD" if ctrl.grip > GRIP_CLUTCH_THRESHOLD else "----"
            txt = (
                f"pos=({ctrl.position[0]:+.2f},{ctrl.position[1]:+.2f},{ctrl.position[2]:+.2f}) "
                f"trig={ctrl.trigger:.2f} grip={grip_txt}"
            )
            if vr_ik_info[side]:
                txt += " " + vr_ik_info[side]
            return txt

        if vr_state["hmd_base"] and vr_state["hmd_suspended"]:
            hmd_status = "HMD: 底盘/升降已暂停 (按住B键调整姿势中, 松开B键恢复)"
        elif hmd.valid:
            hmd_status = (
                f"HMD: pitch={hmd.pitch:+.1f}° yaw={hmd.yaw:+.1f}° roll={hmd.roll:+.1f}° "
                f"(pitch 0=level, ±90=looking straight down/up)"
            )
        else:
            hmd_status = "HMD: [no signal]"
        ui.set_vr_status(
            vr_receiver.is_connected(),
            _status_text(right, "right"),
            _status_text(left, "left"),
            hmd_status,
        )

    # ─────────────────────────────────────────────────────────────────────────

    ui = DualArmCartesianUI(
        lambda axis, sign: translate("right", axis, sign),
        lambda axis, sign: rotate("right", axis, sign),
        lambda axis, sign: translate("left", axis, sign),
        lambda axis, sign: rotate("left", axis, sign),
        set_gripper,
        reset,
        set_step,
        get_current_pose,
        solve_ik,
        on_get_whole_body=get_whole_body_config,
        on_set_whole_body=set_whole_body_config,
        on_whole_body_solve_ik=solve_whole_body_ik,
        on_vr_toggle=vr_toggle if VR_AVAILABLE else None,
        on_vr_calibrate=vr_calibrate if VR_AVAILABLE else None,
        on_vr_port_change=vr_apply_port if VR_AVAILABLE else None,
        on_hmd_base_toggle=(hmd_base_toggle if (VR_AVAILABLE and base_enabled) else None),
        vr_net_info=vr_net_info,
        on_base_dir=base_set_cmd if base_enabled else None,
        on_base_reset=base_reset if base_enabled else None,
        on_lift_change=set_lift if base_enabled else None,
        on_base_home=base_home if base_enabled else None,
        on_waist_change=(
            set_waist if waist_enabled else None
        ),
        waist_ranges=(
            {k: tuple(np.degrees(v)) for k, v in waist_range.items()} if waist_enabled else None
        ),
        on_collision_toggle=collision_toggle,
        on_collision_threshold=collision_threshold,
    )

    # Connect VR reset smoothing button
    if VR_AVAILABLE:
        ui._on_vr_reset_smoothing = vr_reset_smoothing

    # Sync the lift slider with the initial lift height (e.g. mobile3 ref=0.4 m)
    if base_enabled and hasattr(ui, "lift_slider"):
        ui.lift_slider.set(lift_init * 1000.0)

    print("MuJoCo viewer opened. Dual-arm Cartesian control active.")
    if VR_AVAILABLE:
        print("VR support: Available (Quest 3 on UDP port 5006)")
    else:
        print("VR support: Not available (install vr_receiver.py)")

    dt_print = 1.0 / args.rate
    last_print = 0.0

    latest_left_com = np.zeros(3)
    latest_right_com = np.zeros(3)

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        while viewer.is_running() and ui.is_open():
            # Update VR control if enabled
            if VR_AVAILABLE and vr_state["enabled"]:
                vr_update()

            # Apply Cartesian IK results or sync sliders for each arm
            for side in ("right", "left"):
                state = states[side]
                if state.cartesian_dirty:
                    set_arm_qpos(mj_data, robot, side, state.q_target)
                    state.cartesian_dirty = False
                else:
                    sync_sliders_to_qpos(mj_data, robot, side)

            # Apply gripper values
            for side in ("right", "left"):
                if gripper_dirty[side]:
                    set_finger_qpos(mj_data, side, gripper_values[side])
                    gripper_dirty[side] = False

            # Mobile base kinematic update (omni-drive integration + wheel spin + lift)
            if base_enabled:
                dt_frame = 1.0 / 60.0
                yaw = float(mj_data.qpos[base_qadr["yaw"]])
                c, s = np.cos(yaw), np.sin(yaw)
                # Gripper hold during base/lift motion: snapshot the pre-motion
                # state so the frame's motion can be REVERTED (restriction)
                # when a held arm cannot follow. xpos is still fresh from the
                # previous frame's mj_forward, so the TCP reads are current.
                hold_pos_on, hold_rot_on = hold_switches()
                hold_active = hold_pos_on or hold_rot_on
                snap_base = (
                    float(mj_data.qpos[base_qadr["x"]]),
                    float(mj_data.qpos[base_qadr["y"]]),
                    yaw,
                    float(mj_data.qpos[base_qadr["lift"]]),
                )
                snap_wheels = [float(mj_data.qpos[a]) for a in wheel_qadr] if wheel_qadr else []
                if hold_active:
                    # arm targets may have been applied this frame; refresh
                    # kinematics so the TCP snapshots are current
                    mujoco.mj_forward(mj_model, mj_data)
                    snap_arm = {sd: qpos_to_qarm(mj_data, robot, sd) for sd in ("right", "left")}
                    snap_tcp = {sd: mj_dyn.ee_frame_pose(sd) for sd in ("right", "left")}
                    snap_tgt = {
                        sd: (states[sd].target_translation.copy(), states[sd].target_rotation.copy())
                        for sd in ("right", "left")
                    }
                if base_pose_target["active"]:
                    # servo toward the level-3 IK base pose (rate limited)
                    dx = base_pose_target["x"] - float(mj_data.qpos[base_qadr["x"]])
                    dy = base_pose_target["y"] - float(mj_data.qpos[base_qadr["y"]])
                    dyaw = (base_pose_target["yaw"] - yaw + np.pi) % (2.0 * np.pi) - np.pi
                    step_xy = 0.30 * dt_frame
                    step_yaw = 0.8 * dt_frame
                    sx = float(np.clip(dx, -step_xy, step_xy))
                    sy = float(np.clip(dy, -step_xy, step_xy))
                    syaw = float(np.clip(dyaw, -step_yaw, step_yaw))
                    mj_data.qpos[base_qadr["x"]] += sx
                    mj_data.qpos[base_qadr["y"]] += sy
                    mj_data.qpos[base_qadr["yaw"]] = yaw + syaw
                    # effective body-frame velocities for wheel animation
                    vx = (c * sx + s * sy) / dt_frame
                    vy = (-s * sx + c * sy) / dt_frame
                    om = syaw / dt_frame
                    if abs(dx) < 1e-3 and abs(dy) < 1e-3 and abs(dyaw) < 2e-3:
                        base_pose_target["active"] = False
                else:
                    vx, vy, om = base_cmd["vx"], base_cmd["vy"], base_cmd["omega"]
                    # integrate body-frame velocity into world-frame planar pose
                    mj_data.qpos[base_qadr["x"]] += (c * vx - s * vy) * dt_frame
                    mj_data.qpos[base_qadr["y"]] += (s * vx + c * vy) * dt_frame
                    mj_data.qpos[base_qadr["yaw"]] = yaw + om * dt_frame
                # lead-screw lift: rate-limited motion toward the target height
                lift_cur = float(mj_data.qpos[base_qadr["lift"]])
                dl = float(np.clip(lift_target[0] - lift_cur, -LIFT_RATE * dt_frame, LIFT_RATE * dt_frame))
                mj_data.qpos[base_qadr["lift"]] = float(np.clip(lift_cur + dl, 0.0, LIFT_MAX))
                # wheel visual spinning (inverse kinematics from model geometry)
                if wheel_qadr and (vx != 0.0 or vy != 0.0 or om != 0.0):
                    if omni_wheels:
                        # omni X-layout: w_i = d_i . (v + omega * z x r_i) / R
                        wheel_rates = tuple(
                            (dx * (vx - om * ry) + dy * (vy + om * rx)) / WHEEL_RADIUS
                            for dx, dy, rx, ry in wheel_geom
                        )
                    else:
                        # mecanum: wheel planes along x, 45-deg rollers
                        lw = mecanum_lw
                        wheel_rates = (
                            (vx - vy - lw * om) / WHEEL_RADIUS,  # fl
                            (vx + vy + lw * om) / WHEEL_RADIUS,  # fr
                            (vx + vy - lw * om) / WHEEL_RADIUS,  # rl
                            (vx - vy + lw * om) / WHEEL_RADIUS,  # rr
                        )
                    for adr, w in zip(wheel_qadr, wheel_rates):
                        mj_data.qpos[adr] += w * dt_frame

                # Gripper hold: compensate the arms toward the FIXED anchors
                # for this frame's base/lift delta. When any held arm cannot
                # follow (reach limit / collision), RESTRICT the motion —
                # revert the base/lift (and any compensated arm) so the
                # grippers stay at their anchors; moving the other way works
                # again (compensation becomes reachable).
                if hold_active:
                    moved = (
                        abs(float(mj_data.qpos[base_qadr["x"]]) - snap_base[0]) > 1e-12
                        or abs(float(mj_data.qpos[base_qadr["y"]]) - snap_base[1]) > 1e-12
                        or abs(float(mj_data.qpos[base_qadr["yaw"]]) - snap_base[2]) > 1e-12
                        or abs(float(mj_data.qpos[base_qadr["lift"]]) - snap_base[3]) > 1e-12
                    )
                    if moved:
                        restricted: list[tuple[str, str]] = []
                        for sd in ("right", "left"):
                            hold_engage(sd, hold_pos_on, hold_rot_on, *snap_tcp[sd])
                            st, _ = hold_compensate_side(sd, hold_pos_on, hold_rot_on)
                            if st != "ok":
                                restricted.append((sd, st))
                        if restricted:
                            mj_data.qpos[base_qadr["x"]] = snap_base[0]
                            mj_data.qpos[base_qadr["y"]] = snap_base[1]
                            mj_data.qpos[base_qadr["yaw"]] = snap_base[2]
                            mj_data.qpos[base_qadr["lift"]] = snap_base[3]
                            for adr, w0 in zip(wheel_qadr, snap_wheels):
                                mj_data.qpos[adr] = w0
                            for sd in ("right", "left"):
                                set_arm_qpos(mj_data, robot, sd, snap_arm[sd])
                                states[sd].q_target = snap_arm[sd].copy()
                                states[sd].cartesian_dirty = True
                                states[sd].target_translation, states[sd].target_rotation = (
                                    snap_tgt[sd][0].copy(), snap_tgt[sd][1].copy()
                                )
                            sd0, st0 = restricted[0]
                            why = "臂展极限" if st0 in ("ik", "poor") else "碰撞风险"
                            ui.set_status(
                                f"夹爪锁定: 底盘/升降已限位 ({sd0}臂{why})，反向运动或关闭保持开关可恢复"
                            )

            mujoco.mj_forward(mj_model, mj_data)

            # Collision safety clamp: revert to last safe pose on violation
            # (covers GUI sliders, Cartesian IK, VR, and future VLA outputs)
            if collision_guard.enabled:
                collision_violations = collision_guard.check()
                if collision_violations:
                    collision_guard.restore_safe()
                    mujoco.mj_forward(mj_model, mj_data)
                    collision_guard.check()  # refresh list for display
                    collision_violations = collision_guard.violations
                else:
                    collision_guard.save_safe()

            # Read current TCP poses from MuJoCo (matches the visual model)
            latest_right_pos, latest_right_rot = mj_dyn.ee_frame_pose("right")
            latest_left_pos, latest_left_rot = mj_dyn.ee_frame_pose("left")

            # Compute and print both end-effector poses, torques, and CoM
            now = time.time()
            if now - last_print >= dt_print:
                last_print = now

                right_pos = latest_right_pos
                left_pos = latest_left_pos
                right_quat = pin.Quaternion(latest_right_rot).coeffs()
                left_quat = pin.Quaternion(latest_left_rot).coeffs()
                print(
                    f"\rR: pos=({right_pos[0]:+.3f},{right_pos[1]:+.3f},{right_pos[2]:+.3f}) "
                    f"q=({right_quat[0]:+.3f},{right_quat[1]:+.3f},{right_quat[2]:+.3f},{right_quat[3]:+.3f}) | "
                    f"L: pos=({left_pos[0]:+.3f},{left_pos[1]:+.3f},{left_pos[2]:+.3f}) "
                    f"q=({left_quat[0]:+.3f},{left_quat[1]:+.3f},{left_quat[2]:+.3f},{left_quat[3]:+.3f})",
                    end="",
                    flush=True,
                )

                # Update FK text in UI
                ui.set_fk(right_pos, right_quat, left_pos, left_quat)

                # Update mobile base status in UI
                if base_enabled:
                    ui.set_base_status(
                        float(mj_data.qpos[base_qadr["x"]]),
                        float(mj_data.qpos[base_qadr["y"]]),
                        float(np.degrees(mj_data.qpos[base_qadr["yaw"]])),
                        float(mj_data.qpos[base_qadr["lift"]] * 1000.0),
                        base_cmd["vx"],
                        base_cmd["vy"],
                        base_cmd["omega"],
                        waist_pitch_deg=(
                            float(np.degrees(mj_data.qpos[waist_qadr["pitch"]])) if waist_enabled else None
                        ),
                        waist_yaw_deg=(
                            float(np.degrees(mj_data.qpos[waist_qadr["yaw"]])) if waist_enabled else None
                        ),
                    )

                # Update collision guard status in UI
                ui.set_collision_status(
                    collision_violations if collision_guard.enabled else None,
                    collision_guard.threshold * 1000.0,
                    collision_guard.enabled,
                )

                tau = mj_dyn.gravity_torques()
                left_com, left_mass = mj_dyn.arm_com("left")
                right_com, right_mass = mj_dyn.arm_com("right")
                latest_left_com = left_com
                latest_right_com = right_com
                ui.set_dynamics(
                    tau["right"], tau["left"], right_com, left_com, right_mass, left_mass
                )

            # Draw per-arm CoM markers, EE frames, base frame, and joint frames every frame
            extra_markers = None
            if collision_guard.enabled and ui.is_open() and getattr(ui, "collision_show", None) is not None \
                    and ui.collision_show.get() and collision_violations:
                extra_markers = [
                    (v["pos"], np.array([1.0, 0.1, 0.1, 0.95]), 0.015) for v in collision_violations
                ]
            mj_dyn.draw_markers(
                viewer,
                latest_left_com,
                ui.show_left_com.get(),
                latest_right_com,
                ui.show_right_com.get(),
                left_pos=latest_left_pos,
                left_rot=latest_left_rot,
                right_pos=latest_right_pos,
                right_rot=latest_right_rot,
                show_ee_frames=ui.show_ee_frames.get(),
                show_base_frame=ui.show_base_frame.get(),
                show_joint_frames=ui.show_joint_frames.get(),
                extra_markers=extra_markers,
            )

            viewer.sync()
            ui.update()
            time.sleep(1.0 / 60.0)

    # Cleanup
    if VR_AVAILABLE and vr_receiver:
        vr_receiver.stop()
    ui.close()


if __name__ == "__main__":
    main()
