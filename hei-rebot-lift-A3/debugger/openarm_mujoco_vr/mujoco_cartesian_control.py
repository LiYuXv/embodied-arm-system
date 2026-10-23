from __future__ import annotations

import time
import tkinter as tk
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import mujoco
import mujoco.viewer
import numpy as np
import pinocchio as pin


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


@dataclass
class ArmState:
    """Shared target state for one arm. Updated by the UI callbacks, read by the MuJoCo loop."""

    target_translation: np.ndarray = field(default_factory=lambda: np.zeros(3))
    target_rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    q_target: np.ndarray | None = None
    dirty: bool = False


class CartesianController:
    """Pinocchio-based FK/IK for a single arm."""

    def __init__(self, urdf_path: str, side: str) -> None:
        self.side = side
        self.joint_names = ARM_JOINTS[side]
        self.ee_frame_name = EE_FRAMES[side]

        path = Path(urdf_path)
        if not path.is_absolute():
            path = Path(__file__).parent / path
        self.model = pin.buildModelFromUrdf(str(path))
        self.data = self.model.createData()
        self.ee_frame_id = self.model.getFrameId(self.ee_frame_name)

        self.joint_ids = [self.model.getJointId(n) for n in self.joint_names]
        self.joint_q_offsets = [int(self.model.joints[jid].idx_q) for jid in self.joint_ids]
        self.joint_v_offsets = [int(self.model.joints[jid].idx_v) for jid in self.joint_ids]

        self.lower_limits = np.array([
            self.model.lowerPositionLimit[q] for q in self.joint_q_offsets
        ])
        self.upper_limits = np.array([
            self.model.upperPositionLimit[q] for q in self.joint_q_offsets
        ])

    def neutral(self) -> np.ndarray:
        return pin.neutral(self.model)

    def fk(self, q_arm: np.ndarray) -> pin.SE3:
        q_full = self._full_q(q_arm)
        pin.forwardKinematics(self.model, self.data, q_full)
        pin.updateFramePlacements(self.model, self.data)
        return self.data.oMf[self.ee_frame_id]

    def ik(
        self,
        target: pin.SE3,
        q_init: np.ndarray,
        max_iters: int = 100,
        tol: float = 1e-5,
        dt: float = 0.1,
        damping: float = 1e-3,
    ) -> np.ndarray | None:
        q_arm = np.asarray(q_init, dtype=float).copy()

        q_full = self._full_q(q_arm)
        pin.forwardKinematics(self.model, self.data, q_full)
        pin.updateFramePlacements(self.model, self.data)
        pin.computeJointJacobians(self.model, self.data, q_full)

        for _ in range(max_iters):
            current = self.data.oMf[self.ee_frame_id]
            err = pin.log(target.inverse() * current)
            if np.linalg.norm(err) < tol:
                break

            J = pin.computeFrameJacobian(self.model, self.data, q_full, self.ee_frame_id)
            J_arm = J[:, self.joint_v_offsets]
            H = J_arm.T @ J_arm + damping * np.eye(7)
            dq = np.linalg.solve(H, J_arm.T @ (-err))
            q_arm += dt * dq
            q_arm = np.clip(q_arm, self.lower_limits, self.upper_limits)

            q_full = self._full_q(q_arm)
            pin.forwardKinematics(self.model, self.data, q_full)
            pin.updateFramePlacements(self.model, self.data)
            pin.computeJointJacobians(self.model, self.data, q_full)
        else:
            return None
        return q_arm

    def _full_q(self, q_arm: np.ndarray) -> np.ndarray:
        q_full = pin.neutral(self.model)
        q_arm = np.asarray(q_arm, dtype=float).flatten()
        for offset, value in zip(self.joint_q_offsets, q_arm):
            q_full[offset] = value
        return q_full


class CartesianUI:
    """Tkinter window with Cartesian motion buttons."""

    def __init__(
        self,
        on_translate: Callable[[str, float], None],
        on_rotate: Callable[[str, float], None],
        on_reset: Callable[[], None],
        on_step_change: Callable[[float], None],
    ) -> None:
        self.root = tk.Tk()
        self.root.title("OpenArm Cartesian Control")
        self.root.geometry("420x420")

        self.step = tk.DoubleVar(value=0.02)
        self.on_step_change = on_step_change
        self.on_step_change(self.step.get())

        # Step size
        tk.Label(self.root, text="Step size (m / rad):").pack(pady=(10, 0))
        tk.Scale(
            self.root,
            variable=self.step,
            from_=0.001,
            to_=0.1,
            resolution=0.001,
            orient=tk.HORIZONTAL,
            length=300,
            command=lambda v: self.on_step_change(float(v)),
        ).pack()

        # Translation frame
        trans_frame = tk.LabelFrame(self.root, text="Translation (m)", padx=10, pady=10)
        trans_frame.pack(padx=10, pady=10, fill="both", expand=True)
        for axis in ("X", "Y", "Z"):
            f = tk.Frame(trans_frame)
            f.pack(fill="x", pady=2)
            tk.Label(f, text=f"{axis}:", width=4).pack(side=tk.LEFT)
            tk.Button(f, text="-", width=3, command=lambda a=axis: on_translate(a, -1)).pack(side=tk.LEFT, padx=2)
            tk.Button(f, text="+", width=3, command=lambda a=axis: on_translate(a, +1)).pack(side=tk.LEFT, padx=2)

        # Rotation frame
        rot_frame = tk.LabelFrame(self.root, text="Rotation (rad)", padx=10, pady=10)
        rot_frame.pack(padx=10, pady=10, fill="both", expand=True)
        for axis in ("Roll", "Pitch", "Yaw"):
            f = tk.Frame(rot_frame)
            f.pack(fill="x", pady=2)
            tk.Label(f, text=f"{axis}:", width=6).pack(side=tk.LEFT)
            tk.Button(f, text="-", width=3, command=lambda a=axis: on_rotate(a, -1)).pack(side=tk.LEFT, padx=2)
            tk.Button(f, text="+", width=3, command=lambda a=axis: on_rotate(a, +1)).pack(side=tk.LEFT, padx=2)

        # Reset
        tk.Button(self.root, text="Reset to Neutral", command=on_reset, bg="orange").pack(pady=10)

        # Status
        self.status = tk.StringVar(value="Ready")
        tk.Label(self.root, textvariable=self.status, relief=tk.SUNKEN, anchor=tk.W).pack(side=tk.BOTTOM, fill="x")

    def set_status(self, text: str) -> None:
        self.status.set(text)

    def update(self) -> None:
        self.root.update()

    def is_open(self) -> bool:
        try:
            self.root.winfo_exists()
            return True
        except tk.TclError:
            return False

    def close(self) -> None:
        try:
            self.root.destroy()
        except tk.TclError:
            pass


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Cartesian-space control of OpenArm via Tkinter + MuJoCo")
    parser.add_argument("--urdf", default=URDF_PATH)
    parser.add_argument("--mjcf", default=MJCF_PATH)
    parser.add_argument("--side", default="right", choices=["left", "right"])
    args = parser.parse_args()

    # Load models
    controller = CartesianController(args.urdf, args.side)
    mj_model = mujoco.MjModel.from_xml_path(args.mjcf)
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, mj_data)

    # Initialize current state from neutral pose
    q_current = pin.neutral(controller.model)
    q_arm = np.array([q_current[offset] for offset in controller.joint_q_offsets])
    current_pose = controller.fk(q_arm)
    arm_state = ArmState(
        target_translation=current_pose.translation.copy(),
        target_rotation=current_pose.rotation.copy(),
        q_target=q_arm.copy(),
    )

    step_size = 0.02

    def set_step_size(value: float) -> None:
        nonlocal step_size
        step_size = value

    def update_target() -> None:
        """Recompute IK from the current target translation/rotation and store it."""
        target = pin.SE3(arm_state.target_rotation, arm_state.target_translation)
        q_solution = controller.ik(target, arm_state.q_target, max_iters=200, dt=0.5, damping=1e-4)
        if q_solution is not None:
            arm_state.q_target = q_solution
            arm_state.dirty = True
            ui.set_status(f"IK ok | target={arm_state.target_translation.round(3)}")
        else:
            ui.set_status("IK failed")

    def translate(axis: str, sign: int) -> None:
        idx = {"X": 0, "Y": 1, "Z": 2}[axis]
        arm_state.target_translation[idx] += sign * step_size
        update_target()

    def rotate(axis: str, sign: int) -> None:
        angle = sign * step_size
        axis_vec = {
            "Roll": np.array([1.0, 0.0, 0.0]),
            "Pitch": np.array([0.0, 1.0, 0.0]),
            "Yaw": np.array([0.0, 0.0, 1.0]),
        }[axis]
        delta_rot = pin.exp3(angle * axis_vec)
        arm_state.target_rotation = delta_rot @ arm_state.target_rotation
        update_target()

    def reset() -> None:
        q_arm = np.zeros(7)
        if args.side == "left":
            q_arm[1] = -1.5
        else:
            q_arm[1] = 1.5
        pose = controller.fk(q_arm)
        arm_state.target_translation = pose.translation.copy()
        arm_state.target_rotation = pose.rotation.copy()
        arm_state.q_target = q_arm.copy()
        arm_state.dirty = True
        ui.set_status("Reset")

    # Build UI
    ui = CartesianUI(translate, rotate, reset, set_step_size)

    # Open MuJoCo viewer in the main thread; process Tkinter events via ui.update()
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        while viewer.is_running() and ui.is_open():
            # Apply pending IK solution to MuJoCo
            if arm_state.dirty and arm_state.q_target is not None:
                for name, val in zip(controller.joint_names, arm_state.q_target):
                    jnt_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
                    mj_data.qpos[mj_model.jnt_qposadr[jnt_id]] = val
                    # Also update the position-control actuator so the slider matches
                    act_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, name + "_ctrl")
                    if act_id >= 0:
                        mj_data.ctrl[act_id] = val
                arm_state.dirty = False
            mujoco.mj_forward(mj_model, mj_data)
            viewer.sync()

            # Process Tkinter events without blocking
            ui.update()

            time.sleep(1.0 / 60.0)

    ui.close()
    viewer.close()


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
