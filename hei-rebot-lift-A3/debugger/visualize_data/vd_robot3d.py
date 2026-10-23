# -*- coding: utf-8 -*-
"""机械臂 3D 可视化回放面板（依赖 debugger.widgets.viewer_3d 与 IK 单例）。"""
from vd_common import ARM_JOINT_NAMES, _KIN

import math
import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt


# ==================== 机械臂 3D 可视化面板 ====================
class RobotArmReplayWidget(QWidget):
    """数据回放用的机械臂 3D 可视化面板。

    复用 debugger.widgets.viewer_3d.Viewer3DPanel（含“重置视角/线框/关闭视图”按键），
    在其基础上叠加：根据预反解的关节角度数组，按帧刷新机械臂姿态，并显示各关节角度。
    """

    def __init__(self, parent=None, urdf_path=None, mesh_dir=None):
        super().__init__(parent)
        self._joint_angles = None      # (N, 7) 预反解结果，None 表示未就绪
        self._current_frame = 0
        self._viewer = None
        self._enabled = _KIN is not None
        self._init_ui(urdf_path, mesh_dir)

    def _init_ui(self, urdf_path, mesh_dir):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        if not self._enabled:
            label = QLabel("逆运动学模块 (pinocchio) 未加载，机械臂反解不可用")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet("color: #b00; font-size: 12px; padding: 24px;")
            layout.addWidget(label)
            return

        try:
            from debugger.widgets.viewer_3d import Viewer3DPanel
        except Exception as e:
            label = QLabel(f"无法加载 3D 可视化模块:\n{e}")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setWordWrap(True)
            label.setStyleSheet("color: #b00; font-size: 12px; padding: 20px;")
            layout.addWidget(label)
            self._enabled = False
            return

        self._viewer = Viewer3DPanel(self, urdf_path=urdf_path, mesh_dir=mesh_dir)
        # 回放场景无需拖拽，关闭拖拽相关高亮/事件过滤
        try:
            self._viewer.set_drag_mode(False)
        except Exception:
            pass
        layout.addWidget(self._viewer, stretch=1)

        # 关节角度信息标签
        self._info_label = QLabel("机械臂关节角度: 未加载 Episode")
        self._info_label.setStyleSheet(
            "QLabel { color: #333; font-size: 11px; padding: 3px 6px;"
            " background-color: #f0f0f0; border-radius: 3px; }"
        )
        self._info_label.setWordWrap(True)
        self._info_label.setMinimumHeight(20)
        layout.addWidget(self._info_label)

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def ensure_initialized(self):
        """惰性初始化 3D 模型（首次显示时调用，避免启动卡顿）。"""
        if not self._enabled or self._viewer is None:
            return False
        try:
            if not getattr(self._viewer, "_initialized", False):
                return self._viewer.initialize_model()
        except Exception as e:
            print(f"[RobotArm] 3D 模型初始化失败: {e}")
            return False
        return True

    def clear(self):
        """清空预反解数据并复位姿态。"""
        self._joint_angles = None
        self._current_frame = 0
        if self._enabled and self._viewer is not None:
            try:
                self._render_zero()
            except Exception:
                pass
        if hasattr(self, "_info_label"):
            self._info_label.setText("机械臂关节角度: 未加载 Episode")

    def _render_zero(self):
        if self._viewer is None:
            return
        angles = {name: 0.0 for name in ARM_JOINT_NAMES}
        if self._viewer._model is not None:
            self._viewer._render_pose(angles)

    # ------------------------------------------------------------------
    # 数据与帧更新
    # ------------------------------------------------------------------
    def set_joint_angles(self, angles):
        """设置预反解的整段关节角度数组 (N, 7)。"""
        self._joint_angles = np.asarray(angles) if angles is not None else None
        self._current_frame = 0
        self.show_frame(0)

    def show_frame(self, frame_idx):
        """刷新到指定帧的机械臂姿态。"""
        if not self._enabled or self._viewer is None:
            return
        if self._joint_angles is None or len(self._joint_angles) == 0:
            return
        frame_idx = max(0, min(int(frame_idx), len(self._joint_angles) - 1))
        self._current_frame = frame_idx

        if not self.ensure_initialized():
            return

        row = self._joint_angles[frame_idx]
        angles = {name: float(row[i]) for i, name in enumerate(ARM_JOINT_NAMES)}
        try:
            self._viewer._render_pose(angles)
        except Exception as e:
            print(f"[RobotArm] 渲染失败: {e}")
            return

        # 更新关节角度文本（度）
        degs = [math.degrees(a) for a in row]
        text = "机械臂关节角度 (°):  " + "  ".join(
            f"L{i+1}={d:+.1f}" for i, d in enumerate(degs[:6])
        )
        text += f"  |  夹爪={degs[6]:+.1f}°  |  帧 {frame_idx}"
        self._info_label.setText(text)

    def current_frame(self):
        return self._current_frame

    def has_data(self):
        return self._joint_angles is not None and len(self._joint_angles) > 0
