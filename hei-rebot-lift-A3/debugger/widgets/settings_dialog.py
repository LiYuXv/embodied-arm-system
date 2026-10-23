"""系统设置对话框：分页设置，保存到 settings/main_system_config.json"""

import os

from PyQt6.QtWidgets import (
    QDialog, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QCheckBox, QComboBox, QLabel, QPushButton, QMessageBox,
    QFormLayout, QDoubleSpinBox,
)
from PyQt6.QtCore import pyqtSignal

from debugger.utils.system_config import (
    load_config, save_config, CAMERA_ROLES, CAMERA_ROLE_NAMES,
)

CAN_BITRATE_OPTIONS = [
    ("1M", 1000000),
    ("500K", 500000),
    ("250K", 250000),
    ("125K", 125000),
]


def _list_video_usb_ports():
    """扫描 /sys/class/video4linux/，返回 [(video_idx, usb_id)]，仅保留 USB 摄像头。

    仅读 sysfs 拓扑信息，不打开设备，可与主窗口已打开的摄像头共存，
    不干扰采集。
    """
    base = "/sys/class/video4linux"
    ports = []
    if not os.path.isdir(base):
        return ports
    for name in sorted(os.listdir(base)):
        if not name.startswith("video"):
            continue
        try:
            idx = int(name[5:])
        except ValueError:
            continue
        dev = os.path.join(base, name, "device")
        usb_id = None
        if os.path.islink(dev):
            real = os.path.realpath(dev)
            if "/usb" in real:
                usb_id = os.path.basename(os.readlink(dev))
        if usb_id:
            ports.append((idx, usb_id))
    return ports


class SettingsDialog(QDialog):
    """系统设置对话框（分页：常规设置 + 摄像头设置）"""

    settings_saved = pyqtSignal(dict)

    def __init__(self, parent=None, camera_ports=None):
        super().__init__(parent)
        self.setWindowTitle("设置")
        self.setMinimumWidth(420)

        # 摄像头端口列表 [(video_idx, usb_id)]；由主窗口传入 CameraWidget 的
        # 检测结果，保证与主界面显示的摄像头串口号一致；未传入时回退到
        # sysfs 扫描。
        self._camera_ports = camera_ports

        root = QVBoxLayout(self)

        self.tabs = QTabWidget()
        root.addWidget(self.tabs)

        # ---- 分页 1：常规设置 ----
        tab = QWidget()
        form = QFormLayout(tab)
        form.setContentsMargins(10, 10, 10, 10)
        form.setSpacing(8)

        self.auto_connect_cb = QCheckBox("启动后自动连接串口（连接成功后自动使能）")
        form.addRow(self.auto_connect_cb)

        bitrate_row = QHBoxLayout()
        self.bitrate_combo = QComboBox()
        for label, val in CAN_BITRATE_OPTIONS:
            self.bitrate_combo.addItem(label, val)
        self.bitrate_combo.setCurrentIndex(1)  # 默认 500K
        bitrate_row.addWidget(self.bitrate_combo)
        bitrate_row.addStretch()
        form.addRow(QLabel("CAN 串口控制波特率:"), bitrate_row)

        self.tabs.addTab(tab, "常规设置")

        # ---- 分页 2：摄像头设置 ----
        self._build_camera_tab()

        # ---- 分页 3：VR 控制默认设置 ----
        self._build_vr_tab()

        # ---- 底部按钮 ----
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.save_btn = QPushButton("保存")
        self.save_btn.setFixedWidth(80)
        self.save_btn.clicked.connect(self._on_save)
        btn_row.addWidget(self.save_btn)
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.setFixedWidth(80)
        self.cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(self.cancel_btn)
        root.addLayout(btn_row)

        self._load()

    def _build_camera_tab(self):
        tab = QWidget()
        lay = QVBoxLayout(tab)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)

        info = QLabel(
            "将每个 USB 摄像头端口绑定到对应角色（臂1腕部 / 头部 / 臂2腕部）。\n"
            "同一角色不能被多个端口占用；保存后立即生效，重启后按 USB 端口自动匹配，\n"
            "摄像头插拔后对应关系不会错乱。")
        info.setWordWrap(True)
        lay.addWidget(info)

        # 每一行：端口标签 + 角色下拉框
        self.cam_rows = []  # (usb_id, video_idx, QComboBox)
        ports = self._camera_ports or _list_video_usb_ports()
        for idx, usb_id in ports[:3]:
            row = QHBoxLayout()
            row.setSpacing(6)
            lbl = QLabel(f"usb {usb_id}  (/dev/video{idx})")
            lbl.setMinimumWidth(200)
            combo = QComboBox()
            combo.addItem("未分配", None)
            for role in CAMERA_ROLES:
                combo.addItem(CAMERA_ROLE_NAMES[role], role)
            row.addWidget(lbl)
            row.addWidget(combo, 1)
            lay.addLayout(row)
            self.cam_rows.append((usb_id, idx, combo))

        if not self.cam_rows:
            lay.addWidget(QLabel("未检测到 USB 摄像头（请确认摄像头已连接）"))
        lay.addStretch()
        self.tabs.addTab(tab, "摄像头设置")

    def _build_vr_tab(self):
        tab = QWidget()
        form = QFormLayout(tab)
        form.setContentsMargins(10, 10, 10, 10)
        form.setSpacing(8)

        info = QLabel(
            "VR 控制面板的默认设置，保存后即时应用；运行时仍可在 VR 面板内临时调整。")
        info.setWordWrap(True)
        form.addRow(info)

        self.vr_lift_rpm_spin = QDoubleSpinBox()
        self.vr_lift_rpm_spin.setRange(10.0, 1000.0)
        self.vr_lift_rpm_spin.setSingleStep(10.0)
        self.vr_lift_rpm_spin.setDecimals(0)
        self.vr_lift_rpm_spin.setSuffix(" RPM")
        form.addRow(QLabel("升降转速:"), self.vr_lift_rpm_spin)

        self.vr_gripper_speed_spin = QDoubleSpinBox()
        self.vr_gripper_speed_spin.setRange(0.1, 5.0)
        self.vr_gripper_speed_spin.setSingleStep(0.1)
        self.vr_gripper_speed_spin.setDecimals(1)
        self.vr_gripper_speed_spin.setSuffix(" rad/s")
        form.addRow(QLabel("夹爪速度:"), self.vr_gripper_speed_spin)

        self.vr_scale_spin = QDoubleSpinBox()
        self.vr_scale_spin.setRange(0.1, 3.0)
        self.vr_scale_spin.setSingleStep(0.1)
        self.vr_scale_spin.setDecimals(1)
        self.vr_scale_spin.setSuffix(" x")
        form.addRow(QLabel("运动缩放:"), self.vr_scale_spin)

        self.vr_hmd_pos_scale_spin = QDoubleSpinBox()
        self.vr_hmd_pos_scale_spin.setRange(0.2, 5.0)
        self.vr_hmd_pos_scale_spin.setSingleStep(0.1)
        self.vr_hmd_pos_scale_spin.setDecimals(1)
        self.vr_hmd_pos_scale_spin.setSuffix(" x")
        form.addRow(QLabel("头显位置缩放:"), self.vr_hmd_pos_scale_spin)

        self.vr_hmd_max_lin_spin = QDoubleSpinBox()
        self.vr_hmd_max_lin_spin.setRange(0.05, 1.00)
        self.vr_hmd_max_lin_spin.setSingleStep(0.05)
        self.vr_hmd_max_lin_spin.setDecimals(2)
        self.vr_hmd_max_lin_spin.setSuffix(" m/s")
        form.addRow(QLabel("头显速度跟随:"), self.vr_hmd_max_lin_spin)

        self.tabs.addTab(tab, "VR 控制")

    def _load(self):
        cfg = load_config()
        self.auto_connect_cb.setChecked(
            bool(cfg.get("auto_connect_on_startup", True)))
        bitrate = int(cfg.get("can_bitrate", 500000))
        for i in range(self.bitrate_combo.count()):
            if self.bitrate_combo.itemData(i) == bitrate:
                self.bitrate_combo.setCurrentIndex(i)
                break
        self._load_camera_tab(cfg)
        self._load_vr_tab(cfg)

    def _load_camera_tab(self, cfg):
        roles = cfg.get("camera_roles")
        roles = dict(roles) if isinstance(roles, dict) else {}
        for usb_id, idx, combo in self.cam_rows:
            combo.setCurrentIndex(0)  # 默认未分配
            for role, info in roles.items():
                if isinstance(info, dict) and info.get("usb") == usb_id:
                    i = combo.findData(role)
                    if i >= 0:
                        combo.setCurrentIndex(i)
                    break

    def _load_vr_tab(self, cfg):
        d = cfg.get("vr_defaults")
        d = dict(d) if isinstance(d, dict) else {}
        self.vr_lift_rpm_spin.setValue(float(d.get("lift_speed_rpm", 200.0)))
        self.vr_gripper_speed_spin.setValue(float(d.get("gripper_speed_rad_s", 2.0)))
        self.vr_scale_spin.setValue(float(d.get("scale", 1.0)))
        self.vr_hmd_pos_scale_spin.setValue(float(d.get("hmd_pos_scale", 1.0)))
        self.vr_hmd_max_lin_spin.setValue(float(d.get("hmd_max_lin", 0.30)))

    def _collect_vr_defaults(self):
        return {
            "lift_speed_rpm": self.vr_lift_rpm_spin.value(),
            "gripper_speed_rad_s": self.vr_gripper_speed_spin.value(),
            "scale": self.vr_scale_spin.value(),
            "hmd_pos_scale": self.vr_hmd_pos_scale_spin.value(),
            "hmd_max_lin": self.vr_hmd_max_lin_spin.value(),
        }

    def get_config(self) -> dict:
        return {
            "auto_connect_on_startup": self.auto_connect_cb.isChecked(),
            "can_bitrate": self.bitrate_combo.currentData() or 500000,
        }

    def _collect_camera_roles(self):
        """从界面收集摄像头角色绑定，返回 (roles_dict 或 None, error_msg 或 None)。

        规则：当前显示的端口上旧绑定先移除、按新选择重建；缺席端口的旧绑定
        保留不动；同一角色在本次选择中不允许重复。
        """
        cfg = load_config()
        existing = dict(cfg.get("camera_roles") or {})
        shown_usbs = {usb_id for usb_id, _, _ in self.cam_rows}
        # 移除当前显示端口上的旧绑定（准备重分配），保留缺席端口绑定
        for role, info in list(existing.items()):
            if isinstance(info, dict) and info.get("usb") in shown_usbs:
                existing.pop(role, None)

        assigned = []
        for usb_id, idx, combo in self.cam_rows:
            role = combo.currentData()
            if role:
                assigned.append((role, usb_id, idx))

        seen = {}
        for role, usb_id, idx in assigned:
            if role in seen:
                return None, (
                    f"角色「{CAMERA_ROLE_NAMES[role]}」被多个端口占用"
                    f"（usb {seen[role]} 与 usb {usb_id}），请调整后再保存")
            seen[role] = usb_id

        for role, usb_id, idx in assigned:
            existing[role] = {"usb": usb_id, "index": idx}
        return existing, None

    def _on_save(self):
        roles, err = self._collect_camera_roles()
        if err:
            QMessageBox.warning(self, "摄像头设置", err)
            return
        cfg = load_config()
        cfg.update(self.get_config())
        cfg["camera_roles"] = roles
        cfg["vr_defaults"] = self._collect_vr_defaults()
        if save_config(cfg):
            self.settings_saved.emit(cfg)
            self.accept()
        else:
            QMessageBox.warning(self, "保存失败", "配置文件写入失败，请检查文件权限")
