# -*- coding: utf-8 -*-
"""公共基础模块：sys.path 初始化、日志、matplotlib 中文字体、逆运动学(IK)单例与关节角反解。"""
import sys
import time
import math
import logging
import threading
import warnings
import numpy as np
from pathlib import Path


# ==================== 运行路径初始化（sys.path） ====================
_SDK_ROOT = Path(__file__).resolve().parent.parent.parent

if str(_SDK_ROOT) not in sys.path:
    sys.path.insert(0, str(_SDK_ROOT))

_DBG_DIR = Path(__file__).resolve().parent

if str(_DBG_DIR) not in sys.path:
    sys.path.insert(0, str(_DBG_DIR))

_DEBUGGER_DIR = _DBG_DIR.parent

if str(_DEBUGGER_DIR) not in sys.path:
    sys.path.insert(0, str(_DEBUGGER_DIR))

_DEFAULT_DATA_ROOT = str(_DEBUGGER_DIR / "lerobot_data")


# ==================== 日志 ====================
logger = logging.getLogger("debugger")


# ==================== Matplotlib 嵌入 PyQt6 ====================
import matplotlib

matplotlib.use('QtAgg')

import matplotlib.font_manager as fm

def _setup_matplotlib_chinese_font():
    """挑选系统中支持中文的字体，避免图例/标题中文显示为方框。"""
    candidates = [
        'WenQuanYi Micro Hei',
        'Noto Sans CJK SC',
        'Source Han Sans SC',
        'SimHei',
        'Microsoft YaHei',
        'PingFang SC',
        'Heiti SC',
        'Noto Sans CJK JP',
        'Noto Serif CJK JP',
        'AR PL UMing CN',
        'AR PL UKai CN',
        'Droid Sans Fallback',
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    chosen = None
    for name in candidates:
        if name in available:
            chosen = name
            break

    # 设置 sans-serif 字体列表：优先中文候选，其次 DejaVu Sans 兜底
    if chosen:
        matplotlib.rcParams['font.sans-serif'] = [chosen, 'DejaVu Sans']
    else:
        matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False

    # 抑制字体缺字警告（使用 font fallback 时仍可能产生大量噪音）
    warnings.filterwarnings(
        'ignore',
        message='missing from font',
        category=UserWarning,
    )

_setup_matplotlib_chinese_font()


# ==================== 逆运动学 (IK) 与 3D 机械臂可视化 ====================
_KIN = None

try:
    from el_a3_sdk.kinematics import ELA3Kinematics
    from el_a3_sdk.data_types import ArmEndPose
    _KIN = ELA3Kinematics()
    logger.info("ELA3Kinematics 加载成功，机械臂关节反解可用")
except Exception as _ik_err:
    logger.warning(f"未加载逆运动学模块，机械臂反解不可用: {_ik_err}")

_KIN_LOCK = threading.Lock()

ARM_JOINT_NAMES = [
    "L1_joint", "L2_joint", "L3_joint",
    "L4_joint", "L5_joint", "L6_joint", "L7_joint",
]

def quat_to_rpy_recording(qw, qx, qy, qz):
    """与录制端 euler_to_quat（XYZ 内旋，R=Rx·Ry·Rz）配对的四元数→欧拉角解算。

    训练数据中的四元数 [qx, qy, qz, qw] 由 FK 的 rpy 经 XYZ 内旋编码得到，
    必须用同一约定解出 rpy，才能正确反求关节角度（与 camera_pyqt 中一致）。
    返回 (rx, ry, rz)，单位 rad。
    """
    m = np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])
    ry = math.asin(float(np.clip(m[0, 2], -1.0, 1.0)))
    rx = math.atan2(float(-m[1, 2]), float(m[2, 2]))
    rz = math.atan2(float(-m[0, 1]), float(m[0, 0]))
    return rx, ry, rz

def solve_episode_joint_angles(states, cancel_event=None):
    """对整段 episode 的末端位姿序列批量反求关节角度。

    Args:
        states: (N, M) 数组，自动识别两种格式：
          - M>=8 旧格式 [x,y,z,qx,qy,qz,qw,gripper_deg...]
          - M==7 新格式 [x,y,z,rx,ry,rz,gripper_rad]
        cancel_event: 可选 threading.Event；置位时每帧检查并立即放弃
          （切换 episode 时旧反解任务作废，避免 UI 等待线程收尾）

    Returns:
        (N, 7) float 数组，每行 = [L1..L6 rad, L7 夹爪 rad]；IK 失败的帧沿用上一帧解；
        被取消时返回 None
    """
    if _KIN is None or states is None or len(states) == 0:
        return None
    states = np.asarray(states, dtype=float)
    n = len(states)
    result = np.zeros((n, 7), dtype=float)
    q_prev = [0.0] * 6
    is_quat = states.shape[1] >= 8  # 旧四元数格式
    # 批量反解时降低 kinematics 日志级别，避免个别近奇异帧刷屏（仍沿用上一帧解）
    _kin_logger = logging.getLogger("el_a3_sdk.kinematics")
    _orig_level = _kin_logger.level
    _kin_logger.setLevel(logging.ERROR)
    try:
        for i in range(n):
            # 显式让出 GIL（每帧一次，总开销 <1ms）：pinocchio C++ 绑定调用不
            # 释放 GIL，IK 循环会近乎独占 GIL（实测 1.3s 内其他线程仅被调度
            # 2 次），导致 UI 线程饿死、切换 episode 卡死；让出后 UI 保持响应，
            # 取消标志也能及时被对端置位/被本循环看到
            time.sleep(0)
            if cancel_event is not None and cancel_event.is_set():
                return None
            row = states[i]
            x, y, z = float(row[0]), float(row[1]), float(row[2])
            if is_quat:
                qx, qy, qz, qw = float(row[3]), float(row[4]), float(row[5]), float(row[6])
                grip_deg = float(row[7]) if row.shape[0] > 7 else 0.0
                rx, ry, rz = quat_to_rpy_recording(qw, qx, qy, qz)
                grip_rad = math.radians(max(-90.0, min(90.0, grip_deg)))
            else:
                rx, ry, rz = float(row[3]), float(row[4]), float(row[5])
                grip_rad = float(row[6]) if row.shape[0] > 6 else 0.0
            try:
                pose = ArmEndPose(x=x, y=y, z=z, rx=rx, ry=ry, rz=rz)
                # pinocchio 的 Model/Data 非线程安全：切换 episode 时被取消的
                # 旧线程可能尚未退出，新线程并发调用会数据竞争，需加锁串行化
                with _KIN_LOCK:
                    # 热启动快速路径：ik_step（实时控制级 Jacobian 迭代，1~3 步
                    # 收敛），实测比全量 inverse_kinematics 快约 9 倍（逐帧链路
                    # 残差 ~0.01 rad，3D 回放显示足够）；残差过大（快速运动/
                    # 近奇异帧）时回退全量 IK 保证正确性
                    q, ik_err = _KIN.ik_step(pose, q_prev, damping=5e-3, max_step=0.5)
                    if q is None or ik_err > 0.05:
                        q_full = _KIN.inverse_kinematics(pose, q_init=q_prev)
                        if q_full is not None:
                            q = q_full
                if q is None:
                    q = q_prev
                else:
                    q_prev = list(q)
            except Exception:
                q = q_prev
            for j in range(6):
                result[i, j] = q[j]
            result[i, 6] = grip_rad
    finally:
        _kin_logger.setLevel(_orig_level)
    return result
