# -*- coding: utf-8 -*-
"""
实时正运动学和重力补偿显示模块
支持6轴机械臂，考虑关节安装方向
根据robot_settings.json确定质量和质心
"""

import numpy as np
import math
import json
import os
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QGroupBox, QFormLayout,
    QCheckBox, QHBoxLayout, QGridLayout, QScrollArea
)
from PyQt5.QtCore import Qt, QTimer

import transformations as tf


class RealTimeKinematicsDisplay(QWidget):
    """实时正运动学和重力补偿显示面板"""
    
    def __init__(self, robot, joint_pos_dict, viewer=None, parent=None):
        """
        初始化
        参数:
            robot: URDFRobot实例
            joint_pos_dict: 关节位置字典
            viewer: RobotViewer3D实例（可选）
            parent: 父窗口
        """
        super().__init__(parent)
        self.robot = robot
        self.joint_pos_dict = joint_pos_dict
        self.viewer = viewer
        
        # 检测机械臂类型
        self.dof = self._detect_dof()
        self.is_6dof = (self.dof == 6)
        
        # 加载动力学参数和关节方向
        self._load_robot_settings()
        
        # 设置UI
        self.setup_ui()
        
        # 设置定时器用于实时更新
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_display)
        self.update_timer.start(100)  # 100ms更新一次
    
    def _detect_dof(self):
        """检测机械臂自由度"""
        count = 0
        for jn in self.robot.joint_order:
            jd = self.robot.joints[jn]
            if jd["type"] in ("revolute", "continuous", "prismatic"):
                count += 1
        return count
    
    def _load_robot_settings(self):
        """从robot_settings.json加载动力学参数和关节方向"""
        self.joint_z_directions = np.ones(6, dtype=float)  # 默认所有关节正向
        self.link_masses = np.array([0.1] * 6, dtype=float)  # 默认质量
        self.link_coms = np.zeros((6, 3), dtype=float)  # 默认质心在原点
        self.mdh_params = None  # MDH参数
        
        try:
            settings_path = os.path.join(
                os.path.dirname(__file__), "settings", "robot_settings.json"
            )
            with open(settings_path, "r", encoding="utf-8") as f:
                settings = json.load(f)
            
            # 加载关节Z轴方向（安装方向）
            if "joint_z_directions" in settings:
                directions = settings["joint_z_directions"]
                if len(directions) >= 6:
                    self.joint_z_directions = np.array(directions[:6], dtype=float)
                    print(f"[RealTimeKinematics] 加载关节安装方向: {self.joint_z_directions}")
            
            # 加载动力学参数
            if "dynamics_parameters" in settings:
                dynamics = settings["dynamics_parameters"]
                
                # 加载质量
                if "link_masses" in dynamics:
                    masses = dynamics["link_masses"]
                    while len(masses) < 6:
                        masses.append(0.1)
                    self.link_masses = np.array(masses[:6], dtype=float)
                    print(f"[RealTimeKinematics] 加载连杆质量: {self.link_masses}")
                
                # 加载质心位置
                if "link_coms" in dynamics:
                    coms = dynamics["link_coms"]
                    while len(coms) < 6:
                        coms.append([0.0, 0.0, 0.0])
                    self.link_coms = np.array(coms[:6], dtype=float)
                    print(f"[RealTimeKinematics] 加载连杆质心位置")
            
            # 加载MDH参数
            if "dh_params" in settings:
                dh = settings["dh_params"]
                if all(k in dh for k in ["alpha", "a", "d", "theta_offset"]):
                    self.mdh_params = {
                        "alpha": np.array([np.radians(a) for a in dh["alpha"][:6]], dtype=float),
                        "a": np.array(dh["a"][:6], dtype=float),
                        "d": np.array(dh["d"][:6], dtype=float),
                        "theta_offset": np.array([np.radians(t) for t in dh["theta_offset"][:6]], dtype=float)
                    }
                    print(f"[RealTimeKinematics] 加载MDH参数完成")
            
            # 重力向量
            self.gravity = np.array([0, 0, -9.81], dtype=float)
            
        except Exception as e:
            print(f"[RealTimeKinematics] 加载robot_settings.json失败: {e}")
            print(f"[RealTimeKinematics] 使用默认参数")
    
    def _mdh_transform(self, alpha_prev, a_prev, theta, d):
        """
        MDH单步齐次变换（Craig约定）
        T = Rx(alpha_{i-1}) * Tx(a_{i-1}) * Rz(theta_i) * Tz(d_i)
        """
        ca, sa = np.cos(alpha_prev), np.sin(alpha_prev)
        ct, st = np.cos(theta), np.sin(theta)
        return np.array([
            [ct,      -st,      0,      a_prev ],
            [st*ca,    ct*ca,  -sa,    -sa*d   ],
            [st*sa,    ct*sa,   ca,     ca*d   ],
            [0,        0,       0,      1      ]
        ])
    
    def _get_all_frames(self, q):
        """获取所有关节坐标系（基座+J1~J6）"""
        if self.mdh_params is None:
            return None
        
        frames = [np.eye(4)]
        T = np.eye(4)
        for i in range(6):
            # 应用关节安装方向
            theta = self.joint_z_directions[i] * q[i] + self.mdh_params["theta_offset"][i]
            T = T @ self._mdh_transform(
                self.mdh_params["alpha"][i],
                self.mdh_params["a"][i],
                theta,
                self.mdh_params["d"][i]
            )
            frames.append(T.copy())
        return frames
    
    def _compute_gravity_torques(self, q):
        """
        计算重力补偿力矩
        考虑关节安装方向
        """
        if self.mdh_params is None or not self.is_6dof:
            return np.zeros(6)
        
        # 获取所有关节变换矩阵
        frames = self._get_all_frames(q)
        if frames is None:
            return np.zeros(6)
        
        # 计算各连杆质心的世界坐标
        com_positions = []
        for i in range(6):
            R_parent = frames[i][:3, :3]
            o_parent = frames[i][:3, 3]
            com_world = R_parent @ self.link_coms[i] + o_parent
            com_positions.append(com_world)
        com_positions = np.array(com_positions)
        
        # 初始化重力力矩数组
        tau = np.zeros(6, dtype=float)
        
        # 遍历每个连杆i，对所有上游关节j (0..i) 累加力矩贡献
        for i in range(6):
            F_i = self.link_masses[i] * self.gravity  # 连杆i的重力
            p_ci = com_positions[i]  # 连杆i质心的世界坐标
            
            for j in range(i + 1):
                if j == 0:
                    # 关节0（基座）使用世界系
                    z_j = np.array([0.0, 0.0, 1.0])
                    o_j = np.zeros(3)
                else:
                    # 关节j的Z轴和原点
                    z_j = frames[j][:3, 2]
                    o_j = frames[j][:3, 3]
                
                # 力臂：从关节原点到质心
                r = p_ci - o_j
                # 重力产生的力矩：τ = (r × F) · z
                # 补偿力矩需要抵消重力力矩，所以加负号
                tau[j] += -np.dot(np.cross(r, F_i), z_j)
        
        # 索引修正：整体左移一位，确保 [τ1..τ6] 与 [关节1..6] 一一对应
        tau = np.roll(tau, -1)
        
        # 考虑关节安装方向：反向安装的关节需要反转力矩符号
        for i in range(6):
            tau[i] *= self.joint_z_directions[i]
        
        return tau
    
    def setup_ui(self):
        """设置UI界面"""
        layout = QVBoxLayout(self)
        layout.setSpacing(5)
        
        # 标题
        title = QLabel("<b>实时正运动学</b>")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 12px; color: #2196F3;")
        layout.addWidget(title)
        
        # 机械臂类型标签
        self.type_label = QLabel(f"类型: {self.dof}轴{'(支持重力补偿)' if self.is_6dof else '(通用)'})")
        self.type_label.setAlignment(Qt.AlignCenter)
        self.type_label.setStyleSheet("color: #666; font-size: 10px;")
        layout.addWidget(self.type_label)
        
        # 末端执行器位姿显示
        fk_group = QGroupBox("末端执行器位姿")
        fk_layout = QFormLayout(fk_group)
        fk_layout.setSpacing(3)
        
        # 位置
        self.ee_position = QLabel("X: 0.000, Y: 0.000, Z: 0.000")
        self.ee_position.setStyleSheet("font-family: Consolas; font-size: 10px;")
        fk_layout.addRow("位置(m):", self.ee_position)
        
        # 四元数
        self.ee_quaternion = QLabel("[w:1.0000, x:0.0000, y:0.0000, z:0.0000]")
        self.ee_quaternion.setStyleSheet("font-family: Consolas; font-size: 9px;")
        fk_layout.addRow("四元数:", self.ee_quaternion)
        
        # RPY角度
        self.ee_rpy = QLabel("R:0.0°, P:0.0°, Y:0.0°")
        self.ee_rpy.setStyleSheet("font-family: Consolas; font-size: 10px;")
        fk_layout.addRow("RPY(°):", self.ee_rpy)
        
        layout.addWidget(fk_group)
        
        # 重力补偿显示（仅6轴机械臂）
        if self.is_6dof:
            self._setup_gravity_display(layout)
        
        # 自动更新选项
        auto_update_layout = QHBoxLayout()
        self.auto_update = QCheckBox("自动更新")
        self.auto_update.setChecked(True)
        auto_update_layout.addWidget(self.auto_update)
        
        if self.is_6dof:
            self.auto_update_gravity = QCheckBox("重力补偿")
            self.auto_update_gravity.setChecked(True)
            self.auto_update_gravity.setStyleSheet("color: #FF9800;")
            auto_update_layout.addWidget(self.auto_update_gravity)
        
        auto_update_layout.addStretch()
        layout.addLayout(auto_update_layout)
        
        layout.addStretch()
    
    def _setup_gravity_display(self, layout):
        """设置重力补偿显示区域"""
        # 重力补偿力矩显示组
        gravity_group = QGroupBox("重力补偿力矩 (Nm)")
        gravity_layout = QGridLayout(gravity_group)
        gravity_layout.setSpacing(2)
        
        # 关节方向显示
        direction_str = ", ".join([f"J{i+1}:{'+' if d > 0 else '-'}" 
                                   for i, d in enumerate(self.joint_z_directions)])
        self.direction_label = QLabel(f"安装方向: {direction_str}")
        self.direction_label.setStyleSheet("font-size: 9px; color: #666;")
        gravity_layout.addWidget(self.direction_label, 0, 0, 1, 4)
        
        # 创建6个关节力矩显示标签
        self.gravity_labels = []
        for i in range(6):
            row = (i // 3) + 1
            col = (i % 3) * 2
            
            joint_label = QLabel(f"J{i+1}:")
            joint_label.setStyleSheet("font-size: 9px; color: #666;")
            gravity_layout.addWidget(joint_label, row, col)
            
            torque_label = QLabel("0.00")
            torque_label.setStyleSheet("font-family: Consolas; font-size: 10px; color: #FF9800;")
            gravity_layout.addWidget(torque_label, row, col + 1)
            self.gravity_labels.append(torque_label)
        
        # 最大力矩显示
        self.max_torque_label = QLabel("最大: 0.00 Nm")
        self.max_torque_label.setStyleSheet("font-size: 10px; color: #2196F3; font-weight: bold;")
        gravity_layout.addWidget(self.max_torque_label, 3, 0, 1, 4)
        
        layout.addWidget(gravity_group)
        self.gravity_group = gravity_group
    
    def update_display(self):
        """更新显示（正运动学+重力补偿）"""
        if not self.auto_update.isChecked():
            return
        
        try:
            # 使用URDF计算变换矩阵
            self.robot.compute_link_transforms(self.joint_pos_dict)
            
            # 获取末端执行器变换
            T_ee = self._get_end_effector_transform()
            
            if T_ee is not None:
                # 提取位置
                pos = T_ee[:3, 3]
                
                # 提取四元数
                quat = self._rotation_matrix_to_quaternion(T_ee[:3, :3])
                
                # 提取RPY角度
                rpy = tf.euler_from_matrix(T_ee, axes='sxyz')
                
                # 更新正运动学显示
                self.ee_position.setText(
                    f"X:{pos[0]:.3f}, Y:{pos[1]:.3f}, Z:{pos[2]:.3f}"
                )
                self.ee_quaternion.setText(
                    f"[w:{quat[0]:.4f}, x:{quat[1]:.4f}, y:{quat[2]:.4f}, z:{quat[3]:.4f}]"
                )
                self.ee_rpy.setText(
                    f"R:{np.degrees(rpy[0]):.1f}°, P:{np.degrees(rpy[1]):.1f}°, Y:{np.degrees(rpy[2]):.1f}°"
                )
            
            # 更新重力补偿显示
            if self.is_6dof and hasattr(self, 'auto_update_gravity'):
                if self.auto_update_gravity.isChecked():
                    self._update_gravity_display()
        
        except Exception as e:
            print(f"更新显示时出错: {e}")
    
    def _update_gravity_display(self):
        """更新重力补偿力矩显示"""
        try:
            # 从joint_pos_dict提取6个关节角度
            q = self._extract_joint_angles()
            
            # 计算重力补偿力矩
            gravity_torques = self._compute_gravity_torques(q)
            
            # 更新显示
            max_torque = 0.0
            max_index = 0
            for i in range(6):
                torque = gravity_torques[i]
                self.gravity_labels[i].setText(f"{torque:6.2f}")
                
                # 根据力矩大小设置颜色
                if abs(torque) > 5.0:
                    self.gravity_labels[i].setStyleSheet("font-family: Consolas; font-size: 10px; color: #F44336;")
                elif abs(torque) > 2.0:
                    self.gravity_labels[i].setStyleSheet("font-family: Consolas; font-size: 10px; color: #FF9800;")
                else:
                    self.gravity_labels[i].setStyleSheet("font-family: Consolas; font-size: 10px; color: #4CAF50;")
                
                if abs(torque) > abs(max_torque):
                    max_torque = torque
                    max_index = i
            
            # 更新最大力矩显示
            self.max_torque_label.setText(f"最大: {max_torque:.2f} Nm (J{max_index + 1})")
            
        except Exception as e:
            print(f"更新重力补偿显示时出错: {e}")
    
    def _extract_joint_angles(self):
        """从joint_pos_dict提取6个关节角度"""
        q = np.zeros(6)
        count = 0
        
        for jn in self.robot.joint_order:
            if count >= 6:
                break
            jd = self.robot.joints[jn]
            if jd["type"] in ("revolute", "continuous"):
                q[count] = self.joint_pos_dict.get(jn, 0.0)
                count += 1
        
        return q
    
    def _get_end_effector_transform(self):
        """获取末端执行器变换矩阵"""
        end_link = None
        count = 0
        
        for jn in self.robot.joint_order:
            jd = self.robot.joints[jn]
            if jd["type"] in ("revolute", "continuous", "prismatic"):
                count += 1
                end_link = jd["child"]
        
        if end_link and end_link in self.robot.link_T:
            return self.robot.link_T[end_link]
        
        return None
    
    def _rotation_matrix_to_quaternion(self, R):
        """旋转矩阵转四元数 [w, x, y, z]"""
        trace = R[0, 0] + R[1, 1] + R[2, 2]
        
        if trace > 0:
            s = 0.5 / math.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        
        return np.array([w, x, y, z])
    
    def stop_update(self):
        """停止定时更新"""
        if self.update_timer.isActive():
            self.update_timer.stop()
    
    def start_update(self):
        """开始定时更新"""
        if not self.update_timer.isActive():
            self.update_timer.start(100)
    
    def set_update_interval(self, interval_ms):
        """设置更新间隔"""
        self.update_timer.setInterval(interval_ms)
