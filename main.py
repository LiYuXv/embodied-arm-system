"""
RobStride 电机控制界面 - 实现版本
连接 UI 与 EL-A3 SDK
"""

import sys
import math
import time
import threading
from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtCore import QTimer

from el_a3_sdk import ELA3Interface, RunMode
from main_UI import MotorControlGUI, SystemSettingsDialog


class MotorControlApp(MotorControlGUI):
    """带实现的电机控制界面"""

    def __init__(self):
        self.arm: ELA3Interface = None
        self._status_timer: QTimer = None
        self._poll_timer: QTimer = None
        self._connected = False
        self._enabled = False
        self._current_mode = "运控模式"
        self._slider_control_enabled = False

        super().__init__()

        self._status_timer = QTimer()
        self._status_timer.timeout.connect(self.update_status_display)
        self._status_timer.start(100)

        self._poll_timer = QTimer()
        self._poll_timer.timeout.connect(self.poll_motor_feedback)
        self._poll_timer.start(50)

    def on_connect_clicked(self):
        """连接按钮点击"""
        if not self._connected:
            try:
                interface_type = self.system_config.get("interface_type", "can")
                port = self.system_config.get("default_port", "can0")

                if interface_type == "serial":
                    self.arm = ELA3Interface(
                        can_name=port,
                        backend="slcan",
                        serial_port=port,
                        serial_baudrate=self.system_config.get("default_baudrate", 921600),
                    )
                else:
                    self.arm = ELA3Interface(can_name=port)

                self.arm.ConnectPort()
                self._connected = True
                self.log("✓ 已连接到 CAN 总线")

                self.enable_btn.setEnabled(True)
                self.disable_btn.setEnabled(True)
                self.zero_btn.setEnabled(True)
                self.clear_error_btn.setEnabled(True)
                self.pp_home_btn.setEnabled(True)
                self.home_btn.setEnabled(False)
                self.mode_tabs.setEnabled(True)
                self.switch_mode_btn.setEnabled(True)

                self.connect_btn.setText("断开")
                self.connect_btn.setObjectName("danger")

                self.conn_status_label.setText("● 已连接")
                self.conn_status_label.setStyleSheet("color: green; font-weight: bold;")

                for i in range(self.NUM_MOTORS):
                    self.motor_status_labels[i]["enabled"].setText("○")
                    self.motor_status_labels[i]["enabled"].setStyleSheet("color: gray;")

                self.log("提示: 连接成功，可查看各关节信息，使能后可控制电机")

            except Exception as e:
                self.log(f"✗ 连接失败: {e}")
                QMessageBox.warning(self, "连接错误", f"无法连接: {e}")
        else:
            try:
                if self.arm:
                    if self._enabled:
                        self.arm.DisableArm()
                    self.arm.DisconnectPort()
                self._connected = False
                self._enabled = False
                self.log("✓ 已断开连接")

                self.enable_btn.setEnabled(False)
                self.disable_btn.setEnabled(False)
                self.zero_btn.setEnabled(False)
                self.clear_error_btn.setEnabled(False)
                self.pp_home_btn.setEnabled(False)
                self.mode_tabs.setEnabled(False)
                self.switch_mode_btn.setEnabled(False)
                self.start_btn.setEnabled(False)
                self.stop_btn.setEnabled(False)

                self.connect_btn.setText("🔌 连接")
                self.connect_btn.setObjectName("success")

                self.conn_status_label.setText("● 未连接")
                self.conn_status_label.setStyleSheet("color: red; font-weight: bold;")

            except Exception as e:
                self.log(f"✗ 断开失败: {e}")

    def on_enable_clicked(self):
        """使能按钮点击"""
        if self._connected and self.arm:
            try:
                self.arm.EnableArm()
                self._enabled = True
                self.log("✓ 电机已使能 (运控模式)")
                
                for i in range(self.NUM_MOTORS):
                    self.motor_status_labels[i]["enabled"].setText("●")
                    self.motor_status_labels[i]["enabled"].setStyleSheet("color: green; font-weight: bold;")

                self.home_btn.setEnabled(True)
                self.pp_home_btn.setEnabled(True)
                #self.on_home_clicked()

                self.update_motor_control_buttons(True)
            except Exception as e:
                self.log(f"✗ 使能失败: {e}")

    def on_home_clicked(self):
        """回到初始位置 (运控模式) - 所有关节回零"""
        
        if self._connected and self.arm and self._enabled:
            try:
                self.arm.EnableArm(run_mode=RunMode.MOTION_CONTROL)
                self._enabled = True
                self.log("✓ 已切换到运动模式")
                if not self.arm.control_loop_running:
                    self.arm.start_control_loop(rate_hz=200.0)
                    self.log("✓ 控制循环已启动 (200Hz)")

                self.arm.JointCtrl(0.0, 0.0, -0.3, 0.0, 0.0, 0.0)
                self.log("✓ 回到初始位置: 所有关节归零")
            except Exception as e:
                self.log(f"✗ 失败: {e}")

    def on_pp_home_clicked(self):
        """PP模式下所有关节回零"""
        if self._connected and self.arm and self._enabled:
            try:
                self.arm.EnableArm(run_mode=RunMode.POSITION_PP)
                self._enabled = True
                self.log("✓ 已切换到PP模式")

                speed = self.pos_speed_spin.value()
                acc = self.pos_acc_spin.value()
                driver = self.arm._driver
                for motor_id in range(1, 7):
                    driver.set_pp_velocity(motor_id, speed)
                    driver.set_pp_acceleration(motor_id, acc)
                    driver.set_position_pp(motor_id, 0.0)
                self.log(f"✓ PP回零: 所有关节回0, 速度={speed}, 加速度={acc}")
            except Exception as e:
                self.log(f"✗ PP回零失败: {e}")

    def on_disable_clicked(self):
        """失能按钮点击"""
        if self._connected and self.arm:
            try:
                self.arm.stop_control_loop()
                self.arm.DisableArm()
                self._enabled = False
                self.log("✓ 电机已失能")
                self.update_motor_control_buttons(False)

                for i in range(self.NUM_MOTORS):
                    self.motor_status_labels[i]["enabled"].setText("○")
                    self.motor_status_labels[i]["enabled"].setStyleSheet("color: gray;")

                self.home_btn.setEnabled(False)
                self.pp_home_btn.setEnabled(False)
            except Exception as e:
                self.log(f"✗ 失能失败: {e}")

    def update_motor_control_buttons(self, enabled: bool):
        """更新电机控制按钮状态"""
        self.start_btn.setEnabled(enabled)
        self.stop_btn.setEnabled(enabled)
        self.slider_control_check.setEnabled(enabled)

        mode_index = self.mode_combo.currentIndex()
        if mode_index in [1, 2] and enabled:
            for i in range(self.NUM_MOTORS):
                self.motor_minus_btns[i].setEnabled(True)
                self.motor_plus_btns[i].setEnabled(True)

    def on_zero_clicked(self):
        """置零按钮点击"""
        if self._connected and self.arm:
            try:
                self.arm.SetZeroPosition()
                self.log("✓ 零位已设置")
            except Exception as e:
                self.log(f"✗ 置零失败: {e}")

    def on_clear_error_clicked(self):
        """清错按钮点击"""
        if self._connected and self.arm:
            try:
                self.arm.ClearError()
                self.log("✓ 错误已清除")
            except Exception as e:
                self.log(f"✗ 清错失败: {e}")

    def on_mode_changed(self, index):
        """模式选择改变"""
        super().on_mode_changed(index)
        mode_names = [
            "运控模式", "位置PP模式", "位置CSP模式",
            "速度模式", "电流模式", "JOG模式"
        ]
        self._current_mode = mode_names[index] if index < len(mode_names) else "未知"
        self.current_mode_label.setText(self._current_mode)

    def on_switch_mode_clicked(self):
        """切换模式按钮点击"""
        if self._connected and self.arm:
            mode_index = self.mode_combo.currentIndex()
            mode_names = [
                "运控模式 (Motion Control)",
                "位置模式 - PP (Profile Position)",
                "位置模式 - CSP (Cyclic Synchronous Position)",
                "速度模式 (Speed)",
                "电流模式 (Current)",
                "JOG 模式",
            ]
            try:
                self.arm.SwitchMode(mode_index)
                self.log(f"✓ 已切换到 {mode_names[mode_index]}")
            except Exception as e:
                self.log(f"✗ 切换模式失败: {e}")

    def on_start_clicked(self):
        """启动运动按钮点击"""
        if not self._connected or not self.arm:
            return

        mode_index = self.mode_combo.currentIndex()

        if mode_index == 0:
            pos = self.motion_pos_spin.value()
            vel = self.motion_vel_spin.value()
            torque = self.motion_torque_spin.value()
            kp = self.motion_kp_spin.value()
            kd = self.motion_kd_spin.value()
            try:
                self.arm.JointCtrl(
                    pos, pos, pos, pos, pos, pos,
                    kp=kp, kd=kd, torque_ff=torque
                )
                self.log(f"运控模式: 位置={pos}, 速度={vel}")
            except Exception as e:
                self.log(f"✗ 运动控制失败: {e}")

        elif mode_index in [1, 2]:
            target = self.pos_target_spin.value() if mode_index == 1 else self.csp_target_spin.value()
            speed = self.pos_speed_spin.value()
            acc = self.pos_acc_spin.value()
            try:
                self.arm.MoveJ(
                    [target] * 6,
                    duration=speed / acc if acc > 0 else 2.0
                )
                self.log(f"位置模式: 目标={target}rad, 速度={speed}")
            except Exception as e:
                self.log(f"✗ 运动失败: {e}")

    def on_stop_clicked(self):
        """停止运动按钮点击"""
        if self._connected and self.arm:
            try:
                self.arm.cancel_motion()
                self.log("✓ 运动已停止")
            except Exception as e:
                self.log(f"✗ 停止失败: {e}")

    def on_get_position_clicked(self, spin_box):
        """获取当前位置按钮点击"""
        if self._connected and self.arm:
            try:
                states = self.arm.GetArmJointMsgs()
                if states:
                    pos = states.j1
                    spin_box.setValue(pos)
                    self.log(f"获取位置: {pos:.3f} rad")
            except Exception as e:
                self.log(f"✗ 获取位置失败: {e}")

    def on_speed_slider_changed(self):
        """速度滑块改变"""
        value = self.speed_slider.value() / 10.0
        self.speed_target_spin.setValue(value)
        self.log(f"速度滑块值改变: {value:.1f}")

    def on_current_slider_changed(self):
        """电流滑块改变"""
        value = self.current_slider.value() / 100.0
        self.current_target_spin.setValue(value)
        self.log(f"电流滑块值改变: {value:.2f}")

    def on_jog_clicked(self, direction):
        """JOG按钮点击"""
        if not self._connected or not self.arm:
            return
        speed = self.jog_speed_slider.value() / 100.0 * 10.0
        try:
            if direction == "CW":
                self.arm.JointCtrl(speed, speed, speed, speed, speed, speed)
            else:
                self.arm.JointCtrl(-speed, -speed, -speed, -speed, -speed, -speed)
            self.log(f"JOG {direction}: 速度={speed:.1f}")
        except Exception as e:
            self.log(f"✗ JOG失败: {e}")

    def on_jog_released(self):
        """JOG按钮释放"""
        if self._connected and self.arm and self._enabled:
            try:
                self.arm.JointCtrl(0, 0, 0, 0, 0, 0)
            except:
                pass

    def on_slider_control_changed(self, state):
        """滑条控制开关改变"""
        is_enabled = (state == 2)
        self._slider_control_enabled = is_enabled

        if is_enabled:
            mode_index = self.mode_combo.currentIndex()
            if mode_index not in [1, 2]:
                self.slider_control_check.setChecked(False)
                self.log("滑条控制仅在PP或CSP模式下可用")
                return
            self.log("开启: 滑条实时控制 (50Hz)")

        for i in range(self.NUM_MOTORS):
            self.motor_sliders[i].setEnabled(is_enabled)
            if is_enabled and self._enabled:
                self.motor_minus_btns[i].setEnabled(True)
                self.motor_plus_btns[i].setEnabled(True)

    def on_motor_slider_changed(self, motor_idx, value):
        """单个电机滑条值改变"""
        self.motor_slider_targets[motor_idx] = value / 1000.0
        self.motor_pos_labels[motor_idx].setText(f"{self.motor_slider_targets[motor_idx]:.3f}")

        if self._slider_control_enabled and self._connected and self.arm:
            target = [0.0] * 6
            target[motor_idx] = self.motor_slider_targets[motor_idx]
            try:
                self.arm.JointCtrl(*target)
            except:
                pass

    def on_motor_step_plus(self, motor_idx):
        """单个电机加按钮点击"""
        step_deg = float(self.step_deg_label.text())
        step_rad = math.radians(step_deg)
        current_value = self.motor_sliders[motor_idx].value()
        new_value = current_value + int(step_rad * 1000)
        new_value = max(-3000, min(3000, new_value))
        self.motor_sliders[motor_idx].setValue(new_value)
        self.motor_slider_targets[motor_idx] = new_value / 1000.0

    def on_motor_step_minus(self, motor_idx):
        """单个电机减按钮点击"""
        step_deg = float(self.step_deg_label.text())
        step_rad = math.radians(step_deg)
        current_value = self.motor_sliders[motor_idx].value()
        new_value = current_value - int(step_rad * 1000)
        new_value = max(-3000, min(3000, new_value))
        self.motor_sliders[motor_idx].setValue(new_value)
        self.motor_slider_targets[motor_idx] = new_value / 1000.0

    def update_status_display(self):
        """更新状态显示"""
        if not self._connected or not self.arm:
            return

        try:
            motor_states = self.arm.GetMotorStates()
            if motor_states:
                for i in range(self.NUM_MOTORS):
                    motor_id = i + 1
                    if motor_id in motor_states:
                        m = motor_states[motor_id]
                        self.motor_status_labels[i]["pos"].setText(f"{m.position:.3f}")
                        self.motor_status_labels[i]["vel"].setText(f"{m.velocity:.2f}")
                        self.motor_status_labels[i]["torque"].setText(f"{m.torque:.2f}")

                        if m.mode_state == 2:
                            self.motor_status_labels[i]["enabled"].setText("●")
                            self.motor_status_labels[i]["enabled"].setStyleSheet("color: green; font-weight: bold;")
                        else:
                            self.motor_status_labels[i]["enabled"].setText("○")
                            self.motor_status_labels[i]["enabled"].setStyleSheet("color: gray;")

                joint_states = self.arm.GetArmJointMsgs()
                if joint_states:
                    self.motor_pos_labels[0].setText(f"{joint_states.j1:.3f}")
                    self.motor_pos_labels[1].setText(f"{joint_states.j2:.3f}")
                    self.motor_pos_labels[2].setText(f"{joint_states.j3:.3f}")
                    self.motor_pos_labels[3].setText(f"{joint_states.j4:.3f}")
                    self.motor_pos_labels[4].setText(f"{joint_states.j5:.3f}")
                    self.motor_pos_labels[5].setText(f"{joint_states.j6:.3f}")
                    if self.NUM_MOTORS > 6:
                        self.motor_pos_labels[6].setText(f"{joint_states.j7:.3f}")

        except Exception as e:
            pass

    def poll_motor_feedback(self):
        """定时发送运控命令触发电机反馈"""
        if not self._connected or not self.arm:
            return

        try:
            driver = self.arm._driver
            for motor_id in range(1, self.NUM_MOTORS + 1):
                driver.send_motion_control(motor_id, 0.0, 0.0, 0.0, 0.0, 0.0)
                time.sleep(0.0002)
        except Exception as e:
            pass

    def log(self, message: str):
        """写入日志"""
        self.log_text.append(f"[{time.strftime('%H:%M:%S')}] {message}")


def main():
    app = QApplication(sys.argv)
    window = MotorControlApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()