# VR 控制修改日志

本文件记录 EL-A3 VR（Meta Quest 3）控制相关代码的修改历史。

涉及文件：
- `debugger/widgets/vr_panel.py` — GUI（main.py）VR 控制面板
- `demo/vr_control.py` — 独立命令行 VR 控制程序（参考 `demo/xbox_control.py` 创建）
- `debugger/openarm_mujoco_vr/vr_receiver.py` — Quest 数据接收（本次未改动，保持 MuJoCo 仿真兼容）

---

## 1. 新增独立 VR 控制脚本

**文件**：`demo/vr_control.py`（新建）

参考 `demo/xbox_control.py` 的结构（SDK 接口、Pinocchio IK、2 阶临界阻尼滤波、JointCtrl、MoveJ、急停、零力矩），使用 `vr_receiver.py` 的 `VRControllerReceiver` 接收 Quest 数据，实现相对模式离合控制。

用法：
```bash
python3 demo/vr_control.py              # 默认 can0 / UDP 5006
python3 demo/vr_control.py --scale 1.5  # 运动放大 1.5x
```

---

## 2. 位置方向修正（前后/左右反向）

**原因**：`vr_receiver.py` 的 `_FRAME_ROT` 针对开源 OpenArm 调整，与 EL-A3 基座系相差 180° 偏航，导致手柄移动方向前后、左右都反了。

**修复**：在 EL-A3 控制层（不改 `vr_receiver.py`，避免影响 MuJoCo 仿真）对位置增量做修正：
```python
_POS_FIX = np.array([-1.0, -1.0, 1.0])  # x/y 取反
vr_pos = ctrl.position * _POS_FIX
```

---

## 3. 姿态方向修正（横滚反向、俯仰未映射）

**原因**：接收端 `r_fix = Rz(90)·Rx(180)` 中的 `Rx(180)` 对 EL-A3 末端坐标系不适用。

**修复**：对姿态右乘 `Rx(180)` 抵消该翻转项：
```python
_ROT_FIX = np.diag([1.0, -1.0, -1.0])  # Rx(180)
vr_rot = ctrl.rotation @ _ROT_FIX
```
抵消后映射：俯仰→y 轴、偏航→x 轴、横滚→+z 轴（方向正常）。

---

## 4. 夹爪扳机方向修正

**需求**：松开扳机 → 夹爪张开到 `-1.848`；扣到底 → 夹爪闭合到 `0`。

**修复**：
```python
GRIPPER_OPEN_RAD = -1.848  # 负方向为张开，0 为闭合
angle = (1.0 - trigger) * GRIPPER_OPEN_RAD
# trigger=0 → -1.848（张开），trigger=1 → 0（闭合）
```

---

## 5. 扳机改为无需按住 Grip

**原因**：原逻辑中扳机控制夹爪的代码在"按住 Grip 离合"分支内，松开 Grip 后扳机完全无效。

**修复**：将扳机→夹爪的绝对位置控制移到离合判断之前，只要 VR 已连接就实时生效：
```python
if self._receiver.is_connected():
    trigger = float(np.clip(ctrl.trigger, 0.0, 1.0))
    angle = (1.0 - trigger) * GRIPPER_OPEN_RAD
    if self._gripper_angle is None or abs(angle - self._gripper_angle) > 0.02:
        self._gripper_angle = angle
        self._arm.GripperCtrl(gripper_angle=angle)
```

---

## 6. 横滚/俯仰控制方式调整历史

### 6.1 中间方案（已废弃）：手柄横滚/俯仰 → 关节空间 J4/J5

曾尝试把手柄横滚/俯仰从笛卡尔 IK 改为直接驱动关节 4/5（提取旋转增量的固定轴欧拉角，叠加到 J4/J5，带关节限位保护和每周期限步长）。后因方向反复调整，最终放弃此方案。

### 6.2 最终方案（当前）：横滚/俯仰恢复笛卡尔，摇杆控制关节

**横滚/俯仰**：恢复成原来的笛卡尔模式，经 IK 直接控制末端姿态：
```python
target_rot = self._ee_anchor_rot @ (self._vr_anchor_rot.T @ vr_rot)
```

**摇杆 → 关节空间**：改为速度式直接驱动 J4/J5，覆盖 IK 解：
```python
_JOY_J4_GAIN = 2.0  # 摇杆 Y → 关节4 角速度 (rad/s)
_JOY_J5_GAIN = 2.0  # 摇杆 X → 关节5 角速度 (rad/s)

self._j4_cmd += _JOY_J4_GAIN * joy_y * self._dt
self._j5_cmd += _JOY_J5_GAIN * joy_x * self._dt
# ... 关节限位保护 ...

# IK 求解后覆盖 J4/J5
q_sol[3] = self._j4_cmd
q_sol[4] = self._j5_cmd
```
每次按下 Grip 重新锚定时，`_j4_cmd`/`_j5_cmd` 以当前关节角为基准。

---

## 7. 末端俯仰跟随修正：矩阵 1:1 + 摇杆按需接管 J4/J5

**问题**：6.2 方案中摇杆**始终**用 `_j4_cmd/_j5_cmd` 覆盖 IK 解的 J4/J5，导致即使不碰摇杆，腕关节也被冻结在锚定值 → IK 无法调整腕关节去跟随手柄俯仰，末端俯仰不动作。

**目标**：俯仰走矩阵 1:1 跟随（与横滚/偏航一致），经 IK 直接控制末端姿态；摇杆只在主动推动时才接管 J4/J5。

**修正**（`debugger/widgets/vr_panel.py` 与 `demo/vr_control.py`）：

1. **俯仰/偏航走矩阵 1:1 跟随**（撤回一度尝试的速度累积方案），并修正偏航/俯仰接反：
```python
# 接收端处理后 dR 的 RPY：c[0]=手柄偏航、c[1]=手柄俯仰、c[2]=手柄横滚
dR = self._vr_anchor_rot.T @ vr_rot
c = self._pin.rpy.matrixToRpy(dR)
# 交换 c[0]与c[1] → 手柄偏航→末端偏航、手柄俯仰→末端俯仰、横滚→横滚；
# c[1] 取负纠正俯仰方向
target_rot = self._ee_anchor_rot @ self._pin.rpy.rpyToMatrix(
    float(-c[1]), float(c[0]), float(c[2]))
```
> **验证**：经 Pinocchio 全链路仿真确认，交换后手柄偏航产生世界系垂直轴(Z)旋转=末端偏航，手柄俯仰产生水平轴旋转=末端俯仰（在工具前指的工作位姿下成立）。
> 若某轴方向仍反，可对相应项取负（如偏航反向则 `float(-c[0])`，横滚反向则 `float(-c[2])`）。

2. **摇杆按需接管**：仅当摇杆超出死区时累加并覆盖 J4/J5；摇杆回中时把 J4/J5 交还 IK，使腕关节可自由跟随俯仰/姿态，并同步累加器避免下次推动时跳变：
```python
joy_active = abs(joy_x) > _JOY_DEADZONE or abs(joy_y) > _JOY_DEADZONE
if joy_active:                       # 推动摇杆：累加并覆盖
    self._j4_cmd += _JOY_J4_GAIN * joy_y * self._dt
    self._j5_cmd += _JOY_J5_GAIN * joy_x * self._dt
    # ... 关节限位保护 ...

q_sol, ik_err = self._kin.ik_step(target_pose, self._ik_seed, ...)
if q_sol is not None:
    if joy_active:
        q_sol[3] = self._j4_cmd       # 摇杆接管 J4/J5
        q_sol[4] = self._j5_cmd
    else:
        self._j4_cmd = float(q_sol[3])  # 交还 IK，同步累加器
        self._j5_cmd = float(q_sol[4])
```

**效果**：不碰摇杆时，手柄俯仰 → 矩阵 1:1 → IK 自由调整腕关节 → 末端俯仰实时跟随；推动摇杆时摇杆接管 J4/J5 做腕部微调。

**新增参数**：

| 参数 | 默认 | 说明 |
|------|------|------|
| `_JOY_DEADZONE` | `0.10` | 摇杆激活死区，低于此值视为回中（J4/J5 交还 IK） |

---

## 8. 取消摇杆控臂 + 线程说明（消除卡顿）

**问题**：实测发现手柄控制机械臂卡顿，尤其姿态控制。

**处理**（`debugger/widgets/vr_panel.py` 与 `demo/vr_control.py` 同步）：

1. **完全删除摇杆 → J4/J5 接管逻辑**：移除 `_JOY_J4_GAIN`/`_JOY_J5_GAIN`/`_JOY_DEADZONE` 常量、`_j4_cmd`/`_j5_cmd` 状态、锚定时初始化、`joy_active` 判断及 IK 解覆盖。J4/J5 完全由 IK 决定，末端姿态（偏航/俯仰/横滚）全部走矩阵 1:1 + IK 跟随。
2. **控制循环独立线程（已具备，保留）**：GUI 版在 `VrPanel._start_control` 中启动 `vr_arm_ctrl` 后台线程（100Hz），命令行版在 `main()` 中启动 `ctrl_thread`；VR 接收端 `VRControllerReceiver` 也有独立 UDP 接收线程 + 处理线程。控制循环不与 GUI 主线程争用。

**控制映射总览（右手柄）** 更新为：

| 手柄操作 | 功能 |
|---------|------|
| Grip（侧键）按住 | 离合接合，机械臂跟随手柄运动；松开冻结 |
| 手柄平移 | 末端 XYZ 平移（位置 IK，x/y 已修正） |
| 手柄横滚 | 末端横滚（矩阵 1:1 跟随） |
| 手柄俯仰 | 末端俯仰（矩阵 1:1 跟随） |
| 手柄偏航 | 末端偏航（矩阵 1:1 跟随） |
| 摇杆 Y（上/下） | 夹爪开合（上推=闭合→0，下拉=张开→-1.835，无需按住 Grip） |
| A | 回零位 |
| B | 回工作位（Home） |
| Ctrl+C | 退出（仅 vr_control.py） |

（摇杆不再参与控臂关节，改为控制夹爪。）

---

## 9. 摇杆控制夹爪 + 取消扳机夹爪

**需求**：用手柄摇杆上/下控制夹爪开合（上推=闭合，下拉=张开）；取消扳机控制夹爪；连接时夹爪停留在原来位置，不自动张开；注意夹爪最大/最小限位。

**处理**（`debugger/widgets/vr_panel.py` 与 `demo/vr_control.py` 同步）：

1. **删除扳机→夹爪逻辑**：移除 `angle = (1.0 - trigger) * GRIPPER_OPEN_RAD` 绝对映射块（原逻辑在连接瞬间 `trigger=0` 会立即把夹爪张到全开）。
2. **新增摇杆 Y → 夹爪**（位置速度累积式，钳位在 `[GRIPPER_OPEN_RAD, 0]`）：
```python
joy_y = float(np.clip(ctrl.joystick_y, -1.0, 1.0))
if abs(joy_y) > _JOY_GRIPPER_DEADZONE:
    angle = float(np.clip(
        self._gripper_angle + _JOY_GRIPPER_GAIN * joy_y * self._dt,
        GRIPPER_OPEN_RAD, GRIPPER_CLOSE_RAD))
    self._gripper_angle = angle
    self._arm.GripperCtrl(gripper_angle=angle)
```
3. **夹爪限位**：`GRIPPER_OPEN_RAD = -1.835`（全开，实测值，原 -1.848 超出 SDK 限位 ±1.5708）、`GRIPPER_CLOSE_RAD = 0.0`（全闭）。
4. **连接不突动**：`_initialize()` 以当前夹爪反馈角度为 `_gripper_angle` 基准；摇杆未推动时不发送任何夹爪指令，夹爪停留在原位。

**新增参数**：

| 参数 | 默认 | 说明 |
|------|------|------|
| `GRIPPER_OPEN_RAD` | `-1.835` | 夹爪全开角度（负方向为张开） |
| `GRIPPER_CLOSE_RAD` | `0.0` | 夹爪全闭角度 |
| `_JOY_GRIPPER_GAIN` | `1.0` | 摇杆 Y→夹爪角速度 (rad/s)，方向反了改 `-1.0` |
| `_JOY_GRIPPER_DEADZONE` | `0.10` | 摇杆激活死区，低于此值夹爪保持不动 |

---

## 10. 夹爪力矩限制（力控保位）

**需求**：主程序夹爪面板（`debugger/widgets/gripper_panel.py`）添加力矩限制输入框（默认最大 0.5 Nm、最小 -0.5 Nm）；VR 摇杆夹爪是位置控制（限位 `[GRIPPER_OPEN_RAD, GRIPPER_CLOSE_RAD]`），闭合时若夹爪力矩达到上限，位置将无法继续推进到 0。

**处理**：

1. **夹爪面板新增力矩限制输入框**：`torque_max_spin`（默认 0.5 Nm）/ `torque_min_spin`（默认 -0.5 Nm），改动即发 `torque_limit_changed(max, min)` 信号。
2. **main_window 接线**：`gripper_panel.torque_limit_changed → vr_panel.set_gripper_torque_limits(max, min)`，面板中修改实时同步到 VR 控制器。
3. **VR 摇杆夹爪力矩限制**（`vr_panel.py` 与 `vr_control.py` 同步）：每帧读取 L7 力矩反馈 `GetArmJointEfforts()[6]`，当闭合（joy_y>0）且力矩 ≥ 上限时，目标角度钳位到当前反馈位置（不再继续闭合）；当张开（joy_y<0）且力矩 ≤ 下限时同理。位置控制保位，无法突破力矩限位：
```python
gripper_torque = self._read_gripper_torque()
if gripper_torque is not None:
    if joy_y > 0.0 and gripper_torque >= self.gripper_torque_max:
        angle = min(angle, self._read_gripper_feedback())
    elif joy_y < 0.0 and gripper_torque <= self.gripper_torque_min:
        angle = max(angle, self._read_gripper_feedback())
```
4. **力矩符号约定**：闭合方向力矩为正值（触上限）、张开方向为负值（触下限）。若实测符号相反，交换上限/下限输入值即可。

**参数**：

| 参数 | 默认 | 说明 |
|------|------|------|
| `GRIPPER_TORQUE_MAX` | `0.5` Nm | 夹爪力矩上限：闭合时力矩≥此值停止继续闭合 |
| `GRIPPER_TORQUE_MIN` | `-0.5` Nm | 夹爪力矩下限：张开时力矩≤此值停止继续张开 |

> 面板输入框实时覆盖上述默认值；`demo/vr_control.py` 无界面，直接使用常量。

---

## 11. 夹爪面板"发送"键力矩受限移动

**需求**：夹爪面板的"发送"按键发送目标位置（如发送 0=闭合）时，若夹爪力矩达到上限（默认 0.5 Nm），应停止继续夹闭；张开同理受下限（默认 -0.5 Nm）约束。

**处理**：

1. **面板发送信号携带力矩限制**：`gripper_command` 信号扩展为 `(angle, torque_max, torque_min)`，`_send_command()` 随角度一起发出当前力矩限制输入框的值（全开/全关按钮同样受此约束）。
2. **worker 力矩受限移动**（`debugger/backend/arm_worker.py`）：`_do_gripper_ctrl(angle, torque_max, torque_min)` 在后台线程中逐步逼近目标角度（步进 0.02 rad / 50ms，约 0.4 rad/s），每步读取 L7 反馈位置与力矩：
   - 闭合方向（diff>0）且力矩 ≥ 上限 → 目标钳位到当前反馈位置（保位），停止闭合；
   - 张开方向（diff<0）且力矩 ≤ 下限 → 同样保位停止；
   - 到达目标或超时（30s）结束。
3. **取消机制**：新的夹爪命令、断开连接、worker 停止时自动取消进行中的移动线程；VLA 等其他调用方不传力矩限制时仍为直接 `GripperCtrl`（不受影响）。
4. **力矩符号约定**与 §10 一致：闭合方向力矩为正值（触上限）、张开方向为负值（触下限）。

**参数**：

| 参数 | 默认 | 说明 |
|------|------|------|
| `_GRIPPER_MOVE_STEP_RAD` | `0.02` rad | 单步推进量 |
| `_GRIPPER_MOVE_INTERVAL_S` | `0.05` s | 步进间隔（≈0.4 rad/s） |
| `_GRIPPER_MOVE_TIMEOUT_S` | `30.0` s | 移动总超时保护 |

## 12. 运动学显示面板增加夹爪力矩显示

**需求**：在夹爪弧度值（G: x.xxx rad）的右边加入夹爪力矩显示框。

**处理**：

1. **显示面板**（`debugger/widgets/real_time_kinematics_display.py`）：夹爪显示组在 `rad` 单位右侧新增 `T: [力矩] Nm` 显示（紫色等宽字体，显示到 0.01 Nm）；`update_end_pose()` 新增可选参数 `gripper_torque`，为 `None` 时不刷新。
2. **数据来源**（`debugger/main_window.py`）：新增 `_get_gripper_torque()` 读取 `_last_effort_states.to_list(include_gripper=True)[6]`（L7 夹爪力矩），读不到时返回 `None`；`_on_joint_states_updated` 与 `_on_end_pose_updated` 两处 `update_end_pose()` 调用均传入该值（与面板夹爪角度的力矩来源一致）。
3. 默认参数向后兼容，其他调用方不受影响。

---

## 13. 修复快速闭合夹爪时力矩超调（预减速 + 触限锁定回退）

**问题**：夹爪力矩上限设为 0.5 Nm 时，慢速闭合能正常在 0.5Nm 附近保位；但**快速**闭合时力矩限制失效，力矩冲到 3Nm、夹爪继续闭合，未能及时停住。

**根因**：§10 的力矩限制是纯反应式钳位，存在三处缺陷：
1. **无预减速**：全速闭合 2.0 rad/s，代码只在力矩**已读到** ≥上限时才动作；但力矩反馈有延迟（传感器+CAN+读周期），等读到 0.5Nm 时实际力矩早已更高。
2. **钳位到反馈位置，而反馈本身滞后**：`min(angle, feedback)` 中的 `feedback` 也是延迟值，电机仍继续追这个滞后目标。
3. **无回退、无锁定**：仅"保持位置"无法抵消电机 2 rad/s 的动量；力矩读数瞬时回落后闭合又继续，反复撞击。

**修复**（`debugger/widgets/vr_panel.py` 与 `demo/vr_control.py` 同步），三层加固：

1. **降低闭合速度**：`_JOY_GRIPPER_GAIN` 由 `2.0` 降至 `1.0` rad/s。速度减半后，力矩反馈延迟窗口内的冲量大幅下降，给预减速/触限留出足够反应时间——这是让快速闭合也能受限的必要前提。
2. **预减速软阈值**：力矩达到上限的 `_GRIPPER_TORQUE_SOFT_RATIO`（默认 0.6→0.3Nm）时，推进量按 `(上限-力矩)/(上限-软阈值)` 线性衰减到 0。确保触硬限前速度已趋近 0，从根本消除冲量。
3. **触限锁定保位**：力矩 ≥ 上限（或已锁定）时，锁定该方向（忽略后续同向指令），目标钳位到当前反馈位置保持不动（不回退，避免夹不住物体）。
4. **解锁条件**：摇杆回中（进入死区）或反向操作时清除对应方向锁定，允许用户松开后重新尝试（重新闭合时会再次走预减速，平滑触限）。

```python
if closing:
    soft = torque_max * _GRIPPER_TORQUE_SOFT_RATIO
    if torque >= torque_max or self._gripper_close_locked:
        self._gripper_close_locked = True
        effective_inc = feedback - self._gripper_angle   # 保位，不继续推进、不回退
    elif torque > soft and torque_max > soft:
        k = (torque_max - torque) / (torque_max - soft)
        effective_inc = increment * max(0.0, min(1.0, k))   # 预减速
# ... 张开方向对称处理
```

**效果**：快速闭合时力矩被限制在上限附近（不再冲到 3Nm），触限后稳稳夹住物体保持保位；松开摇杆可重新尝试闭合。

> 注：曾尝试触限后主动回退一小角度释放冲量，实测会导致夹不住物体，已取消；现仅锁定保位。

**新增参数**：

| 参数 | 默认 | 说明 |
|------|------|------|
| `_GRIPPER_TORQUE_SOFT_RATIO` | `0.6` | 软阈值比例：力矩达上限的此比例时开始预减速（调小则更早减速、更柔和） |

> 慢速闭合本就工作正常（低速下反应时间足够、动量可忽略），本节改动不影响其行为；仅对快速闭合的力矩响应做了加固。

---

## 14. VR 摇杆控制底盘：恒定速度 + 加速度斜坡（修复母线电压冲击重启）

**问题**：VR 双手柄控制底盘运动时（左摇杆平移、右摇杆 X 旋转），底盘速度变化太大、太敏感，摇杆轻微拨动或回中都会引起轮速突变，电机反电动势回充母线，导致 Linux 工控机电压过大而重启。

**根因**：指令链路 `vr_panel._poll_status`（20Hz 读摇杆）→ `main_window._on_vr_chassis_cmd` → `chassis_lift_status.set_external_velocity` → `_send_control_commands`（30Hz）中，外部速度指令被**直接**经 `_body_to_wheel` 换算成轮速下发，**没有任何加/减速斜坡**；且摇杆偏心量线性映射速度，操作稍猛即全速。

**修复**（两层）：

1. **摇杆只给方向，目标速度恒定**（`debugger/widgets/vr_panel.py` `_poll_status`）：
   - 平移：左摇杆超死区后归一化为单位向量 × 恒定速度 `_CHASSIS_MOVE_SPEED`（偏心量不再影响速度）；
   - 旋转：右摇杆 X 只取符号 × `_CHASSIS_TURN_SPEED`。
   ```python
   mag = math.hypot(vx, vy)
   if mag < _CHASSIS_JOY_DEADZONE:
       vx = vy = 0.0
   else:
       vx = vx / mag * _CHASSIS_MOVE_SPEED
       vy = vy / mag * _CHASSIS_MOVE_SPEED
   rot = 0.0 if abs(rot) < _CHASSIS_JOY_DEADZONE \
       else math.copysign(_CHASSIS_TURN_SPEED, rot)
   ```

2. **外部速度指令加速度斜坡（缓启缓停）**（`chassis_lift_status.py` `_send_control_commands`）：
   - 外部目标速度（SI 单位）先经 `_ramp` 逐周期逼近再下发：线加速度上限 `CHASSIS_MAX_LIN_ACCEL = 0.5 m/s²`、角加速度上限 `CHASSIS_MAX_ANG_ACCEL = 1.2 rad/s²`（30Hz 循环，步长 = 加速度 × `EXT_CMD_DT`）；
   - 摇杆回中/指令超时失效后不再硬切到 0，而是沿同一斜坡**平滑减速到 0** 再交还面板摇杆控制；
   - 云端推理指令（`set_external_velocity_si`）走同一通道，同样受斜坡保护。
   ```python
   self._ext_filt[0] = self._ramp(self._ext_filt[0], jx_e * max_linear, dlin)
   self._ext_filt[1] = self._ramp(self._ext_filt[1], jy_e * max_linear, dlin)
   self._ext_filt[2] = self._ramp(self._ext_filt[2], rot_e * max_angular, dang)
   wheel_speeds = self._body_to_wheel_vel(*self._ext_filt)
   ```

**效果**：底盘缓启缓停，方向切换时速度平滑过渡，轮速不再突变，消除母线回充电压冲击。

> 注：`demo/vr_control.py` 只控制机械臂/夹爪，不含底盘控制，本次无需修改（底盘控制仅存在于 GUI 主程序链路）。
> 硬件建议：软件斜坡只能缓解无法根治回充问题，建议检查母线是否有制动电阻/泄放回路；若仍有电压冲击，优先把 `CHASSIS_MAX_LIN_ACCEL` 降到 0.3。

**新增参数**：

| 参数 | 默认 | 说明 |
|------|------|------|
| `_CHASSIS_JOY_DEADZONE` | `0.12` | 底盘摇杆死区（vr_panel.py） |
| `_CHASSIS_MOVE_SPEED` | `0.5` | 平移恒定目标速度（0~1，1=面板最大线速度） |
| `_CHASSIS_TURN_SPEED` | `0.4` | 旋转恒定目标速度（0~1，1=面板最大角速度） |
| `EXT_CMD_DT` | `1/30 s` | 底盘命令循环周期（chassis_lift_status.py，与 cmd_timer 33ms 对应） |
| `CHASSIS_MAX_LIN_ACCEL` | `0.5 m/s²` | 外部指令线加速度上限（缓启缓停） |
| `CHASSIS_MAX_ANG_ACCEL` | `2.0 rad/s²` | 外部指令角加速度上限 |

---

## 15. 底盘旋转停车延迟 + 摇杆方向映射修正（§14 后续）

**问题 1：松手后底盘仍旋转约 0.5s**。§14 的斜坡对加速/减速使用同一上限（角 1.2 rad/s²），松手后需按该缓坡慢慢降到 0，停车延迟明显；旋转起步也偏肉。

**修复**：加/减速度分离——起步保持缓坡（防母线回充冲击），松手或反向时改用更大的减速度上限快速停车：
```python
CHASSIS_MAX_LIN_DECEL = 1.5    # m/s² 线减速度上限
CHASSIS_MAX_ANG_DECEL = 4.0    # rad/s² 角减速度上限
step = dec_d if (abs(tgt) < abs(cur) or tgt * cur < 0.0) else acc_d
self._ext_filt[i] = self._ramp(cur, tgt, step)
```
同时将角加速度上限由 1.2 提至 2.0 rad/s²，旋转起步更跟手。

**问题 2：左摇杆方向映射错误**（前推不应后退、左推不应右移）。Quest 摇杆原始值（APK `leftJS`）前推/左推为负值，原代码直接透传给 `set_external_velocity(jx 前进+, jy 左移+)` 导致两轴反向。

**修复**（`vr_panel.py`）：读取时乘符号修正常量，可调：
```python
_CHASSIS_JOY_VX_SIGN = -1.0   # 左摇杆 Y → 底盘前后（前推应前进，反向改 +1.0）
_CHASSIS_JOY_VY_SIGN = -1.0   # 左摇杆 X → 底盘左右（左推应左移，反向改 +1.0）
vx = _CHASSIS_JOY_VX_SIGN * left.joystick_y
vy = _CHASSIS_JOY_VY_SIGN * left.joystick_x
```

**新增/变更参数**：

| 参数 | 默认 | 说明 |
|------|------|------|
| `CHASSIS_MAX_LIN_DECEL` | `1.5 m/s²` | 线减速度上限（松手/反向快速停车） |
| `CHASSIS_MAX_ANG_DECEL` | `4.0 rad/s²` | 角减速度上限（松手快速停转） |
| `CHASSIS_MAX_ANG_ACCEL` | `1.2 → 2.0 rad/s²` | 角加速度上限提升，旋转起步更跟手 |
| `_CHASSIS_JOY_VX_SIGN` | `-1.0` | 左摇杆 Y → 前后方向符号（vr_panel.py） |
| `_CHASSIS_JOY_VY_SIGN` | `-1.0` | 左摇杆 X → 左右方向符号（vr_panel.py） |

---

## 16. 摇杆 45° 解码 + 改走方向滑条通道 + 速度/加减速度下调（§15 后续）

**问题 1：方向映射仍错误**。实测：左摇杆前推 → 只有左前+右后两个轮转动（斜向运动）；左推 → 右前+左后两轮转动。结合麦轮运动学反推：前推时 raw `(lsx,lsy)=(t,-t)`、左推时 `(t,t)` —— **APK 摇杆坐标系相对底盘系旋转了 45°**，单纯的符号翻转无法修正。

**修复**（`vr_panel.py`）：对摇杆原始值做 -45° 旋转解码，还原前进/左移分量：
```python
inv2 = 1.0 / math.sqrt(2.0)
fwd       = (lsx - lsy) * inv2   # 前进分量：前推 raw=(t,-t) → √2·t
leftward  = (lsx + lsy) * inv2   # 左移分量：左推 raw=(t,t)  → √2·t
# 右摇杆同系解码取物理 X 分量（顺带消除上下推动时的旋转串扰）
rot = (rsx + rsy) * inv2
```
整体方向若仍反，改 `_CHASSIS_JOY_SIGN = -1.0` 即可（删除原 `_CHASSIS_JOY_VX/VY_SIGN`）。

**问题 2（需求）：摇杆改走主界面 4 个方向滑条的控制语义**。底盘面板有 前后(fb)/左右(lr)/左斜(ld)/右斜(rd) 四个滑条，方向已经验证正确，要求摇杆与滑条方向严格一致。

**修复**：
1. `chassis_lift_status.py` 新增 `set_external_direction_pads(fb, lr, ld, rd, rot)`：按滑条合成公式（`jx = fb+(ld+rd)/√2`，`jy = -lr+(ld-rd)/√2`）换算为底盘速度，内部仍走 `set_external_velocity` 的加速度斜坡通道（平滑保护不变）。
2. `main_window._on_vr_chassis_cmd` 由 `set_external_velocity` 改为 `set_external_direction_pads(fb, lr, rot=rot)`；`vr_panel` 解码后按滑条语义输出 `fb`（+前进）、`lr`（+右移）。

**问题 3：速度再降一半、加/减速度偏大**。

**调整**：

| 参数 | 旧值 | 新值 |
|------|------|------|
| `_CHASSIS_MOVE_SPEED` | 0.5 | **0.25** |
| `CHASSIS_MAX_LIN_ACCEL` | 0.5 m/s² | **0.3 m/s²** |
| `CHASSIS_MAX_ANG_ACCEL` | 2.0 rad/s² | **1.5 rad/s²** |
| `CHASSIS_MAX_LIN_DECEL` | 1.5 m/s² | **1.0 m/s²** |
| `CHASSIS_MAX_ANG_DECEL` | 4.0 rad/s² | **3.0 rad/s²** |

---

## 17. VR 底盘控制改为上层离散方向控制（8 方向量化，§16 后续）

**问题**：§16 的 45° 解码上机后方向仍不对（摇杆前后/左右仍对应底盘 45° 斜向运动）。两版线性映射均未能命中 APK 摇杆实际坐标系，逐版猜测效率低。

**决策**：按用户要求，VR 控制改为**独立于底盘内部实现的上层控制**——VR 层不再做任何坐标系换算/轮速映射，只把摇杆量化为 8 个离散方向，直接输出与主界面 前后(fb)/左右(lr)/左斜(ld)/右斜(rd) 四个方向滑条完全一致的指令，全向运动合成由底盘面板已有的 4 电机运动学保证（该路径方向已验证正确）。

**实现**：

1. **`vr_panel.py`**：摇杆 raw 值 → `atan2` 求角 → 加校准偏置 `_CHASSIS_JOY_ANGLE_OFFSET` → 量化为 8 扇区 → 查表 `_CHASSIS_SECTOR_PADS` 得滑条组合（前/后/左/右/前左/前右用单滑条，后左/后右用 fb+lr 组合）× `_CHASSIS_MOVE_SPEED`；右摇杆 X 取符号 × `_CHASSIS_TURN_SPEED`。`chassis_cmd` 信号扩展为 5 通道 `(fb, lr, ld, rd, rot)`。
2. **`main_window.py`**：`_on_vr_chassis_cmd(fb, lr, ld, rd, rot)` → `set_external_direction_pads(fb, lr, ld, rd, rot)`。
3. **`chassis_lift_status.py`**：`set_external_direction_pads`（§16 已建）按滑条合成公式换算后走加速度斜坡通道，不变。
4. **校准辅助**：摇杆推动时终端每秒打印一次 `[VR-CHASSIS] raw=(lsx,lsy) ang=.. sector=.. fb=.. lr=.. ...`。若方向不符，按公式 `_CHASSIS_JOY_ANGLE_OFFSET = 90° - atan2(lsy,lsx)（物理前推时）` 计算并填入常量即可（标准摇杆为 0；如前推打印 raw≈(t,-t) 即 ang=-45°，则填 135）。

**参数变更**：

| 参数 | 默认 | 说明 |
|------|------|------|
| `_CHASSIS_JOY_ANGLE_OFFSET` | `0.0°` | 摇杆 raw 系角度偏置（校准用，替代已删除的 `_CHASSIS_JOY_SIGN`） |
| `_CHASSIS_SECTOR_PADS` | 8×4 表 | 扇区 → (fb, lr, ld, rd) 滑条组合 |

---

## 18. 8 方向量化方案落地与上机验证（§17 实现，已验证修复成功）

**背景**：§17 记录了 8 方向量化方案，但仓库代码此前被回退成 §16 的连续 45° 解码版本（`chassis_cmd` 3 通道 → `set_external_velocity`），上机后摇杆前后/左右仍对应底盘 45° 斜向运动。本节为 §17 方案的实际代码落地，**上机验证修复成功，`_CHASSIS_JOY_ANGLE_OFFSET = 0.0`（标准摇杆系，无需偏置）**。

**实际实现**（与 §17 记录的差异已标注）：

1. **`vr_panel.py`**：
   - 常量区：删除 `_CHASSIS_JOY_SIGN`，新增 `_CHASSIS_JOY_ANGLE_OFFSET = 0.0` 与 `_CHASSIS_SECTOR_PADS`（8×4 查表）。扇区角约定：`0°=raw X 正轴，逆时针`，扇区号 `sector = round((degrees(atan2(lsy,lsx)) + offset) / 45) % 8`；扇区→方向：0°右 / 45°前右(rd) / 90°前(fb) / 135°前左(ld) / 180°左(lr=-) / 225°后左(fb=-,lr=-) / 270°后(fb=-) / 315°后右(fb=-,lr=+)。
   - `_poll_status`：摇杆幅值过死区后量化扇区，滑条指令 = 查表值 × `k × _CHASSIS_MOVE_SPEED`，其中 `k = min(1, (mag-死区)/(1-死区))`——**速度为比例式**（§17 记录为恒定速度，实机采用摇杆偏心量比例，起步更柔和，方向不受幅值影响）。
   - 右摇杆 X 旋转保持**比例式**（死区补偿后 × `_CHASSIS_TURN_SPEED`），与 §17 记录的"取符号"不同，实操更跟手且旋转方向此前已验证正确。
   - `chassis_cmd` 信号扩为 5 通道 `(fb, lr, ld, rd, rot)`；回中时补发一次全零指令后停发，不覆盖云端推理底盘指令。
2. **`main_window.py`**：`_on_vr_chassis_cmd(fb, lr, ld, rd, rot)` → `self._chassis_win.set_external_direction_pads(fb, lr, ld, rd, rot)`。
3. **`chassis_lift_status.py`**：未改动；`set_external_direction_pads` 按滑条合成公式换算后走加速度斜坡通道（§16 已建）。

**校准辅助**：摇杆推动时终端每秒打印一次 `[VR-CHASSIS] raw=(lsx,lsy) ang=.. sector=.. fb=.. lr=.. ld=.. rd=..`。若方向不符，物理前推时按 `_CHASSIS_JOY_ANGLE_OFFSET = 90° - 打印的 ang` 填入 vr_panel.py 顶部常量即可。本次上机 `offset=0` 方向即正确。

---

## 19. 头显（HMD）底盘方向映射修正：头部平移按锚定 yaw 旋转到机体系

**问题**：左手柄 X 按住后，头显前移/左移时底盘朝 45° 斜向运动（前后左右方向整体偏转）。

**根因**：`vr_panel._poll_hmd_chassis` 把头部平移增量（沿锚定视线方向分解的 `d_fwd`/`d_lat`）**直接**叠加到里程计坐标轴：
```python
target_x = bx + efwd * scale      # 头前移 → 里程计 +x
target_y = by + STRAFE * elat * scale
```
这隐含假设"锚定时底盘 yaw = 0（里程计 +x = 底盘车头方向）"。一旦锚定时底盘 yaw ≠ 0（如底盘回零后转动过、或摆放方向偏了 45°），头前移对应的里程计 +x 就不再是车头方向，底盘运动方向整体偏转锚定 yaw 角——表现为头前移/左移时底盘走 45° 斜向。

**修复**（`vr_panel.py`，与左摇杆底盘控制的机体系语义对齐）：头部平移先映射到**锚定时刻的底盘机体系**（头前移=底盘前进、头右移=底盘右移），再按锚定 yaw 旋转到里程计世界系：
```python
dx_b = _HMD_FWD_SIGN * efwd * scale    # 机体系：+前进
dy_b = _HMD_STRAFE_SIGN * elat * scale # 机体系：+左移
cos_a, sin_a = math.cos(byaw), math.sin(byaw)
target_x = bx + dx_b * cos_a - dy_b * sin_a
target_y = by + dx_b * sin_a + dy_b * cos_a
```
（旋转公式与底盘里程计积分 `x += vx·cos(yaw) - vy·sin(yaw)`、`y += vx·sin(yaw) + vy·cos(yaw)` 同约定；锚定 yaw=0 时退化为原行为。）转头 yaw 目标本就走 `byaw + dyaw` 相对锚定，不受影响；升降通道不变。

**效果**：无论锚定时底盘朝向如何，头前移=底盘沿车头方向前进、头左移=左移，与左摇杆方向语义一致。

---

## 20. 头显底盘跟随提速：位姿伺服专用斜坡（§19 后续）

**问题**：VR 头显控制底盘运动延迟大、反应迟钝。

**根因**：头显位置跟随走里程计 P 位姿伺服通道（`chassis_lift_status.py` `_ext_pose` 分支），其速度斜坡与摇杆速度指令通道共用 `CHASSIS_MAX_LIN_ACCEL = 0.3 m/s²` / `CHASSIS_MAX_ANG_ACCEL = 1.5 rad/s²`——加速到 0.3 m/s 需约 1 秒，起步/换向明显迟滞。

**修复**：位姿伺服通道改用独立斜坡常量（P 伺服随误差收敛自行限速、目标随头部运动连续变化，不会像摇杆速度指令那样阶跃突变，可安全放宽）：

| 参数 | 值 | 说明 |
|------|-----|------|
| `CHASSIS_POSE_LIN_ACCEL` | `0.8 m/s²` | 位姿伺服线加速度上限（原共用 0.3） |
| `CHASSIS_POSE_LIN_DECEL` | `2.0 m/s²` | 位姿伺服线减速度上限（原共用 1.0） |
| `CHASSIS_POSE_ANG_ACCEL` | `3.0 rad/s²` | 位姿伺服角加速度上限（原共用 1.5） |
| `CHASSIS_POSE_ANG_DECEL` | `5.0 rad/s²` | 位姿伺服角减速度上限（原共用 3.0） |

摇杆/云端速度指令通道仍用原 `CHASSIS_MAX_*` 常量，不受影响。

**提示**：VR 面板已有 **"位置缩放"**（0.2x~5.0x，默认 1.0x，头部平移→底盘位移的放大倍数）与 **"速度跟随"**（0.05~1.00 m/s，默认 0.30 m/s，伺服线速度上限）两个滑条；若仍嫌慢，先把"速度跟随"拉高（如 0.5~0.8）再观察。

---

## 21. 底盘/机械臂摇杆互斥 + 右摇杆 Y 控制底盘升降

**需求**：底盘控制与机械臂控制独立互斥——只要任一手柄 Grip（侧方扳机）按住，手柄摇杆不能控制底盘，摇杆只控制夹爪开合；双手 Grip 都松开时，摇杆恢复控制底盘。右手柄摇杆上下控制底盘升降（注意上下限位），其余底盘控制逻辑不变。

**处理**（`debugger/widgets/vr_panel.py`、`debugger/main_window.py`）：

1. **互斥判定**（`VrPanel._poll_status`）：每周期读取双手柄 `grip` 模拟量，任一 `> GRIP_CLUTCH_THRESHOLD` 即 `grip_held`；按住期间左右摇杆全部置零（底盘指令屏蔽），松开即恢复。取代了原"控臂运行期间底盘整体屏蔽"的粗略逻辑（`self._ctrl_running` 门控已删除）。
   - 夹爪侧本已由各臂控制器 `_tick` 中 `gripped` 门控（§9），与本互斥天然一致：Grip 按住 = 摇杆归夹爪，Grip 松开 = 摇杆归底盘。
   - 按住 Grip 瞬间若底盘在动：本周期发出一次全零指令，底盘沿斜坡平滑停车（§14/§15 斜坡通道不变）。
2. **右摇杆 Y → 底盘升降**（位置伺服，增量式目标）：
   - `chassis_cmd` 信号扩为 6 通道 `(fb, lr, ld, rd, rot, lift_mm)`，`lift_mm=nan` 表示不控升降；
   - 推动右摇杆 Y 时以当前升降反馈为基准（`_lift_provider` = `chassis_win.get_lift_mm`）增量累加：`target += _CHASSIS_LIFT_SIGN × _CHASSIS_LIFT_SPEED × 摇杆有效量 × dt`，钳位在 `[_CHASSIS_LIFT_MIN_MM, _CHASSIS_LIFT_MAX_MM]`（-700~0 mm）；
   - 摇杆回中/Grip 按住：目标置 None（保持当前位置），下次推动重新以反馈为基准，避免跳变；
   - `main_window._on_vr_chassis_cmd` 接收第 6 通道，非 nan 时调 `set_lift_position_mm` 走升降位置伺服（底盘面板内部另有限位开关保护）。
3. **头显模式优先**：左手柄 X 按住的头显底盘/升降控制期间，摇杆升降同样屏蔽（`not hmd_active` 条件），不变。

**新增参数**：

| 参数 | 默认 | 说明 |
|------|------|------|
| `_CHASSIS_LIFT_SPEED` | `120.0 mm/s` | 右摇杆 Y 满量程升降速度 |
| `_CHASSIS_LIFT_MIN_MM` | `-700.0` | 升降下限位 (mm) |
| `_CHASSIS_LIFT_MAX_MM` | `0.0` | 升降上限位 (mm，0=最高) |
| `_CHASSIS_LIFT_SIGN` | `+1.0` | 上推应对应上升，反了改 `-1.0` |

> 注：本次同时修复了上次会话遗留问题——信号已扩为 6 通道但 `_poll_status` 仍按 5 参数 emit，TypeError 被 `except: pass` 吞掉导致摇杆底盘指令完全失效；现已同步为 6 参数。

---

## 22. 取消右摇杆 Y 升降 + 左 X 头显功能移到左扳机 + 左 Y/X 键控底盘升降

**需求**：
1. 取消右手柄摇杆上下扳动控制升降（机械臂随底盘升降柱上下移动）的功能；
2. 左手柄 X 键的头显底盘/升降控制功能移到左手柄前方扳机上；
3. 左手柄 Y 键按住 = 底盘升高，左手柄 X 键按住 = 底盘降低。

**处理**（`debugger/widgets/vr_panel.py`；`demo/vr_control.py` 不含底盘/头显控制，无需改动）：

1. **删除右摇杆 Y → 升降**：移除 `_poll_status` 中 `rsy` 读取及右摇杆 Y 升降位置目标整块逻辑；右摇杆 Y 不再占用（Grip 按住时仍归夹爪，§9/§21 不变）。
2. **头显控制激活改左扳机**：`_poll_hmd_chassis` 的激活条件由 `left.button_a`（X 键）改为 `left.trigger > _HMD_TRIGGER_HOLD_THRESHOLD`（新增常量，默认 0.5，模拟量判定）；锚定/跟随/释放逻辑不变，日志提示同步改为"左扳机"。
3. **左 Y/X 键 → 底盘升降**（复用原右摇杆 Y 的增量式位置伺服通道）：双手 Grip 都松开且头显未激活时，Y 按住目标以 `_CHASSIS_LIFT_SPEED`（120 mm/s）递增（升高→0），X 按住递减（降低→负），钳位在 `[_CHASSIS_LIFT_MIN_MM, _CHASSIS_LIFT_MAX_MM]`（-700~0 mm）；两键同按视为无效（不动）；松开按键目标保持、置 None，下次按键重新以升降反馈为基准避免跳变。指令仍走 `chassis_cmd` 第 6 通道 `lift_mm` → `set_lift_position_mm` 位置伺服，链路不变。

**新增参数**：

| 参数 | 默认 | 说明 |
|------|------|------|
| `_HMD_TRIGGER_HOLD_THRESHOLD` | `0.5` | 左扳机模拟量超过此值判定为按住（头显底盘控制激活） |

**参数语义微调**：

| 参数 | 说明 |
|------|------|
| `_CHASSIS_LIFT_SPEED` | 原为"右摇杆 Y 满量程升降速度"，现为"左 Y/X 键按住时的升降速度"（120 mm/s 不变） |
| `_CHASSIS_LIFT_SIGN` | 原为"上推=上升"，现为"Y 键=上升"，反了改 `-1.0` |

---

## 23. 右手柄左右方向调换 + 左 X/Y 键升降提速

**需求**：
1. 右手柄摇杆左右（X 轴）控制方向调换；
2. 左手柄 X/Y 按键控制底盘升降的速度提高 2 倍。

**修正说明**：曾按初稿将右手柄摇杆 Y 置空；收到反馈后已撤销，右手柄摇杆 Y→夹爪的原有控制保持不变。

**处理**：

1. **右手柄摇杆 Y 保持夹爪控制**：`debugger/widgets/vr_panel.py` 与 `demo/vr_control.py` 均保留原有摇杆 Y→夹爪逻辑（上推闭合、下拉张开、力矩限制保护不变）。
2. **右手柄摇杆左右反向**：新增 `_CHASSIS_TURN_SIGN = -1.0`，读取底盘旋转输入时使用 `rsx = _CHASSIS_TURN_SIGN * right.joystick_x`；恢复旧方向时改回 `+1.0`。
3. **左 X/Y 键升降提速 2 倍**：`_CHASSIS_LIFT_SPEED` 由 `120.0` 提高到 `240.0 mm/s`；增量式位置伺服、上下限位 `-700~0 mm`、松开保持逻辑不变。

**参数变更**：

| 参数 | 旧值 | 新值 | 说明 |
|------|------|------|------|
| `_CHASSIS_TURN_SIGN` | 无 | `-1.0` | 右摇杆左右方向调换（恢复默认改 `+1.0`） |
| `_CHASSIS_LIFT_SPEED` | `120.0 mm/s` | `240.0 mm/s` | 左手柄 Y/X 键按住时的升/降速度 |

---

## 24. 左 Y/X 键改为连续速度控制（200 RPM，松开发 0 停止）

**需求**：VR 左手柄 X/Y 键升降控制由「按住连续移动（增量式位置伺服）」改为「连续速度控制」——Y 按住 = 上升、X 按住 = 下降（转速默认 200 RPM，界面可调），松开 X/Y 发送速度 0 目标停止。借鉴 `chassis_lift_status.py` 的速度控制机制。

**处理**（`debugger/widgets/vr_panel.py`、`debugger/main_window.py`、`chassis_lift_status.py`）：

1. **`chassis_lift_status.py`**：
   - 新增 `set_lift_velocity(direction, speed_rpm=200.0)` 公开接口（direction -1=升 / +1=降 / 0=停），把转速换算为 rad/s 存入 `_ext_lift_vel_rad`，并记录 `_ext_lift_vel_time` 供保鲜期判定；
   - `_lift_command_velocity` 增加最高优先级分支：外部升降速度在保鲜期 `_ext_lift_vel_timeout_s`（0.5s）内直接下发；超时自动失效交还下方「点动 > 位置伺服 > 界面」路径，松开/断连后自动停车；
   - 调用时清除 `_lift_target_mm`，避免与位置伺服目标冲突；限位开关保护不变。
2. **`vr_panel.py`**：
   - 删除按住增量位置伺服逻辑（`_vr_lift_target_mm`、`_lift_speed_mms`、`_CHASSIS_LIFT_SPEED`、`_CHASSIS_LIFT_SIGN`）与点动边沿逻辑；
   - 新增信号 `lift_vel_cmd(direction, speed_rpm)`：Y 按住 = -1、X 按住 = +1、松开 = 0；按住期间每周期下发续期，松开/进入 HMD/握住 Grip 自动发 0；
   - 原「按键升降速度 mm/s」滑条改为「升降转速」`QDoubleSpinBox`（10~1000 RPM，默认 200，步进 10），写 `_lift_rpm`。
3. **`main_window.py`**：`lift_vel_cmd` → `_on_vr_lift_vel_cmd` → `chassis.set_lift_velocity`。

**方向约定**：Y=上升、X=降低（保持现状）；`direction=-1` 升 / `+1` 降，与 `chassis._jog_lift` 一致。

**参数变更**：

| 参数 | 旧值 | 新值 | 说明 |
|------|------|------|------|
| `_CHASSIS_LIFT_SPEED` | `240.0 mm/s` | 删除 | 改为连续速度控制 |
| `_CHASSIS_LIFT_SIGN` | `+1.0` | 删除 | 方向并入 `direction` 约定 |
| `_CHASSIS_LIFT_SPEED_RPM` | 无 | `200.0` | 升降转速默认值（界面「升降转速」可调） |
| `_ext_lift_vel_timeout_s` | 无 | `0.5 s` | 外部升降速度保鲜期，超时自动停车（chassis_lift_status.py） |

---

## 25. VR 连接后自动启动控臂

**需求**：开启 VR 连接即自动启动 VR 控制（无需再手动点「启动控制」）。

**处理**（`debugger/widgets/vr_panel.py`）：
1. 新增信号 `auto_zero_finished`，在 `__init__` 中连接到 `_start_control`（自动队列连接，回主线程执行，避免后台线程直接改 UI）；
2. `_auto_move_zero` 双臂回零位完成后 `emit(auto_zero_finished)`，从而在回零完成后自动启动控臂；
3. `_start_control` 增加 `if self._ctrl_running: return` 防重复启动；
4. 断开 VR 时 `_stop_link` 仍调用 `_stop_control`，连接/断开行为对称。

> 若连接 VR 时机械臂尚未就绪（`_arm is None`），则不自动回零也不自动启动控臂，仍可后续手动点「启动控制」。

---

## 26. 摇杆 Y 方向调换 + 夹爪速度可调（默认 2 倍速）

**需求**：
1. 调换 VR 手柄摇杆 Y 的夹爪开合方向——下拉=闭合(→0)，上推=张开(→-1.835)；
2. VR 控制界面新增「夹爪速度」设置，默认 2 倍速（2.0 rad/s）。

**处理**（`debugger/widgets/vr_panel.py` 与 `demo/vr_control.py` 同步）：

1. **方向调换**：新增 `_JOY_GRIPPER_SIGN = -1.0`（反向改 `+1.0`），`_tick` 中 `increment = _JOY_GRIPPER_SIGN * gain * joy_y * dt`，并把 `closing = increment > 0.0`（跟随增量方向，力矩限位逻辑自动正确）。
2. **速度可调**：`_JOY_GRIPPER_GAIN` 默认 `1.0 → 2.0`（幅度）；GUI 侧改为 `VrArmController.gripper_joy_gain` 实例属性，控制区新增「夹爪速度」`QDoubleSpinBox`（0.1~5.0 rad/s，默认 2.0），改动实时写入双臂控制器，`_start_control` 启动时同步；`demo/vr_control.py` 无 UI，直接使用常量。

**参数变更**：

| 参数 | 旧值 | 新值 | 说明 |
|------|------|------|------|
| `_JOY_GRIPPER_GAIN` | `1.0` | `2.0` | 夹爪角速度默认值（2 倍速） |
| `_JOY_GRIPPER_SIGN` | 无 | `-1.0` | 下拉=闭合、上推=张开；反向改 `+1.0` |

> 注：§13 曾为修复快速闭合力矩超调把 gain 降到 1.0；现默认调回 2.0，若快速闭合力矩再超调，可界面下调速度或依赖 §13 的预减速+锁定兜底。

---

## 27. 系统设置新增「VR 控制」默认设置分页

**需求**：在主界面「设置」对话框中新增 VR 控制界面默认设置分页，包含升降转速、夹爪速度、运动缩放、头显位置缩放、头显速度跟随等默认值。

**处理**（`debugger/utils/system_config.py`、`debugger/widgets/settings_dialog.py`、`debugger/widgets/vr_panel.py`、`debugger/main_window.py`）：

1. **`system_config.py`**：`DEFAULT_CONFIG` 新增嵌套字典 `vr_defaults`（`lift_speed_rpm` / `gripper_speed_rad_s` / `scale` / `hmd_pos_scale` / `hmd_max_lin`），随 `main_system_config.json` 持久化。
2. **`settings_dialog.py`**：新增第三个分页「VR 控制」，5 个 `QDoubleSpinBox`；`_load` 回填、`_on_save` 收集写入 `cfg["vr_defaults"]`。
3. **`vr_panel.py`**：新增 `apply_defaults(dict)`，把配置写入对应控件并同步内部状态（控件 `setValue` 自动触发 valueChanged 传播到运行中控制器）。
4. **`main_window.py`**：`VrPanel()` 构造后与 `_on_settings_saved` 中均调用 `apply_defaults`，启动与保存后即时生效。

> 这些是「默认值」：启动时按此初始化 VR 面板，运行时仍可在 VR 面板内临时调整。

---

## 当前控制映射总览（右手柄）

| 手柄操作 | 功能 |
|---------|------|
| Grip（侧键）按住 | 离合接合，机械臂跟随手柄运动；松开冻结 |
| 手柄平移 | 末端 XYZ 平移（位置 IK，x/y 已修正） |
| 手柄横滚 | 末端横滚（矩阵 1:1 跟随） |
| 手柄俯仰 | 末端俯仰（矩阵 1:1 跟随） |
| 手柄偏航 | 末端偏航（矩阵 1:1 跟随） |
| 摇杆 Y（上/下） | 夹爪开合（下拉=闭合→0，上推=张开→-1.835，仅按住本手柄 Grip 时，§21/§23/§26） |
| A | 回零位 |
| B | 回工作位（Home） |
| Ctrl+C | 退出（仅 vr_control.py） |

> 注：摇杆关节控制已按 §8 移除，摇杆按 §9 控制夹爪；右手柄摇杆 Y 夹爪控制在 §23 中确认保留；扳机不再控制夹爪。

**底盘（双手 Grip 都松开时，§21）**：

| 手柄操作 | 功能 |
|---------|------|
| 左摇杆（任意方向） | 底盘 8 方向平移（方向滑条语义，比例速度） |
| 右摇杆 X（左/右） | 底盘旋转（比例速度，左右方向已调换，§23） |
| 左手柄 Y 按住 | 底盘升高（连续速度，默认 200 RPM，界面可调，松开发 0，§24） |
| 左手柄 X 按住 | 底盘降低（连续速度，默认 200 RPM，界面可调，松开发 0，§24） |
| 左手柄前方扳机按住 | 头显位姿控制底盘/升降（优先于摇杆/按键，§19/§20/§22） |
| 任一 Grip 按住 | 底盘摇杆/按键指令屏蔽，摇杆归夹爪 |

---

## 可调参数速查

文件顶部常量（`vr_panel.py` 与 `vr_control.py` 同步）：

| 参数 | 默认 | 说明 |
|------|------|------|
| `_POS_FIX` | `[-1,-1,1]` | 位置方向修正（前后/左右反了调符号） |
| `_ROT_FIX` | `diag(1,-1,-1)` | 姿态方向修正（Rx180） |
| `GRIPPER_OPEN_RAD` | `-1.835` | 夹爪全开角度（负方向为张开） |
| `GRIPPER_CLOSE_RAD` | `0.0` | 夹爪全闭角度 |
| `_JOY_GRIPPER_GAIN` | `2.0` | 摇杆 Y→夹爪角速度 (rad/s)，默认 2 倍速，界面「夹爪速度」可调（§26） |
| `_JOY_GRIPPER_SIGN` | `-1.0` | 下拉=闭合、上推=张开；反向改 `+1.0`（§26） |
| `_JOY_GRIPPER_DEADZONE` | `0.10` | 摇杆激活死区，低于此值夹爪保持不动 |
| `GRIPPER_TORQUE_MAX` | `0.5` Nm | 夹爪力矩上限（闭合停止推进） |
| `GRIPPER_TORQUE_MIN` | `-0.5` Nm | 夹爪力矩下限（张开停止推进） |
| `_GRIPPER_TORQUE_SOFT_RATIO` | `0.6` | 力矩预减速软阈值比例（§13，快速闭合防超调） |
| `GRIP_CLUTCH_THRESHOLD` | `0.5` | Grip 判定阈值 |

底盘控制相关（§14~§18）：

| 参数 | 默认 | 说明 |
|------|------|------|
| `_CHASSIS_JOY_DEADZONE` | `0.12` | 底盘摇杆死区（vr_panel.py） |
| `_CHASSIS_MOVE_SPEED` | `0.25` | 平移满量程速度比例（1.0=面板最大线速度，§18 起为比例式） |
| `_CHASSIS_TURN_SPEED` | `0.4` | 旋转满量程速度比例（1.0=面板最大角速度） |
| `_CHASSIS_TURN_SIGN` | `-1.0` | 右摇杆左右方向调换（恢复默认改 `+1.0`，§23） |
| `_CHASSIS_JOY_ANGLE_OFFSET` | `0.0°` | 摇杆 raw 系角度偏置（方向校准用，已上机验证 0° 正确） |
| `_CHASSIS_SECTOR_PADS` | 8×4 表 | 8 扇区 → (fb, lr, ld, rd) 滑条组合 |
| `_CHASSIS_LIFT_SPEED_RPM` | `200.0` | 升降转速默认值，界面「升降转速」可调（§24） |
| `_CHASSIS_LIFT_MIN/MAX_MM` | `-700.0 / 0.0` | 升降下/上限位 (mm，仅参考) |
| `_HMD_TRIGGER_HOLD_THRESHOLD` | `0.5` | 左扳机按住判定阈值（头显底盘控制，§22） |
| `CHASSIS_MAX_LIN_ACCEL` | `0.3 m/s²` | 外部指令线加速度上限（chassis_lift_status.py） |
| `CHASSIS_MAX_ANG_ACCEL` | `1.5 rad/s²` | 外部指令角加速度上限 |
| `CHASSIS_MAX_LIN_DECEL` | `1.0 m/s²` | 线减速度上限（松手/反向快速停车） |
| `CHASSIS_MAX_ANG_DECEL` | `3.0 rad/s²` | 角减速度上限（松手快速停转） |
