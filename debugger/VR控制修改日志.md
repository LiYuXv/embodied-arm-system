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

## 当前控制映射总览（右手柄）

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

> 注：摇杆关节控制已按 §8 移除，摇杆现按 §9 控制夹爪；扳机不再控制夹爪。

---

## 可调参数速查

文件顶部常量（`vr_panel.py` 与 `vr_control.py` 同步）：

| 参数 | 默认 | 说明 |
|------|------|------|
| `_POS_FIX` | `[-1,-1,1]` | 位置方向修正（前后/左右反了调符号） |
| `_ROT_FIX` | `diag(1,-1,-1)` | 姿态方向修正（Rx180） |
| `GRIPPER_OPEN_RAD` | `-1.835` | 夹爪全开角度（负方向为张开） |
| `GRIPPER_CLOSE_RAD` | `0.0` | 夹爪全闭角度 |
| `_JOY_GRIPPER_GAIN` | `1.0` | 摇杆 Y→夹爪角速度 (rad/s)，方向反了改 `-1.0` |
| `_JOY_GRIPPER_DEADZONE` | `0.10` | 摇杆激活死区，低于此值夹爪保持不动 |
| `GRIPPER_TORQUE_MAX` | `0.5` Nm | 夹爪力矩上限（闭合停止推进） |
| `GRIPPER_TORQUE_MIN` | `-0.5` Nm | 夹爪力矩下限（张开停止推进） |
| `_GRIPPER_TORQUE_SOFT_RATIO` | `0.6` | 力矩预减速软阈值比例（§13，快速闭合防超调） |
| `GRIP_CLUTCH_THRESHOLD` | `0.5` | Grip 判定阈值 |
