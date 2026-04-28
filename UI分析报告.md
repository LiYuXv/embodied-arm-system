# RobStride 电机控制界面分析报告

## 一、界面整体结构

程序使用 PyQt5 构建，包含一个主窗口 `MotorControlGUI`，布局分为左右两大部分：

```
┌─────────────────────────────────────────────────────────────┐
│                      主窗口 (950x750)                      │
├─────────────────────────┬───────────────────────────────────┤
│    左侧控制面板       │       右侧状态面板               │
│    (约50%宽度)        │       (约50%宽度)                │
├─────────────────────────┴───────────────────────────────────┤
```

---

## 二、左侧控制面板

### 2.1 连接控制组

| 控件 | 对象名 | 作用 | 逻辑 |
|------|--------|------|------|
| 连接按钮 | `connect_btn` | 连接/断开串口或CAN | 点击切换连接状态，调用 `toggle_connection()` |
| 系统设置按钮 | `system_settings_btn` | 打开系统设置对话框 | 打开 `SystemSettingsDialog` 对话框 |
| 安全状态标签 | `safety_status_label` | 显示安全限制状态 | 显示"安全限制: 开启/关闭" |
| 端口选择(隐藏) | `port_combo` | 选择串口/CAN通道 | 默认为 can0 |
| 波特率选择(隐藏) | `baud_combo` | 选择波特率 | 默认 921600 |
| 电机ID选择(隐藏) | `motor_id_spin` | 选择电机ID | 范围 1-127 |

### 2.2 电机控制组

| 控件 | 对象名 | 作用 | 逻辑 |
|------|--------|------|------|
| 使能按钮 | `enable_btn` | 使能电机 | 调用 `enable_motor()`，成功使能后启用运动控制 |
| 失能按钮 | `disable_btn` | 失能电机 | 调用 `disable_motor()`，停止所有电机活动 |
| 置零按钮 | `zero_btn` | 设置零点位置 | 调用 `set_zero_position()` |
| 清错按钮 | `clear_error_btn` | 清除故障 | 调用 `clear_error()`，发送清除错误命令 |

### 2.3 模式选择组

#### 模式选择下拉框

| 索引 | 模式名称 | 对应API模式 |
|------|----------|-----------|
| 0 | 运控模式 (Motion Control) | `MODE_MOTION` |
| 1 | 位置模式 - PP (Profile Position) | `MODE_POSITION_PP` |
| 2 | 位置模式 - CSP (Cyclic Synchronous Position) | `MODE_POSITION_CSP` |
| 3 | 速度模式 (Speed) | `MODE_SPEED` |
| 4 | 电流模式 (Current) | `MODE_CURRENT` |
| 5 | JOG 模式 | `MODE_JOG` |

#### 模式参数标签页

每个模式对应一个参数标签页，通过 `mode_tabs` TabWidget 管理：

**运控参数页 (运动控制模式)**
| 控件 | 作用 | 范围/默认值 |
|------|------|------------|
| 目标位置 | 设置目标位置 (rad) | -12.57 ~ 12.57, 默认 0 |
| 目标速度 | 设置目标速度 (rad/s) | -33.0 ~ 33.0, 默认 0 |
| 力矩前馈 | 设置前馈力矩 (Nm) | -14.0 ~ 14.0, 默认 0 |
| 位置增益 Kp | PID中P参数 | 0 ~ 500, 默认 0 |
| 速度增益 Kd | PID中D参数 | 0 ~ 5, 默认 0 |
| 控制频率 | 运控控制频率 (Hz) | 10 ~ 1000, 默认 500 |
| 获取当前位置按钮 | 读取当前位置到输入框 | - |

**位置PP参数页**
| 控件 | 作用 | 范围/默认值 |
|------|------|------------|
| 目标位置 | 设置目标位置 (rad) | -12.57 ~ 12.57, 默认 0 |
| 目标速度 | 最大速度 (rad/s) | 0.1 ~ 33.0, 默认 5.0 |
| 加速度 | 加速度 (rad/s²) | 0.1 ~ 100, 默认 10.0 |

**位置CSP参数页**
| 控件 | 作用 | 范围/默认值 |
|------|------|------------|
| 目标位置 | 设置目标位置 (rad) | -12.57 ~ 12.57, 默认 0 |
| 速度限制 | 速度限制 (rad/s) | 0.1 ~ 33.0, 默认 5.0 |

**速度参数页**
| 控件 | 作用 | 范围/默认值 |
|------|------|------------|
| 目标速度 | 目标速度 (rad/s) | -33.0 ~ 33.0, 默认 0 |
| 速度滑块 | 速度滑块 (-330 ~ 330) | 与输入框同步 |
| 加速度 | 加速度 (rad/s²) | 0.1 ~ 100, 默认 10.0 |
| 电流限制 | 电流限制 (A) | 0.1 ~ 16.0, 默认 5.0 |

**电流参数页**
| 控件 | 作用 | 范围/默认值 |
|------|------|------------|
| 目标电流 | 目标电流 (A) | -16.0 ~ 16.0, 默认 0 |
| 电流滑块 | 电流滑块 (-1600 ~ 1600) | 与输入框同步 |

**JOG参数页**
| 控件 | 作用 | 范围/默认值 |
|------|------|------------|
| JOG速度滑块 | 速度百分比 | 0 ~ 100%, 默认 20% |
| JOG正转按钮 | 正转 (CW) | 按下启动，松开停止 |
| JOG反转按钮 | 反转 (CCW) | 按下启动，松开停止 |

#### 模式控制按钮
| 控件 | 作用 | 逻辑 |
|------|------|------|
| 切换模式按钮 | 切换到选定模式 | `switch_mode()` - 失能→切换模式→使能流程 |
| 启动运动按钮 | 使用当前参数启动 | `start_motion()` - 根据当前模式启动 |
| 停止运动按钮 | 停止运动 | `stop_motion()` - 保持使能但停止 |

### 2.4 日志显示组

| 控件 | 作用 |
|------|------|
| 日志文本框 | 显示操作日志和时间戳 |
| 清空日志按钮 | 清空日志区域 |

---

## 三、右侧状态面板

### 3.1 7个关节状态显示组

| 控件 | 作用 |
|------|------|
| 连接状态标签 | 显示"● 已连接"/"● 未连接" |
| 当前模式标签 | 显示当前运行模式名称 |

每个关节的状态帧包含：
- 使能状态指示 (● 灰色/绿色)
- 位置显示 (蓝色, 3位小数)
- 速度显示 (绿色, 2位小数)
- 力矩显示 (红色, 2位小数)

### 3.2 7个关节位置控制组

| 控件 | 作用 |
|------|------|
| 控制方式选择 | 直接控制/直线插值/三次多项式 |
| 步长显示 | 显示配置的步长角度 |
| 步长时间显示 | 显示配置的步进时间 |

每个关节的控制组件：
- 减按钮 (−) - 步减操作
- 滑条 - 位置滑条 (-3000 ~ 3000 对应 -3 ~ 3 rad)
- 加按钮 (+) - 步加操作
- 位置标签 - 显示当前位置

| 全局控件 | 作用 |
|---------|------|
| 滑条实时控制复选框 | 开启/关闭50Hz实时控制 |
| 跟踪速度 | PP/CSP模式跟踪速度 |
| PP加速度 | PP模式加速度 |

---

## 四、界面外观与配色

### 4.1 全局配色方案

```css
主色调:     #3498db (蓝色)      - 普通按钮
成功色:    #27ae60 (绿色)    - 连接/使能/启动
危险色:    #e74c3c (红色)    - 断开/失能/停止/减按钮
禁用色:    #bdc3c7 (灰色)    - 禁用状态按钮
背景色:    #2c3e50 (深灰)    - 日志区域背景
文字色:    #ecf0f1 (浅灰)    - 日志文字
边框色:    #3498db (蓝色)      - GroupBox边框
```

### 4.2 按钮样式

```css
/* 普通按钮 */
QPushButton {
    background-color: #3498db;
    color: white;
    border: none;
    padding: 8px;
    border-radius: 4px;
    font-weight: bold;
}
QPushButton:hover { background-color: #2980b9; }

/* 成功按钮 (#success) */
QPushButton#success { background-color: #27ae60; }
QPushButton#success:hover { background-color: #219a52; }

/* 危险按钮 (#danger) */
QPushButton#danger { background-color: #e74c3c; }
QPushButton#danger:hover { background-color: #c0392b; }

/* 禁用按钮 */
QPushButton:disabled { background-color: #bdc3c7; }
```

### 4.3 特殊控件样式

**加按钮 (+)**
```css
background-color: #27ae60;  /* 绿色 */
color: white;
border-radius: 12px;
font-weight: bold;
```

**减按钮 (-)**
```css
background-color: #e74c3c;  /* 红色 */
color: white;
border-radius: 12px;
font-weight: bold;
```

**状态显示颜色**
- 位置: `#3498db` (蓝色)
- 速度: `#27ae60` (绿色)
- 力矩: `#e74c3c` (红色)
- 使能指示-绿: `green`
- 使能指示-灰: `gray`

---

## 五、电机控制逻辑分析

### 5.1 控制模式与API对应关系

| 界面模式索引 | 模式名称 | MotorController常量 | 说明 |
|------------|----------|-------------------|------|
| 0 | 运控模式 | `MODE_MOTION` | 高频率闭环控制，需持续发送指令 |
| 1 | 位置PP模式 | `MODE_POSITION_PP` | 位置设定后自动轨迹规划 |
| 2 | 位置CSP模式 | `MODE_POSITION_CSP` | 周期性同步位置，立即生效 |
| 3 | 速度模式 | `MODE_SPEED` | 恒定速度控制 |
| 4 | 电流模式 | `MODE_CURRENT` | 力矩控制 (Iq电流) |
| 5 | JOG模式 | `MODE_JOG` | 点动控制，按住运行 |

### 5.2 模式切换流程

```
switch_mode() 执行流程:
1. 获取目标模式
2. 逐个电机执行:
   a. disable() - 失能电机
   b. stop_motion_control() - 停止运控线程
   c. set_run_mode(target_mode) - 切换模式
   d. enable() - 使能电机
3. 更新UI显示
```

### 5.3 位置控制方式

| 方式索引 | 名称 | 算法 |
|---------|------|------|
| 0 | 直接控制 | 直接发送目标位置命令 |
| 1 | 直线插值 | 线性插值，期间持续发送 |
| 2 | 三次多项式 | S曲线插值，平滑加减速 |

---

## 六、系统设置对话框

通过 `SystemSettingsDialog` 类实现，包含两个标签页：

### 6.1 通信设置标签页
| 设置项 | 说明 |
|--------|------|
| 接口类型 | 串口 (Serial) 或 CAN总线 |
| 端口/通道 | 串口号或CAN通道选择 |
| 波特率 | 仅串口使用，默认 921600 |
| 状态更新频率 | 电机状态查询频率 (10-1000Hz) |
| 运控控制频率 | 运控模式控制频率 (10-1000Hz) |
| 步长 | 每次点击±按钮的角度 (0.1-90°) |
| 步长时间 | 步进运动时间 (0.1-10s) |
| 滑条控制频率 | 实时控制频率 (10-200Hz) |

### 6.2 安全限制标签页
每个关节 (1-7) 单独设置：
- 启用/禁用该关节安全限制
- 位置限制 (min/max, rad)
- 速度限制 (min/max, rad/s)
- 力矩限制 (min/max, Nm)

---

## 七、后续更换关节的修改建议

如果需要更换其他关节电机（控制逻辑有差异），需要修改以下部分：

### 7.1 协议和参数适配
```python
# 修改位置: MotorController.__init__() 和协议初始化
# 1. 电机型号 (motor_type) - 从 RS00 改为其他型号
# 2. 参数范围 - 修改 V_MAX, V_MIN, P_MAX, P_MIN, T_MAX, T_MIN
# 3. 波特率配置
```

### 7.2 安全限制配置
```python
# 修改位置: SystemSettingsDialog.create_joint_safety_tab()
# 针对新关节调整:
# - 位置限制范围
# - 速度限制范围
# - 力矩限制范围
```

### 7.3 控制模式支持
```python
# 修改位置: mode_combo 和各模式标签页
# 如果新电机不支持某些模式，需要:
# - 隐藏或禁用相关选项
# - 调整参数范围
```

### 7.4 配色主题适配
```python
# 修改位置: setStyleSheet() 调用
# 如果需要匹配其他设备的主题:
# - 修改主色调
# - 修改成功/危险色
# - 修改字体
```

---

## 八、关键函数速查表

| 函数名 | 功能 | 所在类 |
|--------|------|--------|
| `toggle_connection()` | 切换连接状态 | MotorControlGUI |
| `connect_serial()` | 连接串口/CAN | MotorControlGUI |
| `disconnect_serial()` | 断开连接 | MotorControlGUI |
| `enable_motor()` | 使能所有电机 | MotorControlGUI |
| `disable_motor()` | 失能所有电机 | MotorControlGUI |
| `switch_mode()` | 切换运行模式 | MotorControlGUI |
| `start_motion()` | 启动运动 | MotorControlGUI |
| `stop_motion()` | 停止运动 | MotorControlGUI |
| `jog_start/d jog_stop()` | JOG控制 | MotorControlGUI |
| `set_zero_position()` | 设置零点 | MotorControlGUI |
| `clear_error()` | 清除故障 | MotorControlGUI |
| `check_safety_limits()` | 安全检查 | MotorControlGUI |
| `on_motor_status_updated()` | 状态更新回调 | MotorControlGUI |
| `on_multi_motor_slider_timer()` | 滑条定时器 | MotorControlGUI |
| `move_linear_interpolation()` | 直线插值 | MotorControlGUI |
| `move_cubic_polynomial()` | 三次多项式 | MotorControlGUI |