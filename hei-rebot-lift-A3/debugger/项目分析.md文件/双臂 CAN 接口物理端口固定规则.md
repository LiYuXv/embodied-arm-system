# 双臂 CAN 接口物理端口固定规则

## 背景

主界面控制程序 `debugger/main_window.py` 中双臂 CAN 连接约定：

- **臂1 → can0**
- **臂2 → can1**

Linux 下 CAN 接口名（can0/can1）默认按 USB 设备枚举顺序分配，插拔顺序或重启后可能互换，
导致臂1/臂2 接反。通过 udev 规则按**物理 USB 端口路径**固定接口名，解决该问题。

## 当前硬件映射（2026-10-15 记录）

| 臂 | CAN 接口 | 适配器型号 | VID:PID | 物理 USB 端口 | ID_PATH |
|----|---------|-----------|---------|--------------|---------|
| 臂1 | can0 | canable2 gs_usb | 1d50:606f | 1-1.1.4 | `pci-0000:00:15.0-usb-0:1.1.4:1.0` |
| 臂2 | can1 | candleLight USB to CAN | 1d50:606f | 1-1.1.3 | `pci-0000:00:15.0-usb-0:1.1.3:1.0` |

> 免密 sudo 与开机自动拉起配置见文末「免密 sudo」「开机自动拉起」两节（2026-10-16 新增）。

> 两个适配器 VID:PID 相同，只能靠物理端口路径区分。

## udev 规则

规则文件：`/etc/udev/rules.d/99-hei-rebot-lift.rules`（追加在原有底盘/升降串口规则之后）

```udev
# HEI ReBot Lift — 双臂 CAN 适配器按物理 USB 端口固定命名（防止插拔/重启后 can0/can1 互换）
# 臂1 = canable2 gs_usb，物理端口 1-1.1.4 → can0
ACTION=="add", SUBSYSTEM=="net", KERNEL=="can*", ENV{ID_PATH}=="pci-0000:00:15.0-usb-0:1.1.4:1.0", NAME="can0"
# 臂2 = candleLight，物理端口 1-1.1.3 → can1
ACTION=="add", SUBSYSTEM=="net", KERNEL=="can*", ENV{ID_PATH}=="pci-0000:00:15.0-usb-0:1.1.3:1.0", NAME="can1"
```

## 常用操作命令

```bash
# 查看当前 CAN 接口及其物理端口路径
for i in /sys/class/net/can*; do
  echo "$i:"
  udevadm info -q property "$i" | grep -E 'ID_PATH|ID_MODEL'
done

# 修改规则后重新加载（不需要重启）
sudo udevadm control --reload-rules

# 重新触发命名（接口需先 down，UP 状态无法改名）
sudo ip link set can0 down
sudo ip link set can1 down
# 然后拔插 USB 适配器，或重启

# 验证规则是否被读取
sudo udevadm test /sys/class/net/can0 2>&1 | grep 99-hei

# 拉起 CAN 接口（1 Mbps）
sudo bash scripts/setup_can.sh can0 1000000
sudo bash scripts/setup_can.sh can1 1000000
```

## 修改参考（换端口/换适配器时）

1. 将适配器插到新端口后，运行上面的 `udevadm info` 命令拿到新的 `ID_PATH`。
2. 编辑 `/etc/udev/rules.d/99-hei-rebot-lift.rules`，替换对应行的 `ID_PATH` 值。
3. `sudo udevadm control --reload-rules`，重新拔插适配器（或重启）生效。
4. 若更换了工控机/主板，PCI 路径前缀（`pci-0000:00:15.0-usb-0:`）可能变化，需整体替换。

## 免密 sudo（界面手动 up/down 必需）

`debugger/utils/can_utils.py` 的 `_run_privileged()` 使用 `sudo -n`（非交互模式，无法读密码），
必须配置 NOPASSWD 才能成功。已配置 `/etc/sudoers.d/can-up`：

```sudoers
wl ALL=(ALL) NOPASSWD: /usr/sbin/ip link set can[0-9] *
```

注意：早期只有 `/etc/sudoers.d/can0-up`（仅 can0 免密），导致界面里 can1 手动开启永远失败，
已删除旧文件并合并到 `can-up` 覆盖所有 canN。修改后验证：

```bash
sudo -n /usr/sbin/ip link set can1 txqueuelen 1000 && echo 免密OK
```

## 开机自动拉起（防止枚举竞争）

原 `can-reset.service`（`/usr/local/bin/reset_can.sh`）开机只执行一次，
与 USB 枚举存在竞争：can1 还没出现脚本就跑完，导致重启后 can1 常处于 DOWN。

已改为**事件驱动**：udev 规则在接口出现时立即拉起，追加在
`/etc/udev/rules.d/99-hei-rebot-lift.rules` 末尾：

```udev
# 接口出现时自动拉起（事件驱动，避免开机脚本与 USB 枚举竞争导致 can1 拉起失败）
ACTION=="add", SUBSYSTEM=="net", KERNEL=="can*", ENV{ID_PATH}=="pci-0000:00:15.0-usb-0:1.1.4:1.0", RUN+="/usr/sbin/ip link set can0 txqueuelen 1000", RUN+="/usr/sbin/ip link set can0 up type can bitrate 1000000"
ACTION=="add", SUBSYSTEM=="net", KERNEL=="can*", ENV{ID_PATH}=="pci-0000:00:15.0-usb-0:1.1.3:1.0", RUN+="/usr/sbin/ip link set can1 txqueuelen 1000", RUN+="/usr/sbin/ip link set can1 up type can bitrate 1000000"
```

同时 `/usr/local/bin/reset_can.sh` 已加固为循环重试（最多 15 次，每次间隔 1s，
直到接口 UP），作为 udev 规则的兜底；断电恢复场景下 USB 重新授权触发的 add 事件
也会命中上面的 udev 规则自动拉起。

## 注意事项

- 适配器必须始终插在各自对应的物理 USB 口，否则规则不匹配，接口名会回落到默认枚举顺序。
- 程序代码（`debugger/main_window.py`、`debugger/main.py`）无需改动，臂1=can0、臂2=can1 约定不变。
- 接口处于 UP 状态时无法改名，改名操作前先 `ip link set <iface> down`。
