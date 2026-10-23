# Sonix 摄像头 (0c45:6360) Linux 驱动修复与测试

## 问题描述

Sonix USB 2.0 摄像头（`lsusb` 显示 `0c45:6360`，设备名 "USB 2.0 Camera"）
在 Linux 下能正常枚举出 `/dev/video*`，但**无法启动视频流**：

- OpenCV `cap.read()` 永远返回 `False`
- `v4l2-ctl --stream-mmap` 报错：`VIDIOC_STREAMON returned -1 (Protocol error)`

原因：该摄像头固件的 USB 带宽协商有 bug，Linux UVC 驱动默认参数下握手失败。

## 解决方案

给 `uvcvideo` 驱动加 `quirks=64`（`UVC_QUIRK_FIX_BANDWIDTH`）参数，跳过有 bug 的带宽协商。

## 安装（迁移到新 Linux 机器）

```bash
cd sonix_camera
bash install.sh
```

脚本会：
1. 把 `10-uvcvideo-sonix.conf` 复制到 `/etc/modprobe.d/`（重启后仍生效）
2. 重载 `uvcvideo` 驱动
3. 验证 quirks 参数已生效

**安装后如摄像头已连接，需拔插一次 USB 使其重新枚举。**

卸载：`bash uninstall.sh`

## 测试画面

```bash
python3 test_sonix_pyqt6.py   # 需要 PyQt6 + opencv-python
```

自动检测最多 3 路摄像头并横向显示，可切换分辨率。

命令行快速验证（无需 GUI）：

```bash
# 应输出 64
cat /sys/module/uvcvideo/parameters/quirks

# 抓 5 帧到文件，文件大小 > 0 即正常
v4l2-ctl -d /dev/video0 \
  --set-fmt-video=width=640,height=480,pixelformat=MJPG \
  --stream-mmap --stream-count=5 --stream-to=test.jpg
```

## 注意事项

### 1. 分辨率只能用离散值

该摄像头**只支持固定几档分辨率**（MJPG）：

| MJPG | YUYV |
|---|---|
| 640x480 @30fps | 640x480 @30fps |
| 1280x720 @30fps | 800x600 @20fps |
| 1280x960 @30fps | 1280x720 @10fps |
| 1920x1080 @30fps | 1920x1080 @5fps |
| 2048x1536 / 2592x1944 | 2592x1944 @2fps |

**不要设置 848x480 等非标分辨率**，实测会导致摄像头固件卡死，
之后所有分辨率都无法出流，只能**物理拔插 USB** 恢复（软件重置无效）。

### 2. 必须用 MJPG 格式

部分情况下默认 YUYV 无法出流。OpenCV 打开后先设置：

```python
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
```

检测摄像头时若默认格式 `read()` 失败，应设置 MJPG 后重试再判定不可用
（参考 `test_sonix_pyqt6.py` 中 `CameraDetectThread`）。

### 3. USB 带宽

- 每个摄像头会枚举出 2 个 `/dev/video*` 节点（视频流 + 元数据），
  只有第一个能出流，程序检测时会自动跳过元数据节点。
- 多个摄像头不要插在同一条 USB 2.0 (480M) 总线上，否则会因带宽不足
  无法出流（dmesg 出现 `Not enough bandwidth for altsetting`）。
  **建议把摄像头分到不同总线**（如一个插 USB 3.0 蓝色口）。
- `quirks=64` 是全局参数，对所有 UVC 摄像头生效，实测对正常摄像头无害。

### 4. 排错

| 现象 | 处理 |
|---|---|
| 所有分辨率都读不到帧 | 固件卡死，**物理拔插 USB** |
| `quirks` 不是 64 | 重跑 `install.sh` 或 `sudo rmmod uvcvideo && sudo modprobe uvcvideo` |
| dmesg 报 `Not enough bandwidth` | 摄像头插到了同一总线，换 USB 3.0 口 |
| 程序里检测不到摄像头 | 检测逻辑需加 MJPG 回退（见上） |

查看内核日志：`sudo dmesg | grep -i "uvc\|usb"`

## 文件清单

| 文件 | 说明 |
|---|---|
| `10-uvcvideo-sonix.conf` | modprobe 配置（quirks=64） |
| `install.sh` | 一键安装驱动修复 |
| `uninstall.sh` | 卸载修复 |
| `test_sonix_pyqt6.py` | PyQt6 多路摄像头画面测试程序 |
