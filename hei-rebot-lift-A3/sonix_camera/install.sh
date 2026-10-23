#!/bin/bash
# Sonix 摄像头 (0c45:6360) Linux 驱动修复安装脚本
# 原理：该摄像头固件带宽协商有 bug，需给 uvcvideo 驱动加 quirks=64 (FIX_BANDWIDTH)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[1/3] 安装 modprobe 配置到 /etc/modprobe.d/ ..."
sudo cp "$SCRIPT_DIR/10-uvcvideo-sonix.conf" /etc/modprobe.d/10-uvcvideo-sonix.conf
sudo chmod 644 /etc/modprobe.d/10-uvcvideo-sonix.conf

echo "[2/3] 重载 uvcvideo 驱动 ..."
sudo rmmod uvcvideo 2>/dev/null || true
sleep 1
sudo modprobe uvcvideo

echo "[3/3] 验证 quirks 参数 ..."
Q=$(cat /sys/module/uvcvideo/parameters/quirks)
echo "当前 quirks = $Q (期望 64)"

echo ""
echo "安装完成！如摄像头之前已连接，请拔插一次 USB 使其重新枚举。"
echo "测试画面: python3 $SCRIPT_DIR/test_sonix_pyqt6.py"
