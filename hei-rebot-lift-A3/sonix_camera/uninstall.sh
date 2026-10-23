#!/bin/bash
# 卸载 Sonix 摄像头驱动修复
set -e
sudo rm -f /etc/modprobe.d/10-uvcvideo-sonix.conf
sudo rmmod uvcvideo 2>/dev/null || true
sleep 1
sudo modprobe uvcvideo
echo "已卸载，quirks 恢复为: $(cat /sys/module/uvcvideo/parameters/quirks)"
