#!/bin/bash
# HEI ReBot Lift 底盘升降监控程序启动脚本
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR" || exit 1
export QT_ACCESSIBILITY=1
export GTK_MODULES=gail:atk-bridge
export QT_LINUX_ACCESSIBILITY_ALWAYS_ON=1
exec "$DIR/.venv/bin/python" "$DIR/debugger/main.py"
