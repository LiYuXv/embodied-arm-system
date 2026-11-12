#!/bin/bash
# HEI ReBot Lift 底盘升降监控程序启动脚本
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR" || exit 1
export QT_ACCESSIBILITY=1
export GTK_MODULES=gail:atk-bridge
export QT_LINUX_ACCESSIBILITY_ALWAYS_ON=1
# 当前机器沿用 lerobot Python 环境；复制的 .venv 不包含解释器。
PYTHON="$HOME/miniforge3/envs/lerobot/bin/python"
if [ ! -x "$PYTHON" ]; then
    echo "未找到 Python: $PYTHON" >&2
    exit 1
fi
# Ubuntu ROS Humble 的 Python 3.10 包不要混入 Python 3.12。
PYTHON_SITE="$("$PYTHON" -c 'import site; print(site.getsitepackages()[0])')"
unset PYTHONPATH
export LD_LIBRARY_PATH="$PYTHON_SITE/cmeel.prefix/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$PYTHON" "$DIR/debugger/main.py" "$@"
