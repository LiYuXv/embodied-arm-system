"""EL-A3 机械臂调试上位机 - 入口"""

import sys
import os
import argparse
import logging
from pathlib import Path

SDK_ROOT = Path(__file__).parent.parent
REPO_ROOT = SDK_ROOT.parent

if str(SDK_ROOT) not in sys.path:
    sys.path.insert(0, str(SDK_ROOT))
if str(SDK_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(SDK_ROOT.parent))

os.environ.setdefault("QT_API", "pyqt6")


def main():
    parser = argparse.ArgumentParser(description="EL-A3 Robot Arm Debugger")
    parser.add_argument("--can", default="can0", help="CAN interface name (default: can0)")
    parser.add_argument("--sim", action="store_true", help="Simulation mode (no hardware)")
    parser.add_argument(
        "--urdf",
        default=str(SDK_ROOT / "resources" / "urdf" / "el_a3.urdf"),
        help="URDF file path",
    )
    parser.add_argument(
        "--meshes",
        default=str(SDK_ROOT / "resources" / "meshes"),
        help="STL mesh directory",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    from debugger.utils.system_config import (
        CONFIG_PATH, DEFAULT_CONFIG, load_config, save_config,
    )
    try:
        if not CONFIG_PATH.is_file() or not CONFIG_PATH.read_text(
                encoding="utf-8").strip():
            save_config(dict(DEFAULT_CONFIG))
    except Exception:
        pass
    _sys_cfg = load_config()

    from PyQt6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    app.setApplicationName("EL-A3 Debugger")

    from debugger.utils.theme_manager import ThemeManager
    from debugger.utils.style import THEMES

    tm = ThemeManager.instance()
    app.setStyleSheet(THEMES[tm.theme])
    tm.theme_changed.connect(lambda t: app.setStyleSheet(THEMES[t]))

    from debugger.main_window import MainWindow
    window = MainWindow(
        urdf_path=args.urdf,
        mesh_dir=args.meshes,
        auto_connect=bool(_sys_cfg.get("auto_connect_on_startup", True)),
    )
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
