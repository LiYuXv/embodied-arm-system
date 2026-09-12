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
    parser.add_argument(
        "--can", default="can0",
        help="Initial CAN interface in single-arm mode (default: can0)",
    )
    parser.add_argument("--sim", action="store_true", help="Simulation mode (no hardware)")
    parser.add_argument(
        "--config",
        default=str(SDK_ROOT / "resources" / "config" / "multi_arm_config.yaml"),
        help="Dual-arm configuration file",
    )
    parser.add_argument(
        "--mode", choices=("single", "dual"), default="dual",
        help="Initial GUI mode (default: dual)",
    )
    parser.add_argument(
        "--urdf",
        default=str(SDK_ROOT / "resources" / "urdf" / "el_a3.urdf"),
        help="URDF file used by single-arm forward kinematics",
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

    from PyQt6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    app.setApplicationName("EL-A3 Debugger")

    from debugger.utils.theme_manager import ThemeManager
    from debugger.utils.style import THEMES

    tm = ThemeManager.instance()
    app.setStyleSheet(THEMES[tm.theme])
    tm.theme_changed.connect(lambda t: app.setStyleSheet(THEMES[t]))

    from el_a3_sdk import initialize_dual_arms
    from debugger.window_coordinator import WindowCoordinator

    manager = initialize_dual_arms(args.config, auto_connect=False)
    coordinator = WindowCoordinator(
        manager,
        urdf_path=args.urdf,
        single_can_name=args.can,
        sim_mode=args.sim,
    )
    coordinator.show(args.mode)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
