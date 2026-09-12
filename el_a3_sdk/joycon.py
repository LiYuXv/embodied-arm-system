"""Joy-Con evdev 输入、设备发现和相对姿态适配层。"""

import math
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from el_a3_sdk.controller_profiles import (
    JOYCON_PROFILES,
    JOYCON_VENDOR_ID,
    JoyConProfile,
)


class JoyConUnavailableError(RuntimeError):
    """evdev 不可用或左右 Joy-Con 节点不完整。"""


class LinuxEventCodes:
    """测试和无 evdev 导入环境使用的 Linux input-event-codes 子集。"""

    EV_SYN = 0
    EV_KEY = 1
    EV_ABS = 3
    SYN_REPORT = 0
    ABS_X = 0
    ABS_Y = 1
    ABS_Z = 2
    ABS_RX = 3
    ABS_RY = 4
    ABS_RZ = 5
    INPUT_PROP_ACCELEROMETER = 6
    BTN_SOUTH = 304
    BTN_EAST = 305
    BTN_NORTH = 307
    BTN_WEST = 308
    BTN_TL = 310
    BTN_TR = 311
    BTN_TL2 = 312
    BTN_TR2 = 313
    BTN_SELECT = 314
    BTN_START = 315
    BTN_THUMBL = 317
    BTN_THUMBR = 318
    BTN_DPAD_UP = 544
    BTN_DPAD_DOWN = 545
    BTN_DPAD_LEFT = 546
    BTN_DPAD_RIGHT = 547


@dataclass(frozen=True)
class JoyConDeviceBinding:
    side: str
    control_device: object
    imu_device: object

    @property
    def control_path(self) -> str:
        return str(getattr(self.control_device, "path", ""))

    @property
    def imu_path(self) -> str:
        return str(getattr(self.imu_device, "path", ""))


@dataclass(frozen=True)
class JoyConPair:
    left: JoyConDeviceBinding
    right: JoyConDeviceBinding

    def for_side(self, side: str) -> JoyConDeviceBinding:
        if side == "left":
            return self.left
        if side == "right":
            return self.right
        raise KeyError(f"Unknown Joy-Con side: {side}")


@dataclass(frozen=True)
class JoyConSnapshot:
    side: str
    valid: bool
    stick_x: float
    stick_y: float
    rotation: np.ndarray
    clutch: bool
    z_positive: bool
    z_negative: bool
    gripper_close: bool
    gripper_open: bool
    zero: bool
    home: bool
    exit: bool


@dataclass(frozen=True)
class RelativePoseInput:
    """VRArmController 可直接消费的通用相对位姿输入。"""

    valid: bool
    position: np.ndarray
    rotation: np.ndarray
    grip: float
    joystick_y: float
    button_a: bool
    button_b: bool
    translation_without_clutch: bool = False


def _device_identity(device: object) -> Tuple[int, int]:
    info = getattr(device, "info", None)
    return (
        int(getattr(info, "vendor", 0)),
        int(getattr(info, "product", 0)),
    )


def _detect_side(device: object) -> Optional[str]:
    vendor, product = _device_identity(device)
    name = str(getattr(device, "name", "")).lower()
    if vendor == JOYCON_VENDOR_ID:
        for side, profile in JOYCON_PROFILES.items():
            if product == profile.product_id:
                return side
    if "joy-con" in name or "joycon" in name:
        if "left" in name or "(l)" in name:
            return "left"
        if "right" in name or "(r)" in name:
            return "right"
    return None


def _is_imu_device(device: object, ecodes: object) -> bool:
    if "imu" in str(getattr(device, "name", "")).lower():
        return True
    try:
        return getattr(ecodes, "INPUT_PROP_ACCELEROMETER") in device.input_props()
    except (AttributeError, OSError, TypeError):
        return False


def _same_physical_controller(control: object, imu: object) -> bool:
    control_uniq = str(getattr(control, "uniq", "") or "")
    imu_uniq = str(getattr(imu, "uniq", "") or "")
    if control_uniq and imu_uniq:
        return control_uniq == imu_uniq
    control_phys = str(getattr(control, "phys", "") or "")
    imu_phys = str(getattr(imu, "phys", "") or "")
    return bool(control_phys and imu_phys and control_phys == imu_phys)


def discover_joycons(
    devices: Optional[Iterable[object]] = None,
    ecodes: Optional[object] = None,
) -> JoyConPair:
    """扫描 evdev 节点并按 VID/PID、uniq/phys 自动绑定左右 Joy-Con。

    hid-nintendo 会为每只 Joy-Con 创建控制节点和独立 IMU 节点。这里不使用
    eventX 编号；优先以 Nintendo VID 和左右产品 ID 分类，再用 MAC ``uniq``
    （或 ``phys``）将同一只手柄的两个节点配对。
    """
    owns_devices = devices is None
    if owns_devices:
        try:
            import evdev
        except ImportError as exc:
            raise JoyConUnavailableError(
                "缺少 evdev，请安装: pip install evdev"
            ) from exc
        ecodes = evdev.ecodes
        devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
    elif ecodes is None:
        ecodes = LinuxEventCodes

    candidates: Dict[str, Dict[str, list]] = {
        "left": {"control": [], "imu": []},
        "right": {"control": [], "imu": []},
    }
    devices = list(devices)
    for device in devices:
        side = _detect_side(device)
        if side is None:
            continue
        kind = "imu" if _is_imu_device(device, ecodes) else "control"
        candidates[side][kind].append(device)

    bindings = {}
    try:
        for side in ("left", "right"):
            controls = candidates[side]["control"]
            imus = candidates[side]["imu"]
            matches = [
                (control, imu)
                for control in controls
                for imu in imus
                if _same_physical_controller(control, imu)
            ]
            if len(matches) != 1:
                detail = (
                    f"control={[getattr(d, 'path', '?') for d in controls]}, "
                    f"imu={[getattr(d, 'path', '?') for d in imus]}"
                )
                raise JoyConUnavailableError(
                    f"无法唯一绑定 Joy-Con ({side}) 的控制/IMU 节点: {detail}"
                )
            bindings[side] = JoyConDeviceBinding(side, *matches[0])
    except JoyConUnavailableError:
        if owns_devices:
            for device in devices:
                try:
                    device.close()
                except OSError:
                    pass
        raise

    pair = JoyConPair(left=bindings["left"], right=bindings["right"])
    if owns_devices:
        selected = {
            pair.left.control_device,
            pair.left.imu_device,
            pair.right.control_device,
            pair.right.imu_device,
        }
        for device in devices:
            if device not in selected:
                try:
                    device.close()
                except OSError:
                    pass
    return pair


def discover_joycon(
    side: str,
    devices: Optional[Iterable[object]] = None,
    ecodes: Optional[object] = None,
) -> JoyConDeviceBinding:
    """Discover one Joy-Con without requiring its opposite-side partner.

    This is the single-arm equivalent of :func:`discover_joycons`.  Device
    numbers remain irrelevant: control and IMU nodes are paired by ``uniq``
    (or ``phys``) after filtering by the requested left/right product ID.
    """
    if side not in JOYCON_PROFILES:
        raise ValueError("side must be 'left' or 'right'")

    owns_devices = devices is None
    if owns_devices:
        try:
            import evdev
        except ImportError as exc:
            raise JoyConUnavailableError(
                "缺少 evdev，请安装: pip install evdev"
            ) from exc
        ecodes = evdev.ecodes
        devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
    elif ecodes is None:
        ecodes = LinuxEventCodes

    all_devices = list(devices)
    controls = []
    imus = []
    for device in all_devices:
        if _detect_side(device) != side:
            continue
        if _is_imu_device(device, ecodes):
            imus.append(device)
        else:
            controls.append(device)

    matches = [
        (control, imu)
        for control in controls
        for imu in imus
        if _same_physical_controller(control, imu)
    ]
    if len(matches) != 1:
        if owns_devices:
            for device in all_devices:
                try:
                    device.close()
                except OSError:
                    pass
        detail = (
            f"control={[getattr(d, 'path', '?') for d in controls]}, "
            f"imu={[getattr(d, 'path', '?') for d in imus]}"
        )
        raise JoyConUnavailableError(
            f"无法唯一绑定 Joy-Con ({side}) 的控制/IMU 节点: {detail}"
        )

    binding = JoyConDeviceBinding(side, *matches[0])
    if owns_devices:
        selected = {binding.control_device, binding.imu_device}
        for device in all_devices:
            if device not in selected:
                try:
                    device.close()
                except OSError:
                    pass
    return binding


def _skew(vector: Sequence[float]) -> np.ndarray:
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _rotation_vector_matrix(rotation_vector: Sequence[float]) -> np.ndarray:
    vector = np.asarray(rotation_vector, dtype=float)
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        return np.eye(3) + _skew(vector)
    axis_skew = _skew(vector / angle)
    return (
        np.eye(3)
        + math.sin(angle) * axis_skew
        + (1.0 - math.cos(angle)) * (axis_skew @ axis_skew)
    )


def _align_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = source / np.linalg.norm(source)
    target = target / np.linalg.norm(target)
    cross = np.cross(source, target)
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    cross_norm = float(np.linalg.norm(cross))
    if cross_norm < 1e-9:
        if dot > 0.0:
            return np.eye(3)
        orthogonal = np.array([1.0, 0.0, 0.0])
        if abs(source[0]) > 0.9:
            orthogonal = np.array([0.0, 1.0, 0.0])
        axis = np.cross(source, orthogonal)
        axis /= np.linalg.norm(axis)
        return _rotation_vector_matrix(axis * math.pi)
    axis = cross / cross_norm
    return _rotation_vector_matrix(axis * math.atan2(cross_norm, dot))


class ImuOrientationFilter:
    """陀螺积分＋加速度重力校正的互补姿态滤波器。"""

    def __init__(
        self,
        gravity_correction: float = 2.5,
        gyro_bias_time_constant: float = 3.0,
        max_dt: float = 0.05,
    ):
        self.gravity_correction = float(gravity_correction)
        self.gyro_bias_time_constant = float(gyro_bias_time_constant)
        self.max_dt = float(max_dt)
        self._rotation = np.eye(3)
        self._reference = np.eye(3)
        self._gyro_bias = np.zeros(3)
        self._last_timestamp: Optional[float] = None
        self.initialized = False

    @property
    def rotation(self) -> np.ndarray:
        return self._reference.T @ self._rotation

    @property
    def gyro_bias(self) -> np.ndarray:
        return self._gyro_bias.copy()

    def recenter(self):
        self._reference = self._rotation.copy()

    def update(
        self,
        accel_g: Sequence[float],
        gyro_rad_s: Sequence[float],
        timestamp: float,
    ) -> np.ndarray:
        accel = np.asarray(accel_g, dtype=float)
        gyro = np.asarray(gyro_rad_s, dtype=float)
        accel_norm = float(np.linalg.norm(accel))

        if not self.initialized:
            if accel_norm < 1e-6:
                return self.rotation
            self._rotation = _align_vectors(accel / accel_norm, np.array([0.0, 0.0, 1.0]))
            self._reference = self._rotation.copy()
            self._last_timestamp = float(timestamp)
            self.initialized = True
            return self.rotation

        dt = float(timestamp) - float(self._last_timestamp)
        self._last_timestamp = float(timestamp)
        if dt <= 0.0:
            return self.rotation
        dt = min(dt, self.max_dt)

        stationary = abs(accel_norm - 1.0) < 0.12 and np.linalg.norm(gyro) < 0.05
        if stationary:
            alpha = min(dt / max(self.gyro_bias_time_constant, dt), 0.02)
            self._gyro_bias += alpha * (gyro - self._gyro_bias)

        corrected_gyro = gyro - self._gyro_bias
        self._rotation = self._rotation @ _rotation_vector_matrix(corrected_gyro * dt)

        if 0.75 < accel_norm < 1.25:
            measured_world = self._rotation @ (accel / accel_norm)
            gravity_error_world = np.cross(measured_world, np.array([0.0, 0.0, 1.0]))
            correction = self.gravity_correction * gravity_error_world * dt
            self._rotation = _rotation_vector_matrix(correction) @ self._rotation

        u, _, vh = np.linalg.svd(self._rotation)
        self._rotation = u @ vh
        if np.linalg.det(self._rotation) < 0.0:
            u[:, -1] *= -1.0
            self._rotation = u @ vh
        return self.rotation


class JoyConInput:
    """读取一只 Joy-Con 的控制节点与 IMU 节点并维护线程安全快照。"""

    ACCEL_RESOLUTION = 4096.0
    GYRO_RESOLUTION_PER_DPS = 14247.0

    def __init__(
        self,
        binding: JoyConDeviceBinding,
        profile: Optional[JoyConProfile] = None,
        ecodes: Optional[object] = None,
        clock: Callable[[], float] = time.monotonic,
        stale_timeout: float = 0.5,
    ):
        self.binding = binding
        self.profile = profile or JOYCON_PROFILES[binding.side]
        self.ecodes = ecodes or self._load_ecodes()
        self._clock = clock
        self._stale_timeout = float(stale_timeout)
        self._lock = threading.Lock()
        self._running = False
        self._threads = []
        self._stick = np.zeros(2)
        self._buttons = set()
        self._axis_values = np.zeros(6)
        self._last_imu_time: Optional[float] = None
        self._orientation = ImuOrientationFilter()
        self._stick_calibration = self._read_stick_calibration()
        self._accel_resolution = self._axis_resolution("ABS_X", self.ACCEL_RESOLUTION)
        self._gyro_resolution = self._axis_resolution(
            "ABS_RX", self.GYRO_RESOLUTION_PER_DPS
        )

    @staticmethod
    def _load_ecodes():
        try:
            from evdev import ecodes
        except ImportError as exc:
            raise JoyConUnavailableError(
                "缺少 evdev，请安装: pip install evdev"
            ) from exc
        return ecodes

    def _code(self, name: str) -> int:
        return int(getattr(self.ecodes, name))

    def _axis_resolution(self, name: str, default: float) -> float:
        try:
            resolution = float(self.binding.imu_device.absinfo(self._code(name)).resolution)
            return resolution if resolution > 0.0 else default
        except (AttributeError, OSError, TypeError, ValueError):
            return default

    def _read_stick_calibration(self) -> Dict[int, Tuple[float, float, float]]:
        calibration = {}
        for axis_name in (self.profile.stick_x, self.profile.stick_y):
            code = self._code(axis_name)
            try:
                info = self.binding.control_device.absinfo(code)
                center = (float(info.min) + float(info.max)) * 0.5
                half_range = max((float(info.max) - float(info.min)) * 0.5, 1.0)
                flat = max(float(getattr(info, "flat", 0.0)), 0.0)
            except (AttributeError, OSError, TypeError, ValueError):
                center, half_range, flat = 0.0, 32767.0, 500.0
            calibration[code] = (center, half_range, flat)
        return calibration

    def _normalize_stick(self, code: int, value: float) -> float:
        center, half_range, flat = self._stick_calibration[code]
        centered = float(value) - center
        if abs(centered) <= flat:
            return 0.0
        return float(np.clip(centered / half_range, -1.0, 1.0))

    def start(self):
        if self._running:
            return
        self._running = True
        self._threads = [
            threading.Thread(
                target=self._read_loop,
                args=(self.binding.control_device, self.handle_control_event),
                daemon=True,
                name=f"joycon-{self.binding.side}-control",
            ),
            threading.Thread(
                target=self._read_loop,
                args=(self.binding.imu_device, self.handle_imu_event),
                daemon=True,
                name=f"joycon-{self.binding.side}-imu",
            ),
        ]
        for thread in self._threads:
            thread.start()

    def stop(self):
        self._running = False
        for device in (self.binding.control_device, self.binding.imu_device):
            try:
                device.close()
            except (AttributeError, OSError):
                pass
        for thread in self._threads:
            thread.join(timeout=0.5)
        self._threads = []

    def _read_loop(self, device: object, handler: Callable):
        try:
            for event in device.read_loop():
                if not self._running:
                    break
                try:
                    timestamp = float(event.timestamp())
                except (AttributeError, TypeError, ValueError):
                    timestamp = self._clock()
                handler(event.type, event.code, event.value, timestamp)
        except OSError:
            self._running = False

    def handle_control_event(
        self, event_type: int, code: int, value: int, timestamp: Optional[float] = None
    ):
        del timestamp
        with self._lock:
            if event_type == self._code("EV_ABS"):
                if code == self._code(self.profile.stick_x):
                    self._stick[0] = self._normalize_stick(code, value)
                elif code == self._code(self.profile.stick_y):
                    self._stick[1] = self._normalize_stick(code, value)
            elif event_type == self._code("EV_KEY"):
                if value:
                    self._buttons.add(int(code))
                else:
                    self._buttons.discard(int(code))

    def handle_imu_event(
        self, event_type: int, code: int, value: int, timestamp: Optional[float] = None
    ):
        if event_type == self._code("EV_ABS"):
            imu_codes = (
                self._code("ABS_X"), self._code("ABS_Y"), self._code("ABS_Z"),
                self._code("ABS_RX"), self._code("ABS_RY"), self._code("ABS_RZ"),
            )
            if code in imu_codes:
                with self._lock:
                    self._axis_values[imu_codes.index(code)] = float(value)
            return

        if event_type != self._code("EV_SYN") or code != self._code("SYN_REPORT"):
            return
        sample_time = self._clock() if timestamp is None else float(timestamp)
        with self._lock:
            accel_g = self._axis_values[:3] / self._accel_resolution
            gyro_dps = self._axis_values[3:] / self._gyro_resolution
            gyro_rad_s = np.deg2rad(gyro_dps)
            self._orientation.update(accel_g, gyro_rad_s, sample_time)
            if self._orientation.initialized:
                self._last_imu_time = self._clock()

    def recenter_orientation(self):
        with self._lock:
            self._orientation.recenter()

    def is_connected(self) -> bool:
        return self._running

    def snapshot(self) -> JoyConSnapshot:
        with self._lock:
            valid = (
                self._orientation.initialized
                and self._last_imu_time is not None
                and self._clock() - self._last_imu_time <= self._stale_timeout
            )
            pressed = self._buttons.__contains__
            return JoyConSnapshot(
                side=self.binding.side,
                valid=valid,
                stick_x=float(self._stick[0]),
                stick_y=float(self._stick[1]),
                rotation=self._orientation.rotation.copy(),
                # ZL/ZR 是按住式的姿态离合；IMU 无效时故障安全地释放。
                clutch=valid and pressed(self._code(self.profile.clutch)),
                z_positive=pressed(self._code(self.profile.z_positive)),
                z_negative=pressed(self._code(self.profile.z_negative)),
                gripper_close=pressed(self._code(self.profile.gripper_close)),
                gripper_open=pressed(self._code(self.profile.gripper_open)),
                zero=pressed(self._code(self.profile.zero)),
                home=pressed(self._code(self.profile.home)),
                exit=pressed(self._code(self.profile.exit)),
            )


class JoyConPoseAdapter:
    """将单只 Joy-Con 状态适配为 VR 相对位姿控制器的输入。"""

    def __init__(
        self,
        source: JoyConInput,
        translation_speed: float = 0.12,
        deadzone: float = 0.15,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.source = source
        self.translation_speed = float(translation_speed)
        self.deadzone = float(deadzone)
        self._clock = clock
        self._position = np.zeros(3)
        self._last_time = self._clock()
        self.exit_requested = False

    def start(self):
        self._last_time = self._clock()
        self.source.start()

    def stop(self):
        self.source.stop()

    def is_connected(self) -> bool:
        return self.source.is_connected()

    def _deadzone(self, value: float) -> float:
        if abs(value) <= self.deadzone:
            return 0.0
        sign = 1.0 if value > 0.0 else -1.0
        return sign * (abs(value) - self.deadzone) / (1.0 - self.deadzone)

    def get_state(self) -> RelativePoseInput:
        now = self._clock()
        dt = min(max(now - self._last_time, 0.0), 0.05)
        self._last_time = now
        state = self.source.snapshot()

        connected = self.source.is_connected()
        if connected:
            velocity = np.array([
                -self._deadzone(state.stick_y),
                -self._deadzone(state.stick_x),
                float(state.z_positive) - float(state.z_negative),
            ]) * self.translation_speed
            self._position += velocity * dt

        self.exit_requested = self.exit_requested or state.exit
        return RelativePoseInput(
            # 平移只依赖按键/摇杆连接，不依赖 IMU 或离合。
            valid=connected,
            position=self._position.copy(),
            rotation=state.rotation.copy(),
            grip=1.0 if state.valid and state.clutch else 0.0,
            joystick_y=float(state.gripper_close) - float(state.gripper_open),
            button_a=state.zero,
            button_b=state.home,
            translation_without_clutch=True,
        )

    # 与 VRControllerReceiver 的默认读取接口兼容，方便直接复用 VRArmController。
    def get_right_state(self) -> RelativePoseInput:
        return self.get_state()
