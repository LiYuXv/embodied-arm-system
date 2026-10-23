#!/usr/bin/env python3
"""Meta Quest 3 VR controller receiver for OpenArm MuJoCo simulation.

Extracted from dora-openarm-vr (https://github.com/enactic/dora-openarm-vr)
and adapted to work standalone without dora-rs dependency.

Protocol: UDP JSON on port 5006 (default)
Data format from Quest APK:
  - lc/rc/rf: left controller / right controller / reference pose
    {x, y, z, qx, qy, qz, qw} in Unity left-handed coordinates
  - lt/rt: left/right trigger 0.0-1.0
  - lg/rg: left/right grip 0.0-1.0
  - lsx/lsy/rsx/rsy: thumbsticks -1.0 to 1.0
  - a/b/x/y: buttons
  - v/vl/vr: validity (0=OK, 1=STALE, 2=INVALID)
"""

from __future__ import annotations

import collections
import json
import select
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.spatial.transform import Rotation


# ── Frame alignment (tune these for your setup) ─────────────────────────────
# Maps rectified VR-frame vectors to the robot workspace frame.
# The rectified VR frame is gravity-aligned (see gravity_aligned_rotation):
#   +y = up, +z = backward (away from where the user faces), +x = user right.
#   ws_x (robot forward) = -ref_z (user forward)
#   ws_y (robot left)    = -ref_x (user left)
#   ws_z (robot up)      = +ref_y (up)
_FRAME_ROT: np.ndarray = np.array(
    [
        [0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)

# Offset applied after workspace mapping. Kept at zero: relative-mode control
# is offset-invariant (only deltas matter), and absolute mode applies its own
# waist-to-robot-base offset downstream in the control script.
FRAME_OFFSET_NECK: np.ndarray = np.array([0.0, 0.0, 0.0], dtype=np.float64)
# ─────────────────────────────────────────────────────────────────────────────

_R_FRAME = Rotation.from_matrix(_FRAME_ROT)
_IDENTITY_REF = {
    "x": 0.0,
    "y": 0.0,
    "z": 0.0,
    "qx": 0.0,
    "qy": 0.0,
    "qz": 0.0,
    "qw": 1.0,
}

VALID_OK = 0
VALID_STALE = 1
VALID_INVALID = 2
_VALID_NAMES = {VALID_OK: "OK", VALID_STALE: "STALE", VALID_INVALID: "INVALID"}


def _slerp_quat(q1: np.ndarray, q2: np.ndarray, alpha: float) -> np.ndarray:
    """Spherical linear interpolation between two quaternions."""
    dot = np.dot(q1, q2)
    if dot < 0.0:
        q2 = -q2
        dot = -dot
    if dot > 0.9995:
        res = q1 + alpha * (q2 - q1)
        return res / np.linalg.norm(res)

    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * alpha
    sin_theta = np.sin(theta)

    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return s0 * q1 + s1 * q2


class OneEuroPoseSmoother:
    """1 Euro Filter applied to position (adaptive cutoff) and rotation (SLERP)."""

    def __init__(
        self, min_cutoff: float = 2.0, beta: float = 0.04, d_cutoff: float = 1.5
    ):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.p_prev = None
        self.q_prev = None
        self.dp_prev = np.zeros(3)
        self.t_prev = None

    def reset(self) -> None:
        """Clear state so next sample is treated as a fresh start."""
        self.p_prev = None
        self.q_prev = None
        self.dp_prev = np.zeros(3)
        self.t_prev = None

    def smooth(self, t: float, target_pose: np.ndarray | None) -> np.ndarray | None:
        if target_pose is None:
            return None

        t_p = target_pose[0:3]
        t_q = target_pose[3:7]

        if self.t_prev is None or self.p_prev is None:
            self.p_prev = t_p.copy()
            self.q_prev = t_q.copy()
            self.t_prev = t
            return target_pose.copy()

        dt = t - self.t_prev
        if dt <= 0.0:
            return target_pose.copy()

        def get_alpha(dt: float, cutoff: float) -> float:
            tau = 1.0 / (2 * np.pi * cutoff)
            return 1.0 / (1.0 + tau / dt)

        dp_raw = (t_p - self.p_prev) / dt
        alpha_d = get_alpha(dt, self.d_cutoff)
        dp_filtered = alpha_d * dp_raw + (1.0 - alpha_d) * self.dp_prev

        speed = np.linalg.norm(dp_filtered)
        cutoff_p = self.min_cutoff + self.beta * speed

        alpha_p = get_alpha(dt, cutoff_p)
        p_hat = self.p_prev + alpha_p * (t_p - self.p_prev)
        q_hat = _slerp_quat(self.q_prev, t_q, alpha_p)

        self.p_prev = p_hat
        self.q_prev = q_hat
        self.dp_prev = dp_filtered
        self.t_prev = t

        return np.array(
            [p_hat[0], p_hat[1], p_hat[2], q_hat[0], q_hat[1], q_hat[2], q_hat[3]],
            dtype=np.float32,
        )


def parse_lh_to_rh(c: dict) -> tuple[np.ndarray, Rotation]:
    """Convert Unity left-handed pose dict to right-handed (position, Rotation).

    Input keys: x, y, z (meters), qx, qy, qz, qw (Unity quaternion, scalar-last).
    Flip: z → -z, qx → -qx, qy → -qy.
    """
    pos = np.array([c["x"], c["y"], -c["z"]], dtype=np.float64)
    rot = Rotation.from_quat([-c["qx"], -c["qy"], c["qz"], c["qw"]])
    return pos, rot


def gravity_aligned_rotation(rot: Rotation) -> Rotation:
    """Extract the yaw-only (gravity-aligned) component of a headset rotation.

    The rectified world frame has +Y up (gravity-referenced by the Quest's
    IMU/visual tracking). The gaze direction is projected onto the horizontal
    plane, discarding pitch and roll. This makes the controller rectification
    independent of headset tilt — e.g. when the headset hangs around the neck
    during teleoperation.
    """
    gaze = rot.apply(np.array([0.0, 0.0, -1.0]))  # HMD forward in world frame
    fwd = np.array([gaze[0], 0.0, gaze[2]], dtype=np.float64)
    n = np.linalg.norm(fwd)
    if n < 1e-6:
        # Gaze pointing straight up/down: fall back to HMD right vector
        r = rot.apply(np.array([1.0, 0.0, 0.0]))
        fwd = np.array([r[0], 0.0, r[2]], dtype=np.float64)
        n = np.linalg.norm(fwd)
        if n < 1e-6:
            return Rotation.identity()
    fwd /= n
    up = np.array([0.0, 1.0, 0.0])
    back = -fwd
    right = np.cross(up, back)
    m = np.column_stack([right, up, back])
    return Rotation.from_matrix(m)


def pose_to_array(pos: np.ndarray, rot: Rotation) -> np.ndarray:
    """Convert to [x, y, z, qw, qx, qy, qz] array."""
    q = rot.as_quat()
    return np.array([pos[0], pos[1], pos[2], q[3], q[0], q[1], q[2]], dtype=np.float32)


def map_trigger_to_gripper(trigger: float, side: str) -> float:
    """Map trigger value (0.0-1.0) to gripper joint value (0.0-0.044).

    OpenArm gripper: 0.0 = closed, 0.044 = fully open.
    Trigger 0.0 (released) = open, Trigger 1.0 (pressed) = closed.
    """
    trigger = float(np.clip(trigger, 0.0, 1.0))
    # Inverted: release trigger = open gripper, press trigger = close gripper
    return float(0.044 * (1.0 - trigger))


@dataclass
class VRControllerState:
    """Current state of one VR controller."""

    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    quaternion: np.ndarray = field(default_factory=lambda: np.array([1.0, 0, 0, 0]))  # w,x,y,z
    trigger: float = 0.0
    grip: float = 0.0
    joystick_x: float = 0.0
    joystick_y: float = 0.0
    button_a: bool = False
    button_b: bool = False
    valid: bool = False
    timestamp: float = 0.0


@dataclass
class HeadsetState:
    """Current state of the VR headset (reference pose)."""

    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    quaternion: np.ndarray = field(default_factory=lambda: np.array([1.0, 0, 0, 0]))  # w,x,y,z
    yaw: float = 0.0    # degrees, heading about the up axis
    pitch: float = 0.0  # degrees, tilt up/down (0 = level)
    roll: float = 0.0   # degrees, tilt sideways (0 = level)
    valid: bool = False


class JsonUdpReceiver:
    """Background thread that binds a UDP socket and keeps the latest parsed JSON packet."""

    def __init__(self, host: str, port: int, buf_size: int = 4096) -> None:
        self._host = host
        self._port = port
        self._buf_size = buf_size
        self._lock = threading.Lock()
        self._latest: dict | None = None
        self._recv_ts: collections.deque[int] = collections.deque(maxlen=512)
        self._recv_count = 0
        self._running = True
        self._connected = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def latest(self) -> dict | None:
        with self._lock:
            return self._latest

    def is_alive(self) -> bool:
        """Check if the background receive thread is running."""
        return self._running and self._thread.is_alive()

    def is_connected(self) -> bool:
        """Check if we've received data recently."""
        with self._lock:
            if not self._recv_ts:
                return False
            # Consider connected if we received data in the last 500ms
            return (time.time_ns() - self._recv_ts[-1]) < 500_000_000

    def drain_recv_timestamps(self) -> list[int]:
        """Return and clear the arrival timestamps (ns) collected since last call."""
        with self._lock:
            items = list(self._recv_ts)
            self._recv_ts.clear()
            return items

    def packet_count(self) -> int:
        """Total number of valid packets received since start (monotonic)."""
        with self._lock:
            return self._recv_count

    def close(self) -> None:
        self._running = False

    def _parse_packet(self, data: bytes) -> dict | None:
        try:
            line = data.decode("utf-8", errors="replace").strip()
            if not line:
                return None
            return json.loads(line)
        except json.JSONDecodeError:
            return None

    def _loop(self) -> None:
        while self._running:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as srv:
                    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    srv.bind((self._host, self._port))
                    srv.settimeout(1.0)
                    print(f"[VR] Listening on UDP {self._host}:{self._port}")

                    while self._running:
                        try:
                            data, _ = srv.recvfrom(self._buf_size)
                            recv_ns = time.time_ns()
                            last_msg = self._parse_packet(data)
                            arrivals = [recv_ns] if last_msg is not None else []

                            # Drain queued datagrams, keep only the freshest
                            while select.select([srv], [], [], 0.0)[0]:
                                data, _ = srv.recvfrom(self._buf_size)
                                recv_ns = time.time_ns()
                                parsed = self._parse_packet(data)
                                if parsed is not None:
                                    arrivals.append(recv_ns)
                                    last_msg = parsed

                            with self._lock:
                                self._recv_ts.extend(arrivals)
                                if last_msg is not None:
                                    self._latest = last_msg
                                    self._connected = True
                                    self._recv_count += max(len(arrivals), 1)

                        except TimeoutError:
                            continue
                        except Exception:
                            pass
            except OSError:
                if self._running:
                    time.sleep(1.0)


class QuestPoseProcessor:
    """Process raw Quest JSON data into robot workspace poses."""

    def __init__(self, frame_offset: np.ndarray | None = None):
        self.frame_offset = frame_offset if frame_offset is not None else FRAME_OFFSET_NECK

    def process(
        self, msg: dict
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        """Process raw message, return (right_pose, left_pose, reference_pose).

        Each pose is [x, y, z, qw, qx, qy, qz] or None if not available.
        """
        ref_raw = msg.get("rf")
        right_raw = msg.get("rc")
        left_raw = msg.get("lc")

        p_ref, r_ref = parse_lh_to_rh(ref_raw or _IDENTITY_REF)
        # Rectify against the gravity-aligned (yaw-only) component of the HMD
        # pose, so headset tilt (e.g. hung around the neck) does not skew the
        # controller→workspace axis mapping.
        r_ref_level = gravity_aligned_rotation(r_ref)
        active_p_ref = p_ref
        active_r_ref_inv = r_ref_level.inv()

        # Controller orientation fix applied after workspace mapping.
        # Base: Rot_z(90) aligns the Quest controller grip frame to the EE frame.
        # Extra Rot_x(180): flips the pitch direction (conjugating the relative
        # rotation by Rot_x(180) negates Y/Z-axis rotation components).
        # If roll also feels inverted, remove the x-180 term and try y-180 instead.
        r_fix = Rotation.from_euler("z", 90, degrees=True) * Rotation.from_euler("x", 180, degrees=True)

        def _rectify(raw: dict) -> np.ndarray:
            p, r = parse_lh_to_rh(raw)
            p_rel = active_r_ref_inv.apply(p - active_p_ref)
            r_rel = active_r_ref_inv * r
            p_out = _R_FRAME.apply(p_rel) + self.frame_offset
            r_out = _R_FRAME * r_rel * r_fix
            return pose_to_array(p_out, r_out)

        pose_right = _rectify(right_raw) if right_raw is not None else None
        pose_left = _rectify(left_raw) if left_raw is not None else None
        pose_reference = pose_to_array(p_ref, r_ref) if ref_raw is not None else None
        return pose_right, pose_left, pose_reference


class VRControllerReceiver:
    """High-level VR controller receiver with smoothing and state management.

    Usage:
        receiver = VRControllerReceiver(port=5006)
        receiver.start()

        # In your control loop:
        right_state = receiver.get_right_state()
        left_state = receiver.get_left_state()
        if right_state.valid:
            # Use right_state.position, right_state.quaternion, etc.

        receiver.stop()
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 5006,
        smoothing_enabled: bool = True,
        min_cutoff: float = 2.0,
        beta: float = 0.04,
    ):
        self.host = host
        self.port = port
        self.smoothing_enabled = smoothing_enabled

        self._receiver: JsonUdpReceiver | None = None
        self._processor = QuestPoseProcessor()

        self._smoother_right = OneEuroPoseSmoother(min_cutoff=min_cutoff, beta=beta)
        self._smoother_left = OneEuroPoseSmoother(min_cutoff=min_cutoff, beta=beta)

        self._lock = threading.Lock()
        self._right_state = VRControllerState()
        self._left_state = VRControllerState()
        self._headset_state = HeadsetState()
        self._running = False
        self._process_thread: threading.Thread | None = None

        self._prev_v_right = VALID_OK
        self._prev_v_left = VALID_OK

    def start(self) -> None:
        """Start the background processing thread."""
        if self._running:
            return
        if self._receiver is None or not self._receiver.is_alive():
            self._receiver = JsonUdpReceiver(self.host, self.port)
        self._running = True
        self._process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self._process_thread.start()
        print(f"[VR] VR receiver started on {self.host}:{self.port}")

    def stop(self) -> None:
        """Stop the receiver."""
        self._running = False
        if self._process_thread:
            self._process_thread.join(timeout=2.0)
            self._process_thread = None
        if self._receiver:
            self._receiver.close()
            self._receiver = None
        print("[VR] VR receiver stopped")

    def is_connected(self) -> bool:
        """Check if VR headset is sending data."""
        return self._receiver is not None and self._receiver.is_connected()

    def get_right_state(self) -> VRControllerState:
        """Get current right controller state (thread-safe copy)."""
        with self._lock:
            return VRControllerState(
                position=self._right_state.position.copy(),
                rotation=self._right_state.rotation.copy(),
                quaternion=self._right_state.quaternion.copy(),
                trigger=self._right_state.trigger,
                grip=self._right_state.grip,
                joystick_x=self._right_state.joystick_x,
                joystick_y=self._right_state.joystick_y,
                button_a=self._right_state.button_a,
                button_b=self._right_state.button_b,
                valid=self._right_state.valid,
                timestamp=self._right_state.timestamp,
            )

    def get_left_state(self) -> VRControllerState:
        """Get current left controller state (thread-safe copy)."""
        with self._lock:
            return VRControllerState(
                position=self._left_state.position.copy(),
                rotation=self._left_state.rotation.copy(),
                quaternion=self._left_state.quaternion.copy(),
                trigger=self._left_state.trigger,
                grip=self._left_state.grip,
                joystick_x=self._left_state.joystick_x,
                joystick_y=self._left_state.joystick_y,
                button_a=self._left_state.button_a,
                button_b=self._left_state.button_b,
                valid=self._left_state.valid,
                timestamp=self._left_state.timestamp,
            )

    def get_headset_state(self) -> HeadsetState:
        """Get current headset state (thread-safe copy)."""
        with self._lock:
            return HeadsetState(
                position=self._headset_state.position.copy(),
                quaternion=self._headset_state.quaternion.copy(),
                yaw=self._headset_state.yaw,
                pitch=self._headset_state.pitch,
                roll=self._headset_state.roll,
                valid=self._headset_state.valid,
            )

    def reset_smoothing(self) -> None:
        """Reset pose smoothers (call when re-enabling VR control)."""
        self._smoother_right.reset()
        self._smoother_left.reset()

    def _process_loop(self) -> None:
        """Background loop that processes incoming VR data."""
        while self._running:
            receiver = self._receiver
            if receiver is None:
                time.sleep(0.01)
                continue
            msg = receiver.latest()
            if msg is None:
                time.sleep(0.001)
                continue

            now = time.perf_counter()

            # Parse validity
            v_right = int(msg.get("vr", VALID_OK))
            v_left = int(msg.get("vl", VALID_OK))

            # Process poses
            pose_right_raw, pose_left_raw, _ = self._processor.process(msg)

            # Apply smoothing and validity filtering
            if v_right == VALID_INVALID:
                if self._prev_v_right != VALID_INVALID:
                    self._smoother_right.reset()
                pose_right = None
            else:
                if self.smoothing_enabled:
                    pose_right = self._smoother_right.smooth(now, pose_right_raw)
                else:
                    pose_right = pose_right_raw

            if v_left == VALID_INVALID:
                if self._prev_v_left != VALID_INVALID:
                    self._smoother_left.reset()
                pose_left = None
            else:
                if self.smoothing_enabled:
                    pose_left = self._smoother_left.smooth(now, pose_left_raw)
                else:
                    pose_left = pose_left_raw

            self._prev_v_right = v_right
            self._prev_v_left = v_left

            # Parse headset (reference) pose for tilt display
            ref_raw = msg.get("rf")
            if ref_raw is not None:
                hmd_pos, hmd_rot = parse_lh_to_rh(ref_raw)
                hmd_yaw, hmd_pitch, hmd_roll = (
                    float(a) for a in hmd_rot.as_euler("YXZ", degrees=True)
                )
                hmd_quat = hmd_rot.as_quat()  # x,y,z,w
                headset_valid = True
            else:
                headset_valid = False

            # Update states
            with self._lock:
                if headset_valid:
                    self._headset_state.position = hmd_pos.copy()
                    self._headset_state.quaternion = np.array(
                        [hmd_quat[3], hmd_quat[0], hmd_quat[1], hmd_quat[2]]
                    )
                    self._headset_state.yaw = hmd_yaw
                    self._headset_state.pitch = hmd_pitch
                    self._headset_state.roll = hmd_roll
                    self._headset_state.valid = True
                else:
                    self._headset_state.valid = False

                if pose_right is not None:
                    self._right_state.position = pose_right[0:3].copy()
                    self._right_state.quaternion = pose_right[3:7].copy()  # w,x,y,z
                    self._right_state.rotation = Rotation.from_quat(
                        [pose_right[4], pose_right[5], pose_right[6], pose_right[3]]
                    ).as_matrix()
                    self._right_state.valid = True
                    self._right_state.timestamp = now
                else:
                    self._right_state.valid = False

                if pose_left is not None:
                    self._left_state.position = pose_left[0:3].copy()
                    self._left_state.quaternion = pose_left[3:7].copy()
                    self._left_state.rotation = Rotation.from_quat(
                        [pose_left[4], pose_left[5], pose_left[6], pose_left[3]]
                    ).as_matrix()
                    self._left_state.valid = True
                    self._left_state.timestamp = now
                else:
                    self._left_state.valid = False

                # Update buttons/triggers (always forwarded)
                self._right_state.trigger = float(msg.get("rt", 0.0))
                self._right_state.grip = float(msg.get("rg", 0.0))
                self._right_state.joystick_x = float(msg.get("rsx", 0.0))
                self._right_state.joystick_y = float(msg.get("rsy", 0.0))
                self._right_state.button_a = bool(msg.get("a", False))
                self._right_state.button_b = bool(msg.get("b", False))

                self._left_state.trigger = float(msg.get("lt", 0.0))
                self._left_state.grip = float(msg.get("lg", 0.0))
                self._left_state.joystick_x = float(msg.get("lsx", 0.0))
                self._left_state.joystick_y = float(msg.get("lsy", 0.0))
                self._left_state.button_a = bool(msg.get("x", False))
                self._left_state.button_b = bool(msg.get("y", False))

            time.sleep(0.001)  # ~1000 Hz processing rate


def get_local_ips() -> list[str]:
    """Return this machine's IPv4 addresses (for configuring the Quest APK)."""
    ips: list[str] = []
    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith("127.") and ip not in ips:
                ips.append(ip)
    except OSError:
        pass
    if not ips:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                ips.append(s.getsockname()[0])
        except OSError:
            pass
    return ips


# Convenience function for quick testing
def test_receiver(port: int = 5006, duration: float = 10.0, host: str = "0.0.0.0"):
    """Test the VR receiver for a specified duration."""
    receiver = VRControllerReceiver(host=host, port=port)
    receiver.start()

    ips = get_local_ips()
    print(f"Testing VR receiver for {duration} seconds...")
    print(f"Listening on UDP {host}:{port}")
    if ips:
        print(f"Local IPs (enter one of these in the Quest APK): {', '.join(ips)}")
    print("Move your Quest controllers and press buttons.")
    print()

    start_time = time.time()
    last_rate_t = start_time
    last_pkt_total = 0
    pkt_rate = 0.0
    while time.time() - start_time < duration:
        now = time.time()
        if now - last_rate_t >= 1.0:
            total = receiver._receiver.packet_count() if receiver._receiver else 0
            pkt_rate = (total - last_pkt_total) / (now - last_rate_t)
            last_pkt_total = total
            last_rate_t = now

        right = receiver.get_right_state()
        left = receiver.get_left_state()

        status = []
        if right.valid:
            status.append(
                f"R: pos=({right.position[0]:+.3f},{right.position[1]:+.3f},{right.position[2]:+.3f}) "
                f"trig={right.trigger:.2f} grip={right.grip:.2f} "
                f"joy=({right.joystick_x:+.2f},{right.joystick_y:+.2f}) "
                f"btn={'A' if right.button_a else '-'}{'B' if right.button_b else '-'}"
            )
        else:
            status.append("R: [INVALID]")

        if left.valid:
            status.append(
                f"L: pos=({left.position[0]:+.3f},{left.position[1]:+.3f},{left.position[2]:+.3f}) "
                f"trig={left.trigger:.2f} grip={left.grip:.2f} "
                f"joy=({left.joystick_x:+.2f},{left.joystick_y:+.2f}) "
                f"btn={'X' if left.button_a else '-'}{'Y' if left.button_b else '-'}"
            )
        else:
            status.append("L: [INVALID]")

        conn = "CONNECTED" if receiver.is_connected() else f"waiting... ({pkt_rate:.0f} pkt/s)"
        print(f"\r[{conn}] {' | '.join(status)}          ", end="", flush=True)
        time.sleep(0.05)

    print()
    receiver.stop()
    print("Test complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test VR controller receiver")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5006, help="UDP port (default: 5006)")
    parser.add_argument("--duration", type=float, default=30.0, help="Test duration in seconds")
    args = parser.parse_args()

    test_receiver(port=args.port, duration=args.duration, host=args.host)
