#!/usr/bin/env python
"""CloudInferenceThread 端到端测试（云端本机运行，直连真实推理服务）。

覆盖：
  A. 完整链路：握手(server_info/model_loaded) → request_chunk → chunk_ready
  B. trained RTC：_pending_rtc=(5,3) → _last_used_delay==3
  C. reset 语义：_last_raw_chunk=None（暂停恢复路径）→ 下次请求 reset
  D. 单次诊断：request_single_chunk → single_chunk_ready
  E. 故障注入：错误 token → error；服务不可达 → error；
     运行中杀服务 → link_lost；服务恢复 → link_restored
  （E 用轻量 dummy server，避免真实模型反复加载）

用法：
  # 终端1（真实服务，已在跑则跳过）：
  python cloud_inference/server.py --host 127.0.0.1 --port 50051 --token ela3-vla-2026
  # 终端2：
  python cloud_inference/test_client_thread.py
"""

import os
import sys
import time
import threading
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt6.QtCore import QCoreApplication, Qt

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from widgets.camera_pyqt import CloudInferenceThread  # noqa: E402

PASS, FAIL = "PASS", "FAIL"
results = []


def report(name, ok, detail=""):
    results.append((name, ok))
    print(f"[{PASS if ok else FAIL}] {name}" + (f"  ->  {detail}" if detail else ""))


def wait_for(pred, timeout, desc=""):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.05)
    print(f"  (等待超时: {desc})")
    return False


_rng = np.random.default_rng(7)


def get_obs():
    img0 = _rng.integers(0, 255, (720, 720, 3), dtype=np.uint8)
    img1 = _rng.integers(0, 255, (720, 720, 3), dtype=np.uint8)
    state = np.array([0.3, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return img0, img1, state


DIRECT = Qt.ConnectionType.DirectConnection


def test_real_server():
    """A~D：连真实推理服务（127.0.0.1:50051）。"""
    print("===== A~D: 真实服务端到端 =====")
    events = {"loaded": False, "info": None, "chunks": [], "singles": [],
              "errors": [], "status": []}
    thr = CloudInferenceThread("127.0.0.1:50051", "ela3-vla-2026",
                               "pick up the cube", get_obs, fps=20,
                               rtc_trained=True)
    thr.model_loaded.connect(lambda: events.__setitem__("loaded", True), DIRECT)
    thr.server_info.connect(lambda d: events.__setitem__("info", d), DIRECT)
    thr.chunk_ready.connect(lambda c, lat: events["chunks"].append((c, lat)), DIRECT)
    thr.single_chunk_ready.connect(lambda c, info: events["singles"].append((c, info)), DIRECT)
    thr.error.connect(lambda e: events["errors"].append(e), DIRECT)
    thr.status.connect(lambda s: events["status"].append(s), DIRECT)
    thr.start()

    ok = wait_for(lambda: events["loaded"] or events["errors"], 30, "握手")
    report("A1 握手+model_loaded", ok and events["loaded"] and not events["errors"],
           f"info={events['info']}, err={events['errors'][:1]}")
    report("A2 trained RTC 能力识别",
           thr._trained_rtc_active and thr._trained_max_delay == 6,
           f"active={thr._trained_rtc_active}, max_delay={thr._trained_max_delay}")

    thr.request_chunk()
    ok = wait_for(lambda: len(events["chunks"]) >= 1, 15, "首个 chunk")
    if ok:
        c, lat = events["chunks"][0]
        report("A3 chunk_ready", c.shape == (50, 7) and np.isfinite(c).all(),
               f"shape={c.shape}, latency={lat * 1000:.0f}ms")
    else:
        report("A3 chunk_ready", False)

    # trained RTC：(k=5, d=3)
    thr._pending_rtc = (5, 3)
    thr.request_chunk()
    ok = wait_for(lambda: len(events["chunks"]) >= 2, 15, "RTC chunk")
    report("B1 trained RTC (k=5,d=3)", ok and thr._last_used_delay == 3,
           f"used_delay={thr._last_used_delay}")

    # reset 语义（暂停恢复路径：_last_raw_chunk=None → 下次请求 reset）
    thr._last_raw_chunk = None
    thr._pending_rtc = None
    thr.request_chunk()
    ok = wait_for(lambda: len(events["chunks"]) >= 3, 15, "reset chunk")
    report("C1 reset 后正常返回", ok, f"chunks={len(events['chunks'])}")

    # 单次诊断
    thr.request_single_chunk()
    ok = wait_for(lambda: len(events["singles"]) >= 1, 45, "single chunk")
    if ok:
        c, info = events["singles"][0]
        report("D1 single_chunk_ready",
               c.shape == (50, 7) and info["state"].shape == (7,)
               and info["rgb0"].shape == (720, 720, 3),
               f"shape={c.shape}, info keys={sorted(info.keys())}")
    else:
        report("D1 single_chunk_ready", False)

    thr.stop()
    thr.wait(5000)


def _start_dummy_server(port, token, trained_max_delay=6):
    """轻量 dummy 服务：不加载模型，InferChunk 返回固定 chunk，可快速杀/起。"""
    import grpc
    from concurrent import futures
    import vla_inference_pb2 as pb2
    import vla_inference_pb2_grpc as pb2_grpc

    class Dummy(pb2_grpc.VLAInferenceServicer):
        def Ping(self, request, context):
            if request.token != token:
                context.abort(grpc.StatusCode.UNAUTHENTICATED, "bad token")
            return pb2.PingReply(server_time_ms=0, policy_type="dummy",
                                 model_path="dummy", device="cpu",
                                 trained_max_delay=trained_max_delay,
                                 image_height=720, image_width=720)

        def InferChunk(self, request, context):
            if request.token != token:
                context.abort(grpc.StatusCode.UNAUTHENTICATED, "bad token")
            actions = np.zeros((50, 7), dtype=np.float32).ravel().tolist()
            return pb2.InferenceReply(request_id=request.request_id,
                                      actions=actions, num_steps=50,
                                      action_dim=7, server_latency_ms=1.0,
                                      used_delay=min(request.rtc_d, trained_max_delay))

    srv = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    pb2_grpc.add_VLAInferenceServicer_to_server(Dummy(), srv)
    srv.add_insecure_port(f"127.0.0.1:{port}")
    srv.start()
    return srv


def test_fault_injection():
    """E：故障注入（dummy server，端口 50099）。"""
    print("===== E: 故障注入（dummy server） =====")
    port = 50099

    # E1: 服务不可达 → error
    events = {"errors": [], "loaded": False}
    thr = CloudInferenceThread(f"127.0.0.1:{port}", "ela3-vla-2026",
                               "t", get_obs, fps=20)
    thr.error.connect(lambda e: events["errors"].append(e), DIRECT)
    thr.model_loaded.connect(lambda: events.__setitem__("loaded", True), DIRECT)
    thr.start()
    wait_for(lambda: events["errors"] or events["loaded"], 20, "不可达握手")
    report("E1 服务不可达 → error", bool(events["errors"]) and not events["loaded"],
           events["errors"][:1] and events["errors"][0].splitlines()[0])
    thr.stop(); thr.wait(5000)

    srv = _start_dummy_server(port, "ela3-vla-2026")
    try:
        # E2: 错误 token → error（UNAUTHENTICATED）
        events2 = {"errors": [], "loaded": False}
        thr = CloudInferenceThread(f"127.0.0.1:{port}", "wrong-token",
                                   "t", get_obs, fps=20)
        thr.error.connect(lambda e: events2["errors"].append(e), DIRECT)
        thr.model_loaded.connect(lambda: events2.__setitem__("loaded", True), DIRECT)
        thr.start()
        wait_for(lambda: events2["errors"] or events2["loaded"], 15, "错误token握手")
        report("E2 错误 token → error", bool(events2["errors"]) and not events2["loaded"],
               events2["errors"][:1] and events2["errors"][0].splitlines()[0])
        thr.stop(); thr.wait(5000)

        # E3: 运行中杀服务 → link_lost；重启 → link_restored
        events3 = {"loaded": False, "lost": [], "restored": [], "chunks": 0}
        thr = CloudInferenceThread(f"127.0.0.1:{port}", "ela3-vla-2026",
                                   "t", get_obs, fps=20)
        thr.model_loaded.connect(lambda: events3.__setitem__("loaded", True), DIRECT)
        thr.link_lost.connect(lambda m: events3["lost"].append(m), DIRECT)
        thr.link_restored.connect(lambda: events3["restored"].append(1), DIRECT)
        thr.chunk_ready.connect(lambda c, l: events3.__setitem__("chunks", events3["chunks"] + 1), DIRECT)
        thr.start()
        wait_for(lambda: events3["loaded"], 15, "dummy 握手")
        thr.request_chunk()
        wait_for(lambda: events3["chunks"] >= 1, 10, "dummy chunk")

        srv.stop(grace=0).wait()  # 杀服务
        srv = None
        thr.request_chunk()  # 应触发 link_lost
        ok = wait_for(lambda: len(events3["lost"]) >= 1, 20, "link_lost")
        report("E3 杀服务 → link_lost", ok,
               events3["lost"][:1] and events3["lost"][0])

        srv = _start_dummy_server(port, "ela3-vla-2026")
        ok = wait_for(lambda: len(events3["restored"]) >= 1, 60, "link_restored")
        report("E4 重启服务 → link_restored（心跳重连）", ok)
        if ok:
            thr.request_chunk()
            ok2 = wait_for(lambda: events3["chunks"] >= 2, 10, "重连后 chunk")
            report("E5 重连后推理恢复", ok2, f"chunks={events3['chunks']}")
        thr.stop(); thr.wait(5000)
    finally:
        if srv is not None:
            srv.stop(grace=0).wait()


def main():
    app = QCoreApplication(sys.argv)
    test_real_server()
    test_fault_injection()
    n_fail = sum(1 for _, ok in results if not ok)
    print(f"\n===== 端到端测试完成: {len(results) - n_fail}/{len(results)} 通过 =====")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
