#!/usr/bin/env python
"""云端推理服务冒烟测试（在云端本机运行，直连 127.0.0.1）。

测试项：
  1. Ping 握手：token 正确 → 返回模型能力；token 错误 → UNAUTHENTICATED
  2. 首次 InferChunk（reset=true）：返回 (50,7) chunk，记录耗时
  3. trained RTC：第二次请求带 (rtc_k=5, rtc_d=3) → used_delay=3；
     并验证 prefix clamp 生效（服务端日志应有 rtc_d=3->used=3）
  4. rtc_d 超限钳制：rtc_d=99 → used_delay <= trained_max_delay
  5. avg_n=2 多次采样平均：正常返回
  6. 连续 5 次推理统计耗时分布（评估看门狗阈值）

用法：python cloud_inference/smoke_test.py [--addr 127.0.0.1:50051] [--token ...]
"""

import argparse
import sys
import time
from pathlib import Path

import grpc
import numpy as np
import cv2

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import vla_inference_pb2 as pb2
import vla_inference_pb2_grpc as pb2_grpc

PASS, FAIL = "PASS", "FAIL"
results = []


def report(name, ok, detail=""):
    results.append((name, ok))
    print(f"[{PASS if ok else FAIL}] {name}" + (f"  ->  {detail}" if detail else ""))


def make_request(rid, token, reset=False, rtc_k=0, rtc_d=0, avg_n=1):
    rng = np.random.default_rng(rid)
    imgs = []
    for _ in range(2):
        rgb = rng.integers(0, 255, (720, 720, 3), dtype=np.uint8)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        assert ok
        imgs.append(buf.tobytes())
    return pb2.InferenceRequest(
        request_id=rid, token=token,
        cam1_jpeg=imgs[0], cam2_jpeg=imgs[1],
        state=[0.3, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0],
        torque=0.1, task="pick up the cube",
        rtc_k=rtc_k, rtc_d=rtc_d, avg_n=avg_n, reset=reset,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--addr", default="127.0.0.1:50051")
    ap.add_argument("--token", default="ela3-vla-2026")
    args = ap.parse_args()

    channel = grpc.insecure_channel(args.addr)
    stub = pb2_grpc.VLAInferenceStub(channel)

    # 等待服务就绪（模型加载+预热可能耗时较长）
    deadline = time.time() + 300
    trained_max_delay = 0
    while True:
        try:
            rep = stub.Ping(pb2.PingRequest(token=args.token), timeout=10)
            trained_max_delay = rep.trained_max_delay
            report("Ping 握手", True,
                   f"policy={rep.policy_type}, device={rep.device}, "
                   f"trained_max_delay={rep.trained_max_delay}, "
                   f"image={rep.image_width}x{rep.image_height}")
            break
        except grpc.RpcError as e:
            if time.time() > deadline:
                report("Ping 握手", False, f"等待服务就绪超时: {e}")
                sys.exit(1)
            time.sleep(5)

    # 1. 错误 token
    try:
        stub.Ping(pb2.PingRequest(token="wrong-token"), timeout=5)
        report("错误 token 拒绝", False, "未抛出 UNAUTHENTICATED")
    except grpc.RpcError as e:
        report("错误 token 拒绝", e.code() == grpc.StatusCode.UNAUTHENTICATED,
               str(e.code()))

    # 2. 首次推理（reset）
    t0 = time.perf_counter()
    rep = stub.InferChunk(make_request(1, args.token, reset=True), timeout=30)
    dt = (time.perf_counter() - t0) * 1000
    ok = not rep.error and rep.num_steps > 0 and rep.action_dim == 7
    chunk = np.asarray(rep.actions, dtype=np.float64).reshape(rep.num_steps, rep.action_dim) if ok else None
    finite = bool(np.isfinite(chunk).all()) if chunk is not None else False
    report("首次推理 (reset)", ok and finite,
           f"chunk=({rep.num_steps},{rep.action_dim}), roundtrip={dt:.0f}ms, "
           f"server={rep.server_latency_ms:.0f}ms, finite={finite}, err={rep.error!r}")

    # 3. trained RTC prefix（rtc_k=5, rtc_d=3）
    if trained_max_delay > 0:
        rep = stub.InferChunk(make_request(2, args.token, rtc_k=5, rtc_d=3), timeout=30)
        report("trained RTC prefix (k=5,d=3)",
               not rep.error and rep.used_delay == 3,
               f"used_delay={rep.used_delay}, err={rep.error!r}")

        # 4. rtc_d 超限钳制
        rep = stub.InferChunk(make_request(3, args.token, rtc_k=10, rtc_d=99), timeout=30)
        report("rtc_d 超限钳制",
               not rep.error and rep.used_delay <= trained_max_delay,
               f"used_delay={rep.used_delay} (max={trained_max_delay})")
    else:
        report("trained RTC prefix (k=5,d=3)", True, "checkpoint 不支持，跳过")
        report("rtc_d 超限钳制", True, "checkpoint 不支持，跳过")

    # 5. avg_n=2
    rep = stub.InferChunk(make_request(4, args.token, avg_n=2), timeout=60)
    report("avg_n=2 多次采样", not rep.error and rep.num_steps > 0,
           f"server={rep.server_latency_ms:.0f}ms, err={rep.error!r}")

    # 6. 连续 5 次耗时分布
    lat = []
    for i in range(5):
        t0 = time.perf_counter()
        rep = stub.InferChunk(make_request(10 + i, args.token), timeout=30)
        lat.append((time.perf_counter() - t0) * 1000)
        if rep.error:
            report("连续推理稳定性", False, f"第 {i} 次失败: {rep.error}")
            break
    else:
        report("连续推理稳定性", True,
               f"roundtrip ms: min={min(lat):.0f}, avg={sum(lat)/len(lat):.0f}, "
               f"max={max(lat):.0f}")

    n_fail = sum(1 for _, ok in results if not ok)
    print(f"\n===== 冒烟测试完成: {len(results) - n_fail}/{len(results)} 通过 =====")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
