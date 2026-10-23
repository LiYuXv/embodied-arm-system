#!/usr/bin/env python
"""VLA 云端推理服务（方案 A：全量云端推理）—— pi0.5 / SmolVLA / ACT。

在云服务器上常驻运行：加载 checkpoint、预热推理，通过 gRPC 接收本地端
（widgets/camera_pyqt.py 的 CloudInferenceThread）发来的观测
（2×JPEG + 位姿 + 力矩 + 任务文本），返回完整动作 chunk。

特性（与本地 VLAInferenceThread 的推理逻辑逐点对应）：
- 按 checkpoint config.json 的 type 自动识别策略（pi05/smolvla 走外部
  pre/post processor + task 文本 + 可选独立力矩键；ACT 图像缩 512、
  力矩并入 state、无 processor）；
- trained RTC（arXiv:2512.05964）：checkpoint 带 rtc_training_max_delay>0
  时自动注入 RTCConfig(mode="trained")；上一段归一化 chunk 保存在服务端
  会话内，客户端每次请求只发 (rtc_k, rtc_d)；reset=true 清空 prefix 状态；
- avg_n 多次采样平均（对 _get_action_chunk 包装，与本地一致）；
- token 校验（UNAUTHENTICATED）、单推理锁串行化、启动 2 次预热。

启动（tmux 常驻）：
  cd /root/autodl-tmp/debugger
  python cloud_inference/server.py \
      --host 0.0.0.0 --port 50051 \
      --model_path ckpt/pi05_ela3_fullft/checkpoints/last \
      --token ela3-vla-2026
"""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")       # 离线缓存 tokenizer，避免 HF 空转重试
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import logging
import signal
import sys
import threading
import time
from concurrent import futures
from pathlib import Path

import grpc
import numpy as np
import cv2
import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
_ROOT = _HERE.parent  # debugger/：注册 smolvla_torque 自定义策略用
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import vla_inference_pb2 as pb2
import vla_inference_pb2_grpc as pb2_grpc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("vla_cloud_server")

# ACT 训练时的图像输入尺寸（与 camera_pyqt.py VLAInferenceThread.ACT_IMAGE_SIZE 一致）
ACT_IMAGE_SIZE = 512

DEFAULT_TOKEN = "ela3-vla-2026"
DEFAULT_MODEL = "ckpt/pi05_ela3_fullft/checkpoints/last"


class VLAInferenceServicer(pb2_grpc.VLAInferenceServicer):
    """gRPC 服务实现：会话内保存 policy 与 _last_raw_chunk（归一化空间）。"""

    def __init__(self, model_path, token, device=None):
        self._token = token
        self._infer_lock = threading.Lock()  # 单 GPU，推理串行化
        self._last_raw_chunk = None          # 上次推理的归一化 chunk (H,A) tensor
        self._avg_n = 1                      # 当前请求的多次采样平均次数
        self._load_model(model_path, device)

    # ==================== 模型加载（镜像 VLAInferenceThread.run 的加载段） ====================

    def _load_model(self, model_path, device):
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class
        from lerobot.processor import DataProcessorPipeline
        try:
            import configuration_smolvla_torque  # noqa: F401 注册 smolvla_torque 策略
        except Exception:
            logger.warning("smolvla_torque 注册失败", exc_info=True)

        ckpt = Path(model_path).resolve()
        if (ckpt / "pretrained_model").is_dir():
            ckpt = ckpt / "pretrained_model"
        self._model_path = str(ckpt)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device

        try:
            config = PreTrainedConfig.from_pretrained(str(ckpt))
            policy_type = str(getattr(config, "type", "") or "").lower()
        except Exception:
            config = None
            with open(ckpt / "config.json") as f:
                policy_type = str(json.load(f).get("type", "")).lower()
        self._policy_type = policy_type
        self._is_act = (policy_type == "act")
        policy = get_policy_class(policy_type).from_pretrained(str(ckpt))
        policy.to(device)
        policy.eval()
        cfg = config if config is not None else policy.config

        # 多次采样平均包装（与本地一致）：同一观测采 avg_n 次取均值。
        # avg_n 由每个请求携带，实例属性 self._avg_n 在 InferChunk 中更新。
        if hasattr(policy, "_get_action_chunk"):
            orig_get_chunk = policy._get_action_chunk
            servicer = self

            def _averaged_get_chunk(batch, noise=None, **kwargs):
                n = max(1, int(servicer._avg_n))
                if n == 1:
                    return orig_get_chunk(batch, noise, **kwargs)
                outs = [orig_get_chunk(batch, None, **kwargs) for _ in range(n)]
                return torch.stack(outs, dim=0).mean(dim=0)

            policy._get_action_chunk = _averaged_get_chunk
        policy.reset()

        # trained RTC：checkpoint 带 rtc_training_max_delay>0 时自动启用。
        # 是否使用 prefix 由客户端每个请求的 rtc_d 决定（0=不用），
        # 因此服务端总是准备好能力即可。
        self._trained_max_delay = 0
        if not self._is_act and hasattr(policy, "init_rtc_processor"):
            md = int(getattr(cfg, "rtc_training_max_delay", 0) or 0)
            if md > 0:
                from lerobot.policies.rtc.configuration_rtc import RTCConfig
                policy.config.rtc_config = RTCConfig(enabled=True, mode="trained")
                policy.init_rtc_processor()
                self._trained_max_delay = md
                logger.info(f"trained RTC 已启用 (max_delay={md})")

        if self._is_act:
            self._preprocessor = None
            self._postprocessor = None
            self._torque_key = None
            state_ft = (getattr(cfg, "input_features", None) or {}
                        ).get("observation.state")
            try:
                self._act_state_dim = int(state_ft.shape[0])
            except Exception:
                self._act_state_dim = 7
        else:
            self._preprocessor = DataProcessorPipeline.from_pretrained(
                str(ckpt), config_filename="policy_preprocessor.json")
            self._postprocessor = DataProcessorPipeline.from_pretrained(
                str(ckpt), config_filename="policy_postprocessor.json")
            torque_key = getattr(cfg, "torque_key", "observation.gripper_torque")
            input_features = getattr(cfg, "input_features", None) or {}
            self._torque_key = torque_key if torque_key in input_features else None

        # 模型期望的图像输入尺寸（握手时告知客户端同步采集分辨率）
        img_ft = (getattr(cfg, "input_features", None) or {}
                  ).get("observation.images.cam1")
        try:
            _, self._img_h, self._img_w = [int(v) for v in img_ft.shape]
        except Exception:
            self._img_h = self._img_w = 720

        self._policy = policy
        logger.info(
            f"模型加载完成: type={policy_type}, device={device}, "
            f"ckpt={self._model_path}, torque_key={self._torque_key}, "
            f"trained_max_delay={self._trained_max_delay}, "
            f"image={self._img_w}x{self._img_h}"
        )
        self._warmup()

    def _warmup(self, n=2):
        """预热推理：触发 CUDA 初始化与 cudnn autotune，避免首个真实请求卡顿。"""
        logger.info(f"预热推理 {n} 次 ...")
        for i in range(n):
            rng = np.random.default_rng(i)
            rgb = rng.integers(0, 255, (self._img_h, self._img_w, 3),
                               dtype=np.uint8)
            state = np.zeros(7, dtype=np.float32)
            t0 = time.perf_counter()
            try:
                chunk, ms, _ = self._infer(rgb, rgb, state, 0.0, "warm up",
                                           rtc_k=0, rtc_d=0, reset=True,
                                           update_raw=False)
                logger.info(f"预热 #{i + 1}: chunk {chunk.shape}, "
                            f"耗时 {ms:.0f}ms (含冷启动 {time.perf_counter() - t0:.2f}s)")
            except Exception:
                logger.warning(f"预热 #{i + 1} 失败", exc_info=True)
        self._policy.reset()
        self._last_raw_chunk = None
        logger.info("预热完成，服务就绪")

    # ==================== 推理核心（镜像 _build_batch / _rtc_inference_loop） ====================

    def _build_batch(self, rgb0, rgb1, state, torque, task):
        state = np.asarray(state, dtype=np.float32)
        dev = self._device
        if self._is_act:
            s = ACT_IMAGE_SIZE
            if rgb0.shape[0] != s or rgb0.shape[1] != s:
                rgb0 = cv2.resize(rgb0, (s, s), interpolation=cv2.INTER_LINEAR)
                rgb1 = cv2.resize(rgb1, (s, s), interpolation=cv2.INTER_LINEAR)
            if self._act_state_dim == state.shape[0] + 1:
                state = np.concatenate([state, [float(torque)]]).astype(np.float32)
            return {
                "observation.images.cam1": torch.from_numpy(
                    np.ascontiguousarray(rgb0)).permute(2, 0, 1).unsqueeze(0).float().div_(255.0).to(dev),
                "observation.images.cam2": torch.from_numpy(
                    np.ascontiguousarray(rgb1)).permute(2, 0, 1).unsqueeze(0).float().div_(255.0).to(dev),
                "observation.state": torch.from_numpy(state).unsqueeze(0).to(dev),
            }
        batch = {
            "observation.images.cam1": torch.from_numpy(
                np.ascontiguousarray(rgb0)).permute(2, 0, 1).unsqueeze(0).float().div_(255.0).to(dev),
            "observation.images.cam2": torch.from_numpy(
                np.ascontiguousarray(rgb1)).permute(2, 0, 1).unsqueeze(0).float().div_(255.0).to(dev),
            "observation.state": torch.from_numpy(state).unsqueeze(0).to(dev),
            "task": [task],
        }
        if self._torque_key is not None:
            batch[self._torque_key] = torch.tensor(
                [[float(torque)]], dtype=torch.float32, device=dev)
        return batch

    def _infer(self, rgb0, rgb1, state, torque, task, rtc_k, rtc_d,
               reset, update_raw):
        """返回 (chunk (H,D) float64, 服务端推理耗时 ms, used_delay)。"""
        if reset:
            self._policy.reset()
            self._last_raw_chunk = None
        batch = self._build_batch(rgb0, rgb1, state, torque, task)
        t0 = time.perf_counter()
        with torch.no_grad():
            if self._preprocessor is not None:
                batch = self._preprocessor(batch)
            # trained RTC：从上一段归一化 chunk 第 k 步起取剩余动作作 prefix
            rtc_kwargs = {}
            used_delay = 0
            if (self._trained_max_delay > 0 and rtc_d > 0
                    and self._last_raw_chunk is not None):
                raw = self._last_raw_chunk
                d = min(int(rtc_d), self._trained_max_delay)
                if d > 0 and raw.shape[0] > rtc_k:
                    leftover = raw[rtc_k:, :]
                    if leftover.shape[0] < d:
                        pad = leftover[-1:].expand(d - leftover.shape[0], -1)
                        leftover = torch.cat([leftover, pad], dim=0)
                    rtc_kwargs = {"inference_delay": d,
                                  "prev_chunk_left_over": leftover}
                    used_delay = d
            if hasattr(self._policy, "predict_action_chunk"):
                action = self._policy.predict_action_chunk(batch, **rtc_kwargs)
            else:
                action = self._policy.select_action(batch)
            if update_raw and self._trained_max_delay > 0:
                # 存归一化空间输出（postprocessor 之前），供下一段作 prefix
                self._last_raw_chunk = action[0].detach()
            if self._postprocessor is not None:
                out = self._postprocessor({"action": action})
                action = out["action"] if isinstance(out, dict) else out
        latency_ms = (time.perf_counter() - t0) * 1000.0
        c = action.detach().cpu().numpy().astype(np.float64)
        c = np.reshape(c, (-1, c.shape[-1]))  # (B,H,D) 或 (B,D) -> (H,D)
        return c, latency_ms, used_delay

    @staticmethod
    def _decode_jpeg(data):
        """JPEG 字节 -> RGB numpy (H,W,3)。客户端以 BGR 编码，解码后转回 RGB。"""
        arr = np.frombuffer(bytes(data), dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("JPEG 解码失败")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # ==================== gRPC 处理器 ====================

    def _check_token(self, request, context):
        if request.token != self._token:
            context.abort(grpc.StatusCode.UNAUTHENTICATED, "token 校验失败")

    def Ping(self, request, context):
        self._check_token(request, context)
        return pb2.PingReply(
            server_time_ms=int(time.time() * 1000),
            policy_type=self._policy_type,
            model_path=self._model_path,
            device=self._device,
            trained_max_delay=self._trained_max_delay,
            image_height=self._img_h,
            image_width=self._img_w,
        )

    def InferChunk(self, request, context):
        self._check_token(request, context)
        t_recv = time.perf_counter()
        try:
            rgb0 = self._decode_jpeg(request.cam1_jpeg)
            rgb1 = self._decode_jpeg(request.cam2_jpeg)
        except Exception as e:
            return pb2.InferenceReply(request_id=request.request_id,
                                      error=f"图像解码失败: {e}")
        if len(request.state) == 0:
            return pb2.InferenceReply(request_id=request.request_id,
                                      error="state 为空")
        decode_ms = (time.perf_counter() - t_recv) * 1000.0

        with self._infer_lock:
            self._avg_n = 1 if self._is_act else max(1, int(request.avg_n))
            reset = bool(request.reset or request.is_single)
            try:
                chunk, infer_ms, used_delay = self._infer(
                    rgb0, rgb1,
                    np.asarray(list(request.state), dtype=np.float32),
                    float(request.torque), request.task,
                    int(request.rtc_k), int(request.rtc_d),
                    reset=reset, update_raw=not request.is_single)
            except Exception as e:
                logger.exception("推理失败")
                return pb2.InferenceReply(request_id=request.request_id,
                                          error=f"{type(e).__name__}: {e}")
        total_ms = (time.perf_counter() - t_recv) * 1000.0
        logger.info(
            f"req#{request.request_id}: chunk {chunk.shape}, "
            f"decode={decode_ms:.0f}ms, infer={infer_ms:.0f}ms, "
            f"total={total_ms:.0f}ms, rtc_d={request.rtc_d}->used={used_delay}, "
            f"avg_n={self._avg_n}, reset={reset}"
        )
        return pb2.InferenceReply(
            request_id=request.request_id,
            actions=chunk.astype(np.float32).ravel().tolist(),
            num_steps=int(chunk.shape[0]),
            action_dim=int(chunk.shape[1]),
            server_latency_ms=float(infer_ms),
            used_delay=int(used_delay),
        )


def serve():
    parser = argparse.ArgumentParser(description="VLA 云端推理服务")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--model_path", default=DEFAULT_MODEL,
                        help="checkpoint 目录（服务端本地路径）")
    parser.add_argument("--token", default=DEFAULT_TOKEN)
    parser.add_argument("--device", default=None, help="cuda / cpu（默认自动）")
    args = parser.parse_args()

    # 限制 torch CPU 线程数，避免挤占 gRPC/编解码线程
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))

    servicer = VLAInferenceServicer(args.model_path, args.token, args.device)
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=4),
        options=[
            ("grpc.max_send_message_length", 32 * 1024 * 1024),
            ("grpc.max_receive_message_length", 32 * 1024 * 1024),
            # 服务端 keepalive：快速发现断连的客户端
            ("grpc.keepalive_time_ms", 5000),
            ("grpc.keepalive_timeout_ms", 3000),
        ],
    )
    pb2_grpc.add_VLAInferenceServicer_to_server(servicer, server)
    bound = server.add_insecure_port(f"{args.host}:{args.port}")
    if bound == 0:
        logger.error(f"端口绑定失败: {args.host}:{args.port}")
        sys.exit(1)
    server.start()
    logger.info(f"gRPC 服务已启动: {args.host}:{args.port} "
                f"(模型 {servicer._policy_type} @ {servicer._device})")

    def _shutdown(*_):
        logger.info("收到退出信号，正在关闭 ...")
        server.stop(grace=2).wait()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
