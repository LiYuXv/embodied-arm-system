#!/usr/bin/env python3
"""
基于 Qwen3-VL 的子任务自动语言标注（本地 GPU，dense 模型）。

适配 lerobot 0.6.1 的 SARM 标注思路（data_processing/sarm_annotations/subtask_annotation.py），
但改用 **dense 模型类**（Qwen3VLForConditionalGeneration）以支持 Qwen3-VL-2B-Instruct 等
非 MoE 模型，并直接加载**本地 LeRobot 数据集**（无需 HF repo-id / 联网下载）。

流程：
    本地数据集视频  →  Qwen3-VL 看视频  →  输出带时间戳的子任务 JSON
    →  时间戳转帧号  →  写入 LanguageAnnotationStore（与 visualize_data_qt.py 界面对接）

输出（meta/language_annotations.json）会被 visualize_data_qt.py 的“自动语言标注”面板
直接读取并在夹爪曲线上着色显示。

用法：
    # 预览（不写盘，仅打印）
    HF_ENDPOINT=https://hf-mirror.com python qwen3vl_subtask_annotate.py \
        --root ../lerobot_data/ela3_demo --episodes 0 \
        --subtasks "靠近物体,抓取,移动到目标,放置" \
        --model Qwen/Qwen3-VL-2B-Instruct

    # 写入数据集
    ... --apply
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

# 复用同目录下的存储与配色
_DBG_DIR = Path(__file__).resolve().parent
if str(_DBG_DIR) not in sys.path:
    sys.path.insert(0, str(_DBG_DIR))
from auto_language_annotate import LanguageAnnotationStore, color_for_index

# 复用 lerobot 的 SARM prompt（经过精心设计的时序定位提示词）
try:
    from lerobot.data_processing.sarm_annotations.subtask_annotation import create_sarm_prompt
except Exception:
    create_sarm_prompt = None  # 降级时用内置简版


def _builtin_prompt(subtask_list: List[str]) -> str:
    """create_sarm_prompt 不可用时的简版降级提示词。"""
    names = "\n".join(f"  - {n}" for n in subtask_list)
    return (
        "You are a robotics vision system. Segment the demonstration video into NON-overlapping "
        "atomic subtasks using ONLY these labels (in order):\n[\n" + names + "\n]\n"
        "Each subtask appears exactly once. Use visual state changes to set boundaries.\n"
        "Return ONLY JSON: {\"subtasks\":[{\"name\":\"...\",\"timestamps\":"
        "{\"start\":\"MM:SS\",\"end\":\"MM:SS\"}}]}\n"
        "First subtask starts at 00:00; last ends at video end. No gaps between subtasks."
    )


def _direct_prompt(subtask_list: List[str], duration_s: float, repeat: bool) -> str:
    """为 2B 小模型设计的精简提示词：直接输出 JSON，不写文本时间线。

    repeat=True 时按“循环模板”处理（同一组子任务在视频中反复出现）。
    """
    names_str = ", ".join(subtask_list)
    dur_str = f"{int(duration_s // 60):02d}:{int(duration_s % 60):02d}"
    if repeat:
        cycle = " → ".join(subtask_list)
        structure = (
            f"This {dur_str} robot video contains SEVERAL repetitions of a manipulation cycle.\n"
            f"The cycle always follows this order: {cycle}\n"
            f"Segment the ENTIRE video into a sequence of these subtasks (names REPEAT each cycle).\n"
            f"Find every occurrence you can see. Set each boundary by a real visual change."
        )
    else:
        structure = (
            f"Divide this {dur_str} robot video into EXACTLY {len(subtask_list)} sequential subtasks.\n"
            f"You MUST use these names IN THIS ORDER: {names_str}\n"
            f"Each appears exactly once. Set boundaries by what you SEE, not equal splits."
        )

    example_name = "<name>"
    return (
        structure + "\n\n"
        "Cover 00:00 to " + dur_str + " with no gaps (each end = next start).\n"
        "Output ONLY valid JSON, no markdown, no explanation:\n"
        '{"subtasks":[{"name":"' + example_name + '","timestamps":{"start":"MM:SS","end":"MM:SS"}},'
        '{"name":"' + example_name + '","timestamps":{"start":"MM:SS","end":"MM:SS"}}]}'
    )


def timestamp_to_seconds(ts: str) -> float:
    """'MM:SS' 或 'SS' → 秒。"""
    ts = str(ts).strip()
    parts = ts.split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    return float(parts[0])


def load_model(model_name: str, device: str = "cuda", dtype: str = "bfloat16",
               quantize: Optional[str] = None):
    """加载 Qwen3-VL 模型。quantize 可为 None(全精度)/"4bit"/"8bit"。

    8B 等大模型在 16G 卡上必须用 4bit/8bit 才放得下。
    """
    import torch
    from transformers import AutoProcessor
    try:
        from transformers import Qwen3VLForConditionalGeneration as _ModelClass
    except ImportError:
        from transformers import AutoModelForImageTextToText as _ModelClass

    compute_dt = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]
    kwargs = {"trust_remote_code": True}
    if quantize in ("4bit", "8bit"):
        from transformers import BitsAndBytesConfig
        if quantize == "4bit":
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dt,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        else:
            bnb = BitsAndBytesConfig(load_in_8bit=True)
        kwargs["quantization_config"] = bnb
        kwargs["device_map"] = device
        print(f"[model] 加载 {model_name} ({quantize} 量化) ...")
    else:
        kwargs["torch_dtype"] = compute_dt
        kwargs["device_map"] = device
        print(f"[model] 加载 {model_name} ({dtype}) ...")
    model = _ModelClass.from_pretrained(model_name, **kwargs)
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model.eval()
    try:
        print(f"[model] 加载完成，显存占用 ≈ {torch.cuda.memory_allocated() / 1e9:.1f} GB")
    except Exception:
        pass
    return model, processor


def _parse_subtasks_lenient(resp: str, duration_s: float) -> List[dict]:
    """宽松解析：先试严格 JSON，失败则用正则逐个提取 name/start/end（容忍语法错误）。"""
    obj = None
    try:
        obj = json.loads(resp)
    except json.JSONDecodeError:
        # 常见修复：相邻对象缺逗号 }{  →  },{
        fixed = re.sub(r"\}\s*\{", "},{", resp)
        # 去尾逗号
        fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
        try:
            obj = json.loads(fixed)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", fixed, re.DOTALL)
            if m:
                try:
                    obj = json.loads(m.group())
                except json.JSONDecodeError:
                    obj = None
    subs: List[dict] = []
    if obj and isinstance(obj, dict) and "subtasks" in obj:
        for s in obj["subtasks"]:
            t = s.get("timestamps", {}) if isinstance(s, dict) else {}
            subs.append({
                "name": str(s.get("name", "")).strip() if isinstance(s, dict) else "",
                "start_s": timestamp_to_seconds(t.get("start", "0")),
                "end_s": timestamp_to_seconds(t.get("end", str(int(duration_s)))),
            })
        if subs:
            return subs
    # 兜底：正则逐条提取（最鲁棒，无视整体 JSON 语法）
    pat = re.compile(
        r'"name"\s*:\s*"([^"]*)"[^}]*?"start"\s*:\s*"([^"]*)"[^}]*?"end"\s*:\s*"([^"]*)"',
        re.DOTALL)
    for name, start, end in pat.findall(resp):
        subs.append({"name": name.strip(), "start_s": timestamp_to_seconds(start),
                     "end_s": timestamp_to_seconds(end)})
    return subs


def annotate_video(model, processor, video_path: Path, subtask_list: List[str],
                   duration_s: float, device: str = "cuda",
                   fps: float = 2.0, resized_h: int = 240, resized_w: int = 320,
                   repeat: bool = False, max_retries: int = 3) -> Optional[dict]:
    """让 Qwen3-VL 看一段视频（低分辨率），返回 {subtasks:[{name,start_s,end_s}]} 或 None。

    fps/resized_h/resized_w 控制采样帧率与分辨率，降低视觉 token 数以适配小模型。
    repeat=True 按“循环模板”处理（同一组子任务反复出现）。
    """
    import torch
    from qwen_vl_utils import process_vision_info

    prompt = _direct_prompt(subtask_list, duration_s, repeat=repeat)

    messages = [
        {"role": "user", "content": [
            {"type": "video", "video": str(video_path),
             "fps": fps, "resized_height": resized_h, "resized_width": resized_w},
            {"type": "text", "text": prompt},
        ]},
    ]

    for attempt in range(max_retries):
        try:
            try:
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            except TypeError:
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt",
            ).to(device)

            # 每次重试换温度，避免卡在同一份错误输出
            temp = [0.3, 0.6, 0.9][attempt] if attempt < 3 else 0.9
            with torch.no_grad():
                out_ids = model.generate(**inputs, max_new_tokens=2048,
                                         do_sample=True, temperature=temp)
            resp = processor.batch_decode(
                [o[len(i):] for i, o in zip(inputs.input_ids, out_ids, strict=True)],
                skip_special_tokens=True,
            )[0].strip()

            if "```json" in resp:
                resp = resp.split("```json", 1)[1].split("```", 1)[0]
            elif "```" in resp:
                resp = resp.split("```", 1)[1].split("```", 1)[0]

            subs = _parse_subtasks_lenient(resp, duration_s)
            if subs:
                return {"subtasks": subs}
            raise ValueError(f"未提取到任何子任务，响应前200字: {resp[:200]!r}")
        except Exception as e:
            print(f"  [重试 {attempt + 1}/{max_retries}] {e}")
            if attempt == max_retries - 1:
                return None
            time.sleep(1)
    return None


def to_frame_segments(vlm_result: dict, fps: int, total_frames: int,
                      subtask_list: List[str], repeat: bool = False,
                      min_seg_frames: int = 5) -> List[dict]:
    """把 VLM 的秒级时间戳转成帧级区间。

    repeat=True 时允许子任务名重复（循环结构）；颜色按预设名固定，便于同色对比。
    会过滤掉超出视频范围的幻觉段、合并过短段。
    """
    if not vlm_result:
        return []
    raw = []
    for s in vlm_result["subtasks"]:
        start_f = max(0, int(round(s["start_s"] * fps)))
        end_f = min(total_frames - 1, int(round(s["end_s"] * fps)))
        # 丢弃完全在视频外的幻觉段
        if start_f >= total_frames:
            continue
        if end_f < start_f:
            end_f = start_f
        name = s["name"]
        if name not in subtask_list:
            name = subtask_list[len(raw) % len(subtask_list)] if subtask_list else name
        raw.append({
            "start": start_f, "end": end_f, "task": name,
            "color": color_for_index(subtask_list.index(name)) if name in subtask_list else color_for_index(len(raw)),
            "note": "qwen3-vl",
        })

    # 合并过短段到前一段（循环模式下短段多为噪声）
    segs: List[dict] = []
    for s in raw:
        if segs and (s["end"] - s["start"] + 1) < min_seg_frames:
            segs[-1]["end"] = s["end"]  # 并入前段
        else:
            segs.append(dict(s))
    # 确保首段从 0 开始、末段到结尾，无间隙
    if segs:
        segs[0]["start"] = 0
        segs[-1]["end"] = total_frames - 1
    return segs


def get_video_duration(path: Path) -> float:
    cap = cv2.VideoCapture(str(path))
    dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 1.0)
    cap.release()
    return max(dur, 0.0)


def _extract_frame(video_path: Path, frame_idx: int, resized_h: int = 240, resized_w: int = 320):
    """从视频读取指定帧并缩放，返回 PIL.Image（RGB）。"""
    from PIL import Image
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fi = max(0, min(frame_idx, total - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    img = img.resize((resized_w, resized_h))
    return img


def generate_labels(model, processor, video_path: Path, segments: List[dict],
                    device: str = "cuda", known_labels: Optional[List[str]] = None,
                    objects_hint: str = "方块, 玩偶, 盒子角, 盖板",
                    cam2_path: Optional[Path] = None) -> List[str]:
    """让 Qwen3-VL 看每段的关键帧，为每段生成“动作+物体”标签。

    一次推理：把 N 段的中点帧按序拼成多图消息，要求输出 N 个简短标签。
    known_labels 来自标签库，作为一致性参考（不强制）。
    cam2_path 提供时，每段同时喂 cam1、cam2 两个视角的关键帧（双摄）。
    """
    import torch
    from qwen_vl_utils import process_vision_info

    n = len(segments)
    if n == 0:
        return []

    dual = cam2_path is not None
    content = [{"type": "text", "text": (
        f"下面按顺序是机器人操作的 {n} 个步骤，每张图是该步骤的关键帧。"
        + ("每个步骤给出摄像头1与摄像头2两个视角的关键帧。" if dual else "")
        + "为每一步生成一个简短中文标签（格式：动作+物体，如“抓取方块”“移动玩偶”“放下盖板”）。"
        f"可能涉及的物体：{objects_hint}。每步只输出一个标签，3-8 个字。"
    )}]
    for i, seg in enumerate(segments):
        mid = (seg["start"] + seg["end"]) // 2
        img = _extract_frame(video_path, mid)
        if img is not None:
            content.append({"type": "image", "image": img})
            if dual:
                img2 = _extract_frame(cam2_path, mid)
                if img2 is not None:
                    content.append({"type": "image", "image": img2})
            content.append({"type": "text", "text": f"步骤{i + 1}"})

    ref = ""
    if known_labels:
        ref = f"\n参考已有标签（可复用或据画面调整）：{', '.join(known_labels[:20])}"
    content.append({"type": "text", "text": ref + f"\n输出严格 JSON：{{\"labels\":[\"步骤1标签\",\"步骤2标签\",...]}}，共 {n} 个。"})

    messages = [{"role": "user", "content": content}]
    try:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    resp = processor.batch_decode([o[len(i):] for i, o in zip(inputs.input_ids, out, strict=True)],
                                  skip_special_tokens=True)[0].strip()

    # 解析标签
    labels: List[str] = []
    if "```json" in resp:
        resp = resp.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in resp:
        resp = resp.split("```", 1)[1].split("```", 1)[0]
    try:
        obj = json.loads(resp)
        labels = [str(x).strip() for x in obj.get("labels", [])]
    except (json.JSONDecodeError, AttributeError):
        for m in re.finditer(r'"([^"]{2,12})"', resp):
            labels.append(m.group(1))
    # 长度对齐
    while len(labels) < n:
        labels.append(f"步骤{len(labels) + 1}")
    return labels[:n]


def interpret_video(model, processor, video_path, cam2_path=None, device: str = "cuda",
                    fps: float = 1.0, resized_h: int = 240, resized_w: int = 320,
                    max_new_tokens: int = 512) -> str:
    """让 Qwen3-VL 观看整段视频并生成中文解读（任务/物体/动作步骤）。

    cam2_path 提供时，同一条消息里同时喂 cam1、cam2 两路视频（双摄），
    模型综合两个视角生成描述。单次推理，低分辨率低帧率，适合小模型；返回自然语言描述文本。
    """
    import torch
    from qwen_vl_utils import process_vision_info

    prompt = (
        "观看这个机器人操作视频，用中文简要解读：\n"
        "1) 机器人在执行什么任务；\n"
        "2) 涉及哪些物体（如方块、玩偶、盒子、盖板等）；\n"
        "3) 主要的动作步骤（按时间顺序，简述即可）。\n"
        "回答控制在 200 字以内，分点说明。"
    )
    content = [
        {"type": "video", "video": str(video_path),
         "fps": fps, "resized_height": resized_h, "resized_width": resized_w},
        {"type": "text", "text": "（以上是摄像头1画面）"},
    ]
    if cam2_path:
        content.append({"type": "video", "video": str(cam2_path),
                        "fps": fps, "resized_height": resized_h, "resized_width": resized_w})
        content.append({"type": "text", "text": "（以上是摄像头2画面。综合两个视角解读）"})
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    try:
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    resp = processor.batch_decode(
        [o[len(i):] for i, o in zip(inputs.input_ids, out, strict=True)],
        skip_special_tokens=True)[0].strip()
    # 去掉可能的 think 标签
    import re
    resp = re.sub(r"<think>.*?</think>", "", resp, flags=re.DOTALL).strip()
    return resp


def main():
    ap = argparse.ArgumentParser(description="Qwen3-VL 本地子任务自动语言标注")
    ap.add_argument("--root", required=True, help="本地 LeRobot 数据集根目录")
    ap.add_argument("--episodes", type=int, nargs="+", default=[0], help="episode 索引列表")
    ap.add_argument("--subtasks", required=True, help="预设子任务名（逗号分隔，按执行顺序）")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--video-key", default=None, help="相机键（默认第一个）")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--apply", action="store_true", help="写入 language_annotations.json（默认仅预览）")
    ap.add_argument("--repeat", action="store_true",
                    help="循环模式：子任务名作为反复出现的循环模板（如 抓取/移动/放置/归位）")
    ap.add_argument("--fps", type=float, default=2.0, help="视频采样帧率（默认 2fps）")
    ap.add_argument("--resized-h", type=int, default=240, help="缩放高度（默认 240p）")
    ap.add_argument("--resized-w", type=int, default=320, help="缩放宽度（默认 320）")
    args = ap.parse_args()

    subtask_list = [s.strip() for s in args.subtasks.replace("、", ",").split(",") if s.strip()]
    print(f"[config] 子任务: {subtask_list}  循环模式={args.repeat}  "
          f"采样={args.fps}fps  分辨率={args.resized_w}x{args.resized_h}")

    # 加载数据集
    from lerobot.datasets import LeRobotDataset
    ds = LeRobotDataset(Path(args.root).name, root=args.root)
    fps = ds.meta.fps
    vkey = args.video_key or ds.meta.video_keys[0]
    print(f"[dataset] {len(ds.meta.episodes)} episodes, fps={fps}, camera={vkey}")

    # 加载模型
    model, processor = load_model(args.model, args.device, args.dtype)

    store = LanguageAnnotationStore(args.root)
    store.load()

    for ep_idx in args.episodes:
        if ep_idx >= len(ds.meta.episodes):
            print(f"[ep {ep_idx}] 超出范围，跳过")
            continue
        vpath = ds.root / ds.meta.get_video_file_path(ep_idx, vkey)
        if not vpath.exists():
            print(f"[ep {ep_idx}] 视频不存在: {vpath}")
            continue
        total_frames = ds.meta.episodes[ep_idx].get("length", 0)
        if not total_frames:
            total_frames = int(get_video_duration(vpath) * fps)
        duration = get_video_duration(vpath)
        print(f"\n[ep {ep_idx}] {vpath.name}  {total_frames}帧  ~{duration:.1f}s")

        result = annotate_video(model, processor, vpath, subtask_list, duration,
                                args.device, fps=args.fps, resized_h=args.resized_h,
                                resized_w=args.resized_w, repeat=args.repeat)
        if not result:
            print(f"[ep {ep_idx}] 标注失败，跳过")
            continue

        segs = to_frame_segments(result, fps, total_frames, subtask_list, repeat=args.repeat)
        print(f"[ep {ep_idx}] VLM 输出 {len(result['subtasks'])} 个子任务 → {len(segs)} 段:")
        for s in segs:
            print(f"    [{s['start']:>4}~{s['end']:<4}] ({s['end']-s['start']+1}帧) {s['color']}  {s['task']}")

        if args.apply:
            store.set_episode(ep_idx, segs)
            store.save()
            print(f"[ep {ep_idx}] 已写入 {store.filepath}")
        else:
            print(f"[ep {ep_idx}] （预览模式，未写入。加 --apply 写入）")

    print("\n完成。可在 visualize_data_qt.py 的“自动语言标注”面板查看结果。")


if __name__ == "__main__":
    main()
