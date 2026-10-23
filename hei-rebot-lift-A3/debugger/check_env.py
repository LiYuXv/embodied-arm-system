#!/usr/bin/env python
"""SmolVLA 训练环境自检脚本（必须在 debugger/ 目录下运行）。

用法：
    cd debugger
    conda run -n VLA --no-capture-output python check_env.py

检查项：
    1. Python 版本（>= 3.10）
    2. PyTorch / CUDA / GPU，并做一次 CUDA 张量运算
    3. 关键依赖可导入并打印版本（lerobot 必须为 0.6.1）
    4. lerobot smolvla 模块存在
    5. 自定义策略 smolvla_torque 注册成功
    6. 预训练权重与数据集文件就位，数据集能被 lerobot 正常加载
全部通过退出码为 0；存在 FAIL 退出码为 1（WARN 不影响退出码）。
"""

import importlib
import importlib.metadata
import os
import shutil
import subprocess
import sys

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
_results = []  # (级别, 项目, 详情)


def report(level, item, detail=""):
    _results.append(level)
    print(f"[{level:4s}] {item}" + (f"  ->  {detail}" if detail else ""))


def run(item, fn):
    """执行单项检查，异常记为 FAIL。"""
    try:
        fn()
    except Exception as e:
        report(FAIL, item, f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------- 1. Python
def check_python():
    v = sys.version_info
    ok = v >= (3, 10)
    report(PASS if ok else FAIL, "Python 版本", f"{v.major}.{v.minor}.{v.micro}（要求 >= 3.10）")


# ---------------------------------------------------------------- 2. PyTorch / CUDA
# CUDA 版本（torch 内置）所需的最低 NVIDIA 驱动版本（Linux x86_64）
CUDA_MIN_DRIVER = {
    (12, 8): 570, (12, 6): 560, (12, 4): 550, (12, 1): 530, (11, 8): 520,
}


def _probe_driver():
    """返回 (driver_version:int|None, nvidia_smi_info:str)。"""
    if not shutil.which("nvidia-smi"):
        return None, ""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version,name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip().splitlines()[0]
        driver_str = out.split(",")[0].strip()
        return int(driver_str.split(".")[0]), out
    except Exception:
        return None, ""


def check_torch():
    import torch

    report(PASS, "PyTorch 版本", f"{torch.__version__}（内置 CUDA {torch.version.cuda}）")

    if not torch.cuda.is_available():
        driver, smi_info = _probe_driver()
        if not smi_info:
            hint = "nvidia-smi 不可用：实例未分配 GPU？（autodl 注意不要用「无卡模式」开机）"
        else:
            cu = tuple(int(x) for x in str(torch.version.cuda).split(".")[:2])
            need = CUDA_MIN_DRIVER.get(cu)
            if need and driver and driver < need:
                hint = (f"nvidia-smi 正常（{smi_info}），但驱动 {driver} < CUDA {torch.version.cuda} "
                        f"所需最低 {need}。请换装低 CUDA 版 torch，如：pip install torch==2.7.0 "
                        f"torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu124")
            else:
                hint = f"nvidia-smi 正常（{smi_info}），但 torch 无法使用 CUDA，请重装匹配的 torch"
        report(FAIL, "CUDA 可用性", f"torch.cuda.is_available() = False。{hint}")
        return
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    level = PASS if vram >= 12 else WARN
    report(level, "GPU", f"{name}，显存 {vram:.1f} GB（建议 >= 16 GB）")

    # 一次真实的 CUDA 运算，验证驱动/运行时正常
    x = torch.randn(1024, 1024, device="cuda")
    torch.cuda.synchronize()
    s = (x @ x).sum().item()
    report(PASS, "CUDA 张量运算", f"matmul 完成，sum={s:.2e}")


# ---------------------------------------------------------------- 3. 依赖导入
# (模块名, pip 包名, 是否必需)
DEPS = [
    ("torch", "torch", True),
    ("torchvision", "torchvision", True),
    ("lerobot", "lerobot", True),
    ("transformers", "transformers", True),
    ("accelerate", "accelerate", True),
    ("datasets", "datasets", True),
    ("tokenizers", "tokenizers", True),
    ("diffusers", "diffusers", True),
    ("draccus", "draccus", True),
    ("einops", "einops", True),
    ("av", "av", True),
    ("safetensors", "safetensors", True),
    ("numpy", "numpy", True),
    ("cv2", "opencv-python-headless", True),
    ("PIL", "pillow", True),
    ("pyarrow", "pyarrow", True),
    ("pandas", "pandas", True),
    ("matplotlib", "matplotlib", True),
    ("gymnasium", "gymnasium", True),
    ("imageio", "imageio", True),
    ("scipy", "scipy", True),
    ("yaml", "pyyaml", True),
    ("wandb", "wandb", False),  # 可选：yaml 中 wandb.enable=false 时不需要
]


def check_deps():
    for module, pkg, required in DEPS:
        try:
            importlib.import_module(module)
            try:
                ver = importlib.metadata.version(pkg)
            except importlib.metadata.PackageNotFoundError:
                ver = "?"
            report(PASS, f"依赖 {pkg}", ver)
        except ImportError as e:
            level = FAIL if required else WARN
            report(level, f"依赖 {pkg}", f"导入失败：{e}" + ("" if required else "（可选，不影响训练）"))


def check_lerobot_version():
    ver = importlib.metadata.version("lerobot")
    ok = ver == "0.6.1"
    report(PASS if ok else FAIL, "lerobot 版本", f"{ver}（必须为 0.6.1，旧版无 smolvla 模块）")


# torch 次版本 -> 兼容的 torchcodec 版本（官方兼容表）
TORCH_TO_TORCHCODEC = {"2.10": "0.10", "2.9": "0.9", "2.8": "0.7", "2.7": "0.5", "2.6": "0.2"}


def check_torchcodec():
    """torchcodec 须与 torch 大版本匹配，且系统装有 FFmpeg 共享库。
    加载失败时 lerobot 自动回退 pyav 解码后端，训练不受影响，故只记 WARN。"""
    try:
        import torchcodec  # noqa: F401

        report(PASS, "torchcodec（视频解码后端）", importlib.metadata.version("torchcodec"))
    except Exception:
        import torch

        mm = ".".join(torch.__version__.split(".")[:2])
        want = TORCH_TO_TORCHCODEC.get(mm, "见官方兼容表")
        report(
            WARN,
            "torchcodec（视频解码后端）",
            f"加载失败，训练将自动回退 pyav 后端（可正常训练）。如需修复："
            f"torch {mm} 对应 torchcodec=={want}（pip install torchcodec=={want}），"
            f"并安装 FFmpeg 共享库：apt-get install -y ffmpeg（或 conda install -c conda-forge ffmpeg）",
        )


def check_ffmpeg():
    """FFmpeg 共享库仅 torchcodec 后端需要；pyav 后端自带解码库，无需系统 FFmpeg。"""
    exe = shutil.which("ffmpeg")
    if exe is None:
        report(WARN, "FFmpeg", "未安装（仅 torchcodec 后端需要；用 pyav 后端可忽略）")
        return
    first = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.splitlines()[0]
    report(PASS, "FFmpeg", first)


# ---------------------------------------------------------------- 4. smolvla 模块
def check_smolvla():
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa: F401
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig  # noqa: F401

    report(PASS, "lerobot smolvla 模块", "SmolVLAPolicy / SmolVLAConfig 可导入")


# ---------------------------------------------------------------- 5. 自定义策略注册
def check_torque_policy():
    import configuration_smolvla_torque  # noqa: F401  # 导入即完成 draccus 注册
    from modeling_smolvla_torque import SmolVLATorquePolicy

    report(PASS, "自定义策略 smolvla_torque", f"注册成功（{SmolVLATorquePolicy.__name__}）")


# ---------------------------------------------------------------- 6. 文件与数据集
FILES = [
    "pretrained_models/smolvla_base/config.json",
    "pretrained_models/smolvla_base/model.safetensors",
    "pretrained_models/smolvlm2-500m-video-instruct/config.json",
    "pretrained_models/smolvlm2-500m-video-instruct/model.safetensors",
    "lerobot_data/ela3_demo/meta/info.json",
    "lerobot_data/ela3_demo/meta/stats.json",
    "smolvla_omy.yaml",
]


def check_files():
    for path in FILES:
        ok = os.path.isfile(path)
        report(PASS if ok else FAIL, f"文件 {path}", "存在" if ok else "缺失，请按部署文档第 4 节上传")


def check_dataset():
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    meta = LeRobotDatasetMetadata("ela3_demo", root="./lerobot_data/ela3_demo")
    info = meta.info
    report(
        PASS,
        "数据集 ela3_demo",
        f"{info.total_episodes} episodes / {info.total_frames} 帧 / {info.fps} fps",
    )


# ---------------------------------------------------------------- main
CHECKS = [
    ("Python 版本", check_python),
    ("PyTorch / CUDA", check_torch),
    ("依赖导入", check_deps),
    ("lerobot 版本", check_lerobot_version),
    ("torchcodec", check_torchcodec),
    ("FFmpeg", check_ffmpeg),
    ("smolvla 模块", check_smolvla),
    ("自定义策略注册", check_torque_policy),
    ("文件就位", check_files),
    ("数据集加载", check_dataset),
]


def main():
    print("=" * 60)
    print("SmolVLA 训练环境自检")
    print("=" * 60)
    for item, fn in CHECKS:
        run(item, fn)

    n_fail = _results.count(FAIL)
    n_warn = _results.count(WARN)
    print("=" * 60)
    print(f"自检完成：共 {len(_results)} 项，FAIL {n_fail}，WARN {n_warn}")
    if n_fail:
        print("存在 FAIL 项，请根据上方提示修复后再启动训练。")
        sys.exit(1)
    print("环境就绪，可以启动训练：")
    print("  conda run -n VLA --no-capture-output python train_model.py "
          "--config_path smolvla_omy.yaml 2>&1 | tee train.log")


if __name__ == "__main__":
    main()
