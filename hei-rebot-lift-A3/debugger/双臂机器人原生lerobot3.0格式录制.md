# 双臂机器人原生 LeRobot v3.0 格式录制（不依赖 lerobot 库）

## 背景与目标

工控机（Celeron J4125 / 3.5GB 内存 / 无 GPU）性能较弱，安装完整 lerobot 0.6.1
会连带 torch/transformers 等重依赖（常驻内存 +300~500MB），危及机械臂实时控制。
因此实现**独立轻量采集包**，产出与 lerobot 0.6.1 完全一致的原生 v3.0 数据集，
训练侧可直接 `LeRobotDataset(...)` 加载。

同时录制规格从「单臂 + 2 路摄像头」升级为「**双臂 + 3 路摄像头**」。

## 录制数据规格

| 键 | 内容 | 维度 |
|----|------|------|
| `observation.images.cam1` | 臂1 腕部摄像头（索引 1，界面右侧图） | 720×720 mp4 |
| `observation.images.cam2` | 头部摄像头（索引 4，界面中间图） | 720×720 mp4 |
| `observation.images.cam3` | 臂2 腕部摄像头（索引 6，界面左侧图） | 720×720 mp4 |
| `observation.state` | arm1_pose7 + arm2_pose7（+底盘 lift/odom_x/odom_y/yaw） | 14（接底盘 18） |
| `observation.gripper_torque` | [臂1夹爪力矩, 臂2夹爪力矩] (Nm) | 2 |
| `action` | 未来第 N 帧的双臂位姿（前瞻帧 N 可调，默认 1） | 同 state |

- 每臂 pose7 = `[x, y, z, rx, ry, rz, gripper_rad]`，欧拉角/夹爪均为弧度
- 界面显示顺序（物理布局）：左=臂2腕部(6)，中=头部(4)，右=臂1腕部(1)，
  由 `CameraWidget.CAMERA_DISPLAY_ORDER = [6, 4, 1]` 控制，仅影响显示
- 三路不齐时丢帧不录（数据集固定 3 路，缺路会产生不一致数据）

## 磁盘格式（对齐 lerobot 0.6.1，codebase_version="v3.0"）

```
<repo_name>/
├── meta/
│   ├── info.json                        # DatasetInfo 全字段（fps/features/counts/路径模板）
│   ├── tasks.parquet                    # index="task", column="task_index"
│   ├── stats.json                       # 聚合统计 mean/std/min/max/count/q01..q99
│   └── episodes/chunk-000/file-000.parquet   # 每 episode 一行元数据
├── data/chunk-XXX/file-YYY.parquet      # 帧数据（每 episode 一个文件）
└── videos/{video_key}/chunk-XXX/file-YYY.mp4  # 每 episode 一个视频（h264/yuv420p）
```

与官方写入器的差异（对读取端透明）：
- 官方按大小滚动打包多 episode 到同一文件；本实现每 episode 独立文件，
  chunk/file 索引按 episode_index 分配（chunks_size=1000）。v3.0 读取端按
  episode 元数据寻址，完全兼容。
- 视频分位数统计为蓄水池采样近似值；数值特征统计为精确值。

## 实现文件

| 文件 | 内容 |
|------|------|
| `debugger/utils/lerobot_lite_dataset.py` | 新增：`LiteLeRobotDataset`（create/resume/add_frame/save_episode/finalize），PyAV 流式编码线程，统计聚合 |
| `debugger/widgets/camera_pyqt.py` | `LeRobotRecordThread` 双臂位姿/力矩/3 路帧；`_ensure_lerobot_dataset` 自动回退 lite；录制需 3 摄像头 |
| `debugger/main_window.py` | 接入臂2回调 `_get_current_end_pose2` / `_get_gripper_torque2` |

## 关键设计

1. **编码性能**：PyAV h264 流式编码（录制期间直接编码，不落地 PNG），
   软编码强制 `preset=veryfast`；实测 J4125 上 3 路 720×720 达 34fps，
   满足 20fps 录制。有 NVIDIA GPU 时自动用 NVENC。
2. **action 前瞻**：action[t] = pose[t+N]，默认 N=1，避免 action==state 退化；
   内置恒等守卫，连续 20 帧 action≈state 会告警。
3. **回退机制**：`_ensure_lerobot_dataset` 优先 `import lerobot`，失败自动用
   lite 写入器；两者 API 对齐，录制线程无感。
4. **兼容性检查**：已有数据集的 state 维度/力矩维度/3 路摄像头/编码器逐项校验，
   旧格式（单臂 7/11 维 + 2 路）会被判不兼容并提示更换文件夹名。

## 依赖

仅需装入 `.venv`（已安装）：`pyarrow`、`pandas`、`av`（合计约 150MB，
无 torch/transformers）。

## 验证步骤

1. 实机录制一个短 episode（开始/停止 LeRobot 录制）。
2. 检查目录结构与本文件“磁盘格式”一致。
3. 拷到训练机，lerobot 0.6.1 环境验证：
   ```python
   from lerobot.datasets.lerobot_dataset import LeRobotDataset
   ds = LeRobotDataset("ela3_demo", root="<数据集路径>")
   print(ds.num_episodes, ds.num_frames, ds.features)
   sample = ds[0]  # 应能取到 3 路图像 + 14/18 维 state + 2 维 torque
   ```
