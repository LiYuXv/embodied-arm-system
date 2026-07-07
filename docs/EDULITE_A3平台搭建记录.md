EDULITE_A3 基础平台搭建记录

1. 搭建目的

本阶段主要完成 EDULITE_A3 机械臂基础平台的搭建与验证，为后续自然语言交互、视觉感知、任务规划和机械臂控制联调提供基础环境。

本阶段的目标是先跑通机械臂基础仿真链路，不涉及 VLA、Agent 或复杂智能规划。

2. 项目结构

本毕设仓库当前结构如下：

embodied-arm-system/
├── docs/                  # 毕设文档记录
├── README.md
├── third_party/
│   └── EDULITE_A3/         # 老师提供的基础平台代码
└── ros2_ws/
    └── src/                # 后续存放自己编写的 ROS2 功能包

其中，third_party/EDULITE_A3 用于存放官方基础平台代码，后续自己编写的语言理解、视觉感知、任务规划和控制封装代码将放在 ros2_ws/src 下，避免与官方平台代码混在一起。

EDULITE_A3 项目主要包括：

EDULITE_A3/
├── el_a3_sdk/       # Python SDK
├── el_a3_ros/       # ROS2 控制系统、MoveIt2 配置和机械臂描述文件
└── hardware/        # 硬件相关资料

本阶段主要使用的是：

EDULITE_A3/el_a3_ros/

该目录本身就是一个 ROS2 工程，可以直接在该目录下使用 colcon build 编译。

3. 基础环境

当前使用环境：

操作系统：Ubuntu 22.04
ROS2 版本：ROS2 Humble
主要工具：colcon、RViz2、MoveIt2、ros2_control
机械臂平台：EDULITE_A3
4. 编译过程

进入 EDULITE_A3 的 ROS2 工程目录：

cd ~/embodied-arm-system/third_party/EDULITE_A3/el_a3_ros

加载 ROS2 Humble 环境：

source /opt/ros/humble/setup.bash

编译工程：

colcon build --symlink-install

编译结果显示：

Summary: 5 packages finished

说明 EDULITE_A3 的 ROS2 工程已经成功编译。

成功编译的包包括：

el_a3_hardware
edulite_a3_description
el_a3_description
el_a3_moveit_config
el_a3_teleop
5. 遇到的问题与解决过程
5.1 依赖安装时出现 404

运行 install_deps.sh 时，终端出现过大量 404 Not Found 和 E: 无法下载 错误。

原因是当前 Ubuntu / ROS2 镜像源中的部分软件包地址失效，导致依赖没有完整下载。

处理思路是更新 apt 缓存、修复依赖，并在必要时切换软件源。

5.2 缺少 Pinocchio 依赖

第一次编译时，el_a3_hardware 报错：

Could not find a package configuration file provided by "pinocchio"

原因是系统缺少 Pinocchio 相关依赖。

通过安装 ros-humble-pinocchio 解决。

5.3 RViz 中 RobotModel 加载几何模型失败

第一次启动 RViz 后，可以看到机械臂 TF 骨架，但 RobotModel 报错：

URDF: Errors loading geometries

排查发现，el_a3_description/meshes 下的 STL 文件只有 131B 或 132B，说明这些文件不是实际模型文件，而是 Git LFS 指针文件。

解决方式是安装并执行 Git LFS，拉取真实 STL 模型文件。

处理后重新编译并启动 RViz，机械臂完整模型能够正常显示。

6. MoveIt2 仿真启动

编译完成后，加载当前工作空间：

source install/setup.bash

启动 MoveIt2 仿真环境：

ros2 launch el_a3_moveit_config demo.launch.py

启动后 RViz 正常打开，MotionPlanning 面板正常显示，RobotModel 能够加载完整机械臂模型。

7. 运动规划测试

在 RViz 的 MotionPlanning 面板中进行了基础运动规划测试：

选择 Planning Group 为 arm；
在 Joints 面板中调整目标关节状态；
点击 Plan 进行路径规划；
点击 Plan & Execute 执行规划结果。

测试结果表明，机械臂模型能够按照规划结果运动。

说明当前链路已经跑通：

MoveIt2 运动规划
→ ros2_control mock hardware
→ RViz 可视化执行
8. 当前阶段成果

当前已经完成：

EDULITE_A3 基础平台代码引入；
EDULITE_A3 ROS2 工程编译；
MoveIt2 + RViz 仿真环境启动；
机械臂完整模型显示；
MoveIt2 中基础 Plan 与 Plan & Execute 测试。

该阶段说明 EDULITE_A3 基础仿真平台已经初步搭建完成，可以作为后续语言交互、视觉感知和机械臂操作系统集成的基础环境。

9. 后续计划

下一阶段计划：

创建自己的 ROS2 工作空间 ros2_ws；
编写机械臂控制封装包；
学习 MoveIt2 Python/C++ 调用接口；
实现简单目标位姿输入下的机械臂运动控制；
后续逐步接入语言指令解析和视觉感知模块。

后续整体技术路线为：

自然语言输入
→ 指令解析
→ 目标识别
→ 任务规划
→ MoveIt2 运动规划
→ 仿真执行
→ 真实机械臂迁移


## 10. 上位机安装与启动记录

EDULITE_A3 项目中的 `el_a3_sdk` 提供了 Python SDK 和 Debugger 上位机工具。该工具主要用于机械臂状态监控、关节调试和可视化操作，后续真实机械臂调试阶段可能会使用。

本阶段在独立 Python 虚拟环境中安装上位机，避免影响系统 Python 与 ROS2 环境。

虚拟环境创建与激活命令如下：

```bash
python3 -m venv ~/.venvs/el_a3_debugger
source ~/.venvs/el_a3_debugger/bin/activate
```


进入 SDK 目录：

<pre class="overflow-visible! px-0!" data-start="1085" data-end="1154"><div class="relative w-full mt-4 mb-1"><div class=""><div class="contents"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="relative h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute inset-x-4 top-12 bottom-4"><div class="pointer-events-none sticky z-40 shrink-0 z-1!"><div class="sticky bg-token-border-light"></div></div></div><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class=""><div class="relative"><div class=""><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼd ͼr"><div class="cm-scroller"><pre class="cm-content q9tKkq_readonly m-0"><code><span class="ͼl">cd</span><span> ~/embodied-arm-system/third_party/EDULITE_A3/el_a3_sdk</span></code></pre></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></div></div></div></pre>

安装依赖：

<pre class="overflow-visible! px-0!" data-start="1163" data-end="1285"><div class="relative w-full mt-4 mb-1"><div class=""><div class="contents"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="relative h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute inset-x-4 top-12 bottom-4"><div class="pointer-events-none sticky z-40 shrink-0 z-1!"><div class="sticky bg-token-border-light"></div></div></div><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class=""><div class="relative"><div class=""><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼd ͼr"><div class="cm-scroller"><pre class="cm-content q9tKkq_readonly m-0"><code><span>python </span><span class="ͼn">-m</span><span> pip install </span><span class="ͼn">-U</span><span> pip setuptools wheel</span><br/><span>pip install pyyaml jinja2 typeguard</span><br/><span>pip install </span><span class="ͼn">-e</span><span></span><span class="ͼk">".[debugger]"</span></code></pre></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></div></div></div></pre>

安装过程中遇到 Qt xcb 插件缺失问题，补充安装相关系统依赖后解决。

启动上位机时，官方命令 `el-a3-debugger --sim` 出现入口模块错误：

<pre class="overflow-visible! px-0!" data-start="1371" data-end="1430"><div class="relative w-full mt-4 mb-1"><div class=""><div class="contents"><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute end-1.5 top-1 z-2 md:end-2 md:top-1"></div><div class="relative"><div class="pe-11 pt-3"><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼd ͼr"><div class="cm-scroller"><pre class="cm-content q9tKkq_readonly m-0"><code><span>ModuleNotFoundError: No module named 'debugger'</span></code></pre></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></div></pre>

因此暂时采用以下方式启动：

<pre class="overflow-visible! px-0!" data-start="1447" data-end="1491"><div class="relative w-full mt-4 mb-1"><div class=""><div class="contents"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="relative h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute inset-x-4 top-12 bottom-4"><div class="pointer-events-none sticky z-40 shrink-0 z-1!"><div class="sticky bg-token-border-light"></div></div></div><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class=""><div class="relative"><div class=""><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼd ͼr"><div class="cm-scroller"><pre class="cm-content q9tKkq_readonly m-0"><code><span>python </span><span class="ͼn">-m</span><span> MotorStudio.main </span><span class="ͼn">--sim</span></code></pre></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></div></div></div></pre>

启动过程中还遇到 NumPy 2.x 与 ROS2 Humble 中 Pinocchio 不兼容的问题，报错信息包含：

<pre class="overflow-visible! px-0!" data-start="1554" data-end="1672"><div class="relative w-full mt-4 mb-1"><div class=""><div class="contents"><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute end-1.5 top-1 z-2 md:end-2 md:top-1"></div><div class="relative"><div class="pe-11 pt-3"><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼd ͼr"><div class="cm-scroller"><pre class="cm-content q9tKkq_readonly m-0"><code><span>A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x</span><br/><span>AttributeError: _ARRAY_API not found</span></code></pre></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></div></pre>

通过在虚拟环境中将 NumPy 降级到 1.x 解决：

<pre class="overflow-visible! px-0!" data-start="1703" data-end="1754"><div class="relative w-full mt-4 mb-1"><div class=""><div class="contents"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="relative h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute inset-x-4 top-12 bottom-4"><div class="pointer-events-none sticky z-40 shrink-0 z-1!"><div class="sticky bg-token-border-light"></div></div></div><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class=""><div class="relative"><div class=""><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼd ͼr"><div class="cm-scroller"><pre class="cm-content q9tKkq_readonly m-0"><code><span>pip install </span><span class="ͼk">"numpy<2"</span><span></span><span class="ͼn">--force-reinstall</span></code></pre></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></div></div></div></pre>

最终执行：

<pre class="overflow-visible! px-0!" data-start="1763" data-end="1807"><div class="relative w-full mt-4 mb-1"><div class=""><div class="contents"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="relative h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute inset-x-4 top-12 bottom-4"><div class="pointer-events-none sticky z-40 shrink-0 z-1!"><div class="sticky bg-token-border-light"></div></div></div><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class=""><div class="relative"><div class=""><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼd ͼr"><div class="cm-scroller"><pre class="cm-content q9tKkq_readonly m-0"><code><span>python </span><span class="ͼn">-m</span><span> MotorStudio.main </span><span class="ͼn">--sim</span></code></pre></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></div></div></div></pre>

成功打开 EDULITE_A3 Debugger 上位机窗口。

当前阶段仅完成上位机软件启动验证，尚未连接真实机械臂，也未配置 CAN 接口。
