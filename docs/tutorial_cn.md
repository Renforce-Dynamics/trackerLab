# 🧭 TrackerLab Tutorial

> One repo unify IsaacLab & whole-body control, fully built on managers!

---

## 📌 Introduction

**TrackerLab** 是一个模块化框架，集成了人体动作的 **Retargeting**、**轨迹跟踪（Tracking）** 与 **技能控制（Skill-based Control）**，构建于 [IsaacLab](https://github.com/NVIDIA-Omniverse/IsaacLab) 之上。它打通了从 **人类动作数据（如 SMPL / FBX）** 到 **机器人控制器（如 Unitree H1）** 的路径，提供了统一的轨迹管理器接口，支持多种动作组织方式。

---

## 🛠️ Environment Setup

### Step 1. 安装 IsaacLab

请先确保你已经完成 [IsaacLab 的配置](https://isaac-sim.github.io/IsaacLab/main/source/setup/quickstart.html)，包括 Isaac Sim 安装与 Python 环境配置。

### Step 2. 克隆并安装 TrackerLab

```bash
git clone https://github.com/interval-package/trackerlab.git
cd trackerlab

conda activate <env_isaaclab>  # 替换为你配置 IsaacLab 的 conda 环境名

# 安装主项目与 poselib（用于姿态处理）
pip install -e .
pip install -e ./poselib
```

---

## 📂 Dataset Preparation

### 方法一：使用 AMASS（推荐）

1. 前往 [AMASS 官网](https://amass.is.tue.mpg.de) 下载数据集。
2. 将解压后的内容放入：

```
./data/amass/
```

3. 结构参考：

```
data/
├── amass/
├── retarget_cfg/
│   └── retar_smpl_2_h1.json
├── tpose/
│   ├── h1_tpose.npy
│   └── smpl_tpose.npy
```

> 更多信息见 [data/README.md](../data/README.md)

### 方法二：使用 CMU FBX 数据

1. 安装 FBX SDK，参见 [issue#61](https://github.com/nv-tlabs/ASE/issues/61)。
2. 参考 [Expressive Humanoid](https://github.com/chengxuxin/expressive-humanoid) 将 `.fbx` 文件转化为 `.npy` 动作数据。


### 机器人资产下载

这里用了宇树开源的数据模型，直接的下载脚本见：
[docs/dataset_download.md](../docs/dataset_download.md)

## 🔁 Motion Retargeting

TrackerLab 支持将标准人体动作数据（如 AMASS）转化为多种机器人骨架格式的轨迹数据。

### Step 1. 检查配置

- 确保 AMASS 数据已放入 `./data/amass/`
- 检查是否有对应的 `retarget_cfg` 配置：
  - 示例：`./data/retarget_cfg/retar_smpl_2_h1.json`
  - 包含关节映射与旋转调整信息
- 确保存在 T-pose 信息：
  - `./data/tpose/smpl_tpose.npy`
  - `./data/tpose/h1_tpose.npy`

### Step 2. 执行 Retargeting 脚本

参考[retarget tutorial](./retarget_tutorial.md)

```bash
python poselib/scripts/amass/retarget_all.py   --folders AMASS_SUBFOLDER1 AMASS_SUBFOLDER2   --robot h1   --max_frames 1000   --workers 8
```

处理完成后，结果保存在：

```
./data/retargeted/amass_results/
```

---

## 🏋️ Training（基于 IsaacLab）

TrackerLab 与 IsaacLab 完全兼容，在训练策略时无需特殊配置。

### 启动训练：


使用自带的脚本：
```bash
python scripts/rsl_rl/base/train.py --task R2TrackWalk --headless 
```

对于使用官方或者是其他框架给的脚本，在训练脚本开始时，加上：
```python
import trackerTask.trackerLab
```

```bash
python IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py   --task H1TrackAll   --headless
```

其中 `H1TrackAll` 是在 `trackerLab` 中定义的新任务环境名。

## ⚙️ 追踪环境配置教程

TrackerLab 的训练环境基于 IsaacLab 的模块化管理器设计，并添加了专用于运动追踪的 `MotionManager` 模块。

### 🧩 注册追踪环境

与原始 IsaacLab 环境不同，TrackerLab 的环境使用新的注册入口 `"trackerLab.tracker_env:ManagerBasedTrackerEnv"`：

```python
gym.register(
    id="H1TrackAll",
    entry_point="trackerLab.tracker_env:ManagerBasedTrackerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.h1_track_cfg:H1TrackAll",
        "rsl_rl_cfg_entry_point": f"{agent.__name__}.rsl_rl_cfg:H1TrackAll",
    },
)
```

此注册方式是集成 TrackerLab 的运动管理器所必需的。

### 🗂️ 构建训练数据集配置

在 `./data/configs` 目录下，使用 YAML 文件格式定义训练所需的运动数据集。例如：

```yaml
motions:
  '02/02_02_poses':
    description: "walk episode"
    difficulty: 4
    trim_beg: -1
    trim_end: -1
    weight: 1.0
  '127/127_28_poses':
    description: Run Jump Over
    difficulty: 4
    trim_beg: -1
    trim_end: -1
    weight: 1.0
  root: "amass_results/r2y/CMU"
```

- `motions` 中的每一项是 `.npy` 运动文件的相对路径（基于 `./data/retargeted`）。
- `root` 是这些运动数据的根目录。

### 🧱 追踪环境配置

在你的环境配置文件中添加 `motion` 字段，类型为 `MotionManagerCfg`：

```python
@configclass
class TrackEnvCfg(TrackEnvBaseCfg):
    rewards: ...
    ...
    motion: MotionManagerCfg = MotionManagerCfg(
        motion_buffer_cfg=MotionBufferCfg(
            motion=MotionBufferCfg.MotionCfg(
                motion_name=<Your *.yaml, left * here>
                ),
            regen_pkl=False,  # 仅第一次加载缓存
        ),
        static_motion=False,
        robot_type="h1",
        obs_from_buffer=False,
        reset_to_pose=False
    )
```

该配置负责连接训练用数据集，并控制缓存使用与初始化方式。


---

## ⚙️ 项目结构参考

详见：[项目结构文档](./docs/project_structure.md) 和 [数据流图](./docs/data_flow.md)

---

## 📚 进一步阅读

| 文档 | 内容 |
|------|------|
| [README](../README.md) | 项目主页说明 |
| [dataset_download.md](./dataset_download.md) | 数据集准备细节 |
| [retarget_tutorial.md](./retarget_tutorial.md) | Retarget 使用说明 |
| [project_structure.md](./project_structure.md) | 项目结构与模块功能 |
| [data_flow.md](./data_flow.md) | TrackerLab 数据流与关键流程图 |
| [new_env_setup.md](./new_env_setup.md) | TrackerLab Create new env. |

---

## 🙋 常见问题

[problems](./problems.md)
