<p align="center">
  <img src="docs/images/trackerLab_icon.png" width="80%" alt="TrackerLab Icon"/>
</p>

<h1 align="center">TrackerLab</h1>
<h3 align="center">Unifying IsaacLab and Whole-Body Control in One Modular Framework</h3>
<h3 align="center">Powered by Managers – Built for Motion Intelligence</h3>

---

## 🦿 What is TrackerLab?

**TrackerLab** is a cutting-edge modular framework for humanoid motion **retargeting**, **trajectory tracking**, and **skill-level control**, built on top of [IsaacLab](https://github.com/NVIDIA-Omniverse/IsaacLab).

Whether you're working with **SMPL/FBX motion data**, designing low-level **whole-body controllers**, or building **skill graphs** for high-level motion planning — TrackerLab brings everything together with a clean, extensible **manager-based design**.

> Built to **track**, **compose**, and **control** humanoid motions — seamlessly from dataset to deployment.

### Snapshots

| G1 Debug | G1 Running | G1 Jump |
|----------|------------|------------|
| <video src="./docs/videos/g1_debug.mp4" width="320" controls><a href="./docs/videos/g1_debug.mp4">Video at here</a></video>  | <video src="./docs/videos/g1_running.mp4" width="320" controls><a href="./docs/videos/g1_running.mp4">Video at here</a></video> | <video src="./docs/videos/g1_jump_over.mp4" width="320" controls><a href="./docs/videos/g1_jump_over.mp4">Video at here</a></video> |


## 🚀 Key Features

* 🧠 **IsaacLab-Integrated Motion Tracking**
  Seamlessly plugs motion tracking into IsaacLab's simulation and control framework using manager-based abstraction.

* 🔁 **Full Motion Retargeting Pipeline**
  Converts SMPL/AMASS/FBX human motions into robot-specific trajectories with support for T-pose alignment, filtering, and interpolation.

* 🎮 **Versatile Command Control Modes**
  Switch between multiple control paradigms like ex-body pose control, PHC, and more—using the powerful **CommandManager**.

* 🔀 **Skill Graph via FSM Composition**
  Design complex motion behaviors using FSM-based skill graphs; supports manual triggers, planners, or joystick interfaces.

<p align="center">
  <img src="docs/images/features.jpg" width="100%" alt="TrackerLab Features"/>
</p>

## ⚡ Quick Start

> 🎓 Want to understand TrackerLab quickly?
> 👉 Check out our full [Tutorial (EN)](./docs/tutorial_en.md) or [教程 (中文版)](./docs/tutorial_cn.md)

### ✅ Prerequisites

TrackerLab extends IsaacLab. Make sure IsaacLab and its dependencies are installed properly.
Follow the official [IsaacLab setup guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/quickstart.html) if needed.

### 🚀 Installation

```bash
# Clone TrackerLab
git clone https://github.com/interval-package/trackerlab.git
cd trackerlab

# Activate IsaacLab conda environment
conda activate <env_isaaclab>

# Install TrackerLab and poselib
pip install -e .
pip install -e ./poselib
```

> 💡 No extra packages or repos required — it's fully self-contained!

### 📁 Dataset Preparation

1. Download motion datasets: AMASS or CMU FBX.
2. Apply the retargeting process (see tutorial).
3. Organize data under `./data/` as shown in [data README](./data/README.md).

---

## 🧭 Project Highlights

* ✨ Fully modular and extensible
* 🤖 Designed for real-world humanoid control (e.g., Unitree H1)
* 📚 Clean codebase and manager-based environment design
* 🛠️ Easy integration of new motion sets and control modes

---

## 📂 Project Structure & Data Flow

* [📁 Project Structure](./docs/project_structure.md)
  Understand TrackerLab’s layout and modular system.

* [🔄 Data Flow](./docs/data_flow.md)
  Learn how data flows through the tracking, retargeting, and control pipeline.

* [🔄 Problems](./docs/problems.md)
 Problems you may encounter is recorded.


---

## 🔧 Tasks and Environments

New training and testing tasks are registered under:

```
trackerLab/tasks/
```

Custom Gym environments are recursively registered, including `H1TrackAll`, and can be used directly with IsaacLab's training scripts.

Just add following lines into your train script:

```python
import trackerLab.tasks
```

We also provide a copy from the orginal repo, for which you could directly run:
```bash
python scripts/rsl_rl/base/train.py --task H1TrackingWalk --headless 
# H1 tasks that do not require generate usd, sine we use the isaaclab's usd, however it ruins the performance.
```

For play just directly play it like:

```bash
# on gui
python scripts/rsl_rl/base/play.py --task <Your task> --num_envs 32 # In your work space dir

# on saved video
python scripts/rsl_rl/base/play.py --task <Your task> --num_envs 32 --headless --video --video_length 500 # In your work space dir

```

---

## 📜 Citation

If you find TrackerLab helpful for your work or research, please consider citing:

```bibtex
@software{zheng2025@trackerLab,
  author = {Ziang Zheng},
  title = {TrackerLab: One step unify IsaacLab with multi-mode whole-body control.},
  url = {https://github.com/interval-package/trackerLab},
  year = {2025}
}
```

---

## 👨‍💻 Author

**Zaterval**
📧 [ziang\_zheng@foxmail.com](mailto:ziang_zheng@foxmail.com)

> Looking for collaborators and contributors — feel free to reach out or open an issue!

### Contact

You can join the Wechat group for detialed contact!

![Wechat Group](https://youke1.picui.cn/s1/2025/08/20/68a568770e11f.jpg)


## 📄 License

This project is licensed under the **MIT License**. See [`LICENSE`](./LICENSE) for details.
