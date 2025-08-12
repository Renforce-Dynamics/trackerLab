# 🧭 TrackerLab Tutorial

> One repo to unify IsaacLab & whole-body control, fully built on managers!

---

## 📌 Introduction

**TrackerLab** is a modular framework integrating **motion retargeting**, **trajectory tracking**, and **skill-based control** for humanoid motion. Built on top of [IsaacLab](https://github.com/NVIDIA-Omniverse/IsaacLab), it bridges the gap between **human motion data** (e.g., SMPL / FBX) and **robot control systems** (e.g., Unitree H1). It provides a unified trajectory management interface, supporting various modes of motion organization.

---

## 🛠️ Environment Setup

### Step 1. Install IsaacLab

Please follow the official [IsaacLab Quickstart Guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/quickstart.html), including Isaac Sim installation and Python environment setup.

### Step 2. Clone and Install TrackerLab

```bash
git clone https://github.com/interval-package/trackerlab.git
cd trackerlab

conda activate <env_isaaclab>  # Replace with your IsaacLab conda env name

# Install main project and poselib (for pose processing)
pip install -e .
pip install -e ./poselib
```

---

## 📂 Dataset Preparation

### Option 1: Use AMASS (Recommended)

1. Download AMASS dataset from the [official site](https://amass.is.tue.mpg.de).
2. Place extracted content in:

```
./data/amass/
```

3. Directory structure:

```
data/
├── amass/
├── retarget_cfg/
│   └── retar_smpl_2_h1.json
├── tpose/
│   ├── h1_tpose.npy
│   └── smpl_tpose.npy
```

> For details, refer to [data/README.md](../data/README.md)

### Option 2: Use CMU FBX

1. Install FBX SDK (see [issue#61](https://github.com/nv-tlabs/ASE/issues/61)).
2. Use [Expressive Humanoid](https://github.com/chengxuxin/expressive-humanoid) to convert `.fbx` to `.npy`.

---

## 🔁 Motion Retargeting

TrackerLab supports converting standard motion datasets (e.g., AMASS) into robot-compatible trajectory formats.

### Step 1. Check Config

* Ensure AMASS is under `./data/amass/`
* Check for proper retarget config like:

  * `./data/retarget_cfg/retar_smpl_2_h1.json`
* T-pose files required:

  * `./data/tpose/smpl_tpose.npy`
  * `./data/tpose/h1_tpose.npy`

### Step 2. Run Retargeting Script

See [retarget tutorial](./retarget_tutorial.md):

```bash
python poselib/scripts/amass/retarget_all.py \
  --folders AMASS_SUBFOLDER1 AMASS_SUBFOLDER2 \
  --robot h1 \
  --max_frames 1000 \
  --workers 8
```

Results saved in:

```
./data/retargeted/amass_results/
```

---

## 🏋️ Training (Based on IsaacLab)

TrackerLab is fully compatible with IsaacLab. No extra configuration is required.

### Launch training:

```bash
python scripts/rsl_rl/base/train.py --task R2TrackWalk --headless 
```

If using other framework's training scripts, add
```
import trackerLab.tasks
```

```bash
python IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py \
  --task H1TrackAll \
  --headless
```

Here, `H1TrackAll` is the custom environment defined in `trackerLab`.

---

## ⚙️ Tracking Environment Configuration

TrackerLab's training environment extends IsaacLab's manager system with a dedicated `MotionManager` module.

### 🧩 Register a Tracking Environment

Unlike default IsaacLab environments, TrackerLab uses a new entry point:

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

### 🗂️ Define Training Motion Set

Place YAML configs in `./data/configs/`. Example:

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

* `motions` points to `.npy` files relative to `./data/retargeted`
* `root` specifies the dataset base path

### 🧱 Configure the Tracking Environment

Set `motion` in the env config file:

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
            regen_pkl=False,  # load once if True
        ),
        static_motion=False,
        robot_type="h1",
        obs_from_buffer=False,
        reset_to_pose=False
    )
```

This connects the motion dataset with the policy training.

---

## 🧾 Project Structure

See [project\_structure.md](./docs/project_structure.md) and [data\_flow.md](./docs/data_flow.md) for more details.

---

## 📚 Further Reading

| Document                                        | Description                    |
| ----------------------------------------------- | ------------------------------ |
| [README](../README.md)                          | Project introduction           |
| [dataset\_download.md](./dataset_download.md)   | Dataset details                |
| [retarget\_tutorial.md](./retarget_tutorial.md) | Retargeting usage              |
| [project\_structure.md](./project_structure.md) | Structure and modules          |
| [data\_flow.md](./data_flow.md)                 | Data flow and key logic        |
| [new\_env\_setup.md](./new_env_setup.md)        | Creating new env in TrackerLab |

---

## 🙋 FAQ

| Problem                 | Solution                                                 |
| ----------------------- | -------------------------------------------------------- |
| poselib import fails    | Run `pip install -e ./poselib`                           |
| retarget\_cfg not found | Check `./data/retarget_cfg` and ensure the config exists |
| tpose files missing     | Generate via script or use example tposes                |

