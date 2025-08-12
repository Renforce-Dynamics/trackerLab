# Tracking Env Configuration Tutorial

The env configuration follows the basic design of the isaaclab, which utilize the manager mechanism.
For a tracking env, we adpot the manager design by adding the motion manager.

## Register a tracker env

Should note that, different from the isaaclab original env, the tracker env requires a new entry point at `"trackerLab.tracker_env:ManagerBasedTrackerEnv"`.

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

## Setup a tracker env

If you want to setup new tracking training set and tracking cfg, following steps might be helpful.

### Tracking Set Setup

Configuring your training set configuration in `*.yaml` files under `./data/configs`, the tructure should be organized as follow:

```
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

where the motions contains the target motions that will be used in training while the `root` shows where the `*.npy` files is stored. The `root` is relative based on `./data/retargeted`.

### Tracking Env Configure

The only different from original env configuration is that we should add motion attribute to the env configuration file, like the rewards et. al. do.

```python
@configclass
class TrackEnvCfg(TrackEnvBaseCfg):
    rewards: ...
    ...
    motion: MotionManagerCfg = MotionManagerCfg(
        motion_buffer_cfg=MotionBufferCfg(
                motion=<The *.yaml file>,
                regen_pkl=False, # only load once if True
            ),
        static_motion=False,
        robot_type="h1",
        obs_from_buffer=False,
        reset_to_pose=False
    )
```