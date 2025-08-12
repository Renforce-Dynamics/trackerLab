import gymnasium as gym

from . import agent

##
# Register Gym environments.
##

gym.register(
    id="H1Track",
    entry_point="trackerLab.tracker_env:ManagerBasedTrackerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.h1_track_cfg:H1TrackEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agent.__name__}.rsl_rl_cfg:H1TrackerPPORunnerCfg",
    },
)

gym.register(
    id="H1TrackAll",
    entry_point="trackerLab.tracker_env:ManagerBasedTrackerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.h1_track_cfg:H1TrackAll",
        "rsl_rl_cfg_entry_point": f"{agent.__name__}.rsl_rl_cfg:H1TrackAll",
    },
)

gym.register(
    id="H1TrackDebugPunch",
    entry_point="trackerLab.tracker_env:ManagerBasedTrackerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.h1_track_cfg:H1TrackDebugPunch",
        "rsl_rl_cfg_entry_point": f"{agent.__name__}.rsl_rl_cfg:H1TrackDebugPunch",
    },
)

gym.register(
    id="H1TrackDebugWalk",
    entry_point="trackerLab.tracker_env:ManagerBasedTrackerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.h1_track_cfg:H1TrackDebugWalk",
        "rsl_rl_cfg_entry_point": f"{agent.__name__}.rsl_rl_cfg:H1TrackDebugWalk",
    },
)

gym.register(
    id="H1TrackRun",
    entry_point="trackerLab.tracker_env:ManagerBasedTrackerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.h1_track_cfg:H1TrackRun",
        "rsl_rl_cfg_entry_point": f"{agent.__name__}.rsl_rl_cfg:H1TrackRun",
    },
)

gym.register(
    id="H1TrackJump",
    entry_point="trackerLab.tracker_env:ManagerBasedTrackerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.h1_track_cfg:H1TrackJump",
        "rsl_rl_cfg_entry_point": f"{agent.__name__}.rsl_rl_cfg:H1TrackJump",
    },
)

gym.register(
    id="H1TrackJumpOver",
    entry_point="trackerLab.tracker_env:ManagerBasedTrackerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.h1_track_cfg:H1TrackJumpOver",
        "rsl_rl_cfg_entry_point": f"{agent.__name__}.rsl_cmd:H1TrackJumpOver",
    },
)