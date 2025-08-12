import gymnasium as gym

from . import agent

##
# Register Gym environments.
##

gym.register(
    id="R2TrackWalk",
    entry_point="trackerLab.tracker_env:ManagerBasedTrackerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.r2_track_cfg:R2TrackWalk",
        "rsl_rl_cfg_entry_point": f"{agent.__name__}.rsl_rl_cfg:R2TrackWalk",
    },
)

gym.register(
    id="R2TrackRun",
    entry_point="trackerLab.tracker_env:ManagerBasedTrackerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.r2_track_cfg:R2TrackRun",
        "rsl_rl_cfg_entry_point": f"{agent.__name__}.rsl_rl_cfg:R2TrackRun",
    },
)

gym.register(
    id="R2TrackJumpOver",
    entry_point="trackerLab.tracker_env:ManagerBasedTrackerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.r2_track_cfg:R2TrackJumpOver",
        "rsl_rl_cfg_entry_point": f"{agent.__name__}.rsl_rl_cfg:R2TrackJumpOver",
    },
)