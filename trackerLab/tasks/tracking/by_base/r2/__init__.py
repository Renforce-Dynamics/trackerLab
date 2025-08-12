import gymnasium as gym

from . import agent

##
# Register Gym environments.
##

gym.register(
    id="R2TrackingWalk",
    entry_point="trackerLab.tracker_env:ManagerBasedTrackerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.r2_tracking_env_cfg:R2TrackingWalk",
        "rsl_rl_cfg_entry_point": f"{agent.__name__}.rsl_rl_cfg:R2TrackingWalk",
    },
)