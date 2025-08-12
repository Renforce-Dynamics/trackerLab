import gymnasium as gym

from . import agent

##
# Register Gym environments.
##

gym.register(
    id="H1TrackingWalk",
    entry_point="trackerLab.tracker_env:ManagerBasedTrackerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.h1_tracking_env_cfg:H1TrackingWalk",
        "rsl_rl_cfg_entry_point": f"{agent.__name__}.rsl_rl_cfg:H1TrackingWalk",
    },
)