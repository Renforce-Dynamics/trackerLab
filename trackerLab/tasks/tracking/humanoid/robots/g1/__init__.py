import gymnasium as gym

from . import agent

##
# Register Gym environments.
##

gym.register(
    id="G123DTrackingWalk",
    entry_point="trackerLab.tracker_env:ManagerBasedTrackerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_23d_tracking_env_cfg:G1TrackingWalk",
        "rsl_rl_cfg_entry_point": f"{agent.__name__}.g1_23d_rsl_rl_cfg:G1TrackingWalk",
    },
)

gym.register(
    id="G129DLocoTrackingWalk",
    entry_point="trackerLab.tracker_env:ManagerBasedTrackerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_29d_loco_tracking_env_cfg:G1TrackingWalk",
        "rsl_rl_cfg_entry_point": f"{agent.__name__}.g1_29d_loco_rsl_rl_cfg:G1TrackingWalk",
    },
)

gym.register(
    id="G129DTrackingWalk",
    entry_point="trackerLab.tracker_env:ManagerBasedTrackerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_29d_tracking_env_cfg:G1TrackingWalk",
        "rsl_rl_cfg_entry_point": f"{agent.__name__}.g1_29d_rsl_rl_cfg:G1TrackingWalk",
    },
)