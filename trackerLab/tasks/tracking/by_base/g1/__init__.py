import gymnasium as gym

from . import agent
from . import skrl_agent

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
    id="G123DAMPTrackingWalk",
    entry_point="trackerLab.tracker_env:ManagerBasedAMPTrackerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_23d_tracking_env_cfg:G1AMPTrackingWalk",
        "skrl_amp_cfg_entry_point": f"{skrl_agent.__name__}:skrl_walk_amp_cfg.yaml",
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