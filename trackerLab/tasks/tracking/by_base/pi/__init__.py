import gymnasium as gym

from . import agent

##
# Register Gym environments.
##

gym.register(
    id="PiPlus27DofTrackingWalk",
    entry_point="trackerLab.tracker_env:ManagerBasedTrackerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pi_plus_27dof_tracking_env_cfg:PiTrackingWalk",
        "rsl_rl_cfg_entry_point": f"{agent.__name__}.pi_plus_27dof_rsl_rl_cfg:PiTrackingWalk",
    },
)