import gymnasium as gym

from . import agent

##
# Register Gym environments.
##

gym.register(
    id="PiPlus25DofTrackingWalk",
    entry_point="trackerLab.tracker_env:ManagerBasedTrackerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pi_plus_25dof_tracking_env_cfg:PiTrackingWalk",
        "rsl_rl_cfg_entry_point": f"{agent.__name__}.pi_plus_25dof_rsl_rl_cfg:PiTrackingWalk",
    },
)

gym.register(
    id="PiPlus25DofTrackingRun",
    entry_point="trackerLab.tracker_env:ManagerBasedTrackerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pi_plus_25dof_tracking_env_cfg:PiTrackingRun",
        "rsl_rl_cfg_entry_point": f"{agent.__name__}.pi_plus_25dof_rsl_rl_cfg:PiTrackingRun",
    },
)

gym.register(
    id="PiPlus25DofTrackingJump",
    entry_point="trackerLab.tracker_env:ManagerBasedTrackerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pi_plus_25dof_tracking_env_cfg:PiTrackingJump",
        "rsl_rl_cfg_entry_point": f"{agent.__name__}.pi_plus_25dof_rsl_rl_cfg:PiTrackingJump",
    },
)

gym.register(
    id="PiPlus27DofTrackingWalk",
    entry_point="trackerLab.tracker_env:ManagerBasedTrackerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pi_plus_27dof_tracking_env_cfg:PiTrackingWalk",
        "rsl_rl_cfg_entry_point": f"{agent.__name__}.pi_plus_27dof_rsl_rl_cfg:PiTrackingWalk",
    },
)

gym.register(
    id="PiPlus27DofTrackingRun",
    entry_point="trackerLab.tracker_env:ManagerBasedTrackerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pi_plus_27dof_tracking_env_cfg:PiTrackingRun",
        "rsl_rl_cfg_entry_point": f"{agent.__name__}.pi_plus_27dof_rsl_rl_cfg:PiTrackingRun",
    },
)

gym.register(
    id="PiPlus27DofTrackingJump",
    entry_point="trackerLab.tracker_env:ManagerBasedTrackerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pi_plus_27dof_tracking_env_cfg:PiTrackingJump",
        "rsl_rl_cfg_entry_point": f"{agent.__name__}.pi_plus_27dof_rsl_rl_cfg:PiTrackingJump",
    },
)