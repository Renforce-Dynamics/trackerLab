import gymnasium as gym

gym.register(
    id="TrackerLab-Tracking-Pi-Plus-25D-Walk",
    entry_point="trackerLab.tracker_env:ManagerBasedTrackerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pi_plus_25dof_tracking_env_cfg:PiTrackingWalk",
        "play_env_cfg_entry_point": f"{__name__}.pi_plus_25dof_tracking_env_cfg:PiTrackingWalk_Play",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:PiPlus25DWalkPPOCfg",
    },
)

gym.register(
    id="TrackerLab-Tracking-Pi-Plus-25D-Run",
    entry_point="trackerLab.tracker_env:ManagerBasedTrackerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pi_plus_25dof_tracking_env_cfg:PiTrackingRun",
        "play_env_cfg_entry_point": f"{__name__}.pi_plus_25dof_tracking_env_cfg:PiTrackingRun_Play",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:PiPlus25DRunPPOCfg",
    },
)

gym.register(
    id="TrackerLab-Tracking-Pi-Plus-25D-Jump",
    entry_point="trackerLab.tracker_env:ManagerBasedTrackerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pi_plus_25dof_tracking_env_cfg:PiTrackingJump",
        "play_env_cfg_entry_point": f"{__name__}.pi_plus_25dof_tracking_env_cfg:PiTrackingJump_Play",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:PiPlus25DJumpPPOCfg",
    },
)

gym.register(
    id="TrackerLab-Tracking-Pi-Plus-27D-Walk",
    entry_point="trackerLab.tracker_env:ManagerBasedTrackerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pi_plus_27dof_tracking_env_cfg:PiTrackingWalk",
        "play_env_cfg_entry_point": f"{__name__}.pi_plus_27dof_tracking_env_cfg:PiTrackingWalk_Play",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:PiPlus27DWalkPPOCfg",
    },
)

gym.register(
    id="TrackerLab-Tracking-Pi-Plus-27D-Run",
    entry_point="trackerLab.tracker_env:ManagerBasedTrackerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pi_plus_27dof_tracking_env_cfg:PiTrackingRun",
        "play_env_cfg_entry_point": f"{__name__}.pi_plus_27dof_tracking_env_cfg:PiTrackingRun_Play",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:PiPlus27DRunPPOCfg",
    },
)

gym.register(
    id="TrackerLab-Tracking-Pi-Plus-27D-Jump",
    entry_point="trackerLab.tracker_env:ManagerBasedTrackerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pi_plus_27dof_tracking_env_cfg:PiTrackingJump",
        "play_env_cfg_entry_point": f"{__name__}.pi_plus_27dof_tracking_env_cfg:PiTrackingJump_Play",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:PiPlus27DJumpPPOCfg",
    },
)