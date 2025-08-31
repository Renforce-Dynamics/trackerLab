import gymnasium as gym

gym.register(
    id="TrackerLab-Tracking-Pi-Plus-25D-Walk",
    entry_point="trackerLab.tracker_env:ManagerBasedTrackerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pi_plus_25dof_tracking_env_cfg:PiTrackingWalk",
        "play_env_cfg_entry_point": f"{__name__}.pi_plus_25dof_tracking_env_cfg:PiTrackingWalk_Play",
        "rsl_rl_cfg_entry_point": f"trackerLab.tasks.tracking.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="TrackerLab-Tracking-Pi-Plus-25D-Run",
    entry_point="trackerLab.tracker_env:ManagerBasedTrackerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pi_plus_25dof_tracking_env_cfg:PiTrackingRun",
        "play_env_cfg_entry_point": f"{__name__}.pi_plus_25dof_tracking_env_cfg:PiTrackingRun_Play",
        "rsl_rl_cfg_entry_point": f"trackerLab.tasks.tracking.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="TrackerLab-Tracking-Pi-Plus-25D-Jump",
    entry_point="trackerLab.tracker_env:ManagerBasedTrackerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pi_plus_25dof_tracking_env_cfg:PiTrackingJump",
        "play_env_cfg_entry_point": f"{__name__}.pi_plus_25dof_tracking_env_cfg:PiTrackingJump_Play",
        "rsl_rl_cfg_entry_point": f"trackerLab.tasks.tracking.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="TrackerLab-Tracking-Pi-Plus-27D-Walk",
    entry_point="trackerLab.tracker_env:ManagerBasedTrackerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pi_plus_27dof_tracking_env_cfg:PiTrackingWalk",
        "play_env_cfg_entry_point": f"{__name__}.pi_plus_27dof_tracking_env_cfg:PiTrackingWalk_Play",
        "rsl_rl_cfg_entry_point": f"trackerLab.tasks.tracking.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="TrackerLab-Tracking-Pi-Plus-27D-Run",
    entry_point="trackerLab.tracker_env:ManagerBasedTrackerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pi_plus_27dof_tracking_env_cfg:PiTrackingRun",
        "play_env_cfg_entry_point": f"{__name__}.pi_plus_27dof_tracking_env_cfg:PiTrackingRun_Play",
        "rsl_rl_cfg_entry_point": f"trackerLab.tasks.tracking.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="TrackerLab-Tracking-Pi-Plus-27D-Jump",
    entry_point="trackerLab.tracker_env:ManagerBasedTrackerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pi_plus_27dof_tracking_env_cfg:PiTrackingJump",
        "play_env_cfg_entry_point": f"{__name__}.pi_plus_27dof_tracking_env_cfg:PiTrackingJump_Play",
        "rsl_rl_cfg_entry_point": f"trackerLab.tasks.tracking.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)