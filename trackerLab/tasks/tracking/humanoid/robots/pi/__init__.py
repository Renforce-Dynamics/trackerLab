import gymnasium as gym

gym.register(
    id="Tracking-Pi-Plus-25D-Walk",
    entry_point="trackerLab.tracker_env:ManagerBasedTrackerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pi_plus_25dof_tracking_env_cfg:PiTrackingWalk",
        "rsl_rl_cfg_entry_point": f"trackerLab.tasks.tracking.agents.rsl_rl_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Pi-Plus-25D-Run",
    entry_point="trackerLab.tracker_env:ManagerBasedTrackerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pi_plus_25dof_tracking_env_cfg:PiTrackingRun",
        "rsl_rl_cfg_entry_point": f"trackerLab.tasks.tracking.agents.rsl_rl_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Pi-Plus-25D-Jump",
    entry_point="trackerLab.tracker_env:ManagerBasedTrackerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pi_plus_25dof_tracking_env_cfg:PiTrackingJump",
        "rsl_rl_cfg_entry_point": f"trackerLab.tasks.tracking.agents.rsl_rl_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Pi-Plus-27D-Walk",
    entry_point="trackerLab.tracker_env:ManagerBasedTrackerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pi_plus_27dof_tracking_env_cfg:PiTrackingWalk",
        "rsl_rl_cfg_entry_point": f"trackerLab.tasks.tracking.agents.rsl_rl_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Pi-Plus-27D-Run",
    entry_point="trackerLab.tracker_env:ManagerBasedTrackerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pi_plus_27dof_tracking_env_cfg:PiTrackingRun",
        "rsl_rl_cfg_entry_point": f"trackerLab.tasks.tracking.agents.rsl_rl_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Pi-Plus-27D-Jump",
    entry_point="trackerLab.tracker_env:ManagerBasedTrackerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pi_plus_27dof_tracking_env_cfg:PiTrackingJump",
        "rsl_rl_cfg_entry_point": f"trackerLab.tasks.tracking.agents.rsl_rl_cfg:BasePPORunnerCfg",
    },
)