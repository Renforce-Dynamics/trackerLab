import gymnasium as gym

gym.register(
    id="Tracking-R2-Walk",
    entry_point="trackerLab.tracker_env:ManagerBasedTrackerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.r2b_tracking_env_cfg:R2TrackingWalk",
        "play_env_cfg_entry_point": f"{__name__}.r2b_tracking_env_cfg:R2TrackingWalk_Play",
        "rsl_rl_cfg_entry_point": f"trackerLab.tasks.tracking.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="Tracking-R2-Run",
    entry_point="trackerLab.tracker_env:ManagerBasedTrackerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.r2b_tracking_env_cfg:R2TrackingRun",
        "play_env_cfg_entry_point": f"{__name__}.r2b_tracking_env_cfg:R2TrackingRun_Play",
        "rsl_rl_cfg_entry_point": f"trackerLab.tasks.tracking.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)