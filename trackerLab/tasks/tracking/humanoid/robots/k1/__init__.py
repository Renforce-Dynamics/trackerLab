import gymnasium as gym

gym.register(
    id="TrackerLab-Tracking-Booster-K1-Walk",
    entry_point="trackerLab.tracker_env:ManagerBasedTrackerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.k1_22dof_tracking_env_cfg:Booster_K1_TrackingWalk",
        "play_env_cfg_entry_point": f"{__name__}.k1_22dof_tracking_env_cfg:Booster_K1_TrackingWalk_Play",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:Booster_K1_Walk_PPOCfg",
    },
)

gym.register(
    id="TrackerLab-Tracking-Booster-K1-Walk-Full",
    entry_point="trackerLab.tracker_env:ManagerBasedTrackerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.k1_22dof_tracking_env_cfg:Booster_K1_TrackingWalk_Full",
        "play_env_cfg_entry_point": f"{__name__}.k1_22dof_tracking_env_cfg:Booster_K1_TrackingWalk_Full_Play",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:Booster_K1_Walk_PPOCfg",
    },
)

gym.register(
    id="TrackerLab-Tracking-Booster-K1-Run",
    entry_point="trackerLab.tracker_env:ManagerBasedTrackerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.k1_22dof_tracking_env_cfg:Booster_K1_TrackingRun",
        "play_env_cfg_entry_point": f"{__name__}.k1_22dof_tracking_env_cfg:Booster_K1_TrackingRun_Play",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:Booster_K1_Walk_PPOCfg",
    },
)

