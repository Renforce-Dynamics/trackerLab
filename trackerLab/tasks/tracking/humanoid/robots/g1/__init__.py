import gymnasium as gym

gym.register(
    id="Tracking-Unitree-G1-23D-Walk",
    entry_point="trackerLab.tracker_env:ManagerBasedTrackerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_23d_tracking_env_cfg:G1TrackingWalk",
        "play_env_cfg_entry_point": f"{__name__}.g1_23d_tracking_env_cfg:G1TrackingWalk_Play",
        "rsl_rl_cfg_entry_point": f"trackerLab.tasks.tracking.agents.rsl_rl_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Unitree-G1-29D-Walk",
    entry_point="trackerLab.tracker_env:ManagerBasedTrackerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_29d_tracking_env_cfg:G1TrackingWalk",
        "play_env_cfg_entry_point": f"{__name__}.g1_29d_tracking_env_cfg:G1TrackingWalk_Play",
        "rsl_rl_cfg_entry_point": f"trackerLab.tasks.tracking.agents.rsl_rl_cfg:BasePPORunnerCfg",
    },
)