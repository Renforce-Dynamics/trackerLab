import gymnasium as gym

gym.register(
    id="TrackerLab-Tracking-Unitree-H1-Walk",
    entry_point="trackerLab.tracker_env:ManagerBasedTrackerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.h1_tracking_env_cfg:H1TrackingWalk",
        "play_env_cfg_entry_point": f"{__name__}.h1_tracking_env_cfg:H1TrackingWalk_Play",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:H1WalkPPOCfg",
    },
)