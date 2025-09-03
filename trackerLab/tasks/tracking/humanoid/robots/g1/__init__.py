import gymnasium as gym

gym.register(
    id="TrackerLab-Tracking-Unitree-G1-23D-Walk",
    entry_point="trackerLab.tracker_env:ManagerBasedTrackerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_23d_tracking_env_cfg:G1TrackingWalk",
        "play_env_cfg_entry_point": f"{__name__}.g1_23d_tracking_env_cfg:G1TrackingWalk_Play",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:G123DWalkPPOCfg",
    },
)

# gym.register(
#     id="G123DAMPTrackingWalk",
#     entry_point="trackerLab.tracker_env:ManagerBasedAMPEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.g1_23d_tracking_env_cfg:G1TrackingWalk",
#         "rsl_rl_cfg_entry_point": f"{agent.__name__}.g1_23d_rsl_rl_cfg:G1AMPTrackingWalk",
#     },
# )

gym.register(
    id="TrackerLab-Tracking-Unitree-G1-29D-Walk",
    entry_point="trackerLab.tracker_env:ManagerBasedTrackerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_29d_tracking_env_cfg:G1TrackingWalk",
        "play_env_cfg_entry_point": f"{__name__}.g1_29d_tracking_env_cfg:G1TrackingWalk_Play",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:G129DWalkPPOCfg",
    },
)


gym.register(
    id="TrackerLab-Tracking-Unitree-G1-23D-Walk-Replay",
    entry_point="trackerLab.tracker_env:ManagerBasedTrackerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_23d_tracking_env_cfg:G1TrackingWalk_Replay",
        "play_env_cfg_entry_point": f"{__name__}.g1_23d_tracking_env_cfg:G1TrackingWalk_Replay",
        "rsl_rl_cfg_entry_point": f"trackerLab.tasks.tracking.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)