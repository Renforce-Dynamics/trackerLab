import gymnasium as gym

gym.register(
    id="Booster-k1-22dof-Velocity",
    entry_point="trackerLab.tracker_env:ManagerBasedTrackerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"trackerTask.trackerLab.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)
