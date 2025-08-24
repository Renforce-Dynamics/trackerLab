import gymnasium as gym

gym.register(
    id="Unitree-G1-29dof-Velocity",
    entry_point="trackerLab.tracker_env:ManagerBasedTrackerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotEnvCfg",
        "rsl_rl_cfg_entry_point": f"trackerLab.tasks.velocity.agents.rsl_rl_ppo_cfg:G1_Velocity_PPORunnerCfg",
    },
)

gym.register(
    id="Unitree-G1-29dof-Velocity-Play",
    entry_point="trackerLab.tracker_env:ManagerBasedTrackerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"trackerLab.tasks.velocity.agents.rsl_rl_ppo_cfg:G1_Velocity_PPORunnerCfg",
    },
)
