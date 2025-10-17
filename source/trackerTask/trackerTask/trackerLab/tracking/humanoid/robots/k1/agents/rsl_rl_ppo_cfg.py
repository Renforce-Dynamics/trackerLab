from isaaclab.utils import configclass
from trackerTask.trackerLab.tracking.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg

@configclass
class Booster_K1_Walk_PPOCfg(BasePPORunnerCfg):
    experiment_name = "tracking_booster_k1_walk"
    max_iterations = 7500
    
@configclass
class Booster_K1_Walk_Full_PPOCfg(BasePPORunnerCfg):
    experiment_name = "tracking_booster_k1_walk_full"
    max_iterations = 7500

@configclass
class Booster_K1_Walk_Full_Deploy_PPOCfg(BasePPORunnerCfg):
    experiment_name = "tracking_booster_k1_walk_full_deploy"
    max_iterations = 30000

@configclass
class Booster_K1_Walk_Full_Deploy_WithoutHistory_PPOCfg(BasePPORunnerCfg):
    experiment_name = "tracking_booster_k1_walk_full_deploy_without_history"
    max_iterations = 30000

@configclass
class Booster_K1_Run_PPOCfg(BasePPORunnerCfg):
    experiment_name = "tracking_booster_k1_run"
    max_iterations = 15000