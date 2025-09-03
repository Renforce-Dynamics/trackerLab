from isaaclab.utils import configclass
from trackerLab.tasks.tracking.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg

@configclass
class Booster_K1_Walk_PPOCfg(BasePPORunnerCfg):
    experiment_name = "booster_k1_walk"
    max_iterations = 15000
    