from isaaclab.utils import configclass
from trackerLab.tasks.tracking.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg

@configclass
class R2bWalkPPOCfg(BasePPORunnerCfg):
    experiment_name = "r2b_walk"
    
@configclass
class R2bRunPPOCfg(BasePPORunnerCfg):
    experiment_name = "r2b_run"