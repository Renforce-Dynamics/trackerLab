from isaaclab.utils import configclass
from trackerTask.trackerLab.tracking.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg

@configclass
class H1WalkPPOCfg(BasePPORunnerCfg):
    experiment_name = "h1_walk"
    
@configclass
class G123DRunPPOCfg(BasePPORunnerCfg):
    experiment_name = "h1_run"