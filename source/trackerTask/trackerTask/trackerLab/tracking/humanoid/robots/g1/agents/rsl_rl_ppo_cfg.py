from isaaclab.utils import configclass
from trackerTask.trackerLab.tracking.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg

@configclass
class G123DWalkPPOCfg(BasePPORunnerCfg):
    experiment_name = "g1_23d_walk"
    
@configclass
class G123DRunPPOCfg(BasePPORunnerCfg):
    experiment_name = "g1_23d_run"
    
@configclass
class G129DWalkPPOCfg(BasePPORunnerCfg):
    experiment_name = "g1_29d_walk"
    
@configclass
class G129DRunPPOCfg(BasePPORunnerCfg):
    experiment_name = "g1_29d_run"