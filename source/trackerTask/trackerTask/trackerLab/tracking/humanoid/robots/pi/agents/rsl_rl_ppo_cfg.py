from isaaclab.utils import configclass
from trackerTask.trackerLab.tracking.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg

@configclass
class PiPlus25DWalkPPOCfg(BasePPORunnerCfg):
    experiment_name = "pi_plus_25d_walk"
    
@configclass
class PiPlus25DRunPPOCfg(BasePPORunnerCfg):
    experiment_name = "pi_plus_25d_run"
    
@configclass
class PiPlus25DJumpPPOCfg(BasePPORunnerCfg):
    experiment_name = "pi_plus_25d_jump"
    
@configclass
class PiPlus27DWalkPPOCfg(BasePPORunnerCfg):
    experiment_name = "pi_plus_27d_walk"
    
@configclass
class PiPlus27DRunPPOCfg(BasePPORunnerCfg):
    experiment_name = "pi_plus_27d_run"
    
@configclass
class PiPlus27DJumpPPOCfg(BasePPORunnerCfg):
    experiment_name = "pi_plus_27d_jump"