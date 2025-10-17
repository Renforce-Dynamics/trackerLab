from isaaclab.utils import configclass
from trackerTask.trackerLab.tracking.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg

@configclass
class SMPLXWalkPPOCfg(BasePPORunnerCfg):
    experiment_name = "SMPLX_walk"
