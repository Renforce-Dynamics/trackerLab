from isaaclab.utils import configclass
from trackerLab.tasks.tracking.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg

@configclass
class SMPLXWalkPPOCfg(BasePPORunnerCfg):
    experiment_name = "SMPLX_walk"
