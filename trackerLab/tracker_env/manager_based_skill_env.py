import torch
from isaaclab.utils import configclass
from trackerLab.managers import MotionManager, MotionManagerCfg, SkillManager, SkillManagerCfg
from trackerLab.tracker_action.skill_action import SkillAction, SkillActionCfg
from .manager_based_tracker_env import ManagerBasedTrackerEnv, ManagerBasedRLEnv
from .manager_based_tracker_env_cfg import ManagerBasedTrackerEnvCfg


class ManagerBasedSkillEnv(ManagerBasedTrackerEnv):
    """
    Same as the tracker env, the hierarchy is defined by the decimation of the sim & motion & manager.
    
    Every sim decimation step, will apply the low level action. 
    eg. in dynamic step, the low level action will be applied every sim step.
    Every low level action decimation step, will update the low level action. 
    eg. deciamtion = 4, every 4th step will update the low level action.
    Every skill decimation step, will update the skill. 
    eg. decimation = 40, the skill will be updated every 40th sim step. 40/4 = 10 low level action steps.
    """
    cfg: ManagerBasedTrackerEnvCfg 
    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        assert isinstance(self.motion_manager, SkillManager), "Motion manager must be a SkillManager"
        self.motion_manager.motion_dt = self.cfg.sim.dt * self.cfg.actions.skill.low_level_action_decimation
        
    def _post_dynamic_step(self):
        # Not compute motion manager, which it is done in dynamic step.
        # Skip ManagerBasedTrackerEnv post dynamic step
        ManagerBasedRLEnv._post_dynamic_step(self)

