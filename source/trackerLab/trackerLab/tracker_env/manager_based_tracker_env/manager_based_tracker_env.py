import torch
from trackerLab.managers import MotionManager, MotionManagerCfg, SkillManager, SkillManagerCfg
from trackerLab.tracker_env.manager_based_rl_env import ManagerBasedRLEnv
# from isaaclab.ui.widgets import ManagerLiveVisualizer
from isaaclab.assets import Articulation, RigidObject

from .manager_based_tracker_env_cfg import ManagerBasedTrackerEnvCfg

class ManagerBasedTrackerEnv(ManagerBasedRLEnv):
    cfg: ManagerBasedTrackerEnvCfg 
    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.robot:Articulation = self.scene["robot"]

    def load_managers(self):
        if hasattr(self, "motion_manager"):
            pass
        elif getattr(self.cfg, "motion", None) is None:
            print("No motion manager specified.")
            self.motion_manager = None
        elif isinstance(self.cfg.motion, SkillManagerCfg):
            self.motion_manager = SkillManager(self.cfg.motion, self, self.device)
            print("[INFO] Motion Manager: ", self.motion_manager)
        elif isinstance(self.cfg.motion, MotionManagerCfg):
            self.motion_manager = MotionManager(self.cfg.motion, self, self.device)
            print("[INFO] Motion Manager: ", self.motion_manager)
        else:
            raise ValueError("Motion manager not supported: ", self.cfg.motion)
        
        if getattr(self, "motion_manager", None) is not None:
            self.motion_manager.compute()
        
        super().load_managers()
    
    def step(self, action):
        super().step(action)
        if getattr(self, "motion_manager", None) is not None:
            self.motion_manager.compute()
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def reset(self, seed = None, env_ids = None, options = None):
        super().reset(seed, env_ids, options)
        if getattr(self, "motion_manager", None) is not None:
            if env_ids is None:
                env_ids = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)
            self.motion_manager.reset(env_ids)
            # if state:
            #     self.reset_to(state, env_ids, seed, is_relative=False)
        return self.obs_buf, self.extras

    @property
    def joint_subset(self):
        joint_pos = self.robot.data.joint_pos
        return self.motion_manager.get_subset_real(joint_pos)    
    