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
        # prepare the managers
        # -- motion manager
        if isinstance(self.cfg.motion, SkillManagerCfg):
            self.motion_manager = SkillManager(self.cfg.motion, self, self.device)
        elif isinstance(self.cfg.motion, MotionManagerCfg):
            self.motion_manager = MotionManager(self.cfg.motion, self, self.device)
        # else:
        #     raise ValueError("Motion manager not supported: ", self.cfg.motion)
        self.motion_manager.compute()
        # print("[INFO] Motion Manager: ", self.motion_manager)
        super().load_managers()
        
    def _post_dynamic_step(self):
        self.motion_manager.compute()
        return super()._post_dynamic_step()

    def reset(self, seed = None, env_ids = None, options = None):
        super().reset(seed, env_ids, options)
        if seed is not None:
            self.seed(seed)
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)
        state:dict = self.motion_manager.reset(env_ids)
        if state:
            self.reset_to(state, env_ids, seed, is_relative=False)
        return self.obs_buf, self.extras
    
    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)

        # motion manager no log
        # self.motion_manager.reset()
        # self.extras["log"].update()

    @property
    def joint_subset(self):
        joint_pos = self.robot.data.joint_pos
        return self.motion_manager.get_subset_real(joint_pos)    
    