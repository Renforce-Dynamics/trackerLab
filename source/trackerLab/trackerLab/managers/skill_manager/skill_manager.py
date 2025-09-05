import torch
import os
from typing import Callable, List

from trackerLab.managers.motion_manager import MotionManager

from .skill_graph import SkillGraph, SkillGraphCfg, SkillData, SkillEdge
from .skill_buffer import SkillBuffer, SkillBufferCfg
from .skill_manager_cfg import SkillManagerCfg

class SkillManager(MotionManager):
    """
    Skill graph for state transform, and the skill buffer for data gen.
    """
    skill_dt: float
    patch_size: int
    patch_time: float
    skill_transition_policy: Callable
    def __init__(self, cfg: SkillManagerCfg, env, device):
        super().__init__(cfg, env, device)
        self.skill_graph_cfg: SkillGraphCfg = cfg.skill_graph_cfg
        self.skill_buffer_Cfg: SkillBufferCfg = cfg.skill_buffer_cfg
        self.skill_transition_policy = cfg.skill_transition_policy
        self.init_skill()
        self.skill_dt   = self.skill_buffer.skill_dt
        self.patch_size = self.skill_graph_cfg.patch_size
        self.patch_time = self.skill_buffer.skill_dt * self.patch_size
        
        self.env_ids_all = torch.arange(self.num_envs, device=self.device)
        
        self.env_update_target_skill = torch.full((self.num_envs,), -1, device=self.device)
        self.reset_state_mod()

    def init_skill(self):
        if isinstance(self.skill_graph_cfg.skill_edges, dict):
            from .skill_graph.skill_edge import BUILD_GRAPH_METHODS
            func = BUILD_GRAPH_METHODS[self.skill_graph_cfg.skill_edges["func"]]
            default_params = dict(
                threshold=0.4, comp="trans_base", norm="l2", metric="frame", window=3
            )
            default_params.update(self.skill_graph_cfg.skill_edges["kwargs"])
            self.skill_graph_cfg.skill_edges = func(
                self.skill_graph_cfg.skill_list, 
                self.motion_lib, **default_params
            )
        self.skill_graph = SkillGraph(self.skill_graph_cfg, self.device)
        self.skill_buffer = SkillBuffer(
            self.skill_buffer_Cfg, self.skill_graph, self.motion_lib, self.num_envs, self.device)
        
        self.patch_time_curr = torch.zeros(self.num_envs, device=self.device)
        
        if self.skill_transition_policy == "random":
            self.skill_transition_policy = lambda env_ids: self.skill_buffer.update_random(env_ids)
        elif self.skill_transition_policy == "default":
            raise NotImplementedError("Default are todo")
        elif isinstance(self.skill_transition_policy, str) and os.path.isfile(self.skill_transition_policy):
            self._skill_transition_policy = torch.load(self.skill_transition_policy, self.device)
            self.skill_transition_policy = lambda env_ids: self.update_skill_by_policy(env_ids)
        else:
            raise NotImplementedError(f"Type {self.skill_transition_policy} are unknown")

    def update_skill_by_policy(self, env_ids):
        high_level_obs = self._env.observation_manager.compute_group("skill")[env_ids]
        actions = self._skill_transition_policy.act_inference(high_level_obs)
        skill_ids = torch.argmax(actions, dim=-1).to(self.device)
        self.skill_buffer.update_skill(env_ids, skill_ids)

    def update_time(self):
        self.patch_time_curr += self.motion_dt
        update_envs = self.patch_time_curr >= self.patch_time
        if update_envs.any():
            # Update by envs, if given env_update_target_skill update by skill, if not, update by random
            self.patch_time_curr[update_envs] -= self.patch_time
            random_update_target = update_envs
            skill_update_envs = self.env_update_target_skill[self.env_ids_all] > 0
            if skill_update_envs.any():
                # Skill update
                skill_update_target = torch.logical_and(skill_update_envs, update_envs)
                random_update_target = torch.logical_and(~skill_update_envs, update_envs)
                env_ids = self.env_ids_all[skill_update_target]
                self.skill_buffer.update_skill(env_ids, self.env_update_target_skill[env_ids])
                self.env_update_target_skill[env_ids] = -1  # Reset skill target after update
            if random_update_target.any():
                # Random update
                env_ids = self.env_ids_all[random_update_target]
                self.skill_transition_policy(env_ids)
            
    def set_skill(self, env_ids, skill_id):
        self.env_update_target_skill[env_ids] = skill_id

    def reset(self, env_ids):
        self.patch_time_curr[env_ids] = 0
        self.skill_buffer.reset(env_ids)
        self.loc_gen_state(env_ids)
        state = self.calc_current_pose(env_ids)
        return state

    def compute(self):
        self.loc_gen_state()
        # self.loc_state_post_modify()
        self.update_time()
        if self.debug_view:
            self.draw_debug(self.debug_envs)

    def get_current_time(self, env_ids):
        f0l = (self.patch_time_curr[env_ids] / self.skill_dt).int()
        f1l = (f0l + 1).clip(max=self.patch_size - 1)
        blend = (self.patch_time_curr[env_ids] / self.skill_dt) - f0l.float()
        return f0l, f1l, blend

    def calc_loc_terms(self, frame):
        loc_trans_base  = self.skill_buffer.ltbs[self.env_ids_all, frame]
        loc_root_rot    = self.skill_buffer.grs[self.env_ids_all, frame, 0]
        loc_root_pos    = self.skill_buffer.gts[self.env_ids_all, frame, 0]
        loc_local_rot   = self.skill_buffer.lrs[self.env_ids_all, frame]
        loc_dof_vel     = self.skill_buffer.dvs[self.env_ids_all, frame]
        # loc_dof_pos     = self.skill_buffer.dof_pos[frame]
        loc_dof_pos     = None
        loc_root_vel    = self.skill_buffer.lvbs[self.env_ids_all, frame]
        loc_ang_vel     = self.skill_buffer.avbs[self.env_ids_all, frame]
        # self.loc_roll, self.loc_pitch, _ = euler_from_quaternion(self.loc_root_rot)
        loc_dof_pos, loc_dof_vel = \
            self._motion_buffer.reindex_dof_pos_vel(loc_dof_pos, loc_dof_vel)
        loc_dof_pos, loc_dof_vel = \
            loc_dof_pos[:, self.gym2lab_dof_ids], loc_dof_vel[:, self.gym2lab_dof_ids]
        return loc_trans_base, loc_root_rot, loc_root_pos, \
            loc_dof_pos, loc_dof_vel, loc_root_vel, loc_ang_vel, loc_local_rot

    def reset_state_mod(self):
        
        self.height_func = lambda x: torch.where(x <= 1.0, x, x + torch.log(x) * 2)
        
        self.env_update_target_height   = torch.ones((self.num_envs,), device=self.device) * 0.1
        self.env_update_target_vx       = torch.ones((self.num_envs,), device=self.device) * 0.1
        self.env_update_target_vy       = torch.ones((self.num_envs,), device=self.device) * 0.1
        self.env_update_target_ang_z    = torch.ones((self.num_envs,), device=self.device) * 0.1

    def loc_state_post_modify(self):
        """Post modify the """
        self.loc_root_pos[:, 2] = self.height_func(self.loc_height)
        
        # self.loc_root_vel[:, 0] += self.env_update_target_vx
        # self.loc_root_vel[:, 1] += self.env_update_target_vy
        self.loc_root_pos[:, 2]   += self.env_update_target_height
        # self.loc_ang_vel        += self.env_update_target_ang_z
        # self.reset_state_mod()
    
    @property
    def graph_state_curr(self):
        ret = torch.zeros((self.num_envs, self.skill_graph.num_skills), device=self.device)
        ret[:, self.skill_buffer.skill_pid_curr] = 1
        return ret
    
    @property
    def graph_state_prev(self):
        ret = torch.zeros((self.num_envs, self.skill_graph.num_skills), device=self.device)
        ret[:, self.skill_buffer.skill_pid_prev] = 1
        return ret
        