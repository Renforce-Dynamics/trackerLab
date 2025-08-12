import torch
from typing import List
from trackerLab.motion_buffer import MotionBuffer
from trackerLab.motion_buffer.motion_lib import MotionLib
from trackerLab.managers.skill_manager.skill_graph import SkillGraph

from .skill_buffer_cfg import SkillBufferCfg
from trackerLab.utils.smoother import smoother_create, BaseSmoother


class SkillBuffer(object):
    """
    Skillbuffer for shifting observation
    """
    def __init__(self, cfg: SkillBufferCfg, skill_graph: SkillGraph, 
                 motion_lib: MotionLib, num_envs: int, device):
        self.cfg:SkillBufferCfg = cfg
        self.skill_fps = cfg.skill_fps
        self.skill_dt = 1.0 / self.skill_fps
        self.skill_graph: SkillGraph = skill_graph
        self.motion_lib:MotionLib = motion_lib
        self.num_envs = num_envs
        self.device = device
        self.smoother:BaseSmoother = smoother_create(cfg.smoother.type, **cfg.smoother.kwargs)
        
        self.env_ids_all = torch.arange(num_envs, device=device)
        
        self.init_pid = getattr(cfg, "init_pid", 0)
        
        self.init_buffer()
        

    class DataCache:
        trans_base: torch.Tensor = None
        vels_base: torch.Tensor = None
        ang_vels_base: torch.Tensor = None
        lrs: torch.Tensor = None
        gts: torch.Tensor = None
        grs: torch.Tensor = None
        dvs: torch.Tensor = None
        
        def __init__(self, num_envs: int, patch_size: int, comps: List[torch.Tensor]):
            self.trans_base = torch.empty((num_envs, patch_size, *comps[0].shape[1:]), device=comps[0].device)
            self.vels_base = torch.empty((num_envs, patch_size, *comps[1].shape[1:]), device=comps[1].device)
            self.ang_vels_base = torch.empty((num_envs, patch_size, *comps[2].shape[1:]), device=comps[2].device)
            self.lrs = torch.empty((num_envs, patch_size, *comps[3].shape[1:]), device=comps[3].device)
            self.gts = torch.empty((num_envs, patch_size, *comps[4].shape[1:]), device=comps[4].device)
            self.grs = torch.empty((num_envs, patch_size, *comps[5].shape[1:]), device=comps[5].device)
            self.dvs = torch.empty((num_envs, patch_size, *comps[6].shape[1:]), device=comps[6].device)
        
        def update(self, env_ids: torch.Tensor, patch_id: int, buffer):
            for comp in buffer.comp_names:
                if hasattr(self, comp):
                    target = getattr(buffer, "buf_" + comp)
                    getattr(self, comp)[env_ids] = target[patch_id].clone()
                else:
                    raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{comp}'")
        
        def gather(self, targets: List, comps: List[str], patch_size: int):
            for idx, target in enumerate(targets):
                for comp in comps:
                    getattr(self, comp)[:, idx*patch_size:(idx + 1)*patch_size].copy_(getattr(target, comp))

        def apply(self, targets: List, comps: List[str], patch_size: int):
            for idx, target in enumerate(targets):
                for comp in comps:
                    getattr(target, comp)[:] = getattr(self, comp)[:, idx*patch_size:(idx + 1)*patch_size].clone()

        def smooth(self, env_ids: torch.Tensor, smoother, comps: List[str]):
            for comp in comps:
                target: torch.Tensor = getattr(self, comp)
                temp = target[env_ids].clone()
                shape = target.shape
                if len(shape) == 4:
                    _, T, J, D = shape
                    temp = temp.transpose(1, 2).reshape((-1, T, D))
                    temp = smoother.smooth(temp).reshape((-1, J, T, D)).transpose(1, 2)
                elif len(shape) == 3:
                    _, T, D = shape
                    temp = temp.reshape((-1, T, D))
                    temp = smoother.smooth(temp).reshape((-1, T, D))
                else:
                    raise ValueError(f"Invalid shape for {comp}: {shape}")
                getattr(self, comp)[env_ids] = temp.clone()
        
    buf_trans_base: torch.Tensor = None
    buf_vels_base: torch.Tensor = None
    buf_ang_vels_base: torch.Tensor = None
    buf_lrs: torch.Tensor = None
    buf_gts: torch.Tensor = None
    buf_grs: torch.Tensor = None
    buf_dvs: torch.Tensor = None
        
    cache_prev: DataCache = None
    cache_curr: DataCache = None
    cache_next: DataCache = None
    cache_switch: DataCache = None
    
    skill_pid_curr: torch.Tensor = None
    skill_pid_prev: torch.Tensor = None
    skill_pid_next: torch.Tensor = None
    skill_pid_switch: torch.Tensor = None
        
    def init_buffer(self):
        # Initialize the buffer
        target_list = [
            self.motion_lib.trans_base, self.motion_lib.vels_base, self.motion_lib.ang_vels_base,
            self.motion_lib.lrs, self.motion_lib.gts, self.motion_lib.grs, self.motion_lib.dvs]
        
        comps = self.alloc_comp(target_list)
        self.buf_trans_base, self.buf_vels_base, self.buf_ang_vels_base, \
        self.buf_lrs, self.buf_gts, self.buf_grs, self.buf_dvs = comps
        
        # Initialize the skill pid and corresponding cache 
        self.cache_prev = self.DataCache(self.num_envs, self.patch_size, target_list)
        self.cache_curr = self.DataCache(self.num_envs, self.patch_size, target_list)
        self.cache_next = self.DataCache(self.num_envs, self.patch_size, target_list)
        
        self.cache_switch = self.DataCache(self.num_envs, self.patch_size * 3, target_list)
        
        self.skill_pid_curr = torch.full((self.num_envs,), self.init_pid, dtype=torch.long, device=self.device)
        self.skill_pid_prev = torch.full((self.num_envs,), self.init_pid, dtype=torch.long, device=self.device)
        self.skill_pid_next = torch.full((self.num_envs,), self.init_pid, dtype=torch.long, device=self.device)

        self.cache_curr.update(self.env_ids_all, self.skill_pid_curr[self.env_ids_all], self)
        self.cache_next.update(self.env_ids_all, self.skill_pid_next[self.env_ids_all], self)
        self.cache_prev.update(self.env_ids_all, self.skill_pid_prev[self.env_ids_all], self)

    def reset(self, env_ids: torch.Tensor):
        # Reset the buffer
        self.skill_pid_curr[env_ids] = self.init_pid
        self.skill_pid_prev[env_ids] = self.init_pid
        self.skill_pid_next[env_ids] = self.init_pid
        self.cache_curr.update(env_ids, self.skill_pid_curr[env_ids], self)
        self.cache_next.update(env_ids, self.skill_pid_next[env_ids], self)
        self.cache_prev.update(env_ids, self.skill_pid_prev[env_ids], self)
        
    def update_skill(self, env_ids: torch.Tensor, skill_id: torch.Tensor):
        # Update the buffer
        self.skill_pid_prev[env_ids] = self.skill_pid_curr[env_ids]
        self.skill_pid_curr[env_ids], _ = self.skill_graph.step_from_skill_id(self.skill_pid_curr[env_ids], skill_id)
        self.skill_pid_next[env_ids] = self.skill_graph.default_transition_from_patch(self.skill_pid_curr[env_ids])
        
        self.cache_curr.update(env_ids, self.skill_pid_curr[env_ids], self)
        self.cache_next.update(env_ids, self.skill_pid_next[env_ids], self)
        self.cache_prev.update(env_ids, self.skill_pid_prev[env_ids], self)
        self.smooth_cache(env_ids)
        
    def update_random(self, env_ids: torch.Tensor):
        # Update the buffer
        self.skill_pid_prev[env_ids] = self.skill_pid_curr[env_ids].clone()
        self.skill_pid_curr[env_ids], _ = self.skill_graph.random_transition_from_patch(self.skill_pid_curr[env_ids])
        self.skill_pid_next[env_ids] = self.skill_graph.default_transition_from_patch(self.skill_pid_curr[env_ids])
        
        self.cache_curr.update(env_ids, self.skill_pid_curr[env_ids], self)
        self.cache_next.update(env_ids, self.skill_pid_next[env_ids], self)
        self.cache_prev.update(env_ids, self.skill_pid_prev[env_ids], self)
        self.smooth_cache(env_ids)
        
    def smooth_cache(self, env_ids: torch.Tensor):
        if self.smoother is not None:
            self.cache_switch.gather([self.cache_prev, self.cache_curr, self.cache_next],
                            self.comp_names, self.patch_size)
            self.cache_switch.smooth(env_ids, self.smoother, self.comp_names)
            self.cache_switch.apply([self.cache_prev, self.cache_curr, self.cache_next],
                            self.comp_names, self.patch_size)
        
    # Util funcs
    def alloc_comp(self, comps: List[torch.Tensor]) -> List[torch.Tensor]:
        sg, mlib = self.skill_graph, self.motion_lib
        datas:torch.Tensor = \
            [torch.empty((sg.num_patch, self.patch_size, *comp.shape[1:]), device=self.device) for comp in comps]
        
        for start, skill in zip(sg.skill_start_patch_ids, sg.skill_list):
            comp_start = mlib.length_starts[skill.motion_id]
            for i in range(skill.num_patches):
                patch_start = skill.start_frame + comp_start + i * self.patch_size
                for data, comp in zip(datas, comps):
                    assert comp.shape[0] > patch_start + self.patch_size, "Buffer overflow"
                    data[start+i].copy_(comp[patch_start:patch_start + self.patch_size])
            
        return datas
        
    comp_names = ["trans_base", "vels_base", "ang_vels_base", "lrs", "gts", "grs", "dvs"]

    @property
    def patch_size(self): return self.skill_graph.cfg.patch_size
        
    @property
    def trans_base(self):       return self.cache_curr.trans_base
    @property
    def vels_base(self):        return self.cache_curr.vels_base
    @property
    def ang_vels_base(self):    return self.cache_curr.ang_vels_base
    @property
    def lrs(self):              return self.cache_curr.lrs
    @property
    def gts(self):              return self.cache_curr.gts
    @property
    def grs(self):              return self.cache_curr.grs
    @property
    def dvs(self):              return self.cache_curr.dvs
    