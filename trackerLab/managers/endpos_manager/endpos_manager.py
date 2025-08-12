import torch
import os
from typing import Callable, List

from trackerLab.managers.motion_manager import MotionManager

class EndposManager(MotionManager):

    def __init__(self, cfg, env, device):
        super().__init__(cfg, env, device)
        assert self.static_motion, "No motion update for target."

    def compute(self):
        self.loc_gen_state()
        self.loc_state_post_modify()
        
    def loc_gen_state(self, env_ids=None):
        return super().loc_gen_state(env_ids)
    
    def loc_state_post_modify():
        return