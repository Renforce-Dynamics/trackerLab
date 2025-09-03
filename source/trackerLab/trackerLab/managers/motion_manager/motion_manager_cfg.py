from isaaclab.utils import configclass
from trackerLab.motion_buffer import MotionBufferCfg
from dataclasses import MISSING
from typing import Union

@configclass
class MotionManagerCfg:
    motion_buffer_cfg: MotionBufferCfg = MotionBufferCfg()
    
    static_motion:      bool                = False
    loc_gen:            bool                = True
    speed_scale:        float               = 1.0
    robot_type:         str                 = MISSING
    motion_align_cfg:   Union[dict, str]    = None
    
    def set_motion_align_cfg(self, cfg):
        self.motion_align_cfg = cfg