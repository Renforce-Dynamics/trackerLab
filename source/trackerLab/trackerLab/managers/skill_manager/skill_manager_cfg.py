from dataclasses import MISSING
from isaaclab.utils import configclass
from trackerLab.managers.motion_manager import MotionManagerCfg
from .skill_graph import SkillGraphCfg
from .skill_buffer import SkillBufferCfg


@configclass
class SkillManagerCfg(MotionManagerCfg):
    skill_graph_cfg: SkillGraphCfg = MISSING
    skill_buffer_cfg: SkillBufferCfg = MISSING
    
    skill_transition_policy: str = "random"