from isaaclab.utils import configclass
from dataclasses import MISSING

@configclass
class SkillBufferCfg:
    skill_fps: int = 60
    init_pid: int = 0
    @configclass
    class SmootherCfg:
        type: str = "none"
        kwargs: dict = MISSING
    smoother: SmootherCfg = SmootherCfg()
    