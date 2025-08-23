from trackerLab.utils import configclass
from dataclasses import MISSING


@configclass
class MotionBufferCfg:
    @configclass
    class MotionCfg:
        motion_name: str = MISSING
        regen_pkl: bool = False
        
    # n_demo_steps: int = 10
    # interval_demo_steps: int = 2
    
    motion: MotionCfg = MotionCfg()
