from isaaclab.utils import configclass
from trackerLab.motion_buffer import MotionBufferCfg

@configclass
class MotionManagerCfg:
    motion_buffer_cfg: MotionBufferCfg = MotionBufferCfg()
    static_motion: bool = False
    # obs_from_buffer: bool = False
    loc_gen: bool = True
    speed_scale: float = 1.0
    robot_type: str = None
    reset_to_pose: bool = False
