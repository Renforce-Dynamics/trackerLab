from trackerLab.managers.motion_manager import MotionManagerCfg
from isaaclab.utils import configclass
from trackerLab.motion_buffer.motion_buffer_cfg import MotionBufferCfg

@configclass
class MotionCfg(MotionManagerCfg):
    motion_buffer_cfg = MotionBufferCfg(
        motion = MotionBufferCfg.MotionCfg(
            motion_name = None
        ),
        regen_pkl=True
    )
    static_motion: bool = False
    obs_from_buffer: bool = False
    loc_gen: bool = True
    speed_scale: float = 1.0
    robot_type: str = None
    reset_to_pose: bool = False