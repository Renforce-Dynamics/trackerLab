from isaaclab.utils import configclass

@configclass
class MotionBufferCfg:
    @configclass
    class MotionCfg:
        motion_type: str = "yaml"
        motion_name: str = "motions_autogen_debug_punch.yaml"

    # n_demo_steps: int = 10
    # interval_demo_steps: int = 2
    regen_pkl: bool = False
    motion: MotionCfg = MotionCfg()
