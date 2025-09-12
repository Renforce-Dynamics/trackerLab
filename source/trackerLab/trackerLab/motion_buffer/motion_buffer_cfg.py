from trackerLab.utils import configclass
from dataclasses import MISSING
from typing import Literal


@configclass
class MotionBufferCfg:

    motion_name: str = "none"
    motion_type: Literal["poselib", "replay", "GMR"] = "poselib"
    motion_lib_type: Literal["MotionLib", "MotionLibDofPos"] = "MotionLib"
    regen_pkl: bool = False
