import torch
from dataclasses import MISSING
from isaaclab.utils import configclass
from typing import Union
from trackerLab.motion_buffer.motion_lib import MotionLib

@configclass
class SkillData:
    name: str = MISSING
    motion_id: Union[int, str] = MISSING
    start_frame: int = MISSING
    # end_frame: int = MISSING
    num_patches: int = MISSING
    desc: str = None
    
    height_mod: int = 0
    
    def get_data(self, mlib: MotionLib, comp: str, patch_size=10):
        """
        Get the skill data from the motion library.
        """
        comp: torch.Tensor = getattr(mlib, comp)
        comp_start = mlib.length_starts[self.motion_id]
        ret = []
        for i in range(self.num_patches):
            patch_start = self.start_frame + comp_start + i * patch_size
            assert comp.shape[0] > patch_start + patch_size, "Buffer overflow"
            ret.append(comp[patch_start:patch_start + patch_size].clone())
        return torch.cat(ret, dim=0)
    
    def get_terminal_state(self, mlib: MotionLib, comp: str, patch_size=10) -> torch.Tensor:
        comp_data = self.get_data(mlib, comp, patch_size)
        return comp_data[-1]

    def get_start_state(self, mlib: MotionLib, comp: str, patch_size=10) -> torch.Tensor:
        comp_data = self.get_data(mlib, comp, patch_size)
        return comp_data[0]
    
