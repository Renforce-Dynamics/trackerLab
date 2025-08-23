import torch
import numpy as np
from typing import Optional

from trackerLab.managers.motion_manager import MotionManager
from .utils import _interpolate, _slerp

class AMPManager(MotionManager):
    def __init__(self, cfg, env, device):
        super().__init__(cfg, env, device)

        self.duration = sum(self.motion_lib._motion_lengths).item()
        self.num_frames = sum(self.motion_lib._motion_num_frames).item()

    def sample_times(self, num_samples: int, duration: float | None = None) -> torch.Tensor:
        duration = self.duration if duration is None else duration
        assert (
            duration <= self.duration
        ), f"The specified duration ({duration}) is longer than the motion duration ({self.duration})"
        return duration * torch.rand(num_samples, device=self.device)

    def sample(
        self, num_samples: int, times: Optional[np.ndarray] = None, duration: float | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        times = self.sample_times(num_samples, duration) if times is None else times
        # num_frames is a single value now, expand this value to num_samples, as a tensor
        num_frames = torch.full(times.shape, self.num_frames, device=self.device)
        index_0, index_1, blend = self.motion_lib._calc_frame_blend(time = times, len=self.duration, num_frames=num_frames, dt=self.motion_dt)
        blend = torch.tensor(blend, dtype=torch.float32, device=self.device)

        dof_pos         = _interpolate(self.motion_lib.dps, blend=blend, start=index_0, end=index_1)
        dof_vel         = _interpolate(self.motion_lib.dvs, blend=blend, start=index_0, end=index_1)
        body_pos        = _interpolate(self.motion_lib.gts, blend=blend, start=index_0, end=index_1)
        body_rot        = _slerp(      self.motion_lib.grs, blend=blend, start=index_0, end=index_1)
        body_vel        = _interpolate(self.motion_lib.vels_base, blend=blend, start=index_0, end=index_1)
        body_ang_vel    = _interpolate(self.motion_lib.ang_vels_base, blend=blend, start=index_0, end=index_1)

        
        _body_pos = body_pos[:, 0, :]
        _body_rot = body_rot[:, 0, :]

        dof_pos, dof_vel = self._motion_buffer.reindex_dof_pos_vel(dof_pos, dof_vel)
        _dof_pos = dof_pos[:, self.gym2lab_dof_ids]
        _dof_vel = dof_vel[:, self.gym2lab_dof_ids]

        return (
            _dof_pos,
            _dof_vel,
            _body_pos,
            _body_rot,
            body_vel,
            body_ang_vel
        )
    
