import torch
import numpy as np
from typing import Optional

from trackerLab.managers.motion_manager import MotionManager
from .utils import _interpolate, _slerp
from trackerLab.tracker_env.mdp.amp import compute_obs

class AMPManager(MotionManager):
    def __init__(self, cfg, env, device):
        super().__init__(cfg, env, device)

        self.duration = sum(self.motion_lib._motion_lengths).item()
        self.num_frames = sum(self.motion_lib._motion_num_frames).item()
        
        # Build per-frame sampling weights: weight_i / length_i
        # lengths = [len(d) for d in self.motion_lib._motion_num_frames]
        self.lengths = self.motion_lib._motion_num_frames
        per_frame = torch.cat(
            [
                torch.full((L,), w / L, device=self.device)
                for w, L in zip(self.motion_lib._motion_weights, self.lengths)
            ]
        )
        self.per_frame_weights = per_frame / per_frame.sum()

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
    
    def feed_forward_generator(self, num_mini_batch: int, mini_batch_size: int
    ):
        for _ in range(num_mini_batch):
            idx = torch.multinomial(
                self.per_frame_weights, mini_batch_size, replacement=True
            )
            next_idx = torch.clamp(idx + 1, max=self.num_frames - 1)
            (
                dof_positions,
                dof_velocities,
                body_positions,
                body_rotations,
                body_linear_velocities,
                body_angular_velocities,
            ) = self.sample(mini_batch_size, idx)
            
            this_obs = compute_obs(
                dof_positions,
                dof_velocities,
                body_positions,
                body_rotations,
                body_linear_velocities,
                body_angular_velocities,
            )
            
            (
                dof_positions,
                dof_velocities,
                body_positions,
                body_rotations,
                body_linear_velocities,
                body_angular_velocities,
            ) = self.sample(mini_batch_size, next_idx)
            
            next_obs = compute_obs(
                dof_positions,
                dof_velocities,
                body_positions,
                body_rotations,
                body_linear_velocities,
                body_angular_velocities,
            )

            yield this_obs, next_obs
