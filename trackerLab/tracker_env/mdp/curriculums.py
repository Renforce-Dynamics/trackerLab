from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def event_push_robot_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    velocity_range: dict[str, tuple[float, float]],
    rise_threshold: float = 0.9,
    fall_threshold: float = 0.7,
    min_learning_episode: int = 4,
    delta_range: list[float] = [-0.1, 0.1],
    event_term_name: str = "push_robot",
    reward_term_name: str = "alive",
) -> torch.Tensor:
    
    event_term = env.event_manager.get_term_cfg(event_term_name)
    
    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % (min_learning_episode * env.max_episode_length) == 0:
        if reward > reward_term.weight * rise_threshold:
            delta_range = torch.tensor(delta_range, device=env.device)
            event_term.params["velocity_range"]['x'] = torch.clamp(
                torch.tensor(event_term.params["velocity_range"]['x'], device=env.device) + delta_range,
                velocity_range['x'][0],
                velocity_range['x'][1],
            ).tolist()
            event_term.params["velocity_range"]['y'] = torch.clamp(
                torch.tensor(event_term.params["velocity_range"]['y'], device=env.device) + delta_range,
                velocity_range['y'][0],
                velocity_range['y'][1],
            ).tolist()
            event_term.params["velocity_range"]['z'] = torch.clamp(
                torch.tensor(event_term.params["velocity_range"]['z'], device=env.device) + 0.5*delta_range,
                velocity_range['z'][0],
                velocity_range['z'][1],
            ).tolist()
                
        elif reward < reward_term.weight * fall_threshold:
            # lower the event level
            delta_range = torch.tensor(delta_range, device=env.device)
            event_term.params["velocity_range"]['x'] = torch.clamp(
                torch.tensor(event_term.params["velocity_range"]['x'], device=env.device) + delta_range,
                velocity_range['x'][0],
                velocity_range['x'][1],
            ).tolist()
            event_term.params["velocity_range"]['y'] = torch.clamp(
                torch.tensor(event_term.params["velocity_range"]['y'], device=env.device) + delta_range,
                velocity_range['y'][0],
                velocity_range['y'][1],
            ).tolist()
            event_term.params["velocity_range"]['z'] = torch.clamp(
                torch.tensor(event_term.params["velocity_range"]['z'], device=env.device) + 0.5*delta_range,
                velocity_range['z'][0],
                velocity_range['z'][1],
            ).tolist()
            # ensure the lower bound is not greater than the upper bound
            if event_term.params["velocity_range"]['x'][0] > 0:
                event_term.params["velocity_range"]['x'][0] = 0.0
            if event_term.params["velocity_range"]['y'][0] > 0:
                event_term.params["velocity_range"]['y'][0] = 0.0
            if event_term.params["velocity_range"]['z'][0] > 0:
                event_term.params["velocity_range"]['z'][0] = 0.0
            if event_term.params["velocity_range"]['x'][1] < 0:
                event_term.params["velocity_range"]['x'][1] = 0.0
            if event_term.params["velocity_range"]['y'][1] < 0:
                event_term.params["velocity_range"]['y'][1] = 0.0
            if event_term.params["velocity_range"]['z'][1] < 0:
                event_term.params["velocity_range"]['z'][1] = 0.0
                
    return torch.tensor(event_term.params["velocity_range"]["x"][1], device=env.device)
