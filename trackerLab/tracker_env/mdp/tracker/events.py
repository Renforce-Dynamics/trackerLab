import torch

def reset_to_traj(
    env,
    env_ids: torch.Tensor,
):
    state = env.motion_manager.calc_current_pose(env_ids)
    env.reset_to(state=state)
    return
    