import json
import torch
from deploylib.deploy_manager import DeployManager, MotionBufferCfg
from deploylib.utils.configs import get_lab_joint_names

robot_type="pi_plus_27dof"
device = "cuda:0"

lab_joint_names = get_lab_joint_names(robot_type)

cfg = MotionBufferCfg(
    regen_pkl=False,
    motion=MotionBufferCfg.MotionCfg(
        motion_type="yaml",
        motion_name="amass/pi_plus_27dof/simple_walk.yaml"
    )
)
# manager.step()

if __name__ == "__main__":

    manager = DeployManager(
        motion_buffer_cfg=cfg,
        lab_joint_names=lab_joint_names,
        robot_type=robot_type,
        dt=0.01,
        device=device
    )

    manager.init_finite_state_machine()
    manager.set_finite_state_machine_motion_ids(
        motion_ids=torch.tensor([0], device=device, dtype=torch.long)
    )

    while True:
        print(manager.loc_dof_pos)
        is_update = manager.step()
        if torch.any(is_update):
            print("Motion updated.")
            manager.set_finite_state_machine_motion_ids(
                motion_ids=torch.tensor([1], device=device, dtype=torch.long)
            )