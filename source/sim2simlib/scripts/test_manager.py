import json
import torch

from sim2simlib.motion.motion_manager import MotionManagerSim2sim, MotionBufferCfg
from sim2simlib.utils.utils import get_mujoco_joint_names

robot_type="pi_plus_27dof"
device = "cpu"

mujoco_joint_names = get_mujoco_joint_names(robot_type)

cfg = MotionBufferCfg(
    regen_pkl=False,
    motion=MotionBufferCfg.MotionCfg(
        motion_type="yaml",
        motion_name="amass/pi_plus_27dof/simple_walk.yaml"
    )
)
# manager.step()

def test():
    import mujoco
    import mujoco.viewer
    import numpy as np
    import time

    model = mujoco.MjModel.from_xml_path("your_model.xml")
    data = mujoco.MjData(model)

    # assume motion_data shape = (T, nq)
    motion_data = np.load("motion.npy")
    fps = 30

    with mujoco.viewer.launch_passive(model, data) as viewer:
        for qpos in motion_data:
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(1.0 / fps)


if __name__ == "__main__":

    manager = MotionManagerSim2sim(
        motion_buffer_cfg=cfg,
        lab_joint_names=mujoco_joint_names,
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
        print(manager.loc_root_vel)
        is_update = manager.step()
        if torch.any(is_update):
            print("Motion updated.")
            manager.set_finite_state_machine_motion_ids(
                motion_ids=torch.tensor([1], device=device, dtype=torch.long)
            )