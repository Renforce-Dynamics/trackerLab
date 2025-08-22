import json
import torch

from sim2simlib.motion.sim2sim_manager import Motion_Manager, MotionBufferCfg
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

    # 1. 加载模型
    model = mujoco.MjModel.from_xml_path("your_model.xml")
    data = mujoco.MjData(model)

    # 假设 motion_data shape = (T, nq)
    motion_data = np.load("motion.npy")  # 例如保存的qpos序列
    fps = 30  # 播放帧率

    with mujoco.viewer.launch_passive(model, data) as viewer:
        for qpos in motion_data:
            data.qpos[:] = qpos  # 设置关节位置
            mujoco.mj_forward(model, data)  # 前向计算（更新几何）
            viewer.sync()
            time.sleep(1.0 / fps)  # 控制播放速度


if __name__ == "__main__":

    manager = Motion_Manager(
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