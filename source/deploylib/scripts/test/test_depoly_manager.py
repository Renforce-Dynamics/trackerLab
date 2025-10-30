import json
import torch
from deploylib.deploy_manager import DeployManager, MotionBufferCfg
# from deploylib.utils.configs import get_lab_joint_names

robot_type="k1"
device = "cpu"

lab_joint_names = [
    'Head_yaw_joint',
    'Left_Shoulder_Pitch_joint',
    'Right_Shoulder_Pitch_joint',
    'Left_Hip_Pitch_joint',
    'Right_Hip_Pitch_joint',
    'Head_pitch_joint',
    'Left_Shoulder_Roll_joint',
    'Right_Shoulder_Roll_joint',
    'Left_Hip_Roll_joint',
    'Right_Hip_Roll_joint',
    'Left_Elbow_Pitch_joint',
    'Right_Elbow_Pitch_joint',
    'Left_Hip_Yaw_joint',
    'Right_Hip_Yaw_joint',
    'Left_Elbow_Yaw_joint',
    'Right_Elbow_Yaw_joint',
    'Left_Knee_Pitch_joint',
    'Right_Knee_Pitch_joint',
    'Left_Ankle_Pitch_joint',
    'Right_Ankle_Pitch_joint',
    'Left_Ankle_Roll_joint',
    'Right_Ankle_Roll_joint',
]

lab_joint_names = [name[:-6] for name in lab_joint_names]

cfg = MotionBufferCfg(
    regen_pkl=False,
    motion_name="amass/booster_k1/deploy.yaml",
    motion_type="GMR",
    motion_lib_type="MotionLibDofPos"
)
# manager.step()

K1_MOTION_ALIGN_CFG = {
    "gym_joint_names": [
        "Head_yaw:skip", "Head_pitch:skip",
        "Left_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw", 
        "Right_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw",
        "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw",
        "Left_Knee_Pitch",
        "Left_Ankle_Pitch:skip", "Left_Ankle_Roll:skip",
        "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw",
        "Right_Knee_Pitch",
        "Right_Ankle_Pitch:skip", "Right_Ankle_Roll:skip"
    ],
    "dof_body_ids": [],
    "dof_offsets": [22],
    "invalid_dof_id": [],
    "dof_indices_sim": [0],
    "dof_indices_motion": [0]
}


if __name__ == "__main__":

    manager = DeployManager(
        motion_buffer_cfg=cfg,
        lab_joint_names=lab_joint_names,
        motion_align_cfg=K1_MOTION_ALIGN_CFG,
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
                motion_ids=torch.tensor([0], device=device, dtype=torch.long)
            )