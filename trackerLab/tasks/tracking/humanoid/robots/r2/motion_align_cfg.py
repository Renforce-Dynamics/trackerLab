R2B_MOTION_ALIGN_CFG = {
    "gym_joint_names": [
        "left_hip_roll_joint", "left_hip_pitch_joint", "left_hip_yaw_joint", 
        "left_knee_joint", 
        "left_ankle_roll_joint", "left_ankle_pitch_joint", 
        "right_hip_roll_joint", "right_hip_pitch_joint", "right_hip_yaw_joint", 
        "right_knee_joint", 
        "right_ankle_roll_joint", "right_ankle_pitch_joint",
        "skip", "waist_pitch_joint", "waist_yaw_joint",
        "left_shoulder_roll_joint", "left_shoulder_pitch_joint", "left_shoulder_yaw_joint", 
        "left_arm_pitch_joint", 
        "right_shoulder_roll_joint", "right_shoulder_pitch_joint", "right_shoulder_yaw_joint", 
        "right_arm_pitch_joint"
    ],
    "dof_body_ids": [1, 2, 3, 4, 5,  6,  7,  8,   9,  10,  11],
    "dof_offsets": [0, 3, 4, 6, 9, 10, 12, 15, 18,  19,  22, 23],
    "invalid_dof_id": [],
    "dof_indices_sim": [0],
    "dof_indices_motion": [0]
}

R2B_MOTION_ALIGN_CFG_SUB = {
    "gym_joint_names": [
        "left_hip_roll_joint", "left_hip_pitch_joint", "left_hip_yaw_joint", 
        "left_knee_joint", 
        "left_ankle_roll_joint", "left_ankle_pitch_joint", 
        "right_hip_roll_joint", "right_hip_pitch_joint", "right_hip_yaw_joint", 
        "right_knee_joint", 
        "right_ankle_roll_joint", "right_ankle_pitch_joint",
        "skip", "waist_pitch_joint", "waist_yaw_joint:skip",
        "left_shoulder_roll_joint", "left_shoulder_pitch_joint", "left_shoulder_yaw_joint:skip", 
        "left_arm_pitch_joint:skip", 
        "right_shoulder_roll_joint", "right_shoulder_pitch_joint", "right_shoulder_yaw_joint:skip", 
        "right_arm_pitch_joint:skip"
    ],
    "dof_body_ids": [1, 2, 3, 4, 5,  6,  7,  8,   9,  10,  11],
    "dof_offsets": [0, 3, 4, 6, 9, 10, 12, 15, 18,  19,  22, 23],
    "invalid_dof_id": [],
    "dof_indices_sim": [0],
    "dof_indices_motion": [0],
    "reverse_dof": []
}