PI_27D_MOTION_ALIGN_CFG = {
    "gym_joint_names": [
        "waist_joint",
        "l_hip_roll_joint", "l_hip_pitch_joint", "l_thigh_joint",
        "l_calf_joint",
        "l_ankle_roll_joint", "l_ankle_pitch_joint",
        "r_hip_roll_joint", "r_hip_pitch_joint", "r_thigh_joint",
        "r_calf_joint",
        "r_ankle_roll_joint", "r_ankle_pitch_joint",
        "l_shoulder_roll_joint", "l_shoulder_pitch_joint", "l_upper_arm_joint",
        "l_elbow_skip_1", "l_elbow_skip_2", "l_elbow_joint",
        "l_wrist_joint",
        "l_claw_joint",
        "r_shoulder_roll_joint", "r_shoulder_pitch_joint", "r_upper_arm_joint",
        "r_elbow_skip_1", "r_elbow_skip_2", "r_elbow_joint",
        "r_wrist_joint",
        "r_claw_joint",
        "head_pitch_joint"
    ],
    "dof_body_ids": [1, 2, 3, 4, 5,  6,  7,  8,  9,  10,  11,  12,  13,  14,  15,  16],
    "dof_offsets": [0, 1, 4, 5, 7, 10, 11, 13, 16, 19,  20,  21,  24,  27,  28,  29,  30],
    "invalid_dof_id": [],
    "dof_indices_sim": [0],
    "dof_indices_motion": [0]
}

PI_25D_MOTION_ALIGN_CFG = {
    "gym_joint_names": [
        "l_shoulder_roll_joint", "l_shoulder_pitch_joint", "l_upper_arm_joint",
        "l_elbow_skip_1", "l_elbow_skip_2", "l_elbow_joint",
        "r_shoulder_roll_joint", "r_shoulder_pitch_joint", "r_upper_arm_joint",
        "r_elbow_skip_1", "r_elbow_skip_2", "r_elbow_joint",
        "waist_joint",
        "l_hip_roll_joint", "l_hip_pitch_joint", "l_thigh_joint",
        "l_calf_joint",
        "l_ankle_roll_joint", "l_ankle_pitch_joint",
        "r_hip_roll_joint", "r_hip_pitch_joint", "r_thigh_joint",
        "r_calf_joint",
        "r_ankle_roll_joint", "r_ankle_pitch_joint",
    ],
    "dof_body_ids": [1, 2, 3, 4,  5,  6,  7,  8,  9,  10,  11],
    "dof_offsets": [0, 3, 6, 9, 12, 13, 16, 17, 19, 22,  23,   25],
    "invalid_dof_id": [],
    "dof_indices_sim": [0],
    "dof_indices_motion": [0],
    "reverse_dof": [11]
}

PI_25D_MOTION_ALIGN_CFG_WAIST_PITCH = {
    "gym_joint_names": [
        "l_shoulder_roll_joint", "l_shoulder_pitch_joint", "l_upper_arm_joint",
        "l_elbow_skip_1", "l_elbow_skip_2", "l_elbow_joint",
        "r_shoulder_roll_joint", "r_shoulder_pitch_joint", "r_upper_arm_joint",
        "r_elbow_skip_1", "r_elbow_skip_2", "r_elbow_joint",
        "waist_joint",
        "l_hip_roll_joint", "l_hip_pitch_joint", "l_thigh_joint",
        "l_calf_joint",
        "l_ankle_roll_joint", "l_ankle_pitch_joint",
        "r_hip_roll_joint", "r_hip_pitch_joint", "r_thigh_joint",
        "r_calf_joint",
        "r_ankle_roll_joint", "r_ankle_pitch_joint",
    ],
    "dof_body_ids": [1, 2, 3, 4,  5,  6,  7,  8,  9,  10,  11],
    "dof_offsets": [0, 3, 6, 9, 12, 13, 16, 17, 19, 22,  23,   25],
    "invalid_dof_id": [],
    "dof_indices_sim": [0],
    "dof_indices_motion": [0],
    "reverse_dof": [11]
}

PI_25D_MOTION_ALIGN_CFG_WAIST_YAW = {
    "gym_joint_names": [
        "l_shoulder_roll_joint", "l_shoulder_pitch_joint", "l_upper_arm_joint",
        "l_elbow_skip_1", "l_elbow_skip_2", "l_elbow_joint",
        "r_shoulder_roll_joint", "r_shoulder_pitch_joint", "r_upper_arm_joint",
        "r_elbow_skip_1", "r_elbow_skip_2", "r_elbow_joint",
        "waist_skip","waist_skip", "waist_joint",
        "l_hip_roll_joint", "l_hip_pitch_joint", "l_thigh_joint",
        "l_calf_joint",
        "l_ankle_roll_joint", "l_ankle_pitch_joint",
        "r_hip_roll_joint", "r_hip_pitch_joint", "r_thigh_joint",
        "r_calf_joint",
        "r_ankle_roll_joint", "r_ankle_pitch_joint",
    ],
    "dof_body_ids": [1, 2, 3, 4,  5,  6,  7,  8,  9,  10,  11],
    "dof_offsets": [0, 3, 6, 9, 12, 15, 18, 19, 21, 24,  26,   27],
    "invalid_dof_id": [],
    "dof_indices_sim": [0],
    "dof_indices_motion": [0],
    "reverse_dof": [11]
}

PI_25D_MOTION_ALIGN_CFG_KEY = {
    "gym_joint_names": [
        "l_shoulder_roll_joint", "l_shoulder_pitch_joint", "l_upper_arm_joint",
        "l_elbow_skip_1", "l_elbow_skip_2", "l_elbow_joint",
        "r_shoulder_roll_joint", "r_shoulder_pitch_joint", "r_upper_arm_joint",
        "r_elbow_skip_1", "r_elbow_skip_2", "r_elbow_joint",
        "waist_skip","waist_skip", "waist_joint_skip",
        "l_hip_roll_joint", "l_hip_pitch_joint", "l_thigh_joint",
        "l_calf_joint",
        "l_ankle_roll_joint_skip", "l_ankle_pitch_joint_skip",
        "r_hip_roll_joint", "r_hip_pitch_joint", "r_thigh_joint",
        "r_calf_joint",
        "r_ankle_roll_joint_skip", "r_ankle_pitch_joint_skip",
    ],
    "dof_body_ids": [1, 2, 3, 4,  5,  6,  7,  8,  9,  10,  11],
    "dof_offsets": [0, 3, 6, 9, 12, 15, 18, 19, 21, 24,  26,   27],
    "invalid_dof_id": [],
    "dof_indices_sim": [0],
    "dof_indices_motion": [0],
    "reverse_dof": [11]
}