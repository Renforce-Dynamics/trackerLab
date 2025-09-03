G1_23D_MOTION_ALIGN_CFG = {
    "gym_joint_names": [
        "left_hip_roll_joint", "left_hip_pitch_joint", "left_hip_yaw_joint", 
        "left_knee_joint", 
        "left_ankle_roll_joint:skip",  "left_ankle_pitch_joint:skip", 
        "right_hip_roll_joint", "right_hip_pitch_joint", "right_hip_yaw_joint", 
        "right_knee_joint", 
        "right_ankle_roll_joint:skip", "right_ankle_pitch_joint:skip", 
        "waist_yaw_joint",
        "left_shoulder_roll_joint", "left_shoulder_pitch_joint", "left_shoulder_yaw_joint", 
        "left_elbow_joint", 
        "left_wrist_roll_joint",
        "right_shoulder_roll_joint", "right_shoulder_pitch_joint",  "right_shoulder_yaw_joint", 
        "right_elbow_joint",
        "right_wrist_roll_joint"
    ],
    "dof_body_ids": [1, 2, 3, 4, 5,  6,  7,  8,   9,  10,  11,  12,  13],
    "dof_offsets": [0, 3, 4, 6, 9, 10, 12, 13,  16, 17,  18,  21,  22, 23],
    "invalid_dof_id": [],
    "dof_indices_sim": [0],
    "dof_indices_motion": [0]
}

G1_29D_MOTION_ALIGN_CFG = {
    "gym_joint_names": [
        "left_hip_roll_joint", "left_hip_pitch_joint",  "left_hip_yaw_joint", 
        "left_knee_joint", 
        "right_hip_roll_joint", "right_hip_pitch_joint", "right_hip_yaw_joint", 
        "right_knee_joint", 
        "waist_roll_joint", "waist_yaw_joint", "waist_pitch_joint", 
        "left_shoulder_roll_joint", "left_shoulder_pitch_joint", "left_shoulder_yaw_joint", 
        "left_elbow_joint", 
        "right_shoulder_roll_joint", "right_shoulder_pitch_joint", "right_shoulder_yaw_joint", 
        "right_elbow_joint"
    ],
    "dof_body_ids": [1, 2, 4, 5, 7,  8,  9,  10,  11,  12,  13],
    "dof_offsets": [0, 3, 4, 7, 8, 11, 14, 15,  16,  19,  20, 21],
    "invalid_dof_id": [],
    "dof_indices_sim": [0],
    "dof_indices_motion": [0]
}



G1_23D_MOTION_ALIGN_CFG_REPLAY = {
    "gym_joint_names": ['left_hip_pitch_joint',
                        'right_hip_pitch_joint',
                        'waist_yaw_joint',
                        'left_hip_roll_joint',
                        'right_hip_roll_joint',
                        'left_shoulder_pitch_joint',
                        'right_shoulder_pitch_joint',
                        'left_hip_yaw_joint',
                        'right_hip_yaw_joint',
                        'left_shoulder_roll_joint',
                        'right_shoulder_roll_joint',
                        'left_knee_joint',
                        'right_knee_joint',
                        'left_shoulder_yaw_joint',
                        'right_shoulder_yaw_joint',
                        'left_ankle_pitch_joint',
                        'right_ankle_pitch_joint',
                        'left_elbow_joint',
                        'right_elbow_joint',
                        'left_ankle_roll_joint',
                        'right_ankle_roll_joint',
                        'left_wrist_roll_joint',
                        'right_wrist_roll_joint',],
    "dof_body_ids": [],
    "dof_offsets": [23],
    "invalid_dof_id": [],
    "dof_indices_sim": [0],
    "dof_indices_motion": [0],
}