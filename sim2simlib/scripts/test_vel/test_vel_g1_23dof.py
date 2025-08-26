import numpy as np
from sim2simlib.model.config import Sim2Sim_Config, Observations_Config, Actions_Config, Motor_Config
from sim2simlib.model.sim2sim_base import Sim2Sim_Base_Model
from sim2simlib.model.actuator_motor import DC_Motor, PID_Motor
from sim2simlib import MUJOCO_ASSETS, LOGS_DIR

config = Sim2Sim_Config(
    robot_name='g1_23dof',
    simulation_dt=0.001,
    slowdown_factor=1.0,
    control_decimation=20,
    xml_path=MUJOCO_ASSETS["unitree_g1_23dof"],
    policy_path=f"{LOGS_DIR}/rsl_rl/unitree_g1_23dof_velocity/2025-08-25_21-46-18/exported/policy.pt",
    policy_joint_names=['left_hip_pitch_joint', 'right_hip_pitch_joint', 'waist_yaw_joint', 'left_hip_roll_joint', 'right_hip_roll_joint', 'left_shoulder_pitch_joint', 'right_shoulder_pitch_joint', 'left_hip_yaw_joint', 'right_hip_yaw_joint', 'left_shoulder_roll_joint', 'right_shoulder_roll_joint', 'left_knee_joint', 'right_knee_joint', 'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint', 'left_ankle_pitch_joint', 'right_ankle_pitch_joint', 'left_elbow_joint', 'right_elbow_joint', 'left_ankle_roll_joint', 'right_ankle_roll_joint', 'left_wrist_roll_joint', 'right_wrist_roll_joint'],
    observation_cfg=Observations_Config(
        base_observations_terms=['base_ang_vel', 
                             'gravity_orientation', 
                             'cmd', 
                             'joint_pos', 
                             'joint_vel',
                             'last_action'],
        using_base_obs_history=False,
        base_obs_his_length=1,
        scale={
                'base_ang_vel': 0.2,
                'cmd': 1.0,
                'gravity_orientation': 1.0,
                'joint_pos': 1.0,
                'joint_vel': 0.05,
                'last_action': 1.0
            },
        ),
    cmd=[0,0,0],
    action_cfg=Actions_Config(
        action_clip=(-100.0, 100.0),
        scale=0.25
    ),
    motor_cfg=Motor_Config(
        motor_type=PID_Motor,
        effort_limit={
            # "legs"
            ".*_hip_pitch_.*": 88,
            ".*_hip_yaw_.*": 88,
            "waist_yaw_joint": 88,
            ".*_hip_roll_.*": 139,
            ".*_knee_.*": 139,
            # "arms"
            ".*_shoulder_.*": 25, 
            ".*_elbow_.*": 25, 
            ".*_wrist_roll_.*": 25,
            # "feet"
            ".*ankle.*": 35,
            ".*ankle.*": 35
        },
        stiffness={
            # "legs"
            ".*_hip_pitch_.*": 100.0,
            ".*_hip_yaw_.*": 100.0,
            "waist_yaw_joint": 200.0,
            ".*_hip_roll_.*": 100.0,
            ".*_knee_.*": 150.0,
            # "arms"
            ".*_shoulder_.*": 40.0, 
            ".*_elbow_.*": 40.0, 
            ".*_wrist_roll_.*": 40.0,
            # "feet"
            ".*ankle.*": 40.0,
            ".*ankle.*": 40.0,
        },
        damping={
            # "legs"
            ".*_hip_pitch_.*": 2.0,
            ".*_hip_yaw_.*": 2.0,
            "waist_yaw_joint": 5.0,
            ".*_hip_roll_.*": 2.0,
            ".*_knee_.*": 4.0,
            # "arms"
            ".*_shoulder_.*": 10.0, 
            ".*_elbow_.*": 10.0, 
            ".*_wrist_roll_.*": 10.0,
            # "feet"
            ".*ankle.*": 10.0,
            ".*ankle.*": 10.0,
        },
    ),

    default_pos=np.array([0.0, 0.0, 0.8], dtype=np.float32),
    default_angles={
            ".*_hip_pitch_joint": -0.1,
            ".*_knee_joint": 0.3,
            ".*_ankle_pitch_joint": -0.2,
            ".*_shoulder_pitch_joint": 0.3,
            "left_shoulder_roll_joint": 0.25,
            "right_shoulder_roll_joint": -0.25,
            ".*_elbow_joint": 0.97,
            "left_wrist_roll_joint": 0.15,
            "right_wrist_roll_joint": -0.15,
        },
)

mujoco_model = Sim2Sim_Base_Model(config)

mujoco_model.view_run()