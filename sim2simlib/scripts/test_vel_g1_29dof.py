import numpy as np
from sim2simlib.model.config import Sim2Sim_Config, Observations_Config, Actions_Config, Motor_Config
from sim2simlib.model.sim2sim_base import Sim2Sim_Base_Model
from sim2simlib.model.actuator_motor import DC_Motor, PID_Motor

config = Sim2Sim_Config(
    robot_name='g1_29dof',
    simulation_dt=0.001,
    slowdown_factor=1.0,
    control_decimation=20,
    xml_path="/home/ac/Desktop/2025/project_3/unitree_mujoco/unitree_robots/g1/scene_29dof.xml",
    policy_path="/home/ac/Desktop/2025/project_3/unitree_rl_lab/logs/rsl_rl/unitree_g1_29dof_velocity/2025-08-21_20-26-11/exported/policy.pt",
    policy_joint_names=['left_hip_pitch_joint', 
                        'right_hip_pitch_joint', 
                        'waist_yaw_joint', 
                        'left_hip_roll_joint', 
                        'right_hip_roll_joint', 
                        'waist_roll_joint', 
                        'left_hip_yaw_joint', 
                        'right_hip_yaw_joint', 
                        'waist_pitch_joint', 
                        'left_knee_joint', 
                        'right_knee_joint', 
                        'left_shoulder_pitch_joint', 
                        'right_shoulder_pitch_joint', 
                        'left_ankle_pitch_joint', 
                        'right_ankle_pitch_joint', 
                        'left_shoulder_roll_joint', 
                        'right_shoulder_roll_joint', 
                        'left_ankle_roll_joint', 
                        'right_ankle_roll_joint', 
                        'left_shoulder_yaw_joint', 
                        'right_shoulder_yaw_joint', 
                        'left_elbow_joint', 
                        'right_elbow_joint', 
                        'left_wrist_roll_joint', 
                        'right_wrist_roll_joint', 
                        'left_wrist_pitch_joint', 
                        'right_wrist_pitch_joint', 
                        'left_wrist_yaw_joint', 
                        'right_wrist_yaw_joint'],
    observation_cfg=Observations_Config(
        base_observations_terms=['base_ang_vel', 
                             'gravity_orientation', 
                             'cmd', 
                             'joint_pos', 
                             'joint_vel',
                             'last_action'],
        using_base_obs_history=True,
        base_obs_his_length=5,
        scale={
                'base_ang_vel': 0.2,
                'cmd': 1.0,
                'gravity_orientation': 1.0,
                'joint_pos': 1.0,
                'joint_vel': 0.05,
                'last_action': 1.0
            },
        ),
    cmd=[1,0,0],
    action_cfg=Actions_Config(
        action_clip=(-100.0, 100.0),
        scale=0.25
    ),
    motor_cfg=Motor_Config(
        motor_type=PID_Motor,
                effort_limit={
            # "legs"
            ".*_hip_roll_joint": 300,
            ".*_hip_yaw_joint": 300,
            ".*_hip_pitch_joint": 300,
            ".*_knee_joint": 300,
            "waist_.*_joint": 300,
            # "arms"
            ".*_shoulder_pitch_joint": 300,
            ".*_shoulder_roll_joint": 300,
            ".*_shoulder_yaw_joint": 300,
            ".*_elbow_joint": 300,
            ".*_wrist_roll_joint": 300,
            ".*_wrist_pitch_joint": 300,
            ".*_wrist_yaw_joint": 300,
            # "feet"
            ".*_ankle_pitch_joint": 20,
            ".*_ankle_roll_joint": 20
        },
        stiffness={
            # "legs"
            ".*_hip_yaw_joint": 100.0,
            ".*_hip_roll_joint": 100.0,
            ".*_hip_pitch_joint": 100.0,
            ".*_knee_joint": 150.0,
            "waist_.*_joint": 200.0,
            # "arms"
            ".*_shoulder_pitch_joint": 40.0,
            ".*_shoulder_roll_joint": 40.0,
            ".*_shoulder_yaw_joint": 40.0,
            ".*_elbow_joint": 40.0,
            ".*_wrist_roll_joint": 40.0,
            ".*_wrist_pitch_joint": 40.0,
            ".*_wrist_yaw_joint": 40.0,
            # "feet"
            ".*_ankle_pitch_joint": 40.0,
            ".*_ankle_roll_joint": 40.0
        },
        damping={
            # "legs"
            ".*_hip_yaw_joint": 2.0,
            ".*_hip_roll_joint": 2.0,
            ".*_hip_pitch_joint": 2.0,
            ".*_knee_joint": 4.0,
            "waist_.*_joint": 5.0,
            # "arms"
            ".*_shoulder_pitch_joint": 10.0,
            ".*_shoulder_roll_joint": 10.0,
            ".*_shoulder_yaw_joint": 10.0,
            ".*_elbow_joint": 10.0,
            ".*_wrist_roll_joint": 10.0,
            ".*_wrist_pitch_joint": 10.0,
            ".*_wrist_yaw_joint": 10.0,
            # "feet"
            ".*_ankle_pitch_joint": 2.0,
            ".*_ankle_roll_joint": 2.0,
        },
    ),

    default_pos=np.array([0.0, 0.0, 0.8], dtype=np.float32),
    default_angles={
            "left_hip_pitch_joint": -0.1,
            "right_hip_pitch_joint": -0.1,
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