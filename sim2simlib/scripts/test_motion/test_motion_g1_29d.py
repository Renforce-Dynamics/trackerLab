import numpy as np
from sim2simlib.model.config import Sim2Sim_Config, Observations_Config, Actions_Config, Motor_Config
from sim2simlib.model.sim2sim_motion import Sim2Sim_Motion_Model
from sim2simlib.motion.motion_manager import MotionBufferCfg
from sim2simlib.model.actuator_motor import DC_Motor, PID_Motor
from sim2simlib import LOGS_DIR

config = Sim2Sim_Config(
    robot_name='g1_29d',
    simulation_dt=0.001,
    slowdown_factor=1.0,
    control_decimation=20,
    policy_path="",
    xml_path=f"{SIM2SIMLIB_ASSETS_DIR}/g1_description/g1_29dof_rev_1_0.xml",
    policy_joint_names=[
        "left_hip_pitch_joint",
        "right_hip_pitch_joint",
        "waist_yaw_joint",
        "left_hip_roll_joint",
        "right_hip_roll_joint",
        "waist_roll_joint",
        "left_hip_yaw_joint",
        "right_hip_yaw_joint",
        "waist_pitch_joint",
        "left_knee_joint",
        "right_knee_joint",
        "left_shoulder_pitch_joint",
        "right_shoulder_pitch_joint",
        "left_ankle_pitch_joint",
        "right_ankle_pitch_joint",
        "left_shoulder_roll_joint",
        "right_shoulder_roll_joint",
        "left_ankle_roll_joint",
        "right_ankle_roll_joint",
        "left_shoulder_yaw_joint",
        "right_shoulder_yaw_joint",
        "left_elbow_joint",
        "right_elbow_joint",
        "left_wrist_roll_joint",
        "right_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "right_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_wrist_yaw_joint"
    ],
    observation_cfg=Observations_Config(
        base_observations_terms=['base_lin_vel', 
                                 'base_ang_vel', 
                                 'gravity_orientation', 
                                 'joint_pos', 
                                 'joint_vel',
                                 'last_action'],
        using_base_obs_history=True,
        base_obs_his_length=5,
        scale={ 
                'base_lin_vel': 1.0,
                'base_ang_vel': 0.25, # CHECK
                'gravity_orientation': 1.0,
                'joint_pos': 1.0,
                'joint_vel': 0.05, # CHECK
                'last_action': 1.0
            },
        motion_observations_terms=[
            'loc_dof_pos',
            'loc_root_vel'
            ]
        ),
    action_cfg=Actions_Config(
        action_clip= (-3.14, 3.14), # CHECK
        scale=1.0 # CHECK
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

    default_pos=np.array([0.0, 0.0, 1.2], dtype=np.float32),
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
    
    motion_cfg=MotionBufferCfg(
        motion=MotionBufferCfg.MotionCfg(
            motion_name="amass/g1_29d_loco/simple_walk.yaml",
            regen_pkl=True,
        )
    )
)

mujoco_model = Sim2Sim_Motion_Model(config)

# mujoco_model.motion_fk_view()
mujoco_model.view_run()