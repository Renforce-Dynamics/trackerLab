import numpy as np
from sim2simlib.model.config import Sim2Sim_Config, Observations_Config, Actions_Config, Motor_Config
from sim2simlib.model.sim2sim_motion import Sim2Sim_Motion_Model
from sim2simlib.motion.sim2sim_manager import MotionBufferCfg
from sim2simlib.model.actuator_motor import DC_Motor, PID_Motor
from sim2simlib import SIM2SIMLIB_ASSETS_DIR, LOGS_DIR

config = Sim2Sim_Config(
    robot_name='g1_23d',
    simulation_dt=0.001,
    slowdown_factor=1.0,
    control_decimation=20,
    policy_path="",
    xml_path=f"{SIM2SIMLIB_ASSETS_DIR}/g1_description/g1_23dof.xml",
    policy_joint_names=[ 
        "left_hip_pitch_joint",
        "right_hip_pitch_joint",
        "waist_yaw_joint",
        "left_hip_roll_joint",
        "right_hip_roll_joint",
        "left_shoulder_pitch_joint",
        "right_shoulder_pitch_joint",
        "left_hip_yaw_joint",
        "right_hip_yaw_joint",
        "left_shoulder_roll_joint",
        "right_shoulder_roll_joint",
        "left_knee_joint",
        "right_knee_joint",
        "left_shoulder_yaw_joint",
        "right_shoulder_yaw_joint",
        "left_ankle_pitch_joint",
        "right_ankle_pitch_joint",
        "left_elbow_joint",
        "right_elbow_joint",
        "left_ankle_roll_joint",
        "right_ankle_roll_joint",
        "left_wrist_roll_joint",
        "right_wrist_roll_joint"
        ],
    observation_cfg=Observations_Config(
        base_observations_terms=['base_lin_vel', 
                                 'base_ang_vel', 
                                 'gravity_orientation', 
                                 'joint_pos', 
                                 'joint_vel',
                                 'last_action'],
        scale={ 
                'base_lin_vel': 1.0,
                'base_ang_vel': 0.25,
                'gravity_orientation': 1.0,
                'joint_pos': 1.0,
                'joint_vel': 0.05,
                'last_action': 1.0
            },
        # using_base_obs_history=True,
        # base_obs_his_length=5,
        motion_observations_terms=[
            'loc_dof_pos',
            'loc_root_vel'
            ]
        ),
    action_cfg=Actions_Config(
        action_clip=(-100.0, 100.0),
        scale=0.25
    ),            
    motor_cfg=Motor_Config(
        motor_type=PID_Motor,
        effort_limit={
            # "legs"
            ".*_hip_yaw_joint": 300,
            ".*_hip_roll_joint": 300,
            ".*_hip_pitch_joint": 300,
            ".*_knee_joint": 300,
            # "arms"
            ".*_shoulder_pitch_joint": 300,
            ".*_shoulder_roll_joint": 300,
            ".*_shoulder_yaw_joint": 300,
            ".*_elbow_joint": 300,
            # "feet"
            ".*_ankle_pitch_joint": 20,
            ".*_ankle_roll_joint": 20
        },
        stiffness={
            # "legs"
            ".*_hip_yaw_joint": 150.0,
            ".*_hip_roll_joint": 150.0,
            ".*_hip_pitch_joint": 200.0,
            ".*_knee_joint": 200.0,
            # "arms"
            ".*_shoulder_pitch_joint": 40.0,
            ".*_shoulder_roll_joint": 40.0,
            ".*_shoulder_yaw_joint": 40.0,
            ".*_elbow_joint": 40.0,
            # "feet"
            ".*_ankle_pitch_joint": 20.0,
            ".*_ankle_roll_joint": 20.0
        },
        damping={
            # "legs"
            ".*_hip_yaw_joint": 5.0,
            ".*_hip_roll_joint": 5.0,
            ".*_hip_pitch_joint": 5.0,
            ".*_knee_joint": 5.0,
            # "arms"
            ".*_shoulder_pitch_joint": 10.0,
            ".*_shoulder_roll_joint": 10.0,
            ".*_shoulder_yaw_joint": 10.0,
            ".*_elbow_joint": 10.0,
            # "feet"
            ".*_ankle_pitch_joint": 2.0,
            ".*_ankle_roll_joint": 2.0
        },
    ),

    default_pos=np.array([0.0, 0.0, 0.8], dtype=np.float32),
    default_angles={
        ".*_hip_pitch_joint": -0.20,
        ".*_knee_joint": 0.42,
        ".*_ankle_pitch_joint": -0.23,
        ".*_elbow_joint": 0.87,
        "left_shoulder_roll_joint": 0.16,
        "left_shoulder_pitch_joint": 0.35,
        "right_shoulder_roll_joint": -0.16,
        "right_shoulder_pitch_joint": 0.35,
    },
    
    motion_cfg=MotionBufferCfg(
        motion=MotionBufferCfg.MotionCfg(
            regen_pkl=True,
            motion_name="amass/g1_23d/simple_walk.yaml"
        )
    )
)

mujoco_model = Sim2Sim_Motion_Model(config)

# mujoco_model.motion_fk_view()
mujoco_model.view_run()