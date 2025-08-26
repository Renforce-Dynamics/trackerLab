import numpy as np
from sim2simlib.model.config import Sim2Sim_Config, Observations_Config, Actions_Config, Motor_Config
from sim2simlib.model.sim2sim_motion import Sim2Sim_Motion_Model
from sim2simlib.motion.motion_manager import MotionBufferCfg
from sim2simlib.model.actuator_motor import DC_Motor, PID_Motor
from sim2simlib import MUJOCO_ASSETS_DIR, LOGS_DIR
from trackerLab import TRACKERLAB_ASSETS_DIR

config = Sim2Sim_Config(
    robot_name='pi_plus_25dof',
    simulation_dt=0.005,
    slowdown_factor=1.0,
    control_decimation=4,
    policy_path="",
    xml_path=f"{TRACKERLAB_ASSETS_DIR}/pi_plus_25dof/xml/pi_waist.xml",
    policy_joint_names=[
        "l_shoulder_pitch_joint",
        "r_shoulder_pitch_joint",
        "waist_joint",
        "l_shoulder_roll_joint",
        "r_shoulder_roll_joint",
        "l_hip_pitch_joint",
        "r_hip_pitch_joint",
        "l_upper_arm_joint",
        "r_upper_arm_joint",
        "l_hip_roll_joint",
        "r_hip_roll_joint",
        "l_elbow_joint",
        "r_elbow_joint",
        "l_thigh_joint",
        "r_thigh_joint",
        "l_calf_joint",
        "r_calf_joint",
        "l_ankle_pitch_joint",
        "r_ankle_pitch_joint",
        "l_ankle_roll_joint",
        "r_ankle_roll_joint"
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
        action_clip=(-100.0, 100.0), # CHECK
        scale=0.25 # CHECK
    ),            
    motor_cfg=Motor_Config(
        motor_type=PID_Motor,
            effort_limit={
            ".*": 100,
        },
        stiffness={
            ".*": 45.0
        },
        damping={
            ".*": 1.0
        },
    ),

    default_pos=np.array([0.0, 0.0, 0.6], dtype=np.float32),
    default_angles={
        },
    
    motion_cfg=MotionBufferCfg(
        motion=MotionBufferCfg.MotionCfg(
            motion_name="amass/pi_plus_25dof/simple_walk.yaml",
            regen_pkl=True,
        )
    )
)

mujoco_model = Sim2Sim_Motion_Model(config)

# mujoco_model.motion_fk_view()
mujoco_model.view_run()