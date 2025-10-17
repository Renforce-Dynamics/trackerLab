import numpy as np
from sim2simlib.model.config import Sim2Sim_Config, Observations_Config, Actions_Config, Motor_Config
from sim2simlib.model.sim2sim_motion import Sim2SimMotionModel
from sim2simlib.motion import MotionBufferCfg, MotionManagerCfg
from sim2simlib.model.actuator_motor import DCMotor, PIDMotor

from sim2simlib.utils.config import load_from_py, load_from_yaml
from robotlib import TRACKERLAB_ASSETS_DIR

ckpt_dir = ""

env_cfg = load_from_yaml(f"{ckpt_dir}/params/env.yaml")

config = Sim2Sim_Config(
    
    motion_cfg=MotionManagerCfg(
        motion_buffer_cfg = MotionBufferCfg(
            motion_name="amass/pi_plus_25dof/simple_run.yaml",
            regen_pkl=True,
        ),
        robot_type="pi_plus_25dof",
        motion_align_cfg=env_cfg["motion"]["motion_align_cfg"]
    ),
    
    robot_name='pi_plus_25dof',
    simulation_dt=0.005,
    slowdown_factor=1.0,
    control_decimation=4,
    policy_path=f"{ckpt_dir}/exported/policy.pt",
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
        # using_motion_obs_history=True,
        # motion_obs_his_length=5,
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
        motor_type=PIDMotor,
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
)

mujoco_model = Sim2SimMotionModel(config)

# mujoco_model.motion_fk_view()
mujoco_model.view_run()