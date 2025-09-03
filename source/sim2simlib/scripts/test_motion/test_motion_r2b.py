import numpy as np
from sim2simlib.model.config import Sim2Sim_Config, Observations_Config, Actions_Config, Motor_Config
from sim2simlib.model.sim2sim_motion import Sim2SimMotionModel
from sim2simlib.motion.motion_manager import MotionBufferCfg, MotionManagerCfg
from sim2simlib.model.actuator_motor import DCMotor, PIDMotor
from sim2simlib.utils.config import load_from_yaml
from trackerLab import TRACKERLAB_ASSETS_DIR, TRACKERLAB_TASKS_DIR

ckpt_dir = ""

env_cfg = load_from_yaml(f"{ckpt_dir}/params/env.yaml")

config = Sim2Sim_Config(
    
    motion_cfg=MotionManagerCfg(
            motion_buffer_cfg = MotionBufferCfg(
                motion = MotionBufferCfg.MotionCfg(
                    motion_name="amass/r2b/simple_walk.yaml",
                    regen_pkl=True,
                )
            ),
            robot_type="r2b",
            motion_align_cfg=env_cfg["motion"]["motion_align_cfg"]
        ),
    
    robot_name='r2b',
    simulation_dt=0.005,
    slowdown_factor=1.0,
    control_decimation=4,
    policy_path=f"{ckpt_dir}/exported/policy.pt",
    xml_path=f"{TRACKERLAB_ASSETS_DIR}/r2_v3/r2.xml",
    policy_joint_names=[
        "left_hip_pitch_joint",
        "right_hip_pitch_joint",
        "waist_yaw_joint",
        "left_hip_roll_joint",
        "right_hip_roll_joint",
        "waist_pitch_joint",
        "left_hip_yaw_joint",
        "right_hip_yaw_joint",
        "left_shoulder_pitch_joint",
        "right_shoulder_pitch_joint",
        "left_knee_joint",
        "right_knee_joint",
        "left_shoulder_roll_joint",
        "right_shoulder_roll_joint",
        "left_ankle_pitch_joint",
        "right_ankle_pitch_joint",
        "left_shoulder_yaw_joint",
        "right_shoulder_yaw_joint",
        "left_ankle_roll_joint",
        "right_ankle_roll_joint",
        "left_arm_pitch_joint",
        "right_arm_pitch_joint"
    ],
    observation_cfg=Observations_Config(
        base_observations_terms=['base_lin_vel', 
                                 'base_ang_vel', 
                                 'gravity_orientation', 
                                 'joint_pos', 
                                 'joint_vel',
                                 'last_action'],
        # using_base_obs_history=True,
        # base_obs_his_length=5,
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
        motor_type=PIDMotor,
        effort_limit={
            # left leg
            "left_hip_pitch_joint": 130.0,
            "left_hip_roll_joint": 130.0,
            "left_hip_yaw_joint": 90.0,
            "left_knee_joint": 150.0,
            "left_ankle_pitch_joint": 75.0,
            "left_ankle_roll_joint": 75.0,
            # right leg
            "right_hip_pitch_joint": 130.0,
            "right_hip_roll_joint": 130.0,
            "right_hip_yaw_joint": 90.0,
            "right_knee_joint": 150.0,
            "right_ankle_pitch_joint": 75.0,
            "right_ankle_roll_joint": 75.0,
            # arms
            "left_shoulder_pitch_joint": 36.0,
            "left_shoulder_roll_joint": 36.0,
            "left_arm_pitch_joint": 36.0,
            "right_shoulder_pitch_joint": 36.0,
            "right_shoulder_roll_joint": 36.0,
            "right_arm_pitch_joint": 36.0
        },
        stiffness={
            # left leg
            "left_hip_pitch_joint": 100.0,
            "left_hip_roll_joint": 100.0,
            "left_hip_yaw_joint": 100.0,
            "left_knee_joint": 150.0,
            "left_ankle_pitch_joint": 30.0,
            "left_ankle_roll_joint": 30.0,
            # right leg
            "right_hip_pitch_joint": 100.0,
            "right_hip_roll_joint": 100.0,
            "right_hip_yaw_joint": 100.0,
            "right_knee_joint": 150.0,
            "right_ankle_pitch_joint": 30.0,
            "right_ankle_roll_joint": 30.0,
            # arms
            "left_shoulder_pitch_joint": 300.0,
            "left_shoulder_roll_joint": 300.0,
            "left_arm_pitch_joint": 300.0,
            "right_shoulder_pitch_joint": 300.0,
            "right_shoulder_roll_joint": 300.0,
            "right_arm_pitch_joint": 300.0
        },
        damping={
            # left leg
            "left_hip_pitch_joint": 5.0,
            "left_hip_roll_joint": 5.0,
            "left_hip_yaw_joint": 5.0,
            "left_knee_joint": 7.0,
            "left_ankle_pitch_joint": 3.0,
            "left_ankle_roll_joint": 3.0,
            # right leg
            "right_hip_pitch_joint": 5.0,
            "right_hip_roll_joint": 5.0,
            "right_hip_yaw_joint": 5.0,
            "right_knee_joint": 7.0,
            "right_ankle_pitch_joint": 3.0,
            "right_ankle_roll_joint": 3.0,
            # arms
            "left_shoulder_pitch_joint": 3.0,
            "left_shoulder_roll_joint": 3.0,
            "left_arm_pitch_joint": 3.0,
            "right_shoulder_pitch_joint": 3.0,
            "right_shoulder_roll_joint": 3.0,
            "right_arm_pitch_joint": 3.0
        },
    ),

    default_pos=np.array([0.0, 0.0, 0.8], dtype=np.float32),
    default_angles={
        },
    
)

mujoco_model = Sim2SimMotionModel(config)

# mujoco_model.motion_fk_view()
mujoco_model.view_run()