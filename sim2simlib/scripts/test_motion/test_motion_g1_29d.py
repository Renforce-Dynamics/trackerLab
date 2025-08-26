import numpy as np
from sim2simlib.model.config import Sim2Sim_Config, Observations_Config, Actions_Config, Motor_Config
from sim2simlib.model.sim2sim_motion import Sim2Sim_Motion_Model
from sim2simlib.motion.motion_manager import MotionBufferCfg, MotionManagerCfg
from sim2simlib.model.actuator_motor import DC_Motor, PID_Motor
from sim2simlib.utils.config import load_from_py, load_from_yaml
from sim2simlib import LOGS_DIR

ckpt_dir = "/home/ac/Desktop/2025/project_isaac/trackerLab_private/logs/rsl_rl/tracking_unitree_g1_29d_walk/2025-08-26_11-48-11"

env_cfg = load_from_yaml(f"{ckpt_dir}/params/env.yaml")

config = Sim2Sim_Config(
    
    motion_cfg=MotionManagerCfg(
        motion_buffer_cfg = MotionBufferCfg(
            motion = MotionBufferCfg.MotionCfg(
                motion_name="amass/g1_29d_loco/simple_walk.yaml",
                regen_pkl=True,
            )
        ),
        robot_type="g1_29d_loco",
        motion_align_cfg=env_cfg["motion"]["motion_align_cfg"]
    ),
    robot_name='g1_29d_loco',
    simulation_dt=0.005,
    slowdown_factor=1.0,
    control_decimation=4,
    policy_path=f"{ckpt_dir}/exported/policy.pt",
    xml_path="/home/ac/Desktop/2025/project_3/mujoco_sim2sim_assets/unitree_g1/mjcf/scene_29dof.xml",
    policy_joint_names= ['left_hip_pitch_joint', 'right_hip_pitch_joint', 'waist_yaw_joint', 'left_hip_roll_joint', 'right_hip_roll_joint', 'waist_roll_joint', 'left_hip_yaw_joint', 'right_hip_yaw_joint', 'waist_pitch_joint', 'left_knee_joint', 'right_knee_joint', 'left_shoulder_pitch_joint', 'right_shoulder_pitch_joint', 'left_ankle_pitch_joint', 'right_ankle_pitch_joint', 'left_shoulder_roll_joint', 'right_shoulder_roll_joint', 'left_ankle_roll_joint', 'right_ankle_roll_joint', 'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint', 'left_elbow_joint', 'right_elbow_joint', 'left_wrist_roll_joint', 'right_wrist_roll_joint', 'left_wrist_pitch_joint', 'right_wrist_pitch_joint', 'left_wrist_yaw_joint', 'right_wrist_yaw_joint'],
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
        action_clip=(-6.0, 6.0), # CHECK
        scale=0.5 # CHECK
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
            ".*_wrist_roll.*": 25,
            ".*_ankle_.*": 25,
            "waist_roll_joint": 25,
            "waist_pitch_joint": 25,
            # "feet"
            ".*_wrist_pitch.*": 5,
            ".*_wrist_yaw.*": 5
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
            ".*_wrist_roll.*": 40.0,
            ".*_ankle_.*": 40.0,
            "waist_roll_joint": 40.0,
            "waist_pitch_joint": 40.0,
            # "feet"
            ".*_wrist_pitch.*": 40.0,
            ".*_wrist_yaw.*": 40.0,
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
            ".*_wrist_roll.*": 10.0,
            ".*_ankle_.*": 2.0,
            "waist_roll_joint": 5.0,
            "waist_pitch_joint": 5.0,
            # "feet"
            ".*_wrist_pitch.*": 10.0,
            ".*_wrist_yaw.*": 10.0,
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

mujoco_model = Sim2Sim_Motion_Model(config)

# mujoco_model.motion_fk_view()
mujoco_model.view_run()