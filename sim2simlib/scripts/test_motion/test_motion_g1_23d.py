import numpy as np
from sim2simlib.model.config import Sim2Sim_Config, Observations_Config, Actions_Config, Motor_Config
from sim2simlib.model.sim2sim_motion import Sim2Sim_Motion_Model
from sim2simlib.motion.motion_manager import MotionBufferCfg, MotionManagerCfg
from sim2simlib.model.actuator_motor import DC_Motor, PID_Motor
from sim2simlib.utils.config import load_from_py, load_from_yaml
from sim2simlib import LOGS_DIR, MUJOCO_ASSETS

ckpt_dir = f"{LOGS_DIR}/rsl_rl/trackerlab_tracking_unitree_g1_23d_walk/2025-09-02_11-50-12"

env_cfg = load_from_yaml(f"{ckpt_dir}/params/env.yaml")

config = Sim2Sim_Config(
    
    motion_cfg=MotionManagerCfg(
        motion_buffer_cfg = MotionBufferCfg(
            motion = MotionBufferCfg.MotionCfg(
                motion_name="amass/g1_23d/cmu_walk_full.yaml",
                regen_pkl=True,
            ),
            motion_lib_type="MotionLib",
            motion_type="poselib"
        ),
        speed_scale=1.0,
        robot_type="g1_23d",
        motion_align_cfg=env_cfg["motion"]["motion_align_cfg"]
    ),
    motion_id=16,
    robot_name='g1_23d',
    simulation_dt=0.002,
    slowdown_factor=1.0,
    control_decimation=10,
    policy_path=f"{ckpt_dir}/exported/policy.pt",
    xml_path=MUJOCO_ASSETS["unitree_g1_23dof"],
    policy_joint_names=['left_hip_pitch_joint', 'right_hip_pitch_joint', 'waist_yaw_joint', 'left_hip_roll_joint', 'right_hip_roll_joint', 'left_shoulder_pitch_joint', 'right_shoulder_pitch_joint', 'left_hip_yaw_joint', 'right_hip_yaw_joint', 'left_shoulder_roll_joint', 'right_shoulder_roll_joint', 'left_knee_joint', 'right_knee_joint', 'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint', 'left_ankle_pitch_joint', 'right_ankle_pitch_joint', 'left_elbow_joint', 'right_elbow_joint', 'left_ankle_roll_joint', 'right_ankle_roll_joint', 'left_wrist_roll_joint', 'right_wrist_roll_joint'],
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
                'base_ang_vel': 0.25,
                'gravity_orientation': 1.0,
                'joint_pos': 1.0,
                'joint_vel': 0.05,
                'last_action': 1.0
            },
        motion_observations_terms=[
            'loc_dof_pos',
            'loc_root_vel'
            ],
        using_motion_obs_history=True,
        motion_obs_his_length=5,
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

mujoco_model = Sim2Sim_Motion_Model(config)

# mujoco_model.motion_fk_view()
mujoco_model.view_run()