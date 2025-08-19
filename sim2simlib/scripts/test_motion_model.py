import numpy as np
from sim2simlib.model.config import Sim2Sim_Config, Observations_Config, Actions_Config, DC_Motor_Config
from sim2simlib.model.sim2sim_motion import Sim2Sim_Motion_Model
from sim2simlib.motion.sim2sim_manager import MotionBufferCfg

config = Sim2Sim_Config(
    robot_name='pi_plus_27dof',
    simulation_dt=0.005,
    control_decimation=4,
    policy_path="/home/ac/Desktop/2025/project_isaac/trackerLab_private/checkpoints/pi_plus_policy.pth",
    xml_path=None,
    policy_joint_names=[ 
        "head_yaw_joint",
        "l_shoulder_pitch_joint",
        "r_shoulder_pitch_joint",
        "waist_joint",
        "head_pitch_joint",
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
        "l_wrist_joint",
        "r_wrist_joint",
        "l_calf_joint",
        "r_calf_joint",
        "l_claw_joint",
        "r_claw_joint",
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
                                 'action'],
        scale={ 
                'base_lin_vel': 1.0,
                'base_ang_vel': 1.0,
                'cmd': 1.0,
                'gravity_orientation': 1.0,
                'joint_pos': 1.0,
                'joint_vel': 1.0,
                'action': 1.0
            },
        ),
    action_cfg=Actions_Config(
        action_clip=(-100.0, 100.0),
        scale=0.5
    ),            
    dc_motor_cfg=DC_Motor_Config(
        effort_limit=500,
        saturation_effort=500,
        velocity_limit=30.0,
        stiffness=300,
        damping=2
    ),

    default_pos=np.array([0.0, 0.0, 0.4], dtype=np.float32),
    default_angles=np.array([0.0] * 27, dtype=np.float32),
    
    motion_cfg=MotionBufferCfg(
        regen_pkl=False,
        motion=MotionBufferCfg.MotionCfg(
            motion_type="yaml",
            motion_name="amass/pi_plus_27dof/simple_walk.yaml"
        )
    )
)

mujoco_model = Sim2Sim_Motion_Model(config)

mujoco_model.view_run()