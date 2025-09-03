import numpy as np
from sim2simlib.model.config import Sim2Sim_Config, Observations_Config, Actions_Config, Motor_Config
from sim2simlib.model.sim2sim_motion import Sim2SimMotionModel
from sim2simlib.motion.motion_manager import MotionBufferCfg
from sim2simlib.model.actuator_motor import DCMotor, PIDMotor

config = Sim2Sim_Config(
    robot_name='pi_plus_27dof',
    simulation_dt=0.005,
    control_decimation=4,
    policy_path=None,
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
                'base_ang_vel': 0.25,
                'gravity_orientation': 1.0,
                'joint_pos': 1.0,
                'joint_vel': 0.05,
                'action': 1.0
            },
        motion_observations_terms=[
            'loc_dof_pos',
            'loc_root_vel'
            ]
        ),
    action_cfg=Actions_Config(
        action_clip=(-100.0, 100.0),
        scale=0.000025
    ),            
    motor_cfg=Motor_Config(
        motor_type=PIDMotor,
        effort_limit=1,
        saturation_effort=10,
        velocity_limit=300.0,
        stiffness=45.0,
        damping=1.0
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

mujoco_model = Sim2SimMotionModel(config)

# mujoco_model.motion_fk_view()
mujoco_model.view_run()