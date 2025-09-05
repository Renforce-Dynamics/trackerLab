import numpy as np
from sim2simlib.model.config import Sim2Sim_Config, Observations_Config, Actions_Config, Motor_Config
from sim2simlib.model.sim2sim_motion import Sim2Sim_Motion_Model
from sim2simlib.motion import MotionBufferCfg, MotionManagerCfg
from sim2simlib.model.actuator_motor import DC_Motor, PID_Motor
from sim2simlib.utils.config import load_from_py, load_from_yaml
from sim2simlib import MUJOCO_ASSETS

ckpt_dir = ""

env_cfg = load_from_yaml(f"{ckpt_dir}/params/env.yaml")

config = Sim2Sim_Config(
    
    motion_cfg=MotionManagerCfg(
        motion_buffer_cfg = MotionBufferCfg(
            motion = MotionBufferCfg.MotionCfg(
                motion_name="amass/booster_k1/simple_walk.yaml",
                regen_pkl=True
            ),
            motion_lib_type="MotionLibDofPos",
            motion_type="GMR",
        ),
        robot_type="booster_k1",
        motion_align_cfg=env_cfg["motion"]["motion_align_cfg"]

    ),
    robot_name='booster_k1',
    simulation_dt=0.001,
    slowdown_factor=1.0,
    control_decimation=20,
    policy_path=f"{ckpt_dir}/exported/policy.pt",
    xml_path=MUJOCO_ASSETS['booster_k1'],
    policy_joint_names=['AAHead_yaw',
                        'ALeft_Shoulder_Pitch',
                        'ARight_Shoulder_Pitch',
                        'Left_Hip_Pitch',
                        'Right_Hip_Pitch',
                        'Head_pitch',
                        'Left_Shoulder_Roll',
                        'Right_Shoulder_Roll',
                        'Left_Hip_Roll',
                        'Right_Hip_Roll',
                        'Left_Elbow_Pitch',
                        'Right_Elbow_Pitch',
                        'Left_Hip_Yaw',
                        'Right_Hip_Yaw',
                        'Left_Elbow_Yaw',
                        'Right_Elbow_Yaw',
                        'Left_Knee_Pitch',
                        'Right_Knee_Pitch',
                        'Left_Ankle_Pitch',
                        'Right_Ankle_Pitch',
                        'Left_Ankle_Roll',
                        'Right_Ankle_Roll'],
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
            # Head ("AAHead_yaw", "Head_pitch")
            "AAHead_yaw": 7.0,
            "Head_pitch": 7.0,
            # Arms (".*Shoulder_Pitch", ".*Shoulder_Roll", ".*Elbow_Pitch", ".*Elbow_Yaw")
            ".*Shoulder_Pitch": 10.0,
            ".*Shoulder_Roll": 10.0,
            ".*Elbow_Pitch": 10.0,
            ".*Elbow_Yaw": 10.0,
            # Legs (".*Hip_Pitch", ".*Hip_Roll", ".*Hip_Yaw", ".*Knee_Pitch", ".*Ankle_Pitch", ".*Ankle_Roll")
            ".*Hip_Pitch": 45.0,
            ".*Hip_Roll": 30.0,
            ".*Hip_Yaw": 30.0,
            ".*Knee_Pitch": 45.0,
            ".*Ankle_Pitch": 20.0,
            ".*Ankle_Roll": 20.0,
        },
        stiffness={
            # Head
            "AAHead_yaw": 20.0,
            "Head_pitch": 20.0,
            # Arms
            ".*Shoulder_Pitch": 50.0,
            ".*Shoulder_Roll": 50.0,
            ".*Elbow_Pitch": 50.0,
            ".*Elbow_Yaw": 50.0,
            # Legs
            ".*Hip_Pitch": 200.0,
            ".*Hip_Roll": 200.0,
            ".*Hip_Yaw": 200.0,
            ".*Knee_Pitch": 200.0,
            ".*Ankle_Pitch": 200.0,
            ".*Ankle_Roll": 200.0,
        },
        damping={
            # Head
            "AAHead_yaw": 1.0,
            "Head_pitch": 1.0,
            # Arms
            ".*Shoulder_Pitch": 2.0,
            ".*Shoulder_Roll": 2.0,
            ".*Elbow_Pitch": 2.0,
            ".*Elbow_Yaw": 2.0,
            # Legs
            ".*Hip_Pitch": 10.0,
            ".*Hip_Roll": 10.0,
            ".*Hip_Yaw": 10.0,
            ".*Knee_Pitch": 10.0,
            ".*Ankle_Pitch": 10.0,
            ".*Ankle_Roll": 10.0,
        },
    ),

    default_pos=np.array([0.0, 0.0, 0.53], dtype=np.float32),
    default_angles={".*": 0.0},
)

mujoco_model = Sim2Sim_Motion_Model(config)

# mujoco_model.motion_fk_view()
mujoco_model.view_run()