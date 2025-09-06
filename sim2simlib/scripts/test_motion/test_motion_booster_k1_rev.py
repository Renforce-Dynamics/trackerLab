import numpy as np
from sim2simlib.model.config import Sim2Sim_Config, Observations_Config, Actions_Config, Motor_Config
from sim2simlib.model.sim2sim_motion import Sim2SimMotionModel
from sim2simlib.motion.motion_manager import MotionBufferCfg, MotionManagerCfg
from sim2simlib.model.actuator_motor import DCMotor, PIDMotor
from sim2simlib.utils.config import load_from_py, load_from_yaml
from sim2simlib import MUJOCO_ASSETS

ckpt_dir = "/home/ac/Desktop/2025/project_isaac/trackerLab_private/logs/rsl_rl/tracking_booster_k1_walk/2025-09-03_19-09-50"

env_cfg = load_from_yaml(f"{ckpt_dir}/params/env.yaml")

config = Sim2Sim_Config(
    
    motion_cfg=MotionManagerCfg(
        motion_buffer_cfg = MotionBufferCfg(
            motion = MotionBufferCfg.MotionCfg(
                motion_name="amass/booster_k1/cmu_walk_full.yaml",
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
    policy_joint_names=['Head_Yaw_Joint',
                        'Left_Hip_Pitch_Joint',
                        'Left_Shoulder_Pitch_Joint',
                        'Right_Hip_Pitch_Joint',
                        'Right_Shoulder_Pitch_Joint',
                        'Head_Pitch_Joint',
                        'Left_Hip_Roll_Joint',
                        'Left_Shoulder_Roll_Joint',
                        'Right_Hip_Roll_Joint',
                        'Right_Shoulder_Roll_Joint',
                        'Left_Hip_Yaw_Joint',
                        'Left_Shoulder_Yaw_Joint',
                        'Right_Hip_Yaw_Joint',
                        'Right_Shoulder_Yaw_Joint',
                        'Left_Knee_Joint',
                        'Left_Elbow_Joint',
                        'Right_Knee_Joint',
                        'Right_Elbow_Joint',
                        'Left_Ankle_Up_Joint',
                        'Right_Ankle_Up_Joint',
                        'Left_Ankle_Down_Joint',
                        'Right_Ankle_Down_Joint'],
    observation_cfg=Observations_Config(
        base_observations_terms=['base_ang_vel', 
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
        motor_type=PIDMotor,
        effort_limit={
            # Head ("Head_Yaw", "Head_Pitch")
            ".*Head_Yaw.*":         7.0,
            ".*Head_Pitch.*":       7.0,
            # Arms ("Shoulder_Pitch", "Shoulder_Roll", "Shoulder_Yaw", "Elbow")
            ".*Shoulder_Pitch.*":   10.0,
            ".*Shoulder_Roll.*":    10.0,
            ".*Shoulder_Yaw.*":     10.0,
            ".*Elbow.*":            10.0,
            # Legs ("Hip_Pitch", "Hip_Roll", "Hip_Yaw", "Knee", "Ankle_Up", "Ankle_Down")
            ".*Hip_Pitch.*":        45.0,
            ".*Hip_Roll.*":         30.0,
            ".*Hip_Yaw.*":          30.0,
            ".*Knee.*":             45.0,
            ".*Ankle_Up.*":         20.0,
            ".*Ankle_Down.*":       20.0,
        },
        stiffness={
            # Head
            ".*Head_Yaw.*":         50.0,
            ".*Head_Pitch.*":       50.0,
            # Arms
            ".*Shoulder_Pitch.*":   50.0,
            ".*Shoulder_Roll.*":    50.0,
            ".*Shoulder_Yaw.*":     50.0,
            ".*Elbow.*":            50.0,
            # Legs
            ".*Hip_Pitch.*":        200.0,
            ".*Hip_Roll.*":         200.0,
            ".*Hip_Yaw.*":          200.0,
            ".*Knee.*":             200.0,
            ".*Ankle_Up.*":         200.0,
            ".*Ankle_Down.*":       200.0,
        },
        damping={
            # Head
            ".*Head_Yaw.*":         2.0,
            ".*Head_Pitch.*":       2.0,
            # Arms
            ".*Shoulder_Pitch.*":   2.0,
            ".*Shoulder_Roll.*":    2.0,
            ".*Shoulder_Yaw.*":     2.0,
            ".*Elbow.*":            2.0,
            # Legs
            ".*Hip_Pitch.*":        10.0,
            ".*Hip_Roll.*":         10.0,
            ".*Hip_Yaw.*":          10.0,
            ".*Knee.*":             10.0,
            ".*Ankle_Up.*":         10.0,
            ".*Ankle_Down.*":       10.0,
        },
    ),

    default_pos=np.array([0.0, 0.0, 0.53], dtype=np.float32),
    default_angles={".*": 0.0},
)

mujoco_model = Sim2SimMotionModel(config)

mujoco_model.motion_fk_view()
# mujoco_model.view_run()