import numpy as np
from sim2simlib.model.config import Sim2Sim_Config, Observations_Config, Actions_Config, Motor_Config
from sim2simlib.model.sim2sim_base import Sim2SimBaseModel
from sim2simlib.model.actuator_motor import DCMotor, PIDMotor

from sim2simlib.utils.config import load_from_py, load_from_yaml
from sim2simlib import SIM2SIMLIB_ASSETS_DIR, SIM2SIMLIB_MUJOCO_ASSETS

ckpt_dir = ""
env_cfg = load_from_yaml(f"{ckpt_dir}/params/env.yaml")

config = Sim2Sim_Config(
    robot_name='booster_k1',
    simulation_dt=0.001,
    slowdown_factor=1.0,
    control_decimation=20,
    xml_path=SIM2SIMLIB_MUJOCO_ASSETS["booster_k1_rev"],
    policy_path=f"{ckpt_dir}/exported/policy.pt",
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
                             'cmd', 
                             'joint_pos', 
                             'joint_vel',
                             'last_action'],
        using_base_obs_history=False,
        base_obs_his_length=1,
        scale={
                'base_ang_vel': 0.2,
                'cmd': 1.0,
                'gravity_orientation': 1.0,
                'joint_pos': 1.0,
                'joint_vel': 0.05,
                'last_action': 1.0
            },
        ),
    cmd=[0.5,0,0],
    action_cfg=Actions_Config(
        action_clip=(-100.0, 100.0),
        scale=0.5,
        smooth_filter=True
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
    default_angles={
                ".*Shoulder_Pitch.*": 0.25,
                ".*Left_Shoulder_Roll.*": -1.4,
                ".*Right_Shoulder_Roll.*": 1.4,
                ".*Left_Elbow.*": -0.5,
                ".*Right_Elbow.*": 0.5,
                ".*Hip_Pitch.*": -0.1,
                ".*Knee.*": 0.2,
                ".*Ankle_Up.*": -0.1,
            },
)

mujoco_model = Sim2SimBaseModel(config)

mujoco_model.view_run()