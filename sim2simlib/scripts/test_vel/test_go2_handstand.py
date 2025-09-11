import numpy as np
from sim2simlib.model.config import Sim2Sim_Config, Observations_Config, Actions_Config, Motor_Config
from sim2simlib.model.sim2sim_base import Sim2SimBaseModel
from sim2simlib.model.actuator_motor import DCMotor, PIDMotor

from sim2simlib.utils.config import load_from_py, load_from_yaml
from sim2simlib import SIM2SIMLIB_ASSETS_DIR
ckpt_dir = ""
# /home/ac/Desktop/2025/project_isaac/IsaacLab/source/labBundle/factoryIsaac/logs/exp1/Go2_doublehand_flat_action_nonoise/2025-08-04_18-38-07_a0.01_seed114/exported/policy.pt
# /home/ac/Desktop/2025/project_isaac/trackerLab_private/logs/checkpoints/go2_handstand/policy.pt
config = Sim2Sim_Config(
    robot_name='Go2',
    simulation_dt=0.001,
    slowdown_factor=10.0,
    control_decimation=20,
    xml_path=f"{SIM2SIMLIB_ASSETS_DIR}/unitree_go2/mjcf/scene_go2.xml",
    policy_path=f"/home/ac/Desktop/2025/project_isaac/trackerLab_private/logs/checkpoints/go2_handstand/policy.pt",
    policy_joint_names=[ 
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",   
        ],
    observation_cfg=Observations_Config(
        base_observations_terms=['base_ang_vel', 
                             'gravity_orientation', 
                             'cmd', 
                             'joint_pos', 
                             'joint_vel',
                             'last_action'],
        scale={
                'base_ang_vel': 0.2,
                'cmd': 1.0,
                'gravity_orientation': 1.0,
                'joint_pos': 1.0,
                'joint_vel': 0.05,
                'last_action': 1.0
            },
        ),
    cmd=[0.1,0,0],
    action_cfg=Actions_Config(
        action_clip=(-100.0, 100.0),
        scale=0.25
    ),
    motor_cfg=Motor_Config(
        motor_type=PIDMotor,
        effort_limit=23.5,
        saturation_effort=23.5,
        velocity_limit=30.0,
        stiffness=25.0,
        damping=0.5,
    ),

    default_pos=np.array([0.0, 0.0, 0.4], dtype=np.float32),
    default_angles={
        ".*R_hip_joint": -0.1,
        ".*L_hip_joint": 0.1,
        "F[L,R]_thigh_joint": 0.8,
        "R[L,R]_thigh_joint": 1.0,
        ".*_calf_joint": -1.5,
    },
)

mujoco_model = Sim2SimBaseModel(config)

mujoco_model.view_run()