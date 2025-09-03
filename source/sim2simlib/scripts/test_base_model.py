import os
import numpy as np
from sim2simlib.model.config import Sim2Sim_Config, Observations_Config, Actions_Config, Motor_Config
from sim2simlib.model.sim2sim_base import Sim2SimBaseModel
from sim2simlib.model.actuator_motor import DCMotor, PIDMotor
from sim2simlib import SIM2SIMLIB_ASSETS_DIR

config = Sim2Sim_Config(
    robot_name='Go2',
    simulation_dt=0.005,
    slowdown_factor=5.0,
    control_decimation=4,
    xml_path=os.path.join(SIM2SIMLIB_ASSETS_DIR, "unitree_go2", "mjcf", "scene_go2.xml"),
    policy_path=f"",
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
                'base_ang_vel': 0.25,
                'cmd': 1.0,
                'gravity_orientation': 1.0,
                'joint_pos': 1.0,
                'joint_vel': 0.05,
                'last_action': 1.0
            },
        ),
    action_cfg=Actions_Config(
        action_clip=(-100.0, 100.0),
        scale=0.25
    ),
    motor_cfg=Motor_Config(
        motor_type=DCMotor,
        effort_limit=23.5,
        saturation_effort=23.5,
        velocity_limit=30.0,
        stiffness=25.0,
        damping=0.5
    ),

    default_pos=np.array([0.0, 0.0, 0.27], dtype=np.float32),
    default_angles=np.array([ 0.0, 0.9, -1.8, 0.0, 0.9, -1.8, 0.0, 0.9, -1.8, 0.0, 0.9, -1.8 ], dtype=np.float32),
)

mujoco_model = Sim2SimBaseModel(config)

mujoco_model.view_run()