import numpy as np
from sim2simlib.model.config import Sim2Sim_Config, Observations_Config, Actions_Config, Motor_Config
from sim2simlib.model.sim2sim_base import Sim2Sim_Base_Model
from sim2simlib.model.dc_motor import DC_Motor, PID_Motor

config = Sim2Sim_Config(
    robot_name='Go2',
    simulation_dt=0.005,
    control_decimation=4,
    xml_path="/home/ac/Desktop/2025/project_isaac/IsaacLab/source/mujoco_menagerie/unitree_go2/scene.xml",
    policy_path="/home/ac/Desktop/2025/project_isaac/IsaacLab/source/labBundle/factoryIsaac/logs/exp1/Go2_doublehand_flat_action_nonoise/2025-08-06_13-55-54_a0.15_seed423/exported/policy.pt",
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
                             'action'],
        scale={
                'base_ang_vel': 0.25,
                'cmd': 1.0,
                'gravity_orientation': 1.0,
                'joint_pos': 1.0,
                'joint_vel': 0.05,
                'action': 1.0
            },
        ),
    action_cfg=Actions_Config(
        action_clip=(-100.0, 100.0),
        scale=0.25
    ),
    motor_cfg=Motor_Config(
        motor_type=DC_Motor,
        effort_limit=23.5,
        saturation_effort=23.5,
        velocity_limit=30.0,
        stiffness=25.0,
        damping=0.5
    ),

    default_pos=np.array([0.0, 0.0, 0.27], dtype=np.float32),
    default_angles=np.array([ 0.0, 0.9, -1.8, 0.0, 0.9, -1.8, 0.0, 0.9, -1.8, 0.0, 0.9, -1.8 ], dtype=np.float32),
)

mujoco_model = Sim2Sim_Base_Model(config)

mujoco_model.view_run()