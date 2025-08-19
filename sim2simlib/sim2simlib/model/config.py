
from trackerLab.managers.motion_manager import MotionManagerCfg
from dataclasses import dataclass
import numpy as np

@dataclass
class Actions_Config:
    scale: float
    action_clip: tuple[float, float]

@dataclass
class Actions():
    joint_pos: np.ndarray = None
    joint_vel: np.ndarray = None
    joint_efforts: np.ndarray = None
    
@dataclass
class Observations_Config:
    base_observations_terms: list[str]
    scale: dict[str, float]
    motion_observations_terms: list[str] = None
    
@dataclass
class DC_Motor_Config():
    effort_limit: float=23.5
    saturation_effort: float=23.5
    velocity_limit: float=30.0
    stiffness: float=25.0
    damping: float=0.5
    friction: float=0.0

@dataclass
class Sim2Sim_Config:
    robot_name: str
    simulation_dt: float
    control_decimation: int
    xml_path: str
    policy_path: str
    policy_joint_names: list[str]
    default_pos: np.ndarray
    default_angles: np.ndarray
    
    observation_cfg: Observations_Config
    action_cfg: Actions_Config
    dc_motor_cfg: DC_Motor_Config
    
    motion_cfg: MotionManagerCfg = None