"""
sim2simlib config system
follow the isaaclab manager system,
support:
1. Observations Terms order define,
2. Observations Terms history,
3. Motor Joint name Regular expression,
4. Motion Gen.

Author: Lu Zuxing
Data: 2025/8/22
"""

from trackerLab.managers.motion_manager import MotionManagerCfg
from dataclasses import dataclass
import numpy as np


@dataclass
class Actions():
    joint_pos: np.ndarray = None
    joint_vel: np.ndarray = None
    joint_efforts: np.ndarray = None

@dataclass
class Actions_Config:
    scale: float
    action_clip: tuple[float, float]

@dataclass
class Observations_Config:
    base_observations_terms: list[str]
    scale: dict[str, float]
    using_base_obs_history: bool = False
    base_obs_flatten: bool = True
    """
    using base_obs_flatten=True, terms flatten as N*dim(prev->current); 
    else base_obs_flatten=False, terms stack shape as (N, dim)
    """
    base_obs_his_length: int = 1
    
    motion_observations_terms: list[str] = None
    using_motion_obs_history: bool = False
    motion_obs_flatten: bool = True
    motion_obs_his_length: int = 1

@dataclass
class Motor_Config(): 
    motor_type: type = None
    joint_names: list[str] = None
    """joint_names are the mujoco order actuator names, must same as mujoco joint defined."""
    
    effort_limit: float | dict[str, float] = 0.0
    stiffness: float | dict[str, float] = 0.0
    """PID control: P Gain"""
    damping: float | dict[str, float] = 0.0
    """PID control: D Gain"""
    
    saturation_effort: float | dict[str, float] = 0.0
    velocity_limit: float | dict[str, float] = 0.0
    friction: float | dict[str, float] = 0.0
    

@dataclass
class Sim2Sim_Config:
    
    robot_name: str
    simulation_dt: float
    """physics simulation time step, pid control dt"""
    control_decimation: int
    """policy simulation time step"""
    slowdown_factor: float
    """slowdown factor for debug view run, biger is slower"""
    
    xml_path: str
    """robot xml with scene"""
    policy_path: str
    """export policy as *.pt format"""
    policy_joint_names: list[str]
    """isaaclab policy joint order"""
    
    default_pos: np.ndarray
    default_angles: np.ndarray | dict[str, float]
    
    observation_cfg: Observations_Config
    action_cfg: Actions_Config
    motor_cfg: Motor_Config
    
    debug: bool = False
    
    cmd: list[float] = None
    """base sim2sim command for velocity task"""
    
    motion_cfg: MotionManagerCfg = None
    motion_id: int = 0
    motion_update_rise: bool = False
    """motion manager configuration"""