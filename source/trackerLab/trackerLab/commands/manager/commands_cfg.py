import math
from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, FRAME_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.envs.mdp.commands.null_command import NullCommand

from .exbody_command import (
    DofposCommand,
    HeightCommand,
    RootVelCommand,
    RootAngVelCommand
)

from .metrics_command import ActionFluctuationRatioCommand

@configclass
class DofposCommandCfg(CommandTermCfg):
    class_type: type = DofposCommand
    verbose_detail: bool = False
    
    def __post_init__(self):
        self.resampling_time_range = None
        
@configclass
class HeightCommandCfg(CommandTermCfg):
    class_type: type = HeightCommand
    
    def __post_init__(self):
        self.resampling_time_range = None
        
@configclass
class RootVelCommandCfg(CommandTermCfg):
    class_type: type = RootVelCommand
    
    def __post_init__(self):
        self.resampling_time_range = None

@configclass
class RootAngVelCommandCfg(CommandTermCfg):
    class_type: type = RootAngVelCommand
    
    def __post_init__(self):
        self.resampling_time_range = None
        


@configclass
class ActionFluctuationRatioCommandCfg(CommandTermCfg):
    """Configuration for the episode success tracking command generator."""
    
    class_type: type = ActionFluctuationRatioCommand
    
    using_term: bool = False
    term_name: str = "joint_pos"
    
    def __post_init__(self):
        self.resampling_time_range = None