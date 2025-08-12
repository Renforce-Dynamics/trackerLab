import inspect
import math
import sys
import os
from dataclasses import MISSING
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.envs import ManagerBasedRLEnvCfg

from .configs import (
    MySceneCfg,
    CommandsCfg,
    RewardsCfg,
    ObservationsCfg,
    MotionCfg,
    ActionsCfg,
    TerminationsCfg,
    EventCfg,
    CurriculumCfg,
    RecordsCfg
)

@configclass
class ManagerBasedTrackerEnvCfg(ManagerBasedRLEnvCfg):
    scene = MySceneCfg(num_envs=4096, env_spacing=2.5)
    rewards = RewardsCfg()
    commands = CommandsCfg()
    observations = ObservationsCfg()
    motion = MotionCfg()
    actions = ActionsCfg()
    terminations = TerminationsCfg()
    curriculum = CurriculumCfg()
    events = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 4.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.height_scanner is not None:
            self.scene.height_scanner_base.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False
                
        self.observations.policy.set_history(0)

    def adjust_scanner(self, name: str):
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + name

    def set_no_scanner(self):
        self.scene.height_scanner = None
        self.scene.height_scanner_base = None
        self.observations.policy.height_scan = None
    
    def adjust_contact(self, names):
        self.terminations.base_contact.params["sensor_cfg"].body_names = names
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = names

    def adjust_external_events(self, names):
        self.events.add_base_mass.params["asset_cfg"].body_names = names
        self.events.base_external_force_torque.params["asset_cfg"].body_names = names

    def set_test_motion_mode(self):
        self.scene.robot.spawn.articulation_props.fix_root_link = True
        self.scene.robot.spawn.rigid_props.disable_gravity = True
        
    def set_plane(self):
        # Change terrain to flat.
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        
    def set_record(self, dir):
        self.recorders = RecordsCfg()
        self.recorders.dataset_export_dir_path = dir