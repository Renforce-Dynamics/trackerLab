import torch

from isaaclab.utils import configclass
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm

from trackerLab.commands.base_command import BaseCommand
from isaaclab.managers import CommandTermCfg

from dataclasses import MISSING

from trackerLab.tracker_env.manager_based_tracker_env.manager_based_tracker_env_cfg import ManagerBasedTrackerEnvCfg

from isaaclab.managers import ObservationGroupCfg, ObservationManager

class TeacherActionCommand(BaseCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.verbose_detail: bool = getattr(cfg, "verbose_detail", False)
        self.lab_joint_names = self._env.scene.articulations["robot"]._data.joint_names
        self._init_teacher()

    def __str__(self) -> str:
        msg = "Teacher action command."
        return msg

    def _init_teacher(self):
        self.robot: Articulation = self._env.scene[self.cfg.asset_name]

        # load policy
        self.teacher_policy = torch.load(self.cfg.policy_path).to(self._env.device).eval()
        
        self.obs_cfg = self.cfg.teacher_env_cfg.observations
        self.teacher_obs_manager = ObservationManager({"teacher_policy": self.obs_cfg}, self._env)
        
        
    @property
    def command(self):
        self.teacher_observation = self.teacher_obs_manager.compute_group("teacher_policy")
        self.teacher_action = self.teacher_policy(self.teacher_observation)
        return self.teacher_action
        
    def _update_metrics(self):
        student_action = self._env.action_manager.action
        diff_action = student_action - self.teacher_action
        self.metrics["act diff"] = torch.sum(torch.abs(diff_action), dim=1)
        if self.verbose_detail:
            diff_action = torch.abs(diff_action)
            for idx, ptr in enumerate(self._env.motion_manager.shared_subset_lab):
                name = self.lab_joint_names[ptr]
                self.metrics[f"act {name}"] = diff_action[:, idx]
        

@configclass
class TeacherCommandCfg(CommandTermCfg):
    class_type: type = TeacherActionCommand
    teacher_env_cfg: ManagerBasedTrackerEnvCfg = MISSING
    policy_path: str = MISSING
    
    def __post_init__(self):
        self.resampling_time_range = None