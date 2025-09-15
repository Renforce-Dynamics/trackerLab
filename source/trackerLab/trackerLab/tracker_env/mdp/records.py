from isaaclab.utils import configclass
from isaaclab.managers import RecorderManagerBaseCfg, RecorderTermCfg, RecorderTerm

from isaaclab.envs.mdp.recorders import (
    PreStepActionsRecorder,

)

class RecordJointAcc(RecorderTerm):
    
    def record_post_step(self):
        return "joint_acc", self._env.scene.articulations["robot"].data.joint_acc
    
class RecordJointPos(RecorderTerm):
    
    def record_post_step(self):
        return "joint_pos", self._env.scene.articulations["robot"].data.joint_pos
    
class RecordJointVel(RecorderTerm):
    
    def record_post_step(self):
        return "joint_vel", self._env.scene.articulations["robot"].data.joint_vel
    
class RecordJointEffortTarget(RecorderTerm):
    
    def record_post_step(self):
        return "joint_effort_target", self._env.scene.articulations["robot"].data.joint_effort_target
    
class RecordComputedTorque(RecorderTerm):
    
    def record_post_step(self):
        return "computed_torque", self._env.scene.articulations["robot"].data.computed_torque
    
class RecordAppliedTorque(RecorderTerm):
    
    def record_post_step(self):
        return "applied_torque", self._env.scene.articulations["robot"].data.applied_torque
    
    
# For tracking use
class RecordRootLinVelBase(RecorderTerm):
    
    def record_post_step(self):
        return "root_lin_vel_b", self._env.scene.articulations["robot"].data.root_lin_vel_b
    
class RecordRootAngVelBase(RecorderTerm):
    
    def record_post_step(self):
        return "root_ang_vel_b", self._env.scene.articulations["robot"].data.root_ang_vel_b
    
class RecordRootQuatWorld(RecorderTerm):
    
    def record_post_step(self):
        return "root_quat_w", self._env.scene.articulations["robot"].data.root_quat_w

class RecordBodyComQuatBase(RecorderTerm):
    
    def record_post_step(self):
        return "body_com_quat_b", self._env.scene.articulations["robot"].data.body_com_quat_b
    

class RecordBodyLinkPoseWorld(RecorderTerm):
    
    def record_post_step(self):
        return "body_link_pose_w", self._env.scene.articulations["robot"].data.body_link_pose_w
    
@configclass
class TrajMotionRecordCfg(RecorderManagerBaseCfg):

    body_link_pose_w        = RecorderTermCfg(class_type=RecordBodyLinkPoseWorld)
    joint_pos               = RecorderTermCfg(class_type=RecordJointPos)
    joint_vel               = RecorderTermCfg(class_type=RecordJointVel)
    root_lin_vel_b          = RecorderTermCfg(class_type=RecordRootLinVelBase)
    root_ang_vel_b          = RecorderTermCfg(class_type=RecordRootAngVelBase)
    body_com_quat_b         = RecorderTermCfg(class_type=RecordBodyComQuatBase)
    action                  = RecorderTermCfg(class_type=PreStepActionsRecorder)
    
    def __post_init__(self):
        self.dataset_export_mode = 1
