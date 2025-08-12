from isaaclab.utils import configclass
from isaaclab.managers import RecorderManagerBaseCfg, RecorderTermCfg, RecorderTerm

from isaaclab.envs.mdp.recorders import PreStepActionsRecorder

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
    