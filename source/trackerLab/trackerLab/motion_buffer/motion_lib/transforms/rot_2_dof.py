import torch
from trackerLab.joint_id_caster import JointIdCaster
from trackerLab.utils import torch_utils
from trackerLab.utils.torch_utils.isaacgym import normalize_angle, quat_apply_inverse
from poselib.core.rotation3d import quat_mul_norm, quat_inverse, quat_angle_axis


# Using local rotation for calcing dof_pos, indicating that the joints are near.
# Exbody Mode
    

def _local_rotation_to_dof(id_caster:JointIdCaster, local_rot, device):
    body_ids = id_caster.dof_body_ids
    dof_offsets = id_caster.dof_offsets

    n = local_rot.shape[0]
    dof_pos = torch.zeros((n, id_caster.num_dof), dtype=torch.float, device=device)

    for j in range(len(body_ids)):
        body_id = body_ids[j]
        joint_offset = dof_offsets[j]
        joint_size = dof_offsets[j + 1] - joint_offset

        if (joint_size == 3):
            joint_q = local_rot[:, body_id]
            joint_exp_map = torch_utils.quat_to_exp_map(joint_q)
            dof_pos[:, joint_offset:(joint_offset + joint_size)] = joint_exp_map
        elif (joint_size == 2):
            joint_q = local_rot[:, body_id]
            joint_exp_map = torch_utils.quat_to_exp_map(joint_q)
            dof_pos[:, joint_offset:(joint_offset + joint_size)] = joint_exp_map[:, :2]
        elif (joint_size == 1):
            joint_q = local_rot[:, body_id]
            joint_theta, joint_axis = torch_utils.quat_to_angle_axis(joint_q)
            joint_theta = joint_theta * joint_axis[..., 1] # assume joint is always along y axis

            joint_theta = normalize_angle(joint_theta)
            dof_pos[:, joint_offset] = joint_theta
        else:
            print("Unsupported joint type")
            assert(False)

    return dof_pos

def _local_rotation_to_dof_vel(id_caster:JointIdCaster, local_rot0, local_rot1, dt, device):
    body_ids = id_caster.dof_body_ids
    dof_offsets = id_caster.dof_offsets

    dof_vel = torch.zeros([id_caster.num_dof], device=device)

    diff_quat_data = quat_mul_norm(quat_inverse(local_rot0), local_rot1)
    diff_angle, diff_axis = quat_angle_axis(diff_quat_data)
    local_vel = diff_axis * diff_angle.unsqueeze(-1) / dt
    local_vel = local_vel

    for j in range(len(body_ids)):
        body_id = body_ids[j]
        joint_offset = dof_offsets[j]
        joint_size = dof_offsets[j + 1] - joint_offset

        if (joint_size == 3):
            joint_vel = local_vel[body_id]
            dof_vel[joint_offset:(joint_offset + joint_size)] = joint_vel
        elif (joint_size == 2):
            joint_vel = local_vel[body_id]
            dof_vel[joint_offset:(joint_offset + joint_size)] = joint_vel[:2]
        elif (joint_size == 1):
            assert(joint_size == 1)
            joint_vel = local_vel[body_id]
            dof_vel[joint_offset] = joint_vel[1] # assume joint is always along y axis

        else:
            print("Unsupported joint type")
            assert(False)

    return dof_vel

def _compute_motion_dof_vels( motion):
    num_frames = motion.tensor.shape[0]
    dt = 1.0 / motion.fps
    dof_vels = []

    for f in range(num_frames - 1):
        local_rot0 = motion.local_rotation[f]
        local_rot1 = motion.local_rotation[f + 1]
        frame_dof_vel = _local_rotation_to_dof_vel(local_rot0, local_rot1, dt)
        frame_dof_vel = frame_dof_vel
        dof_vels.append(frame_dof_vel)
    
    dof_vels.append(dof_vels[-1])
    dof_vels = torch.stack(dof_vels, dim=0)

    return dof_vels