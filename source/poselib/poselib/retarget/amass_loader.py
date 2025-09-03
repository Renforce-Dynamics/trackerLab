import os
import json
import numpy as np
import torch
from scipy.spatial.transform import Rotation as sRot

from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
from poselib.core.rotation3d import quat_mul, quat_inverse, quat_from_angle_axis

from poselib import POSELIB_TPOSE_DIR, POSELIB_SMPL_SKELETON_PATH

class AMASSLoader:
    def __init__(self, max_frames: int = 2751):
        self.max_frames = max_frames
        self.smpl_config_path = POSELIB_SMPL_SKELETON_PATH
        self.smpl_tpose:SkeletonState = SkeletonState.from_file(os.path.join(POSELIB_TPOSE_DIR, "smpl_tpose.npy"))
        self.smpl_skeleton_amass = self.create_smpl_skeleton()
        self.adjust_skeleton_seq()

    def adjust_skeleton_seq(self):
        amass_nodes:list = [node.lower() for node in self.smpl_skeleton_amass.node_names]
        desti_nodes:list = [node.lower() for node in self.smpl_skeleton.node_names]
        self.idcast_amass2desti = [amass_nodes.index(node) for node in desti_nodes]
        self.idcast_desti2amass = [desti_nodes.index(node) for node in amass_nodes]

    @property
    def smpl_skeleton(self) -> SkeletonTree:
        return self.smpl_tpose._skeleton_tree

    def create_smpl_skeleton(self):
        with open(self.smpl_config_path, 'r') as f:
            config = json.load(f)
        
        skeleton_config = config["skeleton"]
        node_names = skeleton_config["node_names"]
        parent_indices = torch.tensor(skeleton_config["parent_indices"], dtype=torch.long)
        local_translation = torch.tensor(skeleton_config["local_translation"], dtype=torch.float32)
        
        return SkeletonTree(node_names, parent_indices, local_translation)

    def load(self, npz_file: str):
        data = np.load(npz_file)
        available_keys = list(data.keys())
        if 'poses' not in data:
            raise ValueError("No poses data found in file")
        
        poses = data['poses'][:, :66]
        poses = np.concatenate([poses, np.zeros((poses.shape[0], 6))], axis=-1)
        
        trans = data.get('trans', np.zeros((poses.shape[0], 3)))
        fps = data.get('mocap_framerate', 30.0)

        if poses.shape[0] > self.max_frames:
            print(f"Limiting to first {self.max_frames} frames")
            poses = poses[:self.max_frames]
            trans = trans[:self.max_frames]
        
        return poses, trans, fps

    @staticmethod
    def calc_lin_vel(positions, fps):
        vel = (positions[1:] - positions[:-1]) * fps
        vel = torch.cat([vel, vel[-1:].clone()], dim=0)  # shape [T, J, 3]
        return vel

    @staticmethod
    def calc_ang_vel(local_rotations_tensor, fps):
        # Calc the joint angulor velocity
        # local_rotations_tensor: [T, J, 4]
        q = local_rotations_tensor
        dt = 1.0 / fps
        # Diff forward one step
        dq = q[1:] - q[:-1]
        dq = torch.cat([dq, dq[-1:].clone()], dim=0)
        q_inv = quat_inverse(q)  # implement inverse: conjugate / norm
        omega = 2 * quat_mul(dq / dt, q_inv)  # [T, J, 4]
        # take vector part (xyz)
        vel_joint = omega[..., 1:]  # [T, J, 3]
        return vel_joint

    def process_to_motion(self, poses: np.ndarray, trans: np.ndarray, fps: float = 30.0) -> SkeletonMotion:
        num_frames = poses.shape[0]
        num_joints = len(self.smpl_skeleton.node_names) 
        
        # Joint pos (quater)
        body_poses = poses.reshape(num_frames, -1, 3)[:, self.idcast_amass2desti, :]
        body_poses = sRot.from_rotvec(body_poses.reshape(-1, 3)).as_quat().reshape(num_frames, num_joints, 4)
        local_rotations_tensor = torch.tensor(body_poses, dtype=torch.float32)

        # Root translation
        adjusted_trans = trans.copy()
        min_z = np.min(trans[:, 2])
        adjusted_trans[:, 2] += (-min_z + 0.85)
        root_trans_tensor = torch.tensor(adjusted_trans, dtype=torch.float32)
        
        # Feed in the comps into the data tensor
        tensor_size = num_joints * 4  + 3
        tensor_data = torch.zeros(num_frames, tensor_size)

        start_idx = 0
        tensor_data[:, start_idx:start_idx + num_joints*4] = local_rotations_tensor.reshape(num_frames, -1)
        start_idx += num_joints * 4
        tensor_data[:, start_idx:start_idx + 3] = root_trans_tensor
        start_idx += 3

        motion = SkeletonMotion(
            tensor_backend=tensor_data,
            skeleton_tree=self.smpl_skeleton,
            is_local=True,
            fps=fps
        )
        
        return motion
        
        
    @staticmethod
    def fill_motion_vel(source_motion: SkeletonMotion):
        num_joints = source_motion.num_joints
        start_idx = num_joints * 4 + 3
        tensor_size = num_joints * 4 + 3 + num_joints * 3 + num_joints * 3
        
        tensor_data = source_motion.tensor[:, :start_idx].clone()
        num_frames = tensor_data.shape[0]
        fps = source_motion.fps
        # local_rotations_tensor = source_motion.local_rotation.reshape((num_frames, -1))
        
        lin_vel = AMASSLoader.calc_lin_vel(source_motion.global_translation, fps).reshape(num_frames, -1)
        ang_vel = AMASSLoader.calc_ang_vel(source_motion.local_rotation, fps).reshape(num_frames, -1)
        
        re_tensor_data = torch.zeros(num_frames, tensor_size)
        re_tensor_data[..., :start_idx] = tensor_data
        re_tensor_data[..., start_idx:start_idx + num_joints *3] = lin_vel
        start_idx += num_joints *3
        re_tensor_data[..., start_idx:start_idx + num_joints *3] = ang_vel
        
        motion = SkeletonMotion(
            tensor_backend=re_tensor_data,
            skeleton_tree=source_motion.skeleton_tree,
            is_local=True,
            fps=fps
        )
        
        return motion

    @staticmethod
    def get_edge_map(source_tree: SkeletonTree):
        parent_indices = source_tree.parent_indices
        edges = []
        for i in range(1, len(parent_indices)):
            edges.append((i, parent_indices[i]))

        # edges += [(j, i) for (i, j) in edges]
        return edges

    def load_and_process(self, npz_file: str) -> SkeletonMotion:

        poses, trans, fps = self.load(npz_file)
        if poses is None:
            raise ValueError("Unable to load AMASS data")
        
        source_motion = self.process_to_motion(poses, trans, fps)
        # source_motion = self.adjust_motion(source_motion)
        return source_motion
