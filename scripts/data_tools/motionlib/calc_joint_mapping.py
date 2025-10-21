import json
import numpy as np


def compute_joint_reorder(mujoco_joints, isaac_joints, save_path=None):
    """
    根据 MuJoCo 和 IsaacLab 的关节名称，计算 reorder 索引，
    使得 mj_q[reorder] == lab_q（即 Isaac 的顺序）。
    
    Args:
        mujoco_joints (list[str]): MuJoCo 关节名列表
        isaac_joints (list[str]): IsaacLab 关节名列表
        save_path (str, optional): 可选保存路径（JSON）

    Returns:
        reorder_idx (list[int]): 索引数组，使得 lab_q = mj_q[reorder_idx]
        mapping (dict): 映射表 mujoco_joint -> isaac_joint
    """
    reorder_idx = []
    mapping = {}

    for isaac_name in isaac_joints:
        # 精确匹配
        matched = [i for i, mj in enumerate(mujoco_joints) if mj.lower() == isaac_name.lower()]
        if not matched:
            # 模糊匹配
            matched = [i for i, mj in enumerate(mujoco_joints)
                       if isaac_name.lower() in mj.lower() or mj.lower() in isaac_name.lower()]

        if matched:
            j = matched[0]
            reorder_idx.append(j)
            mapping[mujoco_joints[j]] = isaac_name
        else:
            reorder_idx.append(-1)
            mapping[f"unmatched_{isaac_name}"] = None
            print(f"[WARN] No match for Isaac joint: {isaac_name}")

    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump({
                "reorder_index": reorder_idx,
                "mapping": mapping
            }, f, indent=4, ensure_ascii=False)
        print(f"✅ Mapping saved to {save_path}")

    print("\n=== Joint Mapping ===")
    for k, v in mapping.items():
        print(f"{k:35s} → {v}")
    print("\nreorder_idx =", reorder_idx)

    return reorder_idx, mapping


# ===== Example usage =====
if __name__ == "__main__":
    
    mujoco_joints = [
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",

        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",

        "waist_yaw_joint",
        "waist_pitch_joint",

        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_arm_pitch_joint",

        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_arm_pitch_joint"
    ]


    isaac_joints = [
        "left_hip_pitch_joint",
        "right_hip_pitch_joint",
        "waist_yaw_joint",
        "left_hip_roll_joint",
        "right_hip_roll_joint",
        "waist_pitch_joint",
        "left_hip_yaw_joint",
        "right_hip_yaw_joint",
        "left_shoulder_pitch_joint",
        "right_shoulder_pitch_joint",
        "left_knee_joint",
        "right_knee_joint",
        "left_shoulder_roll_joint",
        "right_shoulder_roll_joint",
        "left_ankle_pitch_joint",
        "right_ankle_pitch_joint",
        "left_shoulder_yaw_joint",
        "right_shoulder_yaw_joint",
        "left_ankle_roll_joint",
        "right_ankle_roll_joint",
        "left_arm_pitch_joint",
        "right_arm_pitch_joint"
    ]

    mapping, P = compute_joint_reorder(mujoco_joints, isaac_joints, "joint_mapping.json")

