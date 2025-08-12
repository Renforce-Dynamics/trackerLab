import matplotlib.pyplot as plt
import os
import torch 

def plot_3d_trajs(data, title="3D Trajectories"):
    """
    data: np.ndarray of shape [1, L, 14, 3]
    """
    assert data.shape[0] == 1, "Only supports batch size = 1 for visualization"
    
    trajs = data[0]  # shape: [L, 14, 3]
    L = trajs.shape[0]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for i in range(L):
        traj = trajs[i]  # shape: [14, 3]
        x, y, z = traj[:, 0], traj[:, 1], traj[:, 2]
        ax.plot(x, y, z, marker='o', label=f'Agent {i}' if i != 0 else 'Ego', linewidth=2 if i == 0 else 1)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()

def generate_sliding_trajs(data, traj_start_idx, window_size, stride=1, is_delta=False):
    sliding_trajs = []
    starts = []
    traj_start_idx = traj_start_idx.tolist()
    traj_start_idx.append(len(data))

    for i in range(len(traj_start_idx) - 1):
        start = traj_start_idx[i]
        end = traj_start_idx[i + 1]

        traj: torch.Tensor = data[start:end]

        for j in range(0, len(traj) - window_size + 1, stride):
            window: torch.Tensor = traj[j: j + window_size].clone()
            if is_delta:
                delta = window[1:] - window[:-1]
                sliding_trajs.append(delta)
            else:
                # sliding_trajs.append(window - window[0, 0])
                sliding_trajs.append(window - window[:, 0:1])
            starts.append(window[0] - window[0, 0])

    return torch.stack(sliding_trajs), torch.stack(starts)

def get_edge_index_cmu():
    """
    Return edge_index for CMU skeleton with 14 joints.
    Each edge is bidirectional (i.e., undirected graph represented as directed edges in both directions).
    """
    joint_names = [
        "pelvis",  # 0
        "left_hip_yaw_link",  # 1
        "left_knee_link",     # 2
        "left_ankle_link",    # 3
        "right_hip_yaw_link", # 4
        "right_knee_link",    # 5
        "right_ankle_link",   # 6
        "torso_link",         # 7
        "left_shoulder_pitch_link",  # 8
        "left_elbow_link",           # 9
        "left_hand_link",            #10
        "right_shoulder_pitch_link", #11
        "right_elbow_link",          #12
        "right_hand_link"            #13
    ]

    # Define edges (parent-child pairs)
    edges = [
        (0, 1), (1, 2), (2, 3),         # left leg
        (0, 4), (4, 5), (5, 6),         # right leg
        (0, 7),                         # pelvis to torso
        (7, 8), (8, 9), (9,10),         # left arm
        (7,11), (11,12), (12,13),       # right arm
    ]

    # Make edges bidirectional
    # edges += [(j, i) for (i, j) in edges]

    # Convert to tensor
    edge_index = torch.tensor(edges, dtype=torch.long).contiguous()

    return edge_index