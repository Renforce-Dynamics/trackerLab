import csv
import json
import torch
from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState
from poselib.visualization.common import plot_skeleton_state

def csv_to_skeleton(csv_path):
    """
    Convert a CSV skeleton description into a structured skeleton dictionary.
    Assumptions:
    - The first row is the root node (parent index = -1).
    - 'Parent' column contains the link name of the parent.
    - Output matches the 'skeleton' dict format.
    """

    # Read CSV into a list of dicts
    with open(csv_path, newline='') as csvfile:
        reader = list(csv.DictReader(csvfile))

    # Step 1: Collect all node names and build name -> index mapping
    node_names = [row["Link Name"].strip() for row in reader]
    link_to_index = {name: idx for idx, name in enumerate(node_names)}

    # Step 2: Compute parent indices
    parent_indices = []
    for idx, row in enumerate(reader):
        if idx == 0:
            # First row is root node
            parent_indices.append(-1)
        else:
            parent_name = row["Parent"].strip()
            parent_indices.append(link_to_index.get(parent_name, -1))

    # Step 3: Extract local translations
    local_translation = []
    for row in reader:
        x = float(row["Joint Origin X"])
        y = float(row["Joint Origin Y"])
        z = float(row["Joint Origin Z"])
        local_translation.append([x, y, z])

    # Build skeleton dict
    skeleton = {
        "skeleton": {
            "node_names": node_names,
            "parent_indices": parent_indices,
            "local_translation": local_translation
        }
    }

    # print(skeleton)

    return skeleton

def csv_to_joint_dicts(csv_path):
    """
    Convert joint data from CSV into three dictionaries:
    - effort_dict
    - damping_dict
    - stiffness_dict
    Keys are 'Joint Name', values come from corresponding columns.
    Output is a JSON string containing all three dicts.
    """

    effort_dict = {}
    damping_dict = {}
    stiffness_dict = {}

    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            joint_name = row["Joint Name"].strip()

            # Convert values to float if possible
            effort = float(row["Effort"]) if row["Effort"] else 0.0
            damping = float(row["Damping"]) if row["Damping"] else 0.0
            stiffness = float(row["Stiffness"]) if row["Stiffness"] else 0.0

            effort_dict[joint_name] = effort
            damping_dict[joint_name] = damping
            stiffness_dict[joint_name] = stiffness

    result = {
        "effort": effort_dict,
        "damping": damping_dict,
        "stiffness": stiffness_dict
    }

    # Return as formatted JSON string
    return json.dumps(result, indent=2)

def verbose_list(tar: list):
    for idx, item in enumerate(tar):
        print(f"{idx:02d}:\t{item}")

import argparse

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--robot", required=True)
    args = parser.parse_args()
    csv_file = args.csv
    skeleton_dict = csv_to_skeleton(csv_file)["skeleton"]
    node_names, parent_indices, local_translation = \
        skeleton_dict["node_names"], skeleton_dict["parent_indices"], skeleton_dict["local_translation"]

    tree = SkeletonTree(node_names, torch.tensor(parent_indices), torch.tensor(local_translation))
    verbose_list(tree.node_names)
    tpose = SkeletonState.zero_pose(tree)
    plot_skeleton_state(tpose)
    tpose.to_file(f"./data/tpose/{args.robot}_tpose.npy")
    # Pretty print JSON
