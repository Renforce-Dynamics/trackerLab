import h5py
import numpy as np
from typing import List, Dict, Any

def load_trajectories_from_hdf5(hdf5_path: str) -> List[Dict[str, Any]]:
    trajectories = []

    with h5py.File(hdf5_path, "r") as f:
        if "data" not in f:
            raise ValueError("invalid data format.")

        data_group = f["data"]

        for demo_name in sorted(data_group.keys(), key=lambda x: int(x.split("_")[-1])):
            demo_group = data_group[demo_name]

            traj = {}
            for key in demo_group.keys():
                traj[key] = demo_group[key][:]  # numpy array
            trajectories.append(traj)

    return trajectories