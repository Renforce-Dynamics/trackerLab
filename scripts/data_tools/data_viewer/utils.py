# utils.py
import os
import h5py

def list_hdf5_files(data_dir, allowed_ext):
    """Return sorted list of HDF5 files in given folder."""
    files = []
    for fn in os.listdir(data_dir):
        if fn.lower().endswith(allowed_ext):
            files.append(fn)
    return sorted(files)

def walk_h5_groups(h5node, prefix=""):
    """
    Recursively list all dataset paths in an HDF5 file.
    Returns a flat list of dataset paths.
    """
    items = []
    for key in h5node:
        obj = h5node[key]
        path = prefix + "/" + key if prefix else key
        if isinstance(obj, h5py.Dataset):
            items.append(path)
        elif isinstance(obj, h5py.Group):
            items.extend(walk_h5_groups(obj, path))
    return items
