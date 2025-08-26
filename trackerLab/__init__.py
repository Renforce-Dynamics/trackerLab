"""
Python module serving as a project/extension template.
"""
import os

def config_dir(*args):
    dir = os.path.join(*args)
    os.makedirs(dir, exist_ok=True)
    return dir

TRACKERLAB_REPO_DIR = os.path.dirname(os.path.abspath(__file__))
TRACKERLAB_DATA_DIR = os.path.join(TRACKERLAB_REPO_DIR, "..", "data")

TRACKERLAB_ASSETS_DIR = os.path.join(TRACKERLAB_DATA_DIR, "assets")
TRACKERLAB_USD_DIR = os.path.join(TRACKERLAB_ASSETS_DIR, "usd")

TRACKERLAB_RETARGETED_DATA_DIR  = os.path.join(TRACKERLAB_DATA_DIR, "retargeted")
TRACKERLAB_MOTION_CFG_DIR       = os.path.join(TRACKERLAB_DATA_DIR, "configs")

TRACKERLAB_BUFFER_DIR           = config_dir(TRACKERLAB_DATA_DIR, "pkl_buffer")
TRACKERLAB_LABJOINTS_DIR        = config_dir(TRACKERLAB_DATA_DIR, "lab_joints")

TRACKERLAB_TASKS_DIR            = os.path.join(TRACKERLAB_REPO_DIR, "tasks", "tracking")
# Register Gym environments.
# from .tasks import *

