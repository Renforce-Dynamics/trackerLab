import os

def config_dir(*args):
    dir = os.path.join(*args)
    os.makedirs(dir, exist_ok=True)
    return dir

TRACKERTASK_REPO_DIR                 = os.path.dirname(os.path.abspath(__file__))
TRACKERTASK_DATA_DIR                 = os.path.join(TRACKERTASK_REPO_DIR, "..", "..", "..", "data")