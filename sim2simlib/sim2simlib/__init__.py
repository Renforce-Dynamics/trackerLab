import os
from etils import epath

SIM2SIMLIB_REPO_DIR = epath.Path(__file__).resolve().parent

SIM2SIMLIB_ASSETS_DIR = SIM2SIMLIB_REPO_DIR.parent.parent / "data" / "assets"


MUJOCO_ASSETS = {
    "unitree_go2": f"{SIM2SIMLIB_ASSETS_DIR}/unitree_go2/mjcf/scene_go2.xml",
    "unitree_g1_29dof": f"{SIM2SIMLIB_ASSETS_DIR}/unitree_g1/mjcf/scene_29dof.xml",
    "unitree_g1_23dof": f"{SIM2SIMLIB_ASSETS_DIR}/unitree_g1/mjcf/scene_23dof.xml"
}