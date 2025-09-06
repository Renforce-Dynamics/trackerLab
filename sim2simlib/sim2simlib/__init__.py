import os

SIM2SIMLIB_REPO_DIR     = os.path.dirname(os.path.abspath(__file__))
SIM2SIMLIB_REPO_DIR     = os.path.dirname(os.path.dirname(SIM2SIMLIB_REPO_DIR))
SIM2SIMLIB_ASSETS_DIR   = os.path.join(SIM2SIMLIB_REPO_DIR, "data", "assets")

SIM2SIMLIB_MUJOCO_ASSETS = {
    "unitree_go2":      f"{SIM2SIMLIB_ASSETS_DIR}/unitree_go2/mjcf/scene_go2.xml",
    "unitree_g1_29dof": f"{SIM2SIMLIB_ASSETS_DIR}/unitree_g1/mjcf/scene_29dof.xml",
    "unitree_g1_23dof": f"{SIM2SIMLIB_ASSETS_DIR}/unitree_g1/mjcf/scene_23dof.xml",
    "booster_k1_rev":   f"{SIM2SIMLIB_ASSETS_DIR}/booster_k1_rev/mjcf/K1_serial.xml"
}