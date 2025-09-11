import os

SIM2SIMLIB_REPO_DIR     = os.path.dirname(os.path.abspath(__file__))
SIM2SIMLIB_REPO_DIR     = os.path.dirname(os.path.dirname(SIM2SIMLIB_REPO_DIR))
SIM2SIMLIB_ASSETS_DIR   = os.path.join(SIM2SIMLIB_REPO_DIR, "data", "assets")
SIM2SIMLIB_LOGS_DIR     = os.path.join(SIM2SIMLIB_REPO_DIR, "logs", "rsl_rl")

SIM2SIMLIB_MUJOCO_ASSETS = {
    "unitree_go2":      f"{SIM2SIMLIB_ASSETS_DIR}/unitree_go2/mjcf/scene_go2.xml",
    "unitree_g1_29dof": f"{SIM2SIMLIB_ASSETS_DIR}/unitree_g1/mjcf/scene_29dof.xml",
    "unitree_g1_23dof": f"{SIM2SIMLIB_ASSETS_DIR}/unitree_g1/mjcf/scene_23dof.xml",
    "booster_k1_rev":   f"{SIM2SIMLIB_ASSETS_DIR}/booster_k1_rev/mjcf/K1_serial.xml"
}

SIM2SIMLIB_CHECKPOINTS = {
    "booster_k1_rev":  "tracking_booster_k1_walk_full_deploy/2025-09-06_17-01-31",
    # "booster_k1_rev_withouthistory":  "tracking_booster_k1_walk_full_deploy_without_history/2025-09-06_16-33-05"
    "booster_k1_rev_withouthistory":  "tracking_booster_k1_walk_full_deploy_without_history/2025-09-10_23-00-05"
}