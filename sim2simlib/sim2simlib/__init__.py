from etils import epath

SIM2SIMLIB_REPO_DIR = epath.Path(__file__).parent

LOGS_DIR = SIM2SIMLIB_REPO_DIR.parent.parent / "logs"

MUJOCO_ASSETS_DIR = SIM2SIMLIB_REPO_DIR.parent / "mujoco_assets"

MUJOCO_ASSETS = {
    "unitree_g1_29dof": MUJOCO_ASSETS_DIR / "unitree_g1" / "scene_29dof.xml",
    "unitree_g1_23dof": MUJOCO_ASSETS_DIR / "unitree_g1" / "scene_23dof.xml",
    "unitree_go2": MUJOCO_ASSETS_DIR / "unitree_go2" / "scene_go2.xml",
    
    # "unitree_h1": MUJOCO_ASSETS_DIR / "unitree_h1" / "scene.xml",
    # "unitree_a1": MUJOCO_ASSETS_DIR / "unitree_a1" / "scene.xml",

}