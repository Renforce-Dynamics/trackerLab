"""This script replay a motion from a csv file and output it to a npz file

.. code-block:: bash

    # Usage
    python csv_to_npz.py --input_file LAFAN/dance1_subject2.csv --input_fps 30 --frame_range 122 722 \
    --output_file ./motions/dance1_subject2.npz --output_fps 50
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import numpy as np

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay motion from csv file and output to npz file.")
# parser.add_argument("--input_file", type=str, required=True, help="The path to the input motion csv file.")
# parser.add_argument("--input_fps", type=int, default=30, help="The fps of the input motion.")
parser.add_argument(
    "--frame_range",
    nargs=2,
    type=int,
    metavar=("START", "END"),
    help=(
        "frame range: START END (both inclusive). The frame index starts from 1. If not provided, all frames will be"
        " loaded."
    ),
)
parser.add_argument("--output_name", type=str, required=True, help="The name of the motion npz file.")
parser.add_argument("--output_fps", type=int, default=50, help="The fps of the output motion.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul, quat_slerp

##
# Pre-defined configs
##

from deploylib.deploy_manager import DeployManager
from sim2simlib.motion import MotionBufferCfg, MotionManagerCfg

# from robotlib.beyondMimic.robots.g1 import G1_CYLINDER_CFG
from robotlib.trackerLab.assets.humanoids.r2 import R2_CFG

robot_cfg = R2_CFG
robot_type = "r2b"

from trackerTask.trackerLab.tracking.humanoid.robots.r2.motion_align_cfg import R2B_MOTION_ALIGN_CFG_GMR
motion_cfg = MotionManagerCfg(
    MotionBufferCfg(
        motion_name="GMR/test.yaml",
        motion_type="GMR",
        motion_lib_type="MotionLibDofPos"
    ),
    motion_align_cfg=R2B_MOTION_ALIGN_CFG_GMR
)


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # articulation
    robot: ArticulationCfg = robot_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")
    
    

class MotionLoader():
    def __init__(self, lab_joint_names, sim, fps, cfg:MotionManagerCfg):
        
        self.manager = DeployManager(
            cfg.motion_buffer_cfg, 
            cfg.motion_align_cfg,
            lab_joint_names,
            robot_type,
            dt = 1/fps,
            device=sim.device
            )
        pass
    
    def get_next_state(self):
        flag = self.manager.step()
        return (
            self.manager.loc_root_pos, 
            self.manager.loc_root_rot, 
            self.manager.loc_root_vel, 
            self.manager.loc_ang_vel,
            self.manager.loc_dof_pos,
            self.manager.loc_dof_vel
            ), flag


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    
    
    """Runs the simulation loop."""


    # Extract scene entities
    robot = scene["robot"]
    
    
    # Load motion
    motion = MotionLoader(
        lab_joint_names=robot.data.joint_names,
        sim=sim,
        fps=args_cli.output_fps,
        cfg=motion_cfg,
    )
    
    # robot_joint_indexes = robot.find_joints(joint_names, preserve_order=True)[0]

    # ------- data logger -------------------------------------------------------
    log = {
        "fps": [args_cli.output_fps],
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }
    file_saved = False
    # --------------------------------------------------------------------------

    # Simulation loop
    while simulation_app.is_running():
        (
            (
                motion_base_pos,
                motion_base_rot,
                motion_base_lin_vel,
                motion_base_ang_vel,
                motion_dof_pos,
                motion_dof_vel,
            ),
            reset_flag,
        ) = motion.get_next_state()

        # set root state
        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion_base_pos
        root_states[:, :2] += scene.env_origins[:, :2]
        root_states[:, 3:7] = motion_base_rot
        root_states[:, 7:10] = motion_base_lin_vel
        root_states[:, 10:] = motion_base_ang_vel
        robot.write_root_state_to_sim(root_states)

        # set joint state
        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, :] = motion_dof_pos
        joint_vel[:, :] = motion_dof_vel
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        sim.render()  # We don't want physic (sim.step())
        scene.update(sim.get_physics_dt())

        pos_lookat = root_states[0, :3].cpu().numpy()
        sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)

        if not file_saved:
            log["joint_pos"].append(robot.data.joint_pos[0, :].cpu().numpy().copy())
            log["joint_vel"].append(robot.data.joint_vel[0, :].cpu().numpy().copy())
            log["body_pos_w"].append(robot.data.body_pos_w[0, :].cpu().numpy().copy())
            log["body_quat_w"].append(robot.data.body_quat_w[0, :].cpu().numpy().copy())
            log["body_lin_vel_w"].append(robot.data.body_lin_vel_w[0, :].cpu().numpy().copy())
            log["body_ang_vel_w"].append(robot.data.body_ang_vel_w[0, :].cpu().numpy().copy())

        if reset_flag and not file_saved:
            file_saved = True
            for k in (
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
            ):
                log[k] = np.stack(log[k], axis=0)

            np.savez("./motion.npz", **log)

            import wandb

            COLLECTION = args_cli.output_name
            run = wandb.init(project="GMR_to_npz_r2", name=COLLECTION)

            print(f"[INFO]: Logging motion to wandb: {COLLECTION}")
            REGISTRY = "mocap_datas"
            logged_artifact = run.log_artifact(artifact_or_path="/tmp/motion.npz", name=COLLECTION, type=REGISTRY)
            run.link_artifact(artifact=logged_artifact, target_path=f"wandb-registry-{REGISTRY}/{COLLECTION}")
            print(f"[INFO]: Motion saved to wandb registry: {REGISTRY}/{COLLECTION}")


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.output_fps
    sim = SimulationContext(sim_cfg)
    # Design scene
    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(
        sim,
        scene,
    )


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
