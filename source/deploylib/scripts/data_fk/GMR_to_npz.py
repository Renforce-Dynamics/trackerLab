import argparse
import numpy as np
import torch

from isaaclab.app import AppLauncher

# -----------------------------------------------------------------------------
# Argument Parsing
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Replay motion from CSV and export to NPZ.")
parser.add_argument("--frame_range", nargs=2, type=int, metavar=("START", "END"),
                    help="Frame range (inclusive). If not set, all frames are used.")
parser.add_argument("--output_name", type=str, required=True,
                    help="Name of the output NPZ file.")
parser.add_argument("--output_fps", type=int, default=50,
                    help="Frame rate (fps) of the output motion.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# -----------------------------------------------------------------------------
# Launch Omniverse / IsaacSim
# -----------------------------------------------------------------------------
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from deploylib.deploy_manager import DeployManager
from sim2simlib.motion import MotionBufferCfg, MotionManagerCfg
from robotlib.loader import load_robot_cfg

# -----------------------------------------------------------------------------
# Configurations
# -----------------------------------------------------------------------------
robot_type = "g1_29d"
robot_cfg, motion_align_cfg = load_robot_cfg(robot_type)
motion_name = "/home/ununtu/code/trackerLab/data/configs/GMR/temp.yaml"
COLLECTION = args_cli.output_name

motion_cfg = MotionManagerCfg(
    MotionBufferCfg(
        motion_name=motion_name,
        motion_type="GMR",
        motion_lib_type="MotionLibDofPos",
        regen_pkl=True,
    ),
    motion_align_cfg=motion_align_cfg,
)

# -----------------------------------------------------------------------------
# Scene Configuration
# -----------------------------------------------------------------------------
@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Scene configuration for motion replay."""

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg()
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/"
                         f"kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    robot: ArticulationCfg = robot_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")

# -----------------------------------------------------------------------------
# Motion Loader
# -----------------------------------------------------------------------------
class MotionLoader:
    """Handles motion replay through DeployManager."""

    def __init__(self, lab_joint_names, sim, fps, cfg: MotionManagerCfg):
        self.manager = DeployManager(
            cfg.motion_buffer_cfg,
            cfg.motion_align_cfg,
            lab_joint_names,
            robot_type,
            dt=1 / fps,
            device=sim.device,
        )

    def get_next_state(self):
        """Returns the next motion state and reset flag."""
        flag = self.manager.step()
        return (
            self.manager.loc_root_pos,
            self.manager.loc_root_rot[:, [3, 0, 1, 2]],
            self.manager.loc_root_vel_global,
            self.manager.loc_ang_vel,
            self.manager.loc_dof_pos,
            self.manager.loc_dof_vel,
        ), flag

# -----------------------------------------------------------------------------
# Utility: Save Motion Data
# -----------------------------------------------------------------------------
def save_npz(data_buffer, res_file="./motion.npz"):
    """Save collected motion data to NPZ."""
    for k in (
        "joint_pos", "joint_vel", "body_pos_w",
        "body_quat_w", "body_lin_vel_w", "body_ang_vel_w",
    ):
        data_buffer[k] = np.stack(data_buffer[k], axis=0)
    np.savez(res_file, **data_buffer)

# -----------------------------------------------------------------------------
# Simulation Loop
# -----------------------------------------------------------------------------
def run_simulator(sim: SimulationContext, scene: InteractiveScene, forever=True):
    """Main simulation loop for replay and data recording."""
    robot = scene["robot"]

    motion = MotionLoader(
        lab_joint_names=robot.data.joint_names,
        sim=sim,
        fps=args_cli.output_fps,
        cfg=motion_cfg,
    )

    file_saved = False
    motion.manager.init_finite_state_machine()

    # Initialize data buffer
    data_buffer = {
        "fps": [args_cli.output_fps],
        "joint_pos": [], "joint_vel": [],
        "body_pos_w": [], "body_quat_w": [],
        "body_lin_vel_w": [], "body_ang_vel_w": [],
    }

    while simulation_app.is_running():
        (motion_base_pos, motion_base_rot,
         motion_base_lin_vel, motion_base_ang_vel,
         motion_dof_pos, motion_dof_vel), reset_flag = motion.get_next_state()

        # Root state
        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion_base_pos
        root_states[:, :2] += scene.env_origins[:, :2]
        root_states[:, 3:7] = motion_base_rot
        root_states[:, 7:10] = motion_base_lin_vel
        root_states[:, 10:] = motion_base_ang_vel
        robot.write_root_state_to_sim(root_states)

        # Joint state
        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, :] = motion_dof_pos
        joint_vel[:, :] = motion_dof_vel
        robot.write_joint_state_to_sim(joint_pos, joint_vel)

        # Render (no physics)
        sim.render()
        scene.update(sim.get_physics_dt())

        if reset_flag:
            motion.manager.add_finite_state_machine_motion_ids()

        # Record data
        if not file_saved:
            data_buffer["joint_pos"].append(robot.data.joint_pos[0, :].cpu().numpy().copy())
            data_buffer["joint_vel"].append(robot.data.joint_vel[0, :].cpu().numpy().copy())
            data_buffer["body_pos_w"].append(robot.data.body_pos_w[0, :].cpu().numpy().copy())
            data_buffer["body_quat_w"].append(robot.data.body_quat_w[0, :].cpu().numpy().copy())
            data_buffer["body_lin_vel_w"].append(robot.data.body_lin_vel_w[0, :].cpu().numpy().copy())
            data_buffer["body_ang_vel_w"].append(robot.data.body_ang_vel_w[0, :].cpu().numpy().copy())

        if reset_flag and not file_saved:
            file_saved = True
            save_npz(data_buffer, res_file=f"./{COLLECTION}.npz")

# -----------------------------------------------------------------------------
# Main Entry
# -----------------------------------------------------------------------------
def main():
    """Entry point."""
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.output_fps
    sim = SimulationContext(sim_cfg)

    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    print("[INFO] Setup complete. Starting replay...")

    run_simulator(sim, scene)

if __name__ == "__main__":
    main()
    simulation_app.close()
