"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import pickle
import tqdm
from isaaclab import __version__ as omni_isaac_lab_version
from isaaclab.app import AppLauncher

# local imports
import argtool as cli_rslarg  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
if omni_isaac_lab_version < "0.21.0":
    parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument("--target", type=str, default=None, help="If use, direct point to the target ckpt")

parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# parser.add_argument("--run", type=str, default=".*", help="Name of the run.")
# parser.add_argument("--ckpt", type=str, default=".*", help="Name of the ckpt.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during playing.")
parser.add_argument("--length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--rldevice", type=str, default="cuda:0", help="Device for rl")

parser.add_argument("--web", action="store_true", default=False, help="Web videos during playing.")
parser.add_argument("--local", action="store_true", default=False, help="Using asset in local buffer")

parser.add_argument("--determine",action="store_true", default=False, help="Clear reset terms" )

# append RSL-RL cli arguments
cli_rslarg.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# always enable cameras to record video
if args_cli.video or args_cli.web:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import gymnasium as gym
import os
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab.utils.dict import print_dict
from factoryIsaac.utils.rsl_rl.wrapper import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_onnx,
)

# Import extensions to set up environment tasks
import factoryIsaac.tasks  # noqa: F401  TODO: import lab.<your_extension_name>
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, set_determine_reset
from factoryIsaac.utils.asset.web_asset import modify_matching_strings_deep, web2local, web2str, ASSET_LOCAL_DIR

def main():
    """Play with RSL-RL agent. base branch"""
    task_name = args_cli.task
    if args_cli.target is None:
        raise ValueError("Please using the target specified way.")
    else:
        resume_path = os.path.abspath(args_cli.target)
        log_root_path = os.path.dirname(os.path.dirname(resume_path))
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        log_dir = os.path.dirname(resume_path)
        run_path = os.path.dirname(resume_path)
    
    with open(os.path.join(run_path, "params", "agent.pkl"), "rb") as f:
        agent_cfg: RslRlOnPolicyRunnerCfg = pickle.load(f)
        
    if task_name is None:
        assert os.path.exists(os.path.join(run_path, "params", "args.pkl")), "No task specified."
        with open(os.path.join(run_path, "params", "args.pkl"), "rb") as f:
            args_old = pickle.load(f)
        task_name = args_old.task

        with open(os.path.join(run_path, "params", "env.pkl"), "rb") as f:
            env_cfg = pickle.load(f)
    else:
        env_cfg = load_cfg_from_registry(task_name, "env_cfg_entry_point")
        
        # env_cfg.scene.terrain.terrain_generator.num_cols = 2

    if args_cli.local:
        from factoryIsaac.utils.asset.web_asset import modify_matching_strings_deep, web2local
        # Modify with your args
        env_cfg = modify_matching_strings_deep(env_cfg, modifier_func=web2local)

    env_cfg.sim.device = args_cli.device
    env_cfg.seed = args_cli.seed

    from isaaclab.envs.common import ViewerCfg
    
    
    env_cfg.viewer = ViewerCfg(
        eye = (4.0, 4.0, 4.0),
        # eye = (0.0, 0.0, 10.0),
        lookat = (0.0, 0.0, 0.0),
        env_index = 20,
        origin_type = "asset_root",
        # origin_type = "env",
        asset_name = "robot",
    )
    
    if args_cli.determine:
        set_determine_reset(env_cfg)
    
    # env_cfg.commands.base_velocity.ranges.lin_vel_x = (-0.8, -0.8)
    # env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    # env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    
    # env_cfg.commands.base_velocity.debug_vis = False
    # env_cfg.scene.height_scanner.debug_vis = False

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    render_mode = "rgb_array" if args_cli.video or args_cli.web else None
    # create isaac environment
    env = gym.make(task_name, cfg=env_cfg, render_mode=render_mode)
    
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.length if args_cli.length > 0 else 0,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        # print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    
    if args_cli.web:
        from factoryIsaac.utils.web_viewer.wrapper import RenderEnvWrapper

        if not hasattr(env, "socketio_running"):
            env = RenderEnvWrapper(env)
            env.web_run()
            env.socketio_running = True
            print("[INFO] Running web viewer.")

    agent_cfg.seed = args_cli.seed
    agent_cfg.device = args_cli.rldevice

    env, func_runner, learn_cfg = cli_rslarg.prepare_wrapper(env, args_cli, agent_cfg)

    runner = func_runner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

    runner.load(resume_path)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # export policy
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    os.makedirs(export_model_dir, exist_ok=True)
    torch.save(runner.alg.actor_critic, os.path.join(export_model_dir, "policy.pth"))
    export_policy_as_onnx(runner.alg.actor_critic, export_model_dir, filename="policy.onnx")
    print(f"[INFO]: Saving policy to: {export_model_dir}")


    from beyondMimic.utils.exporter import export_motion_policy_as_onnx, attach_onnx_metadata

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")

    export_motion_policy_as_onnx(
        env.unwrapped,
        runner.alg.actor_critic,
        normalizer=runner.obs_normalizer,
        path=export_model_dir,
        filename="beyond.onnx",
    )
    attach_onnx_metadata(env.unwrapped, "none", export_model_dir, "beyond.onnx")


    # reset environment
    obs, _ = env.get_observations()

    pbar = tqdm.tqdm(range(args_cli.length)) if args_cli.length>0  else tqdm.tqdm()
    
    import numpy as np
    res = [obs[0].clone().cpu().numpy().reshape(1,-1)]
    cmd = env.unwrapped.command_manager.get_term("motion")
    print(cmd.motion.time_step_total)
    
    step = 0
    try:
        # simulate environment
        while simulation_app.is_running():

            # run everything in inference mode
            with torch.inference_mode():
                # agent stepping
                actions = policy(obs)
                # env stepping
                obs, rewards, dones, infos = env.step(actions)

            if step < 100:
                res.append(obs[0].clone().cpu().numpy().reshape(1,-1))
            elif step == 100:
                res = np.concatenate(res, 0)
                np.save("./obs2.npz", res)
                print("saved obs.npy")

            step += 1
            pbar.update() 
            if args_cli.length > 0 and args_cli.length < step:
                break

    except KeyboardInterrupt:
        pass

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main execution
    main()
    # close sim app
    simulation_app.close()
