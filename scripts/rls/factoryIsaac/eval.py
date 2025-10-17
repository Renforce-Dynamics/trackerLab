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
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--length", type=int, default=200, help="Length of the recorded video (in steps).")

parser.add_argument("--collect", action="store_true", default=False, help="Record data during playing.")

# append RSL-RL cli arguments
cli_rslarg.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


import gymnasium as gym
import os
import torch

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from factoryIsaac.utils.rsl_rl.wrapper import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_onnx,
)

# Import extensions to set up environment tasks
import factoryIsaac.tasks  # noqa: F401  TODO: import lab.<your_extension_name>
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

def log_step(infos: dict, iter):
    print("#"*25 + f"iter: {iter} " + "#"*25)
    to_log  = lambda da: getattr(da, "data", da)
    logs: dict = infos["log"]
    for k, v in logs.items():
        print(f"{k}:\t\t{to_log(v)}")

def eval_task(env, runner:OnPolicyRunner, resume_target=None):
    # reset environment
    # obs, info = env.reset()
    obs, _ = env.get_observations()
    policy = runner.get_inference_policy(env.unwrapped.device)
    # run everything in inference mode
    with torch.inference_mode():
        for iter in range(runner.num_steps_per_env * 10):
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, rewards, dones, infos = env.step(actions)
    # On reset, get the ep info
    # obs, infos = env.reset()
    # obs, info = env.get_observations()
    log_step(infos, "last")
    return infos

def main():
    """Play with RSL-RL agent. base branch"""
    task_name = args_cli.task
    if args_cli.target is None:
        raise ValueError("Please using the target specified way.")
        rslrl_cfg: RslRlOnPolicyRunnerCfg = load_cfg_from_registry(task_name, "rsl_rl_cfg_entry_point")
        # specify directory for logging experiments
        log_root_path = os.path.join("logs", "rsl_rl", rslrl_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        resume_path = get_checkpoint_path(log_root_path, args_cli.run, args_cli.ckpt)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        log_dir = os.path.dirname(resume_path)
        run_path = os.path.dirname(resume_path)
    else:
        resume_path = os.path.abspath(args_cli.target)
        log_root_path = os.path.dirname(os.path.dirname(resume_path))
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        log_dir = os.path.dirname(resume_path)
        run_path = os.path.dirname(resume_path)
    
    if args_cli.collect:
        sample_dir = os.path.join(run_path, "samples")
        os.makedirs(sample_dir, exist_ok=True)
        output_file = os.path.join(sample_dir, "total_data.pkl")
    
    with open(os.path.join(run_path, "params", "agent.pkl"), "rb") as f:
        agent_cfg: RslRlOnPolicyRunnerCfg = pickle.load(f)
        
    if task_name is None:
        assert os.path.exists(os.path.join(run_path, "params", "args.pkl")), "No task specified."
        with open(os.path.join(run_path, "params", "args.pkl"), "rb") as f:
            args_old = pickle.load(f)
        task_name = args_old.task

    with open(os.path.join(run_path, "params", "env.pkl"), "rb") as f:
        # This will using old env config
        # env_cfg = pickle.load(f)
        env_cfg = load_cfg_from_registry(task_name, "env_cfg_entry_point")
        # env_cfg.scene.terrain.terrain_generator.num_cols = 2

    env_cfg.viewer.eye = (0.0, 0.0, 50.0)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.0)

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    # create isaac environment
    env = gym.make(task_name, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            # "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        # print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # # obtain the trained policy for inference
    # policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx
    # export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    # export_policy_as_onnx(ppo_runner.alg.actor_critic, export_model_dir, filename="policy.onnx")

    try:
        while simulation_app.is_running():
            with torch.inference_mode():
                eval_task(env, ppo_runner, resume_path)

    except KeyboardInterrupt:
        pass

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main execution
    main()
    # close sim app
    simulation_app.close()
