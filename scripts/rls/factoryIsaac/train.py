"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
from isaaclab import __version__ as omni_isaac_lab_version
assert omni_isaac_lab_version > "0.21.0"
from isaaclab.app import AppLauncher
# local imports
import argtool as cli_rslarg  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")

parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--alg", type=str, default="PPO", help="Name of the algorithm.")
parser.add_argument("--cfg", type=str, default=None, help="Directly using the target cfg object.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")
parser.add_argument("--replicate", type=str, default=None, help="Replicate old experiment with same configuration.")

parser.add_argument("--rldevice", type=str, default="cuda:0", help="Device for rl")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--local", action="store_true", default=False, help="Using asset in local buffer")

# append RSL-RL cli arguments
cli_rslarg.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from datetime import datetime

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

# Import extensions to set up environment tasks
import factoryIsaac

# Loading tasks
# TODO this loading should be in the factory way

import isaaclab_tasks
# import factoryIsaac.tasks
import trackerTask.trackerLab
import trackerTask.beyondMimic

def main():
    if args_cli.replicate is not None:
        task_name, env_cfg, agent_cfg, log_dir = cli_rslarg.load_cfgs(args_cli)
    else:
        from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
        if args_cli.cfg is not None:
            from isaaclab.utils.string import string_to_callable
            agent_cfg = string_to_callable(args_cli.cfg)()
            task_name, env_cfg, agent_cfg, log_dir = cli_rslarg.make_cfgs(args_cli, None, parse_env_cfg, agent_cfg)
        else:
            from factoryIsaac.algorithms.configs.rsl_rl import HYPERPARAMETERS
            task_name, env_cfg, agent_cfg, log_dir = cli_rslarg.make_cfgs(args_cli, HYPERPARAMETERS, parse_env_cfg)
    
    if args_cli.local:
        from factoryIsaac.utils.asset.web_asset import modify_matching_strings_deep, web2local
        # Modify with your args
        env_cfg = modify_matching_strings_deep(env_cfg, modifier_func=web2local)
        
    # env_cfg.scene.terrain.terrain_generator.num_cols = 2
    env_cfg.sim.device = args_cli.device
    env_cfg.seed = args_cli.seed
    # create isaac environment
    env = gym.make(task_name, cfg=env_cfg, 
                   render_mode="rgb_array" if args_cli.video else None)
    
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    agent_cfg.seed = args_cli.seed
    agent_cfg.device = args_cli.rldevice
    
    env, func_runner, learn_cfg = cli_rslarg.prepare_wrapper(env, args_cli, agent_cfg)

    # create runner from rsl-rl, using the base branch format
    runner = func_runner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    
    # write git state to logs
    runner.add_git_repo_to_log(factoryIsaac.__file__)
    
    # save resume path before creating a new log_dir
    if agent_cfg.resume:
        # get path to previous checkpoint
        resume_path = agent_cfg.load_target
        # get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    init_weight = getattr(agent_cfg, "init_weight", None)
    if init_weight: runner.load_weight(init_weight)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_yaml(os.path.join(log_dir, "params", "args.yaml"), vars(args_cli))
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "args.pkl"), args_cli)

    # run training
    runner.learn(**learn_cfg)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main execution
    main()
    # close sim app
    simulation_app.close()
