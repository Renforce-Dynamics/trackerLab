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
# parser.add_argument("--target", type=str, default=None, help="If use, direct point to the target ckpt")

parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--track", type=str, default=None, help="Name of the track.")
# parser.add_argument("--run", type=str, default=".*", help="Name of the run.")
# parser.add_argument("--ckpt", type=str, default=".*", help="Name of the ckpt.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--rldevice", type=str, default="cuda:0", help="Device for rl")
# parser.add_argument("--collect", action="store_true", default=False, help="Record data during playing.")

# append RSL-RL cli arguments
cli_rslarg.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# # always enable cameras to record video
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
from factoryIsaac.tasks.locomotion.dog_playground.tracks import post_modify_track
from factoryIsaac.utils.asset.web_asset import modify_matching_strings_deep, web2local, web2str, ASSET_LOCAL_DIR

from datetime import datetime

def log_step(infos: dict, iter=0, **kwargs):
    print("#"*25 + f"iter: {iter} " + "#"*25)
    to_log  = lambda da: getattr(da, "data", da)
    logs: dict = infos["log"]
    for k, v in logs.items():
        print(f"{k}:\t\t{to_log(v)}")
    print("#"*30 + "#"*28)

import json

class LoggerMetric:
    def __init__(self, save_dir, name, save_res=False):
        self.output_file = self.prepare_file(save_dir, f"{name}.json")
        self.save_res = save_res
        pass
    
    def __call__(self, **kwargs):
        infos:dict = self.extrac(kwargs["infos"], ["log"])
        if self.save_res:
            with open(self.output_file, "wt") as f:
                json.dump(infos, f, indent=2)
        log_step(**kwargs)
    
    @staticmethod
    def extrac(origin:dict, keys=None):
        if keys is None: keys = origin.keys()
        res = {}
        for key in keys:
            val = origin[key]
            if isinstance(val, torch.Tensor): val = val.tolist()
            elif isinstance(val, dict): val = LoggerMetric.extrac(val, None)
            elif isinstance(val, list): val = None
            res[key] = val
        return res

    @staticmethod
    def prepare_file(save_dir, name):
        os.makedirs(save_dir, exist_ok=True)
        target_file = datetime.now().strftime("%Y-%m-%d %H:%M:%S") + f"_{name}"
        return os.path.join(save_dir, target_file)

import tqdm
def eval_task(env, runner:OnPolicyRunner, log_func=log_step, running_time=None):
    """
    Eval one task, the curriculum is update through steping, reset will make the curriculum back
    """
    if running_time is None: running_time = runner.num_steps_per_env * 10
    # reset environment
    obs, infos = env.reset()
    # obs, _ = env.get_observations()
    policy = runner.get_inference_policy(env.unwrapped.device)
    # run everything in inference mode
    reward_total = 0
    pbar = range(running_time)
    with torch.inference_mode():
        for iter in pbar:
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, rewards, dones, infos = env.step(actions)
            reward_total += rewards
            
    reward_avg = reward_total /running_time
    # On reset, get the ep info
    # obs, infos = env.reset()
    # obs, info = env.get_observations()
    log_func(**locals())
    return infos

_terrains = \
[
    "pyramid_stairs",
    "pyramid_stairs_inv",
    "boxes",
    "random_rough",
    "hf_pyramid_slope",
    "hf_pyramid_slope_inv",
]

def prepare_runner(env, target):
    resume_path = os.path.abspath(target)
    run_path = os.path.dirname(resume_path)
    with open(os.path.join(run_path, "params", "agent.pkl"), "rb") as f:
        agent_cfg: RslRlOnPolicyRunnerCfg = pickle.load(f)
    agent_cfg.device = args_cli.rldevice
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    return ppo_runner

"""
Eval one task with multiple policies.
"""
from rsl_rl.utils.config_util import import_file_from_path
import copy


def main(task_name, eval_list, eval_name, tracks=["random_rough"], epi_len=None):
    """Play with RSL-RL agent. base branch"""
    if epi_len is None: epi_len=args_cli.length
    env_cfg = parse_env_cfg(task_name, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg = modify_matching_strings_deep(env_cfg, modifier_func=web2local)
    post_modify_track(env_cfg, tracks)

    # env_cfg.scene.terrain.terrain_generator.num_cols = 2
    # env_cfg.viewer.eye = (0.0, 0.0, 50.0)
    # env_cfg.viewer.lookat = (0.0, 0.0, 0.0)

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    # create isaac environment
    env = gym.make(task_name, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None, seed=args_cli.seed)
    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    total_tasks = len(eval_list)
    step = 0
    print(f"Eval total {total_tasks} tasks")
    _infos = []
    try:
        while simulation_app.is_running():
            if step >= total_tasks: break
            target_task = eval_list[step]
            target_name = eval_name[step]
            print(f"task: {target_task}, with: {target_name}")
            save_func = LoggerMetric("./logs/evals", target_name)
            ppo_runner = prepare_runner(env, target_task)
            with torch.inference_mode():
                info = eval_task(env, ppo_runner, save_func, epi_len)
                # info["target_task"] = target_task
                info["target_name"] = target_name
                info["track"] = tracks[0]
                _infos.append(copy.deepcopy(info))
            step += 1
    except KeyboardInterrupt:
        print("Quit eval")
    # close the simulator
    env.close()
    return _infos

if __name__ == "__main__":
    import pandas as pd
    datalib = import_file_from_path("datalib", "/workspace/isaaclab/logs/eval_cfg.py")
    preload = lambda name: getattr(datalib, name, [])
    loadmod = lambda name, grp: [f"{grp}{attr}" for attr in preload(name)]
    # eval_vars = ["tasks_action_size", "tasks_obs_size", "tasks_transfer_go1", "tasks_transfer_go2"]
    eval_tasks = [
        "Isaac-Velocity-Rough-Unitree-A1-v0",
        "Isaac-Velocity-Rough-Unitree-Go1-v0",
        "Isaac-Velocity-Rough-Unitree-Go2-v0"
    ]
    details = [
        (preload("tasks_action_size")+preload("tasks_obs_size"), 
         loadmod("names_action_size", "act_")+loadmod("names_obs_size", "obs_")),
        (preload("tasks_transfer_go1"), 
         preload("names_transfer_go1")),
        (preload("tasks_transfer_go2"), 
         preload("names_transfer_go2")),
    ]
    def modify(tar):
        res = tar["log"]
        res["target_name"] = tar["target_name"]
        res["track"] = tar["track"]
        return LoggerMetric.extrac(res)
    
    
    task = args_cli.task
    idx = eval_tasks.index(task)
    track = args_cli.track
    eval_list, eval_name = details[idx]
    # run the main execution
    filename = LoggerMetric.prepare_file("./logs/evals/res", f"{task}_{track}.csv")
    logs = []
    infos = main(task, eval_list, eval_name, tracks=[track])
    logs = [modify(info) for info in infos]
    df = pd.DataFrame(logs)
    print(df)
    df.to_csv(filename, index=False)
    # close sim app
    simulation_app.close()
