
# import the skrl components to build the RL system
from skrl.agents.torch.amp import AMP
from skrl.envs.loaders.torch import load_isaaclab_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory

from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

from models.amp import Policy, Value, Discriminator, get_amp_cfg

DEBUG = True
DEBUG = False

RESUME = False
# RESUME = "/home/star/wangtianhao/minimal_humanoid_amp/runs/torch/HumanoidAMP/25-08-18_12-15-23-834360_AMP/checkpoints/agent_80000.pt"
set_seed(42)  # e.g. `set_seed(42)` for fixed seed

# load and wrap the Isaac Lab environment
# env = load_isaaclab_env(task_name="Isaac-Humanoid-AMP-Run-Direct-v0", cli_args=["--headless"],)
env = load_isaaclab_env(task_name="PiPlus27DofAMPTrackingWalk", cli_args=["--headless"],)
# env = load_isaaclab_env(task_name="Isaac-Humanoid-28-AMP-v0", cli_args=["--headless"],)
env = wrap_env(env)

device = env.device

# 必须和cfg["rollouts"]保持一致
cfg = get_amp_cfg(env=env, device=device)

if DEBUG:
    cfg['experiment']["wandb"] = False
    
memory = RandomMemory(memory_size=cfg['rollouts'], num_envs=env.num_envs, device=device)

models = {}
models["policy"] = Policy(env.observation_space, env.action_space, device)
models["value"] = Value(env.observation_space, env.action_space, device)
models["discriminator"] = Discriminator(env.amp_observation_space, env.action_space, device)

# instantiate the agent's models (function approximators).
agent = AMP(models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device,
            amp_observation_space=env.amp_observation_space,
            motion_dataset=RandomMemory(memory_size=200000, device=device),
            reply_buffer=RandomMemory(memory_size=1000000, device=device),
            collect_reference_motions=lambda num_samples: env.collect_reference_motions(num_samples),
)

if RESUME:
    agent.load(RESUME)
# configure and instantiate the RL trainer
# cfg_trainer = {"timesteps": 8, "headless": True}
cfg_trainer = {"timesteps": 80000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

# start training
trainer.train()