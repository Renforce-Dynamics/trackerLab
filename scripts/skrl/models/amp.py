import torch
import torch.nn as nn
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.amp import AMP_DEFAULT_CONFIG
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL

class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 1024),
                                 nn.ReLU(),
                                 nn.Linear(1024, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, self.num_actions))

        # set a fixed log standard deviation for the policy
        self.log_std_parameter = nn.Parameter(torch.full((self.num_actions,), fill_value=-2.9), requires_grad=False)

    def compute(self, inputs, role):
        return torch.tanh(self.net(inputs["states"])), self.log_std_parameter, {}

class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 1024),
                                 nn.ReLU(),
                                 nn.Linear(1024, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 1))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}

class Discriminator(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 1024),
                                 nn.ReLU(),
                                 nn.Linear(1024, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 1))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}


# instantiate a memory as rollout buffer (any memory can be used for this)
def get_amp_cfg(env, device):
    cfg = AMP_DEFAULT_CONFIG.copy()
    cfg["rollouts"] = 16  # memory_size
    cfg["learning_epochs"] = 6
    cfg["mini_batches"] = 2  # 16 * 4096 / 32768
    cfg["discount_factor"] = 0.99
    cfg["lambda"] = 0.95
    cfg["learning_rate"] = 5e-5
    cfg["random_timesteps"] = 0
    cfg["learning_starts"] = 0
    cfg["grad_norm_clip"] = 0
    cfg["ratio_clip"] = 0.2
    cfg["value_clip"] = 0.2
    cfg["clip_predicted_values"] = False
    cfg["entropy_loss_scale"] = 0.0
    cfg["value_loss_scale"] = 2.5 
    cfg["discriminator_loss_scale"] = 5.0
    cfg["amp_batch_size"] = 512
    cfg["task_reward_weight"] = 0.5
    cfg["style_reward_weight"] = 0.5
    cfg["discriminator_batch_size"] = 4096
    cfg["discriminator_reward_scale"] = 2
    cfg["discriminator_logit_regularization_scale"] = 0.05
    cfg["discriminator_gradient_penalty_scale"] = 5
    cfg["discriminator_weight_decay_scale"] = 1.0e-04
    cfg["state_preprocessor"] = RunningStandardScaler
    cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
    cfg["value_preprocessor"] = RunningStandardScaler
    cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}
    cfg["amp_state_preprocessor"] = RunningStandardScaler
    cfg["amp_state_preprocessor_kwargs"] = {"size": env.amp_observation_space, "device": device}
    # logging to TensorBoard and write checkpoints (in timesteps)
    cfg["experiment"]["write_interval"] = 160
    cfg["experiment"]["checkpoint_interval"] = 4000
    cfg["experiment"]["directory"] = "runs/torch/HumanoidAMP"
    # [Optional] Logging with wandb
    cfg['experiment']["wandb"] = True
    cfg['experiment']["wandb_kwargs"] = {"project": "trackerLab-AMP"}
    return cfg
