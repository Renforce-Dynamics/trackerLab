# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="PiPlus27DAMPWalk",
    entry_point="trackerLab.tracker_env:ManagerBasedAMPEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pi_27d_amp_env_cfg:PiAMPRun",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.pi_27d_rsl_rl_cfg:Pi27DAMPRun",
        # "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_amp_cfg:G1AMPTrainerCfg",
    },
)