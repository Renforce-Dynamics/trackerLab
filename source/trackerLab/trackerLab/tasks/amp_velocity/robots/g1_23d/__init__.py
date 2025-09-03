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
    id="G123DAMPWalk",
    entry_point="trackerLab.tracker_env:ManagerBasedAMPEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_23d_amp_env_cfg:G1AMPWalk",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.g1_23d_rsl_rl_cfg:G1AMPWalk",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_amp_cfg:G1AMPTrainerCfg",
    },
)