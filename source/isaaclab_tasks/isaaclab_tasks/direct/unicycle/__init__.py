# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
AMP Humanoid locomotion environment.
"""

import gymnasium as gym

from . import agents
from .unicycle_env import UnicycleEnv
from .unicycle_env_cfg import UnicycleEnvCfg
from .unicycle_g1_env import UnicycleG1Env
from .unicycle_g1_env_cfg import UnicycleG1EnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Unicycle-v0",
    entry_point="isaaclab_tasks.direct.unicycle:UnicycleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point":UnicycleEnvCfg,
        "skrl_cfg_entry_point":
        f"{agents.__name__}:skrl_unicycle_cfg.yaml",
    }
)

gym.register(
    id="Isaac-Unicycle-G1-v0",
    entry_point="isaaclab_tasks.direct.unicycle:UnicycleG1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": UnicycleG1EnvCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_unicycle_cfg.yaml",
    },
)
