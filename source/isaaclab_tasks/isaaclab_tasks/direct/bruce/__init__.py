# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
AMP Humanoid locomotion environment.
"""

import gymnasium as gym

from . import agents
from .bruce_amp_env import BruceAmpEnv
from .bruce_amp_env_cfg import BruceAmpEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-bruce-Direct-v0",
    entry_point=f"{__name__}.bruce_amp_env:BruceAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bruce_amp_env_cfg:BruceAmpEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_walk_amp_cfg.yaml",
    },
)