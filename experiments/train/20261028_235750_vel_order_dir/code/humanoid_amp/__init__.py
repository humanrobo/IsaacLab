# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
AMP Humanoid locomotion environment.
"""

import gymnasium as gym

from . import agents

from .unicycle_human28_env import UnicycleHumanoid28Env
from .unicycle_human28_env_cfg import UnicycleHumanoid28EnvCfg
##
# Register Gym environments.
##

gym.register(
    id="Isaac-Humanoid-AMP-Dance-Direct-v0",
    entry_point=f"{__name__}.humanoid_amp_env:HumanoidAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.humanoid_amp_env_cfg:HumanoidAmpDanceEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_dance_amp_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Humanoid-AMP-Run-Direct-v0",
    entry_point=f"{__name__}.humanoid_amp_env:HumanoidAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.humanoid_amp_env_cfg:HumanoidAmpRunEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_run_amp_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Humanoid-AMP-Walk-Direct-v0",
    entry_point=f"{__name__}.humanoid_amp_env:HumanoidAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.humanoid_amp_env_cfg:HumanoidAmpWalkEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_walk_amp_cfg.yaml",
    },
)

gym.register(
    id="Isaac-h1-AMP-Walk-Direct-v0",
    entry_point=f"{__name__}.h1_amp_env:HumanoidAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.h1_amp_env_cfg:HumanoidAmpWalkEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_walk_amp_cfg.yaml",
    },
)


gym.register(
    id="Isaac-Unicycle-Humanoid28-v0",
    entry_point="isaaclab_tasks.direct.humanoid_amp:UnicycleHumanoid28Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": UnicycleHumanoid28EnvCfg,
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_walk_amp_cfg.yaml",
    },
)