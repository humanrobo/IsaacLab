# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils

@configclass
class UnicycleEnvCfg(DirectRLEnvCfg):
    """Unicycle environment config (cube base model)."""

    # env
    episode_length_s = 5.0
    decimation = 2

    # spaces (ユニサイクルモデルの入力・出力に合わせて変更)
    # 例: 観測空間 = local_lin_vel(3) + local_ang_vel(3) + height(1) + heading_sin(1) + heading_cos(1) + goal_vec_local(2) = 11次元
    observation_space = 11
    action_space = 2
    state_space = 0

    early_termination = True
    termination_height = 0.2

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
        render_interval=decimation,
        physx=PhysxCfg(
            gpu_found_lost_pairs_capacity=2**23,
            gpu_total_aggregate_pairs_capacity=2**23,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, 
        env_spacing=10.0, 
        replicate_physics=True
    )

    # robot (ヒューマノイドの設定を外し、キューブ等のRigidObjectCfgに置き換え)
# robot
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/matsuno/IsaacLab/robots/bruce.usd",
        ),
    )