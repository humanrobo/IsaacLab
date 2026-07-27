# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from dataclasses import MISSING

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils


MOTIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "motions")


@configclass
class BruceAmpEnvCfg(DirectRLEnvCfg):
    """Humanoid AMP environment config (base class)."""

    # env
    episode_length_s = 20.0
    decimation = 2

    # spaces
    observation_space = 67#次元を2+4つ追加
    action_space = 16

    state_space = 0
    # num_amp_observations = 2
    # amp_observation_space = 83#次元を2+4つ追加

    early_termination = True
    termination_height = 0.2

    # motion_file: str = MISSING
    reference_body = "base_link"
    reset_strategy = "default"  # default, random, random-start
    """Strategy to be followed when resetting each environment (humanoid's pose and joint states).

    * default: pose and joint states are set to the initial state of the asset.
    * random: pose and joint states are set by sampling motions at random, uniform times.
    * random-start: pose and joint states are set by sampling motion at the start (time zero).
    """

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
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=10.0, replicate_physics=True)

    # robot (ヒューマノイドの設定を外し、キューブ等のRigidObjectCfgに置き換え)
# robot
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/matsuno/IsaacLab/robots/bruce.usd",
        ),
        # ★ ここを追加（アクチュエータがないとバリデーションエラーになります）
        actuators={
            "body": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit_sim=100.0,
                stiffness=20.0,
                damping=1.0,
                velocity_limit_sim={
                    ".*": 100.0,
                },
            ),
        },
    )


@configclass
class HumanoidAmpDanceEnvCfg(BruceAmpEnvCfg):
    motion_file = os.path.join(MOTIONS_DIR, "humanoid_dance.npz")


@configclass
class HumanoidAmpRunEnvCfg(BruceAmpEnvCfg):
    motion_file = os.path.join(MOTIONS_DIR, "humanoid_run.npz")


@configclass
class HumanoidAmpWalkEnvCfg(BruceAmpEnvCfg):
    motion_file = os.path.join(MOTIONS_DIR, "humanoid_walk.npz")
