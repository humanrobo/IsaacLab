# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from dataclasses import MISSING

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

# 🎯 本物の H1 ロボットアセット（USDパスやモーター設定が全部入っています）をインポート
from isaaclab_assets.robots.unitree import H1_CFG

MOTIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "motions")


@configclass
class HumanoidAmpEnvCfg(DirectRLEnvCfg):
    """Humanoid AMP environment config (base class)."""

    # env
    episode_length_s = 20.0
    decimation = 2

    # 🎯 H1（19関節）のデータ構造に合わせて正確に次元数を計算
    # dof_pos(19) + dof_vel(19) + height(1) + tangent_normal(6) + lin_vel(3) + ang_vel(3) + key_bodies(4*3=12) = 63
    # ここに Matsuno さんが追加した heading_sin(1) + heading_cos(1) = 65 次元
    action_space = 19
    observation_space = 65
    state_space = 0
    num_amp_observations = 2
    amp_observation_space = 65

    early_termination = True
    termination_height = 0.5

    motion_file: str = MISSING
    reference_body = "pelvis"     # H1の基準リンク名
    reset_strategy = "random"    # default, random, random-start

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

    # 🎯 ロボットの設定
    # H1_CFG（本物のH1のUSD・質量・モーターゲイン）をベースにしつつ、出現する場所（パス）だけを指定します
    robot: ArticulationCfg = H1_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    # joint_gears: list = [
    #     50.0,  # left_hip_yaw
    #     50.0,  # right_hip_yaw
    #     50.0,  # torso
    #     50.0,  # left_hip_roll
    #     50.0,  # right_hip_roll
    #     50.0,  # left_shoulder_pitch
    #     50.0,  # right_shoulder_pitch
    #     50.0,  # left_hip_pitch
    #     50.0,  # right_hip_pitch
    #     50.0,  # left_shoulder_roll
    #     50.0,  # right_shoulder_roll
    #     50.0,  # left_knee
    #     50.0,  # right_knee
    #     50.0,  # left_shoulder_yaw
    #     50.0,  # right_shoulder_yaw
    #     50.0,  # left_ankle
    #     50.0,  # right_ankle
    #     50.0,  # left_elbow
    #     50.0,  # right_elbow
    # ]


@configclass
class HumanoidAmpDanceEnvCfg(HumanoidAmpEnvCfg):
    motion_file = os.path.join(MOTIONS_DIR, "humanoid_dance.npz")


@configclass
class HumanoidAmpRunEnvCfg(HumanoidAmpEnvCfg):
    motion_file = os.path.join(MOTIONS_DIR, "humanoid_run.npz")


# 🎯 システム（Hydra/Gym）が探しに来る歩行タスクのクラス名を壊さずに残します
@configclass
class HumanoidAmpWalkEnvCfg(HumanoidAmpEnvCfg):
    # motion_file = os.path.join(MOTIONS_DIR, "humanoid_walk.npz")
    motion_file = os.path.join(MOTIONS_DIR, "h1_walk_for_isaaclab1.npz")