# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject  # キューブ（剛体）用のクラスに変更
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply, quat_apply_inverse, euler_xyz_from_quat, yaw_quat, quat_conjugate

from .unicycle_env_cfg import UnicycleEnvCfg
from tqdm import tqdm
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG, CUBOID_MARKER_CFG, SPHERE_MARKER_CFG # 必要に応じて
from isaaclab.sensors import Camera
from PIL import Image
from .heightmap_generator import HeightMapGenerator
from isaaclab.sensors.ray_caster import MultiMeshRayCaster
from .ray_heightmap_generator import RayHeightmapGenerator

class UnicycleEnv(DirectRLEnv):
    cfg: UnicycleEnvCfg

    def __init__(self, cfg: UnicycleEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        print("--- Unicycle Environment Initialized ---")

        # ユニサイクル探索用の変数を初期化
        self.goal_pos_w = torch.zeros(self.num_envs, 2, device=self.device)  # 指定座標 (x, y)
        self.current_goal_dist = torch.zeros(self.num_envs, device=self.device)
        self.heading_error = torch.zeros(self.num_envs, device=self.device)

        # ユニサイクルモデル用の動作入力 (例: 線速度 v と 角速度 omega)
        # アクションのスケールとオフセット（必要に応じて調整）
        self.action_scale_lin = 1.0   # 最大線速度 [m/s]
        self.action_scale_ang = 2.0   # 最大角速度 [rad/s]
        # ==========================================
        # ゴール表示用マーカーの設定と初期化
        # ==========================================
        marker_cfg = SPHERE_MARKER_CFG.copy()
        marker_cfg.prim_path = "/Visuals/GoalMarker"
        # マーカーのサイズを変更したい場合 (例: 半径20cmの球体)
        marker_cfg.markers["sphere"].scale = (0.4, 0.4, 0.4)
        # 色を赤色などに変更したい場合（オプション）
        # marker_cfg.markers["sphere"].visual_material.diffuse_color = (1.0, 0.0, 0.0)
        self.goal_marker = VisualizationMarkers(marker_cfg)
        # ==========================================
        # 障害物のサイズ
        # ==========================================
        self.obstacle_stage = 3
        self.obstacle_size = torch.tensor(
            self.cfg.obstacle1.spawn.size,
            device=self.device,
            dtype=torch.float32,
        ).view(1, 1, 3).repeat(self.num_envs, 3, 1)
        # ==========================================
        # ヒートマップ作成
        # ==========================================
        self.heightmap_generator = HeightMapGenerator(
            resolution=0.1,
            map_size=8.0,
            device=self.device,
        )
        self.ray_heightmap_generator = RayHeightmapGenerator(
            map_size=8.0,
            output_size=80,
            gui_enabled=True,
            gui_update_interval=10,
        )

    def _setup_scene(self):
        # キューブ（RigidObject）をロボットとしてスポーン
        # ※ cfg.robot にキューブのプリミティブ設定またはUSDパスが指定されている想定
        self.robot = RigidObject(self.cfg.robot)
        self.ray_caster = MultiMeshRayCaster(self.cfg.ray_caster)
        # self.camera = Camera(self.cfg.camera)
        self.obstacle1 = RigidObject(self.cfg.obstacle1)
        self.obstacle2 = RigidObject(self.cfg.obstacle2)
        self.obstacle3 = RigidObject(self.cfg.obstacle3)
        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.0,
                    dynamic_friction=1.0,
                    restitution=0.0,
                ),
            ),
        )
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])
        # シーンに剛体として登録
        self.scene.rigid_objects["robot"] = self.robot
        self.scene.sensors["ray_caster"] = self.ray_caster
        # self.scene.sensors["camera"] = self.camera
        self.scene.rigid_objects["obstacle1"] = self.obstacle1
        self.scene.rigid_objects["obstacle2"] = self.obstacle2
        self.scene.rigid_objects["obstacle3"] = self.obstacle3
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        # actions: [num_envs, 2] -> [線速度, 角速度] を想定
        self.actions = torch.clamp(actions, -1.0, 1.0)

    def _apply_action(self):
        # ユニサイクルモデルへの速度指令（Velocities）を直接適用
        v = self.actions[:, 0] * self.action_scale_lin
        omega = self.actions[:, 1] * self.action_scale_ang

        # 現在の向き（yaw）を取得
        root_rot_w = self.robot.data.root_quat_w
        _, _, robot_yaw = euler_xyz_from_quat(root_rot_w)

        # 2次元平面でのワールド速度成分へ変換 (vx = v * cos(yaw), vy = v * sin(yaw))
        vx = v * torch.cos(robot_yaw)
        vy = v * torch.sin(robot_yaw)

        # 根元（Root）の速度を設定 [vx, vy, vz=0] および [omega_x=0, omega_y=0, omega_z=omega]
        root_lin_vel = torch.stack([vx, vy, torch.zeros_like(vx)], dim=-1)
        root_ang_vel = torch.stack([torch.zeros_like(omega), torch.zeros_like(omega), omega], dim=-1)

        self.robot.write_root_com_velocity_to_sim(
            torch.cat([root_lin_vel, root_ang_vel], dim=-1)
        )

    def _get_observations(self) -> dict:
        # rgb = self.camera.data.output["rgb"][0].cpu().numpy()
        # print(rgb.shape)
        # print(rgb.dtype)
        # print(rgb.min(), rgb.max())
        # print(rgb.mean(axis=(0,1)))
        # depth = self.camera.data.output["distance_to_image_plane"]
        # Image.fromarray(rgb).save("/tmp/unicycle_camera.png")

        root_pos_w = self.robot.data.root_pos_w
        root_rot_w = self.robot.data.root_quat_w
        root_lin_vel_w = self.robot.data.root_lin_vel_w
        root_ang_vel_w = self.robot.data.root_ang_vel_w

        _, _, robot_yaw = euler_xyz_from_quat(root_rot_w)

        # 指定座標（ゴール）への相対ベクトルおよび方位誤差の計算
        goal_vec_w = self.goal_pos_w - root_pos_w[:, :2]
        self.current_goal_dist = torch.norm(goal_vec_w, dim=-1)

        # ==========================================
        # ゴールマーカーの描画位置を更新
        # ==========================================
        # region 2次元のゴール座標 (x, y) に、地面すれすれの高さ (z = 0.2 など) を付与して3次元座標にする
        marker_pos_w = torch.cat([
            self.goal_pos_w, 
            torch.zeros(self.num_envs, 1, device=self.device) + 0.2
        ], dim=-1)
        
        # マーカーの向き（デフォルトの向きでOKなのでクォータニオンを適当に作成）
        marker_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        
        # 可視化を更新
        self.goal_marker.visualize(marker_pos_w, marker_quat)

        target_yaw = torch.atan2(goal_vec_w[:, 1], goal_vec_w[:, 0])
        heading_error = target_yaw - robot_yaw
        heading_error = torch.atan2(torch.sin(heading_error), torch.cos(heading_error))
        self.heading_error = heading_error
        
        heading_sin = torch.sin(heading_error)
        heading_cos = torch.cos(heading_error)
        #endregion

        # ゴールまでのローカル相対座標（キューブ基準のX, Y）
        q_yaw_inv = quat_conjugate(yaw_quat(root_rot_w))
        goal_vec_local = quat_apply(
            q_yaw_inv, 
            torch.cat([goal_vec_w, torch.zeros(self.num_envs, 1, device=self.device)], dim=-1)
        )[:, :2]

        # ローカル速度への変換
        local_lin_vel = quat_apply_inverse(yaw_quat(root_rot_w), root_lin_vel_w)
        local_ang_vel = quat_apply_inverse(yaw_quat(root_rot_w), root_ang_vel_w)
        obstacle_positions = torch.stack(
            [
                self.obstacle1.data.root_pos_w,
                self.obstacle2.data.root_pos_w,
                self.obstacle3.data.root_pos_w,
            ],
            dim=1,
        )
        height_map = self.heightmap_generator.generate(
            root_pos_w,
            robot_yaw,
            obstacle_positions,
            self.obstacle_size,
        )
        # print("height_map shape:", height_map.shape)
        # print("height_map min:", height_map.min().item())
        # print("height_map max:", height_map.max().item())
        # height_map = height_map.flatten(start_dim=1)
        height_map = height_map.unsqueeze(1)  # (N, 1, 80, 80) そのままconv2dへ

        ray_data = self.scene.sensors["ray_caster"].data
        ray_hits_w = ray_data.ray_hits_w
        ray_heightmap = self.ray_heightmap_generator.generate(
            ray_hits_w
        )
        print("ray_heightmap:", ray_heightmap.shape)

        # ポリシー観測値の構築 (キューブの速度、姿勢、ゴールまでの相対位置・方位誤差など)
        policy_obs = torch.cat(
            (
                local_lin_vel,
                local_ang_vel,
                root_pos_w[:, 2:3],  # 高さ
                heading_sin.unsqueeze(-1),
                heading_cos.unsqueeze(-1),
                goal_vec_local,     # ゴールのローカル2次元座標
            ),
            dim=-1,
        )

        return {"policy": {"policy_obs": policy_obs, "heightmap": height_map}}

    def _get_rewards(self) -> torch.Tensor:
        root_pos_w = self.robot.data.root_pos_w
        obstacle_pos = torch.stack(
            [
                self.obstacle1.data.root_pos_w[:, :2],
                self.obstacle2.data.root_pos_w[:, :2],
                self.obstacle3.data.root_pos_w[:, :2],
            ],
            dim=1,
        )

        obstacle_dist = torch.norm(
            root_pos_w[:, None, :2] - obstacle_pos,
            dim=-1,
        )
        root_lin_vel = self.robot.data.root_lin_vel_w
        
        # 1. 距離報酬：ゴールに近づくほど高報酬
        distance_reward = torch.exp(-1.0 * self.current_goal_dist)

        # 2. ヘディング報酬：ゴールの方向を向いているほど高報酬
        heading_reward = torch.exp(-2.0 * self.heading_error ** 2)

        # 3. ゴール方向への進捗速度報酬
        goal_vec_w = self.goal_pos_w - root_pos_w[:, :2]
        goal_dir_w = goal_vec_w / (torch.norm(goal_vec_w, dim=-1, keepdim=True) + 1e-5)
        approach_speed = torch.sum(root_lin_vel[:, :2] * goal_dir_w, dim=-1)
        progress_reward = torch.clamp(approach_speed, min=0.0)

        # 4. 障害物との距離に応じたペナルティ（近づきすぎると負の報酬）
        min_obstacle_dist = obstacle_dist.min(dim=1).values

        collision_penalty = torch.where(
            min_obstacle_dist < 1.0,
            -1.0 * (1.0 - min_obstacle_dist),
            torch.zeros_like(min_obstacle_dist)
        )

        # 報酬の合成
        reward = (
            1.0 * distance_reward
            + 1.0 * progress_reward
            # + 0.5 * heading_reward
            + collision_penalty
        )

        if self.common_step_counter % 1000 == 0:
            tqdm.write(
                f"step={self.common_step_counter} "
                f"dist={self.current_goal_dist.mean().item():.3f} "
                f"progress={progress_reward.mean().item():.3f} "
                f"heading={heading_reward.mean().item():.3f}"
            )

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        # 転倒判定（キューブの高さが低くなりすぎた場合など）
        root_height = self.robot.data.root_pos_w[:, 2]
        died = root_height < 0.2

        # ゴールに十分近づいたら成功終了
        reached_goal = self.current_goal_dist < 0.3
        time_out |= reached_goal

        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]

        self.robot.write_root_link_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids)

        # リセット時にランダムな目標座標（ゴール）を生成
        # init_xy = root_state[:, :2]
        # rand_dist = torch.rand(len(env_ids), device=self.device) * 3.0 + 2.0
        # rand_angle = torch.rand(len(env_ids), device=self.device) * 2.0 * torch.pi - torch.pi
        # self.goal_pos_w[env_ids] = init_xy + torch.stack([
        #     rand_dist * torch.cos(rand_angle),
        #     rand_dist * torch.sin(rand_angle)
        # ], dim=-1)
        self.goal_pos_w[env_ids] = (
            self.scene.env_origins[env_ids, :2]
            + torch.tensor([5.0, 0.0], device=self.device)
        )

        # =====================
        # 障害物配置
        # =====================
        num_envs = len(env_ids)

        if self.obstacle_stage == 0:
            # 障害物なし
            far_away = 100.0
            for obstacle in [self.obstacle1, self.obstacle2, self.obstacle3]:
                obstacle_state = obstacle.data.default_root_state[env_ids].clone()
                obstacle_state[:, :3] = self.scene.env_origins[env_ids] + torch.tensor([far_away, 0.0, 0.25], device=self.device)
                obstacle.write_root_pose_to_sim(obstacle_state[:, :7], env_ids)

        elif self.obstacle_stage == 1:
            # 障害物1個・固定位置
            obstacle1_state = self.obstacle1.data.default_root_state[env_ids].clone()
            obstacle1_state[:, :3] = self.scene.env_origins[env_ids] + torch.tensor([2.5, 0.0, 0.25], device=self.device)
            self.obstacle1.write_root_pose_to_sim(obstacle1_state[:, :7], env_ids)
            for obstacle in [self.obstacle2, self.obstacle3]:
                obstacle_state = obstacle.data.default_root_state[env_ids].clone()
                obstacle_state[:, :3] = self.scene.env_origins[env_ids] + torch.tensor([100.0, 0.0, 0.25], device=self.device)
                obstacle.write_root_pose_to_sim(obstacle_state[:, :7], env_ids)

        elif self.obstacle_stage == 2:
            # 障害物1個・ランダム位置
            pos1 = torch.zeros((num_envs, 3), device=self.device)
            pos1[:, 0] = torch.empty(num_envs, device=self.device).uniform_(1.5, 3.5)
            pos1[:, 1] = torch.empty(num_envs, device=self.device).uniform_(-1.5, 1.5)
            pos1[:, 2] = 0.25
            obstacle1_state = self.obstacle1.data.default_root_state[env_ids].clone()
            obstacle1_state[:, :3] = pos1 + self.scene.env_origins[env_ids]
            self.obstacle1.write_root_pose_to_sim(obstacle1_state[:, :7], env_ids)
            for obstacle in [self.obstacle2, self.obstacle3]:
                obstacle_state = obstacle.data.default_root_state[env_ids].clone()
                obstacle_state[:, :3] = self.scene.env_origins[env_ids] + torch.tensor([100.0, 0.0, 0.25], device=self.device)
                obstacle.write_root_pose_to_sim(obstacle_state[:, :7], env_ids)

        elif self.obstacle_stage == 3:
            # 障害物3個・ランダム位置
            for obstacle in [self.obstacle1, self.obstacle2, self.obstacle3]:
                pos = torch.zeros((num_envs, 3), device=self.device)
                pos[:, 0] = torch.empty(num_envs, device=self.device).uniform_(-3.5, 3.5)
                pos[:, 1] = torch.empty(num_envs, device=self.device).uniform_(-3.5, 3.5)
                pos[:, 2] = 0.25
                obstacle_state = obstacle.data.default_root_state[env_ids].clone()
                obstacle_state[:, :3] = pos + self.scene.env_origins[env_ids]
                obstacle.write_root_pose_to_sim(obstacle_state[:, :7], env_ids)