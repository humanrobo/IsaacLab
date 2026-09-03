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
from pxr import UsdGeom

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
        # 報酬用変数
        # ==========================================
        self.obstacle_passed = torch.zeros((self.num_envs, 3), dtype=torch.bool, device=self.device)
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
        front_marker_cfg = SPHERE_MARKER_CFG.copy()
        front_marker_cfg.prim_path = "/Visuals/RobotFrontMarker"
        front_marker_cfg.markers["sphere"].scale = (0.15, 0.15, 0.15)
        self.front_marker = VisualizationMarkers(front_marker_cfg)
        # ==========================================
        # 障害物のサイズ
        # ==========================================
        self.obstacle_stage = 2
        # self.obstacle_radius = self.cfg.obstacle1.spawn.radius
        # self.obstacle_height = self.cfg.obstacle1.spawn.height
        # ==========================================
        # ヒートマップ作成
        # ==========================================
        self.heightmap_generator = HeightMapGenerator(
            resolution=0.1,
            map_size=8.0,
            gui_enabled=True,
            device=self.device,
        )
        self.ray_heightmap_generator = RayHeightmapGenerator(
            map_size=8.0,
            output_size=80,
            gui_enabled=False,
            gui_update_interval=10,
        )

    def _setup_scene(self):
        # キューブ（RigidObject）をロボットとしてスポーン
        # ※ cfg.robot にキューブのプリミティブ設定またはUSDパスが指定されている想定
        self.robot = RigidObject(self.cfg.robot)
        # self.ray_caster = MultiMeshRayCaster(self.cfg.ray_caster)
        self.camera = Camera(self.cfg.camera)
        self.obstacle1 = RigidObject(self.cfg.obstacle1)
        self.obstacle2 = RigidObject(self.cfg.obstacle2)
        self.obstacle3 = RigidObject(self.cfg.obstacle3)
        self.obstacle_long = RigidObject(self.cfg.obstacle_long)
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
        # self.scene.sensors["ray_caster"] = self.ray_caster
        self.scene.sensors["camera"] = self.camera
        self.scene.rigid_objects["obstacle1"] = self.obstacle1
        self.scene.rigid_objects["obstacle2"] = self.obstacle2
        self.scene.rigid_objects["obstacle3"] = self.obstacle3
        self.scene.rigid_objects["obstacle_long"] = self.obstacle_long
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

        front_offset = 0.8
        front_marker_pos = torch.stack([
            root_pos_w[:, 0] + front_offset * torch.cos(robot_yaw),
            root_pos_w[:, 1] + front_offset * torch.sin(robot_yaw),
            root_pos_w[:, 2] + 0.2 * torch.ones(self.num_envs, device=self.device),
        ], dim=-1)

        self.front_marker.visualize(front_marker_pos, marker_quat)

        # ゴールまでのローカル相対座標（キューブ基準のX, Y）
        q_yaw_inv = quat_conjugate(yaw_quat(root_rot_w))
        goal_vec_local = quat_apply(
            q_yaw_inv, 
            torch.cat([goal_vec_w, torch.zeros(self.num_envs, 1, device=self.device)], dim=-1)
        )[:, :2]

        # ローカル速度への変換
        local_lin_vel = quat_apply_inverse(yaw_quat(root_rot_w), root_lin_vel_w)
        # print("local_lin_vel:", local_lin_vel[0].detach().cpu().numpy())
        # print("yaw:", robot_yaw[0].item())
        local_ang_vel = quat_apply_inverse(yaw_quat(root_rot_w), root_ang_vel_w)
        # obstacle_positions = torch.stack(
        #     [
        #         self.obstacle1.data.root_pos_w,
        #         self.obstacle2.data.root_pos_w,
        #         self.obstacle3.data.root_pos_w,
        #     ],
        #     dim=1,
        # )
        # height_map = self.heightmap_generator.generate(
        #     root_pos_w,
        #     robot_yaw,
        #     obstacle_positions,
        #     self.obstacle_size,
        # )
        depth = self.camera.data.output["distance_to_image_plane"]
        height_map = self.heightmap_generator.generate_from_depth(
            depth,
            self.camera,
            root_pos_w,
            robot_yaw,
        )
        # # print("height_map shape:", height_map.shape)
        # print("height_map min:", height_map.min().item())
        # print("height_map max:", height_map.max().item())
        # height_map = height_map.flatten(start_dim=1)
        height_map = height_map.unsqueeze(1)  # (N, 1, 80, 80) そのままconv2dへ

        # ray_data = self.scene.sensors["ray_caster"].data
        # ray_hits_w = ray_data.ray_hits_w
        # ray_heightmap = self.ray_heightmap_generator.generate(
        #     ray_hits_w
        # )
        # print("ray_heightmap:", ray_heightmap.shape)

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

        return {"policy": {"policy_obs": policy_obs, "ray_heightmap": height_map}}

    def _get_rewards(self) -> torch.Tensor:
        root_pos_w = self.robot.data.root_pos_w
        root_lin_vel = self.robot.data.root_lin_vel_w
        root_lin_vel_w = self.robot.data.root_lin_vel_w
        root_rot_w = self.robot.data.root_quat_w
        local_lin_vel = quat_apply_inverse(yaw_quat(root_rot_w), root_lin_vel_w)
        # ==========================================
        # 障害物の中心位置
        obstacle_pos = torch.stack([
            self.obstacle1.data.root_pos_w[:, :2],
            self.obstacle2.data.root_pos_w[:, :2],
            self.obstacle3.data.root_pos_w[:, :2],
        ], dim=1)
        # robot_radius = self.cfg.robot.spawn.radius
        # obstacle_radius = self.cfg.obstacle1.spawn.radius
        # # 円柱同士の中心距離
        # center_dist = torch.norm(root_pos_w[:, None, :2] - obstacle_pos, dim=-1)
        # # 障害物表面までの距離
        # obstacle_surface_dist = torch.clamp(center_dist - robot_radius - obstacle_radius, min=0.0)
        # min_obstacle_dist = obstacle_surface_dist.min(dim=1).values
        # # 衝突判定
        # collision = (center_dist <= (robot_radius + obstacle_radius)).any(dim=1)
        # ゴール方向
        goal_vec_w = self.goal_pos_w - root_pos_w[:, :2]
        goal_dist = torch.norm(goal_vec_w, dim=-1, keepdim=True)
        goal_dir_w = goal_vec_w / (goal_dist + 1e-5)
        # ゴール方向への進捗
        approach_speed = torch.sum(root_lin_vel[:, :2] * goal_dir_w, dim=-1)
        progress_reward = torch.clamp(approach_speed, min=0.0)
        # 障害物への接近速度
        obstacle_vec = obstacle_pos - root_pos_w[:, None, :2]
        obstacle_dir = obstacle_vec / (torch.norm(obstacle_vec, dim=-1, keepdim=True) + 1e-5)
        obstacle_approach_speed = torch.sum(root_lin_vel[:, None, :2] * obstacle_dir, dim=-1)
        min_obstacle_approach_speed = obstacle_approach_speed.max(dim=1).values
        obstacle_approach_penalty = -torch.clamp(min_obstacle_approach_speed, min=0.0)
        # 障害物接近ペナルティ
        # obstacle_penalty = -torch.clamp(1.0 - min_obstacle_dist, min=0.0)
        # # 衝突ペナルティ
        # collision_penalty = torch.where(collision, torch.full_like(min_obstacle_dist, -2.0), torch.zeros_like(min_obstacle_dist))
        # # 障害物通過判定
        # relative_pos = root_pos_w[:, None, :2] - obstacle_pos
        # forward_dist = torch.sum(relative_pos * goal_dir_w[:, None, :], dim=-1)
        # passed = (forward_dist > obstacle_radius + robot_radius) & (obstacle_surface_dist > 0.1)
        # newly_passed = passed & (~self.obstacle_passed)
        # self.obstacle_passed |= passed
        # obstacle_pass_reward = newly_passed.float().sum(dim=1) * 5.0
        # 障害物通過判定
        relative_pos = root_pos_w[:, None, :2] - obstacle_pos
        forward_dist = torch.sum(relative_pos * goal_dir_w[:, None, :], dim=-1)
        passed = forward_dist > 0.25
        newly_passed = passed & (~self.obstacle_passed)
        self.obstacle_passed |= passed
        obstacle_pass_reward = newly_passed.float().sum(dim=1) * 5.0
        #  障害物前回転報酬
        root_ang_vel = self.robot.data.root_ang_vel_w
        # 障害物への接近速度
        obstacle_vec = obstacle_pos - root_pos_w[:, None, :2]
        obstacle_dist = torch.norm(obstacle_vec, dim=-1)
        obstacle_dir = obstacle_vec / (obstacle_dist.unsqueeze(-1) + 1e-5)
        obstacle_approach_speed = torch.sum(root_lin_vel[:, None, :2] * obstacle_dir, dim=-1)
        min_obstacle_approach_speed = obstacle_approach_speed.max(dim=1).values
        # 障害物が前方にあるか
        forward_dist = torch.sum(obstacle_vec * goal_dir_w[:, None, :], dim=-1)
        front_obstacle = ((forward_dist > 0.0) & (forward_dist < 1.5)).any(dim=1)
        # 旋回しているか
        turning = torch.abs(root_ang_vel[:, 2]) > 0.1
        # 障害物への接近速度が減っているか
        avoiding = min_obstacle_approach_speed < 0.0
        # 障害物回避旋回報酬
        obstacle_turn_reward = (front_obstacle & turning & avoiding).float() * 0.3
        # ゴール到達
        goal_reached = self.current_goal_dist < 0.3
        goal_reward = goal_reached.float() * 50.0
        # 時間ボーナス
        time_bonus = goal_reached.float() * (1.0 - self.episode_length_buf / self.max_episode_length) * 10.0
        #前進速度報酬
        forward_reward = 0.1 * torch.clamp(local_lin_vel[:, 0], min=0.0)

        # ==========================================
        # 報酬合成
        # ==========================================
        reward = (
            0.1 * progress_reward
            # + obstacle_penalty
            # + obstacle_approach_penalty
            # + collision_penalty
            + obstacle_turn_reward
            + obstacle_pass_reward #一個についき5
            + goal_reward #50
            + time_bonus #約5
            + forward_reward
        )

        if self.common_step_counter % 1000 == 0:
            tqdm.write(
                f"step={self.common_step_counter} "
                f"dist={self.current_goal_dist.mean().item():.3f} "
                f"progress={progress_reward.mean().item():.3f} "
                f"goal_dir=({goal_dir_w[:, 0].mean().item():.3f}, "
                f"{goal_dir_w[:, 1].mean().item():.3f}) "
                f"vel=({root_lin_vel[:, 0].mean().item():.3f}, "
                f"{root_lin_vel[:, 1].mean().item():.3f}) "
                f"approach={approach_speed.mean().item():.3f}"
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
        # yaw = torch.empty(len(env_ids), device=self.device).uniform_(-torch.pi, torch.pi)
        yaw = torch.zeros(len(env_ids), device=self.device)
        root_state[:, 3] = torch.cos(yaw / 2.0)
        root_state[:, 4] = 0.0
        root_state[:, 5] = 0.0
        root_state[:, 6] = torch.sin(yaw / 2.0)
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
            + torch.tensor([6.0, 0.0], device=self.device)
        )

        # =====================
        # 障害物配置
        # =====================
        num_envs = len(env_ids)

        if self.obstacle_stage == 0:
            # 障害物なし
            far_away = 100.0
            for obstacle in [self.obstacle1, self.obstacle2, self.obstacle3, self.obstacle_long]:
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
            # 障害物2個・robotから2m先に配置
            robot_pos = self.robot.data.root_pos_w[env_ids, :2]
            goal_pos = self.goal_pos_w[env_ids, :2]
            # robot → goal方向
            direction = goal_pos - robot_pos
            direction = direction / (torch.norm(direction, dim=-1, keepdim=True) + 1e-5)
            # robot → goal方向に対して垂直な方向
            lateral = torch.stack([-direction[:, 1], direction[:, 0]], dim=-1)
            # robotから2m先を基準にする
            mid_pos = robot_pos + direction * 2.5
            # 障害物1：前後-0.3、左右+0.4
            obstacle1_pos = mid_pos + direction * (-1.5) + lateral * 0.4
            # 障害物2：前後+0.3、左右-0.4
            obstacle2_pos = mid_pos + direction * 1.5 + lateral * (-0.4)
            obstacle1_state = self.obstacle1.data.default_root_state[env_ids].clone()
            obstacle1_state[:, :2] = obstacle1_pos
            obstacle1_state[:, 2] = 0.25
            self.obstacle1.write_root_pose_to_sim(obstacle1_state[:, :7], env_ids)
            obstacle2_state = self.obstacle2.data.default_root_state[env_ids].clone()
            obstacle2_state[:, :2] = obstacle2_pos
            obstacle2_state[:, 2] = 0.25
            self.obstacle2.write_root_pose_to_sim(obstacle2_state[:, :7], env_ids)
            # 障害物3は使用しない
            obstacle3_state = self.obstacle3.data.default_root_state[env_ids].clone()
            obstacle3_state[:, :3] = self.scene.env_origins[env_ids] + torch.tensor([100.0, 0.0, 0.25], device=self.device)
            self.obstacle3.write_root_pose_to_sim(obstacle3_state[:, :7], env_ids)
            obstaclelong_state = self.obstacle_long.data.default_root_state[env_ids].clone()
            obstaclelong_state[:, :3] = self.scene.env_origins[env_ids] + torch.tensor([100.0, 0.0, 0.25], device=self.device)
            self.obstacle_long.write_root_pose_to_sim(obstaclelong_state[:, :7], env_ids)

        elif self.obstacle_stage == 3:
            obstacle1_state = self.obstacle1.data.default_root_state[env_ids].clone()
            obstacle1_state[:, :3] = self.scene.env_origins[env_ids] + torch.tensor([2.5, 0.0, 0.25], device=self.device)
            self.obstacle1.write_root_pose_to_sim(obstacle1_state[:, :7], env_ids)
            obstacle_radius = torch.empty(
                len(env_ids),
                device=self.device,
            ).uniform_(0.25, 0.75)
            for i, env_id in enumerate(env_ids):
                prim_path = f"/World/envs/env_{env_id.item()}/Obstacle1"
                prim = self.sim.stage.GetPrimAtPath(prim_path)
                xform = UsdGeom.Xformable(prim)
                for op in xform.GetOrderedXformOps():
                    if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                        scale = obstacle_radius[i].item() / self.cfg.obstacle1.spawn.radius
                        op.Set((scale, scale, 1.0))
                        break
            for obstacle in [self.obstacle2, self.obstacle3]:
                obstacle_state = obstacle.data.default_root_state[env_ids].clone()
                obstacle_state[:, :3] = self.scene.env_origins[env_ids] + torch.tensor([100.0, 0.0, 0.25], device=self.device)
                obstacle.write_root_pose_to_sim(obstacle_state[:, :7], env_ids)

        elif self.obstacle_stage == 4:
            # 障害物1個・ランダム位置
            pos1 = torch.zeros((num_envs, 3), device=self.device)
            pos1[:, 0] = torch.empty(num_envs, device=self.device).uniform_(-3.5, 3.5)
            pos1[:, 1] = torch.empty(num_envs, device=self.device).uniform_(-3.5, 3.5)
            pos1[:, 2] = 0.25
            obstacle1_state = self.obstacle1.data.default_root_state[env_ids].clone()
            obstacle1_state[:, :3] = pos1 + self.scene.env_origins[env_ids]
            self.obstacle1.write_root_pose_to_sim(obstacle1_state[:, :7], env_ids)
            for obstacle in [self.obstacle2, self.obstacle3]:
                obstacle_state = obstacle.data.default_root_state[env_ids].clone()
                obstacle_state[:, :3] = self.scene.env_origins[env_ids] + torch.tensor([100.0, 0.0, 0.25], device=self.device)
                obstacle.write_root_pose_to_sim(obstacle_state[:, :7], env_ids)

        elif self.obstacle_stage == 5:
            # 障害物3個・ランダム位置
            for obstacle in [self.obstacle1, self.obstacle2, self.obstacle3, self.obstacle_long]:
                pos = torch.zeros((num_envs, 3), device=self.device)
                pos[:, 0] = torch.empty(num_envs, device=self.device).uniform_(-3.5, 3.5)
                pos[:, 1] = torch.empty(num_envs, device=self.device).uniform_(-3.5, 3.5)
                pos[:, 2] = 0.25
                obstacle_state = obstacle.data.default_root_state[env_ids].clone()
                obstacle_state[:, :3] = pos + self.scene.env_origins[env_ids]
                obstacle.write_root_pose_to_sim(obstacle_state[:, :7], env_ids)

        elif self.obstacle_stage == 6:
            # 横長障害物1個・固定位置
            obstacle_long_state = self.obstacle_long.data.default_root_state[env_ids].clone()
            obstacle_long_state[:, :3] = self.scene.env_origins[env_ids] + torch.tensor([2.5, 0.0, 0.25], device=self.device)
            self.obstacle_long.write_root_pose_to_sim(obstacle_long_state[:, :7], env_ids)
            for obstacle in [self.obstacle1, self.obstacle2, self.obstacle3]:
                obstacle_state = obstacle.data.default_root_state[env_ids].clone()
                obstacle_state[:, :3] = self.scene.env_origins[env_ids] + torch.tensor([100.0, 0.0, 0.25], device=self.device)
                obstacle.write_root_pose_to_sim(obstacle_state[:, :7], env_ids)

        self.obstacle_passed[env_ids] = False