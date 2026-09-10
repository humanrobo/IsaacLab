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
from pxr import Gf, UsdGeom, UsdPhysics
import ast

class UnicycleEnv(DirectRLEnv):
    cfg: UnicycleEnvCfg

    def __init__(self, cfg: UnicycleEnvCfg, render_mode: str | None = None, **kwargs):
        self.obstacle_stage = 9
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
        self.prev_root_pos = torch.zeros(
            (self.num_envs, 3),
            device=self.device
        )
        self.stuck_count = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.need_avoid_reward = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.stop_count = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
        )
        self.stop_threshold = int(
            1.0 / (self.cfg.sim.dt * self.cfg.decimation)
        )
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
        # self.obstacle_radius = self.cfg.obstacle1.spawn.radius
        # self.obstacle_height = self.cfg.obstacle1.spawn.height
        # ==========================================
        # セマセグ用
        # ==========================================
        self.pushable_color = None
        # ==========================================
        # ヒートマップ作成
        # ==========================================
        self.heightmap_generator = HeightMapGenerator(
            resolution=0.05,
            map_size=3.2,
            gui_enabled=True,
            device=self.device,
        )
        self.ray_heightmap_generator = RayHeightmapGenerator(
            map_size=3.2,
            output_size=64,
            gui_enabled=False,
            gui_update_interval=10,
        )

    def _setup_scene(self):
        self.robot = RigidObject(self.cfg.robot)
        self.camera = Camera(self.cfg.camera)
        # self.ray_caster = MultiMeshRayCaster(self.cfg.ray_caster)
        if self.obstacle_stage in [1, 2, 3, 4, 5, 7, 9]:
            self.obstacle1 = RigidObject(self.cfg.obstacle1)
            self.scene.rigid_objects["obstacle1"] = self.obstacle1
        if self.obstacle_stage in [2, 5, 7, 9]:
            self.obstacle2 = RigidObject(self.cfg.obstacle2)
            self.scene.rigid_objects["obstacle2"] = self.obstacle2
        if self.obstacle_stage == 5:
            self.obstacle3 = RigidObject(self.cfg.obstacle3)
            self.scene.rigid_objects["obstacle3"] = self.obstacle3
        if self.obstacle_stage in [5, 6]:
            self.obstacle_long = RigidObject(self.cfg.obstacle_long)
            self.scene.rigid_objects["obstacle_long"] = self.obstacle_long
        if self.obstacle_stage in [7, 8, 9]:
            self.obstacle_wallr = RigidObject(self.cfg.obstacle_wallr)
            self.scene.rigid_objects["obstacle_wallr"] = self.obstacle_wallr
        if self.obstacle_stage in [7, 8, 9]:
            self.obstacle_walll = RigidObject(self.cfg.obstacle_walll)
            self.scene.rigid_objects["obstacle_walll"] = self.obstacle_walll
        if self.obstacle_stage in [8, 9]:
            self.obstacle_pushable = RigidObject(self.cfg.obstacle_pushable)
            self.scene.rigid_objects["obstacle_pushable"] = self.obstacle_pushable
        stage = self.sim.stage
        barrier_path = "/World/envs/env_0/Robot/Barrier"
        barrier = UsdGeom.Cube.Define(stage, barrier_path)
        barrier.CreateSizeAttr(1.0)
        xform = UsdGeom.Xformable(barrier.GetPrim())
        xform.AddTranslateOp().Set(Gf.Vec3d(0.30, 0.0, 0.25))
        xform.AddScaleOp().Set(Gf.Vec3d(0.025, 0.25, 0.25))
        UsdPhysics.CollisionAPI.Apply(barrier.GetPrim())
        UsdGeom.Imageable(barrier.GetPrim()).MakeInvisible()

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
        self.scene.rigid_objects["robot"] = self.robot
        self.scene.sensors["camera"] = self.camera
        # self.scene.sensors["ray_caster"] = self.ray_caster
        light_cfg = sim_utils.DomeLightCfg(
            intensity=2000.0,
            color=(0.75, 0.75, 0.75),
        )
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        # actions: [num_envs, 2] -> [線速度, 角速度] を想定
        self.actions = torch.clamp(actions, -1.0, 1.0)
        #semseg各クラスがID割り当てられたときに、pushableのIDを探す
        if self.pushable_color is None:
            semantic_info = self.camera.data.info[0]["semantic_segmentation"]["idToLabels"]
            self.pushable_color = next(
                ast.literal_eval(k) for k, v in semantic_info.items()
                if v.get("class") == "pushable"
            )

    def _apply_action(self):
        # ユニサイクルモデルへの速度指令（Velocities）を直接適用
        v = self.actions[:, 0] * self.action_scale_lin
        omega = self.actions[:, 1] * self.action_scale_ang

        self.current_v = v

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
        local_ang_vel = quat_apply_inverse(yaw_quat(root_rot_w), root_ang_vel_w)
        semantic = self.camera.data.output["semantic_segmentation"]
        depth = self.camera.data.output["distance_to_image_plane"]
        height_map = self.heightmap_generator.generate_from_depth(
            depth,
            self.camera,
            root_pos_w,
            robot_yaw,
            semantic,
            self.pushable_color,
        )
        height_map = height_map.unsqueeze(1)  # (N, 1, 80, 80) そのままconv2dへ

        # ray_data = self.scene.sensors["ray_caster"].data
        # ray_hits_w = ray_data.ray_hits_w
        # # ray_heightmap = self.ray_heightmap_generator.generate(
        # #     ray_hits_w
        # # )
        # ray_heightmap = self.heightmap_generator.generate_from_ray(
        #     ray_hits_w,
        #     root_pos_w,
        #     robot_yaw
        # )
        # print("ray_heightmap:", ray_heightmap.shape)

        # ポリシー観測値の構築 (キューブの速度、姿勢、ゴールまでの相対位置・方位誤差など)
        policy_obs = torch.cat(
            (
                local_lin_vel,
                local_ang_vel,
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
        obstacle_pos_list = []
        if self.obstacle_stage in [1, 2, 3, 4, 5, 7, 9]:
            obstacle_pos_list.append(self.obstacle1.data.root_pos_w[:, :2])
        if self.obstacle_stage in [2, 5, 7, 9]:
            obstacle_pos_list.append(self.obstacle2.data.root_pos_w[:, :2])
        if self.obstacle_stage == 5:
            obstacle_pos_list.append(self.obstacle3.data.root_pos_w[:, :2])
        if self.obstacle_stage in [5, 6]:
            obstacle_pos_list.append(self.obstacle_long.data.root_pos_w[:, :2])
        if self.obstacle_stage in [8, 9]:
            obstacle_pos_list.append(self.obstacle_pushable.data.root_pos_w[:, :2])
        obstacle_pos = torch.stack(obstacle_pos_list, dim=1)
        robot_radius = self.cfg.robot.spawn.radius
        obstacle_radius = 0.3#self.cfg.obstacle1.spawn.radius
        # 円柱同士の中心距離
        center_dist = torch.norm(root_pos_w[:, None, :2] - obstacle_pos, dim=-1)
        # 障害物表面までの距離
        obstacle_surface_dist = torch.clamp(center_dist - robot_radius - obstacle_radius, min=0.0)
        min_obstacle_dist = obstacle_surface_dist.min(dim=1).values
        # 衝突判定
        collision = (center_dist <= (robot_radius + obstacle_radius)).any(dim=1)
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
        obstacle_penalty = -torch.clamp(1.0 - min_obstacle_dist, min=0.0)
        # # 衝突ペナルティ
        collision_penalty = torch.where(collision, torch.full_like(min_obstacle_dist, -2.0), torch.zeros_like(min_obstacle_dist))
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
        obstacle_pass_reward = newly_passed.float().sum(dim=1)
        obstacle_pass_reward = torch.clamp(obstacle_pass_reward, max=1.0)
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
        obstacle_turn_reward = (front_obstacle & turning & avoiding).float()
        # ゴール到達
        goal_reached = self.current_goal_dist < 0.3
        goal_reward = goal_reached.float()
        # 時間ボーナス
        time_bonus = goal_reached.float() * (1.0 - self.episode_length_buf / self.max_episode_length)
        #前進速度報酬
        forward_reward = torch.clamp(local_lin_vel[:, 0], min=0.0)
        #回転速度報酬
        turn_reward = (torch.abs(self.actions[:, 1]) > 0.2).float()
        # 前進しようとしているのに進めない
        trying_forward = self.current_v > 0.1
        # 前stepからほとんど位置が変わっていないか
        pos_change = torch.norm(
            root_pos_w[:, :2] - self.prev_root_pos[:, :2],
            dim=-1
        )
        not_moving = pos_change < 0.01
        # 前進指令を出しているのに位置が動かない
        stuck = trying_forward & not_moving
        self.stop_count = torch.where(
            stuck,
            self.stop_count + 1,
            torch.zeros_like(self.stop_count),
        )
        # 1秒以上その場で停止
        stopped_too_long = self.stop_count >= self.stop_threshold
        stuck_penalty = stopped_too_long.float() 
        self.stuck_count = torch.where(
            stuck,
            self.stuck_count + 1,
            torch.zeros_like(self.stuck_count),
        )
        stuck_trigger = self.stuck_count >= 5
        self.need_avoid_reward |= stuck_trigger
        # 今回の位置を次回比較用に保存
        self.prev_root_pos[:] = root_pos_w
        # 旋回 or 後進
        turning = torch.abs(root_ang_vel[:, 2]) > 0.2
        backing = local_lin_vel[:, 0] < -0.05
        avoid_reward_trigger = (
            self.need_avoid_reward
            & (turning | backing)
        )
        avoid_reward = avoid_reward_trigger.float()
        # 実際に動き始めたら解除
        escaped = pos_change > 0.02
        self.need_avoid_reward &= ~escaped

        # ==========================================
        # 報酬合成
        # ==========================================
        reward = (
            1.0 * progress_reward
            + 0.2 * turn_reward
            + obstacle_penalty
            + obstacle_approach_penalty
            + collision_penalty
            + 0.3 * obstacle_turn_reward
            + 5.0 * obstacle_pass_reward #一個についき5
            + 2.0 * avoid_reward
            + 50.0 * goal_reward #50
            + 10.0 * time_bonus #約5
            + 0.1 * forward_reward
            + stuck_penalty * -5.0
        )

        if self.common_step_counter % 1000 == 0:
            tqdm.write(
                f"progress={progress_reward.mean().item():.3f} "
                f"turn={turn_reward.mean().item():.3f} "
                f"obstacle={obstacle_penalty.mean().item():.3f} "
                f"obstacle_approach={obstacle_approach_penalty.mean().item():.3f} "
                f"collision={collision_penalty.mean().item():.3f} "
                f"obstacle_turn={obstacle_turn_reward.mean().item():.3f} "
                f"obstacle_pass={obstacle_pass_reward.mean().item():.3f} "
                f"avoid={avoid_reward.mean().item():.3f} "
                f"goal={goal_reward.mean().item():.3f} "
                f"time={time_bonus.mean().item():.3f} "
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

        # 1秒以上、前進しようとしているのに動かない
        stopped_too_long = self.stop_count >= self.stop_threshold
        terminated = died | stopped_too_long

        return terminated, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)
        self.heightmap_generator.reset(env_ids)


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
            pass

        elif self.obstacle_stage == 1:
            # 障害物1個・固定位置
            obstacle1_state = self.obstacle1.data.default_root_state[env_ids].clone()
            obstacle1_state[:, :3] = self.scene.env_origins[env_ids] + torch.tensor([2.5, 0.0, 0.25], device=self.device)
            self.obstacle1.write_root_pose_to_sim(obstacle1_state[:, :7], env_ids)

        elif self.obstacle_stage == 2:
            # 障害物2個・robotから3m先に配置
            robot_pos = self.robot.data.root_pos_w[env_ids, :2]
            goal_pos = self.goal_pos_w[env_ids, :2]
            direction = goal_pos - robot_pos
            direction = direction / (torch.norm(direction, dim=-1, keepdim=True) + 1e-5)
            lateral = torch.stack([-direction[:, 1], direction[:, 0]], dim=-1)
            mid_pos = robot_pos + direction * 3.0
            obstacle1_pos = mid_pos + direction * (-1.0) + lateral * 0.4
            obstacle2_pos = mid_pos + direction * 1.0 + lateral * (-0.4)
            obstacle1_state = self.obstacle1.data.default_root_state[env_ids].clone()
            obstacle1_state[:, :2] = obstacle1_pos
            obstacle1_state[:, 2] = 0.25
            self.obstacle1.write_root_pose_to_sim(obstacle1_state[:, :7], env_ids)
            obstacle2_state = self.obstacle2.data.default_root_state[env_ids].clone()
            obstacle2_state[:, :2] = obstacle2_pos
            obstacle2_state[:, 2] = 0.25
            self.obstacle2.write_root_pose_to_sim(obstacle2_state[:, :7], env_ids)

        elif self.obstacle_stage == 3:
            # 障害物1個・ランダム半径
            obstacle1_state = self.obstacle1.data.default_root_state[env_ids].clone()
            obstacle1_state[:, :3] = self.scene.env_origins[env_ids] + torch.tensor([2.5, 0.0, 0.25], device=self.device)
            self.obstacle1.write_root_pose_to_sim(obstacle1_state[:, :7], env_ids)
            obstacle_radius = torch.empty(len(env_ids), device=self.device).uniform_(0.25, 0.75)
            for i, env_id in enumerate(env_ids):
                prim_path = f"/World/envs/env_{env_id.item()}/Obstacle1"
                prim = self.sim.stage.GetPrimAtPath(prim_path)
                xform = UsdGeom.Xformable(prim)
                for op in xform.GetOrderedXformOps():
                    if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                        scale = obstacle_radius[i].item() / self.cfg.obstacle1.spawn.radius
                        op.Set((scale, scale, 1.0))
                        break

        elif self.obstacle_stage == 4:
            # 障害物1個・ランダム位置
            pos1 = torch.zeros((num_envs, 3), device=self.device)
            pos1[:, 0] = torch.empty(num_envs, device=self.device).uniform_(0.5, 3.5)
            pos1[:, 1] = torch.empty(num_envs, device=self.device).uniform_(-1.5, 1.5)
            pos1[:, 2] = 0.25
            obstacle1_state = self.obstacle1.data.default_root_state[env_ids].clone()
            obstacle1_state[:, :3] = pos1 + self.scene.env_origins[env_ids]
            self.obstacle1.write_root_pose_to_sim(obstacle1_state[:, :7], env_ids)

        elif self.obstacle_stage == 5:
            # 障害物4個・ランダム位置
            for obstacle in [self.obstacle1, self.obstacle2, self.obstacle3, self.obstacle_long]:
                pos = torch.zeros((num_envs, 3), device=self.device)
                pos[:, 0] = torch.empty(num_envs, device=self.device).uniform_(-0.5, 4.5)
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

        elif self.obstacle_stage == 7:
            # 左右の壁＋障害物1個・ランダム位置
            for obstacle in [self.obstacle1, self.obstacle2]:
                pos = torch.zeros((num_envs, 3), device=self.device)
                pos[:, 0] = torch.empty(num_envs, device=self.device).uniform_(1.0, 5.0)
                pos[:, 1] = torch.empty(num_envs, device=self.device).uniform_(-1.5, 1.5)
                pos[:, 2] = 0.25
                obstacle_state = obstacle.data.default_root_state[env_ids].clone()
                obstacle_state[:, :3] = pos + self.scene.env_origins[env_ids]
                obstacle.write_root_pose_to_sim(obstacle_state[:, :7], env_ids)

            for wall, y in [(self.obstacle_wallr, -2.0), (self.obstacle_walll, 2.0)]:
                wall_state = wall.data.default_root_state[env_ids].clone()
                wall_state[:, :3] = torch.tensor([3.0, y, 0.25], device=self.device) + self.scene.env_origins[env_ids]
                wall.write_root_pose_to_sim(wall_state[:, :7], env_ids)

        elif self.obstacle_stage == 8:
            # 左右の壁＋PushableObstacleを中央に配置
            pos = torch.tensor([3.0, 0.0, 0.1], device=self.device).repeat(num_envs, 1)
            pos += self.scene.env_origins[env_ids]
            obstacle_state = self.obstacle_pushable.data.default_root_state[env_ids].clone()
            obstacle_state[:, :3] = pos
            self.obstacle_pushable.write_root_pose_to_sim(obstacle_state[:, :7], env_ids)
            for wall, y in [(self.obstacle_wallr, -2.0), (self.obstacle_walll, 2.0)]:
                wall_state = wall.data.default_root_state[env_ids].clone()
                wall_state[:, :3] = torch.tensor([3.0, y, 0.25], device=self.device) + self.scene.env_origins[env_ids]
                wall.write_root_pose_to_sim(wall_state[:, :7], env_ids)

        elif self.obstacle_stage == 9:
            # 左右の壁＋障害物1個・ランダム位置
            for obstacle in [self.obstacle1, self.obstacle2, self.obstacle_pushable]:
                pos = torch.zeros((num_envs, 3), device=self.device)
                pos[:, 0] = torch.empty(num_envs, device=self.device).uniform_(1.0, 5.0)
                pos[:, 1] = torch.empty(num_envs, device=self.device).uniform_(-1.5, 1.5)
                pos[:, 2] = 0.25
                obstacle_state = obstacle.data.default_root_state[env_ids].clone()
                obstacle_state[:, :3] = pos + self.scene.env_origins[env_ids]
                obstacle.write_root_pose_to_sim(obstacle_state[:, :7], env_ids)

            for wall, y in [(self.obstacle_wallr, -2.0), (self.obstacle_walll, 2.0)]:
                wall_state = wall.data.default_root_state[env_ids].clone()
                wall_state[:, :3] = torch.tensor([3.0, y, 0.25], device=self.device) + self.scene.env_origins[env_ids]
                wall.write_root_pose_to_sim(wall_state[:, :7], env_ids)

        self.obstacle_passed[env_ids] = False
        self.prev_root_pos[env_ids] = self.robot.data.root_pos_w[env_ids]
        self.stuck_count[env_ids] = 0
        self.need_avoid_reward[env_ids] = False
        self.stop_count[env_ids] = 0