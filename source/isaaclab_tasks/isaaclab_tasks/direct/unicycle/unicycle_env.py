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
from isaaclab.sim import UsdFileCfg

class UnicycleEnv(DirectRLEnv):
    cfg: UnicycleEnvCfg

    def __init__(self, cfg: UnicycleEnvCfg, render_mode: str | None = None, **kwargs):
        self.obstacle_stage = 10
        super().__init__(cfg, render_mode, **kwargs)
        self.obstacle_stages = torch.full(
            (self.num_envs,),
            -1,
            dtype=torch.long,
            device=self.device,
        )
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
        self.obstacle_passed = torch.zeros((self.num_envs, 2), dtype=torch.bool, device=self.device)
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
        self.safe_stop_count = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
        )
        self.safe_stop_done = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )
        self.best_goal_dist = torch.zeros(
            self.num_envs,
            device=self.device,
        )
        self.no_progress_count = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
        )
        self.prev_goal_valid = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )
        self.no_progress_threshold = 20
        self.safe_stop_threshold = 10  # action[0] ≈ 0
        #ヒートマップ用 パーソナルスペース用の距離重み
        H = 64
        W = 64
        yy, xx = torch.meshgrid(
            torch.arange(H, device=self.device),
            torch.arange(W, device=self.device),
            indexing="ij"
        )
        cx, cy = W / 2.0, H / 2.0
        dist = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        personal_radius = min(H, W) * 0.25 #64の16pixelがパーソナルスペース
        self.personal_weight = torch.clamp(
            1.0 - dist / personal_radius,
            min=0.0
        )
        # region Episode累積報酬の初期化（__init__）
        self.episode_progress_reward = torch.zeros(self.num_envs, device=self.device)
        self.episode_goal_reward = torch.zeros(self.num_envs, device=self.device)
        self.episode_time_bonus = torch.zeros(self.num_envs, device=self.device)
        self.episode_forward_reward = torch.zeros(self.num_envs, device=self.device)
        self.episode_turn_reward = torch.zeros(self.num_envs, device=self.device)
        self.episode_back_reward = torch.zeros(self.num_envs, device=self.device)
        self.episode_obstacle_penalty = torch.zeros(self.num_envs, device=self.device)
        self.episode_obstacle_approach_penalty = torch.zeros(self.num_envs, device=self.device)
        self.episode_collision_penalty = torch.zeros(self.num_envs, device=self.device)
        self.episode_obstacle_turn_reward = torch.zeros(self.num_envs, device=self.device)
        self.episode_obstacle_pass_reward = torch.zeros(self.num_envs, device=self.device)
        self.episode_obstacle_space_penalty = torch.zeros(self.num_envs, device=self.device)
        self.episode_safe_reward = torch.zeros(self.num_envs, device=self.device)
        self.episode_blocked_speed_reward = torch.zeros(self.num_envs, device=self.device)
        self.episode_avoid_reward = torch.zeros(self.num_envs, device=self.device)
        self.episode_stuck_penalty = torch.zeros(self.num_envs, device=self.device)
        self.last_episode_rewards = None
        #endregion
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
            num_envs=self.num_envs,
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
        if self.obstacle_stage in [1, 2, 3, 4, 5, 7, 9, 10]:
            self.obstacle1 = RigidObject(self.cfg.obstacle1)
            self.scene.rigid_objects["obstacle1"] = self.obstacle1
        if self.obstacle_stage in [2, 5, 7, 9, 10]:
            self.obstacle2 = RigidObject(self.cfg.obstacle2)
            self.scene.rigid_objects["obstacle2"] = self.obstacle2
        if self.obstacle_stage == 5:
            self.obstacle3 = RigidObject(self.cfg.obstacle3)
            self.scene.rigid_objects["obstacle3"] = self.obstacle3
        if self.obstacle_stage in [5, 6, 11]:
            self.obstacle_long = RigidObject(self.cfg.obstacle_long)
            self.scene.rigid_objects["obstacle_long"] = self.obstacle_long
        if self.obstacle_stage in [7, 8, 9, 10, 11, 12]:
            self.obstacle_wallr = RigidObject(self.cfg.obstacle_wallr)
            self.scene.rigid_objects["obstacle_wallr"] = self.obstacle_wallr
        if self.obstacle_stage in [7, 8, 9, 10, 11, 12]:
            self.obstacle_walll = RigidObject(self.cfg.obstacle_walll)
            self.scene.rigid_objects["obstacle_walll"] = self.obstacle_walll
        if self.obstacle_stage in [12]:
            self.obstacle_wallf = RigidObject(self.cfg.obstacle_wallf)
            self.scene.rigid_objects["obstacle_wallf"] = self.obstacle_wallf
        if self.obstacle_stage in [12]:
            self.obstacle_wallb = RigidObject(self.cfg.obstacle_wallb)
            self.scene.rigid_objects["obstacle_wallb"] = self.obstacle_wallb
        if self.obstacle_stage in [8, 9, 10]:
            self.obstacle_pushable1 = RigidObject(self.cfg.obstacle_pushable1)
            self.scene.rigid_objects["obstacle_pushable1"] = self.obstacle_pushable1
        if self.obstacle_stage in [10]:
            self.obstacle_pushable2 = RigidObject(self.cfg.obstacle_pushable2)
            self.scene.rigid_objects["obstacle_pushable2"] = self.obstacle_pushable2
        if self.obstacle_stage in [10]:
            self.obstacle_pushable3 = RigidObject(self.cfg.obstacle_pushable3)
            self.scene.rigid_objects["obstacle_pushable3"] = self.obstacle_pushable3
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

    # def _setup_scene(self):
    #     self.robot = RigidObject(self.cfg.robot)
    #     # self.camera = Camera(self.cfg.camera)
    #     self.ray_caster = MultiMeshRayCaster(self.cfg.ray_caster)
    #     self.obstacle1 = RigidObject(self.cfg.obstacle1)
    #     self.scene.rigid_objects["obstacle1"] = self.obstacle1
    #     self.obstacle2 = RigidObject(self.cfg.obstacle2)
    #     self.scene.rigid_objects["obstacle2"] = self.obstacle2
    #     self.obstacle3 = RigidObject(self.cfg.obstacle3)
    #     self.scene.rigid_objects["obstacle3"] = self.obstacle3
    #     self.obstacle_long = RigidObject(self.cfg.obstacle_long)
    #     self.scene.rigid_objects["obstacle_long"] = self.obstacle_long
    #     self.obstacle_wallr = RigidObject(self.cfg.obstacle_wallr)
    #     self.scene.rigid_objects["obstacle_wallr"] = self.obstacle_wallr
    #     self.obstacle_walll = RigidObject(self.cfg.obstacle_walll)
    #     self.scene.rigid_objects["obstacle_walll"] = self.obstacle_walll
    #     self.obstacle_wallf = RigidObject(self.cfg.obstacle_wallf)
    #     self.scene.rigid_objects["obstacle_wallf"] = self.obstacle_wallf
    #     self.obstacle_wallb = RigidObject(self.cfg.obstacle_wallb)
    #     self.scene.rigid_objects["obstacle_wallb"] = self.obstacle_wallb
    #     stage = self.sim.stage
    #     barrier_path = "/World/envs/env_0/Robot/Barrier"
    #     barrier = UsdGeom.Cube.Define(stage, barrier_path)
    #     barrier.CreateSizeAttr(1.0)
    #     xform = UsdGeom.Xformable(barrier.GetPrim())
    #     xform.AddTranslateOp().Set(Gf.Vec3d(0.30, 0.0, 0.25))
    #     xform.AddScaleOp().Set(Gf.Vec3d(0.025, 0.25, 0.25))
    #     UsdPhysics.CollisionAPI.Apply(barrier.GetPrim())
    #     UsdGeom.Imageable(barrier.GetPrim()).MakeInvisible()

    #     spawn_ground_plane(
    #         prim_path="/World/ground",
    #         cfg=GroundPlaneCfg(
    #             physics_material=sim_utils.RigidBodyMaterialCfg(
    #                 static_friction=1.0,
    #                 dynamic_friction=1.0,
    #                 restitution=0.0,
    #             ),
    #         ),
    #     )
    #     self.scene.clone_environments(copy_from_source=False)
    #     if self.device == "cpu":
    #         self.scene.filter_collisions(global_prim_paths=["/World/ground"])
    #     self.scene.rigid_objects["robot"] = self.robot
    #     # self.scene.sensors["camera"] = self.camera
    #     self.scene.sensors["ray_caster"] = self.ray_caster
    #     light_cfg = sim_utils.DomeLightCfg(
    #         intensity=2000.0,
    #         color=(0.75, 0.75, 0.75),
    #     )
    #     light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        # actions: [num_envs, 2] -> [線速度, 角速度] を想定
        self.actions = torch.clamp(actions, -1.0, 1.0)
        #semseg各クラスがID割り当てられたときに、pushableのIDを探す
        # if self.pushable_color is None:
        #     semantic_info = self.camera.data.info[0]["semantic_segmentation"]["idToLabels"]
        #     self.pushable_color = next(
        #         ast.literal_eval(k) for k, v in semantic_info.items()
        #         if v.get("class") == "pushable"
        #     )

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
        #ヒートマップ作成
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
        self.current_heightmap = height_map
        height_map = height_map.unsqueeze(1)  # (N, 1, 80, 80) そのままconv2dへ

        # ray_heightmap = torch.zeros(
        #     (self.num_envs, 1, 64, 64),
        #     device=self.device
        # )
        # ray_data = self.scene.sensors["ray_caster"].data
        # ray_hits_w = ray_data.ray_hits_w
        # ray_heightmap = self.ray_heightmap_generator.generate(
        #     ray_hits_w
        # )
        # ray_heightmap = ray_heightmap.squeeze(1)
        # ray_heightmap = self.heightmap_generator.generate_from_ray(
        #     ray_hits_w,
        #     root_pos_w,
        #     robot_yaw
        # )
        # self.current_heightmap = ray_heightmap
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

        return {"policy": {"policy_obs": policy_obs, "ray_heightmap": height_map}} #ray_heightmap

    def _get_rewards(self) -> torch.Tensor:
        # ==========================================
        # 1. 状態量の取得と座標変換
        # ==========================================
        root_pos_w = self.robot.data.root_pos_w
        root_lin_vel = self.robot.data.root_lin_vel_w
        root_rot_w = self.robot.data.root_quat_w
        root_ang_vel = self.robot.data.root_ang_vel_w
        # ロボット座標系における局所的な線形速度
        local_lin_vel = quat_apply_inverse(yaw_quat(root_rot_w), root_lin_vel)
        # ==========================================
        # 2. ゴール方向・距離の計算
        # ==========================================
        goal_vec_w = self.goal_pos_w - root_pos_w[:, :2]
        goal_dist = torch.norm(goal_vec_w, dim=-1, keepdim=True)
        goal_dir_w = goal_vec_w / (goal_dist + 1e-5)
        # ==========================================
        # 3. 障害物関連のペナルティ・報酬計算
        # ==========================================
        if self.obstacle_stage is None:
            # Stage 5 / 12 混合処理
            stage5 = self.obstacle_stages == 5
            stage5_ids = torch.where(stage5)[0]
            min_obstacle_dist = torch.full((self.num_envs,), 999.0, device=self.device)
            collision = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            obstacle_penalty = torch.zeros(self.num_envs, device=self.device)
            obstacle_approach_penalty = torch.zeros(self.num_envs, device=self.device)
            collision_penalty = torch.zeros(self.num_envs, device=self.device)
            obstacle_pass_reward = torch.zeros(self.num_envs, device=self.device)
            obstacle_turn_reward = torch.zeros(self.num_envs, device=self.device)
            if len(stage5_ids) > 0:
                root_pos_stage5 = root_pos_w[stage5_ids, :2]
                goal_dir_stage5 = goal_dir_w[stage5_ids]
                root_lin_vel_stage5 = root_lin_vel[stage5_ids]
                root_ang_vel_stage5 = root_ang_vel[stage5_ids]
                obstacle_pos = torch.stack([
                    self.obstacle1.data.root_pos_w[stage5_ids, :2],
                    self.obstacle2.data.root_pos_w[stage5_ids, :2],
                    self.obstacle3.data.root_pos_w[stage5_ids, :2],
                    self.obstacle_long.data.root_pos_w[stage5_ids, :2],
                ], dim=1)
                robot_radius = self.cfg.robot.spawn.radius
                obstacle_radius = 0.3
                center_dist = torch.norm(root_pos_stage5[:, None, :2] - obstacle_pos, dim=-1)
                obstacle_surface_dist = torch.clamp(center_dist - robot_radius - obstacle_radius, min=0.0)
                min_obstacle_dist[stage5_ids] = obstacle_surface_dist.min(dim=1).values
                collision_stage5 = (center_dist <= (robot_radius + obstacle_radius)).any(dim=1)
                collision[stage5_ids] = collision_stage5
                obstacle_penalty[stage5_ids] = torch.clamp(1.0 - min_obstacle_dist[stage5_ids], min=0.0)
                collision_penalty[stage5_ids] = collision_stage5.float()
                relative_pos = root_pos_stage5[:, None, :] - obstacle_pos
                forward_dist = torch.sum(relative_pos * goal_dir_stage5[:, None, :], dim=-1)
                passed = forward_dist > 0.25
                newly_passed = passed & (~self.obstacle_passed[stage5_ids])
                self.obstacle_passed[stage5_ids] |= passed
                obstacle_pass_reward[stage5_ids] = torch.clamp(newly_passed.float().sum(dim=1), max=1.0)
                obstacle_vec = obstacle_pos - root_pos_stage5[:, None, :2]
                obstacle_dist = torch.norm(obstacle_vec, dim=-1)
                obstacle_dir = obstacle_vec / (obstacle_dist.unsqueeze(-1) + 1e-5)
                obstacle_approach_speed = torch.sum(root_lin_vel_stage5[:, None, :2] * obstacle_dir, dim=-1)
                min_obstacle_approach_speed = obstacle_approach_speed.max(dim=1).values
                obstacle_approach_penalty[stage5_ids] = torch.clamp(min_obstacle_approach_speed, min=0.0)
                forward_dist_obs = torch.sum(obstacle_vec * goal_dir_stage5[:, None, :], dim=-1)
                front_obstacle = ((forward_dist_obs > 0.0) & (forward_dist_obs < 1.5)).any(dim=1)
                turning_obstacle = torch.abs(root_ang_vel_stage5[:, 2]) > 0.1
                avoiding = min_obstacle_approach_speed < 0.0
                obstacle_turn_reward[stage5_ids] = (front_obstacle & turning_obstacle & avoiding).float()
        else:
            # 従来の単一Stage処理
            if self.obstacle_stage in [0, 12]:
                # 障害物なしステージの初期化
                min_obstacle_dist = torch.full((self.num_envs,), 999.0, device=self.device)
                collision = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                obstacle_penalty = torch.zeros(self.num_envs, device=self.device)
                obstacle_approach_penalty = torch.zeros(self.num_envs, device=self.device)
                collision_penalty = torch.zeros(self.num_envs, device=self.device)
                obstacle_pass_reward = torch.zeros(self.num_envs, device=self.device)
                obstacle_turn_reward = torch.zeros(self.num_envs, device=self.device)
            else:
                # ステージに応じた障害物位置の収集
                obstacle_pos_list = []
                if self.obstacle_stage in [1, 2, 3, 4, 5, 7, 9, 10]:
                    obstacle_pos_list.append(self.obstacle1.data.root_pos_w[:, :2])
                if self.obstacle_stage in [2, 5, 7, 9, 10]:
                    obstacle_pos_list.append(self.obstacle2.data.root_pos_w[:, :2])
                if self.obstacle_stage == 5:
                    obstacle_pos_list.append(self.obstacle3.data.root_pos_w[:, :2])
                if self.obstacle_stage in [5, 6, 11]:
                    obstacle_pos_list.append(self.obstacle_long.data.root_pos_w[:, :2])
                if self.obstacle_stage == 8:
                    obstacle_pos_list.append(torch.zeros_like(root_pos_w[:, :2]))
                    
                obstacle_pos = torch.stack(obstacle_pos_list, dim=1)
                robot_radius = self.cfg.robot.spawn.radius
                obstacle_radius = 0.3
                
                # 障害物との距離と衝突判定
                center_dist = torch.norm(root_pos_w[:, None, :2] - obstacle_pos, dim=-1)
                obstacle_surface_dist = torch.clamp(center_dist - robot_radius - obstacle_radius, min=0.0)
                min_obstacle_dist = obstacle_surface_dist.min(dim=1).values
                collision = (center_dist <= (robot_radius + obstacle_radius)).any(dim=1)
                
                # 障害物ペナルティ各種
                obstacle_penalty = torch.clamp(1.0 - min_obstacle_dist, min=0.0)
                collision_penalty = torch.where(
                    collision,
                    torch.full_like(min_obstacle_dist, 1.0),
                    torch.zeros_like(min_obstacle_dist)
                )
                
                # 障害物通過報酬
                relative_pos = root_pos_w[:, None, :2] - obstacle_pos
                forward_dist = torch.sum(relative_pos * goal_dir_w[:, None, :], dim=-1)
                passed = forward_dist > 0.25
                newly_passed = passed & (~self.obstacle_passed)
                self.obstacle_passed |= passed
                obstacle_pass_reward = torch.clamp(newly_passed.float().sum(dim=1), max=1.0)
                
                # 障害物回避・アプローチに関する計算
                obstacle_vec = obstacle_pos - root_pos_w[:, None, :2]
                obstacle_dist = torch.norm(obstacle_vec, dim=-1)
                obstacle_dir = obstacle_vec / (obstacle_dist.unsqueeze(-1) + 1e-5)
                obstacle_approach_speed = torch.sum(root_lin_vel[:, None, :2] * obstacle_dir, dim=-1)
                
                min_obstacle_approach_speed = obstacle_approach_speed.max(dim=1).values
                obstacle_approach_penalty = torch.clamp(min_obstacle_approach_speed, min=0.0) #速度依存1.0 m/s以上で走れば1を超
                
                forward_dist_obs = torch.sum(obstacle_vec * goal_dir_w[:, None, :], dim=-1)
                front_obstacle = ((forward_dist_obs > 0.0) & (forward_dist_obs < 1.5)).any(dim=1)
                turning_obstacle = torch.abs(root_ang_vel[:, 2]) > 0.1
                avoiding = min_obstacle_approach_speed < 0.0
                obstacle_turn_reward = (front_obstacle & turning_obstacle & avoiding).float()
        #ヒートマップ報酬
        height = self.current_heightmap
        obstacle_height = torch.clamp(height - 0.1, min=0.0, max=1.0)
        obstacle_space_penalty = (
            (obstacle_height * self.personal_weight).sum(dim=(1,2))
            / self.personal_weight.sum()
        )
        safe_reward = (
            (height < 0.1).float() * self.personal_weight
        ).mean(dim=(1, 2))
        obstacle_occupancy = (
            (height >= 0.1).float() * self.personal_weight
        ).sum(dim=(1, 2)) / self.personal_weight.sum()
        obstacle_occupancy = torch.clamp(obstacle_occupancy, max=0.5)
        target_speed = 1.0 - 1.4 * obstacle_occupancy                                   #占有率max50%で目標速度0.3m/s
        # front_region = height[:,24:36,26:38]                                            #進行方向領域
        personal_speed_reward = 1.0 - torch.abs(self.actions[:, 0] - target_speed) / target_speed
        # 過去の最短距離を更新できたか
        best_improved = torch.where(
            self.prev_goal_valid,
            self.current_goal_dist < self.best_goal_dist - 0.01,
            torch.zeros_like(self.current_goal_dist, dtype=torch.bool),
        )
        # 最短距離を更新できたらカウントをリセット
        self.no_progress_count = torch.where(
            best_improved,
            torch.zeros_like(self.no_progress_count),
            self.no_progress_count + 1,
        )
        no_progress_done = self.no_progress_count >= self.no_progress_threshold
        stop_reward = torch.where(
            no_progress_done,
            1.0 - torch.abs(self.actions[:, 0]),
            torch.zeros_like(self.actions[:, 0]),
        )
        personal_stop_reward = torch.where(
            no_progress_done,
            stop_reward,
            personal_speed_reward,
        )
        # 最短距離を更新
        self.best_goal_dist = torch.where(
            self.prev_goal_valid,
            torch.minimum(self.best_goal_dist, self.current_goal_dist),
            self.current_goal_dist,
        )
        self.prev_goal_valid[:] = True
        # ==========================================
        # 4. ゴール進行・到達・時間ボーナス
        # ==========================================
        # ゴール方向への進捗速度
        approach_speed = torch.sum(root_lin_vel[:, :2] * goal_dir_w, dim=-1)
        progress_reward = torch.clamp(approach_speed, min=0.0) #速度依存1.0 m/s以上で走れば1を超
        # ゴール到達判定
        goal_reached = self.current_goal_dist < 0.3
        goal_reward = goal_reached.float()
        # 早期到達ボーナス
        time_bonus = goal_reached.float() * (1.0 - self.episode_length_buf / self.max_episode_length)
        # ==========================================
        # 5. 基本的な移動・回転報酬
        # ==========================================
        forward_reward = torch.clamp(local_lin_vel[:, 0], min=0.0) #速度依存1.0 m/s以上で走れば1を超
        turn_reward = (torch.abs(self.actions[:, 1]) > 0.2).float()
        back_reward = (self.actions[:, 0] < -0.05).float()
        # ==========================================
        # 6. スタック（行き詰まり）判定と回避行動の処理
        # ==========================================
        trying_forward = self.current_v > 0.1
        pos_change = torch.norm(root_pos_w[:, :2] - self.prev_root_pos[:, :2], dim=-1)
        not_moving = pos_change < 0.01
        stuck = trying_forward & not_moving
        # stuck判定
        self.stop_count = torch.where(
            stuck,
            self.stop_count + 1,
            torch.zeros_like(self.stop_count),
        )
        stopped_too_long = self.stop_count >= self.stop_threshold
        stuck_penalty = stopped_too_long.float()
        # スタックトリガーの管理
        self.stuck_count = torch.where(
            stuck,
            self.stuck_count + 1,
            torch.zeros_like(self.stuck_count),
        )
        stuck_trigger = self.stuck_count >= 5
        self.need_avoid_reward |= stuck_trigger
        # safe stop判定
        if self.obstacle_stage == 12:
            stopped = torch.abs(self.actions[:, 0]) < 0.25
            self.safe_stop_count = torch.where(
                stopped,
                self.safe_stop_count + 1,
                torch.zeros_like(self.safe_stop_count),
            )
            self.safe_stop_done = self.safe_stop_count >= self.safe_stop_threshold
        else:
            stopped = torch.abs(self.actions[:, 0]) < 0.25
            self.safe_stop_count = torch.where(
                stopped,
                self.safe_stop_count + 1,
                torch.zeros_like(self.safe_stop_count),
            )
            self.safe_stop_done = self.safe_stop_count >= self.safe_stop_threshold
        # 次回比較用の位置を保存
        self.prev_root_pos[:] = root_pos_w
        # 旋回または後進による回避行動の判定
        turning = torch.abs(root_ang_vel[:, 2]) > 0.2
        backing = local_lin_vel[:, 0] < -0.05
        avoid_reward_trigger = self.need_avoid_reward & (turning | backing)
        avoid_reward = avoid_reward_trigger.float()
        # 動き始めたら回避状態を解除
        escaped = pos_change > 0.02
        self.need_avoid_reward &= ~escaped
        # ==========================================
        # 7. 報酬の合成（重み付け合計）
        # ==========================================
        if self.obstacle_stage == 12:
            reward = (
                10.0 * (1.0 - obstacle_space_penalty) * self.safe_stop_done.float()# 停止したときに周囲に障害物が少ないほど報酬
                # --- 基本移動関連 ---
                # + 0.1 * forward_reward              # 前方向へ進んでいることに対する小さな報酬
                # + 0.1 * turn_reward                 # 一定以上の旋回行動を行っていることへの報酬
                + 0.05 * back_reward                # 後進を誘発するための報酬
                # --- 障害物関連（報酬・ペナルティ） ---
                # + 0.1 * safe_reward                 # ロボット周囲に障害物がなければ微小報酬
                # --- スタック・回避関連 ---
                - 5.0 * stuck_penalty               # 前進しようとしているのに動けずスタックしたときのペナルティ
            )
        else:
            reward = (
                # --- ゴール・進行関連 ---
                1.0 * progress_reward               # ゴール方向へ近づくほど高くなる報酬
                + 50.0 * goal_reward                # ゴールに到達したときの大きな報酬
                + 10.0 * (1.0 - obstacle_space_penalty) * self.safe_stop_done.float()# 停止したときに周囲に障害物が少ないほど報酬
                + 10.0 * time_bonus                 # ゴール到達までの早さに応じたボーナス
                # --- 基本移動関連 ---
                + 0.1 * forward_reward              # 前方向へ進んでいることに対する小さな報酬
                + 0.1 * turn_reward                 # 一定以上の旋回行動を行っていることへの報酬
                + 0.05 * back_reward                # 後進を誘発するための報酬
                # --- 障害物関連（報酬・ペナルティ） ---
                - obstacle_penalty                  # 障害物に近づきすぎたときのペナルティ
                - obstacle_approach_penalty         # 障害物へ向かって高速接近しているときのペナルティ
                - 5.0 * collision_penalty           # 障害物と衝突したときのペナルティ
                + 0.3 * obstacle_turn_reward        # 障害物の手前で上手に方向転換（回避）できたときの報酬
                + 5.0 * obstacle_pass_reward        # 障害物を無事に通過したときの報酬（1個につき5.0）
                - 3.0 * obstacle_space_penalty      # ロボットに障害物が近いほど罰則
                + 0.1 * safe_reward                 # ロボット周囲に障害物がなければ微小報酬
                # --- スタック・回避関連 ---
                + 2.0 * avoid_reward                # スタック状態からうまく脱出（旋回・後進）できたときの報酬
                - 5.0 * stuck_penalty               # 前進しようとしているのに動けずスタックしたときのペナルティ
            )
        
        stage12 = self.obstacle_stages == 12
        stage5 = self.obstacle_stages == 5
        reward_stage12 = (
            10.0 * (1.0 - obstacle_space_penalty) * self.safe_stop_done.float()
            + 0.05 * back_reward  
            + personal_stop_reward
            - 5.0 * stuck_penalty
        )
        reward_stage5 = (
            1.0 * progress_reward
            + 50.0 * goal_reward
            + 10.0 * (1.0 - obstacle_space_penalty) * self.safe_stop_done.float()
            + 10.0 * time_bonus
            + 0.1 * forward_reward
            + 0.1 * turn_reward
            + 0.05 * back_reward
            - obstacle_penalty
            - obstacle_approach_penalty
            - 5.0 * collision_penalty
            + 0.3 * obstacle_turn_reward
            + 5.0 * obstacle_pass_reward
            - 3.0 * obstacle_space_penalty
            + 0.1 * safe_reward
            + personal_stop_reward
            + 2.0 * avoid_reward
            - 5.0 * stuck_penalty
        )
        reward = torch.where(stage12, reward_stage12, reward_stage5)

        #actionログを10stepごとに出力
        # if self.common_step_counter % 10 == 0:
        with open("/tmp/env0_action.log", "a") as f:
            f.write(
                f"v={self.actions[0,0].item():+.2f} "
                f"w={self.actions[0,1].item():+.2f} "
                f"no_progress={self.no_progress_count[0].item():+.2f} "
                f"no_progress_done={no_progress_done[0].item():+.2f} "
                f"personal_stop_reward={personal_stop_reward[0].item():.2f} "
                f"personal={obstacle_occupancy[0].item():.2f} "
                f"stop_done={self.safe_stop_done[0].item()}\n "
            )
        # # ==========================================
        # # Episode累積報酬
        # # ==========================================
        # self.episode_progress_reward += progress_reward
        # self.episode_goal_reward += goal_reward
        # self.episode_time_bonus += time_bonus
        # self.episode_forward_reward += forward_reward
        # self.episode_turn_reward += turn_reward
        # self.episode_back_reward += back_reward
        # self.episode_obstacle_penalty += obstacle_penalty
        # self.episode_obstacle_approach_penalty += obstacle_approach_penalty
        # self.episode_collision_penalty += collision_penalty
        # self.episode_obstacle_turn_reward += obstacle_turn_reward
        # self.episode_obstacle_pass_reward += obstacle_pass_reward
        # self.episode_obstacle_space_penalty += obstacle_space_penalty
        # self.episode_safe_reward += safe_reward
        # self.episode_blocked_speed_reward += blocked_speed_reward
        # self.episode_avoid_reward += avoid_reward
        # self.episode_stuck_penalty += stuck_penalty
        
        # # ==========================================
        # # 1000 stepごとの報酬表示
        # # ==========================================
        # if self.common_step_counter % 1000 == 0:
        #     tqdm.write(
        #         f"[AVG] "
        #         f"progress={progress_reward.mean().item():.3f} "
        #         f"goal={goal_reward.mean().item():.3f} "
        #         f"time={time_bonus.mean().item():.3f} "
        #         f"forward={forward_reward.mean().item():.3f} "
        #         f"turn={turn_reward.mean().item():.3f} "
        #         f"back={back_reward.mean().item():.3f} "
        #         f"obstacle={obstacle_penalty.mean().item():.3f} "
        #         f"approach={obstacle_approach_penalty.mean().item():.3f} "
        #         f"collision={collision_penalty.mean().item():.3f} "
        #         f"obstacle_turn={obstacle_turn_reward.mean().item():.3f} "
        #         f"obstacle_pass={obstacle_pass_reward.mean().item():.3f} "
        #         f"space={obstacle_space_penalty.mean().item():.3f} "
        #         f"safe={safe_reward.mean().item():.3f} "
        #         f"blocked_speed={blocked_speed_reward.mean().item():.3f} "
        #         f"avoid={avoid_reward.mean().item():.3f} "
        #         f"stuck={stuck_penalty.mean().item():.3f}"
        #     )
        #     if self.last_episode_rewards is not None:
        #         r = self.last_episode_rewards
        #         tqdm.write(
        #             f"[ENV0 LAST EP] "
        #             f"progress={r['progress']:.2f} "
        #             f"goal={r['goal']:.2f} "
        #             f"time={r['time']:.2f} "
        #             f"forward={r['forward']:.2f} "
        #             f"turn={r['turn']:.2f} "
        #             f"back={r['back']:.2f} "
        #             f"obstacle={r['obstacle']:.2f} "
        #             f"approach={r['approach']:.2f} "
        #             f"collision={r['collision']:.2f} "
        #             f"obstacle_turn={r['obstacle_turn']:.2f} "
        #             f"obstacle_pass={r['obstacle_pass']:.2f} "
        #             f"space={r['space']:.2f} "
        #             f"safe={r['safe']:.2f} "
        #             f"blocked_speed={r['blocked_speed']:.2f} "
        #             f"avoid={r['avoid']:.2f} "
        #             f"stuck={r['stuck']:.2f}"
        #         )

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # 転倒判定（キューブの高さが低くなりすぎた場合など）
        root_height = self.robot.data.root_pos_w[:, 2]
        died = root_height < 0.2
        # ゴールに十分近づいたら成功終了
        reached_goal = self.current_goal_dist < 0.3
        time_out |= reached_goal
        self.extras["reached_goal"] = reached_goal.clone()
        self.extras["time_out"] = time_out.clone()
        # 1秒以上、前進しようとしているのに動かない
        stopped_too_long = self.stop_count >= self.stop_threshold
        terminated = died | stopped_too_long | self.safe_stop_done
        # ==========================================
        # env0のEpisode終了時に最終結果を保存
        # ==========================================
        done = terminated | time_out
        if done[0]:
            self.last_episode_rewards = {
                "progress": self.episode_progress_reward[0].item(),
                "goal": self.episode_goal_reward[0].item(),
                "time": self.episode_time_bonus[0].item(),
                "forward": self.episode_forward_reward[0].item(),
                "turn": self.episode_turn_reward[0].item(),
                "back": self.episode_back_reward[0].item(),
                "obstacle": self.episode_obstacle_penalty[0].item(),
                "approach": self.episode_obstacle_approach_penalty[0].item(),
                "collision": self.episode_collision_penalty[0].item(),
                "obstacle_turn": self.episode_obstacle_turn_reward[0].item(),
                "obstacle_pass": self.episode_obstacle_pass_reward[0].item(),
                "space": self.episode_obstacle_space_penalty[0].item(),
                "safe": self.episode_safe_reward[0].item(),
                "blocked_speed": self.episode_blocked_speed_reward[0].item(),
                "avoid": self.episode_avoid_reward[0].item(),
                "stuck": self.episode_stuck_penalty[0].item(),
            }
            self.episode_progress_reward[0] = 0
            self.episode_goal_reward[0] = 0
            self.episode_time_bonus[0] = 0
            self.episode_forward_reward[0] = 0
            self.episode_turn_reward[0] = 0
            self.episode_back_reward[0] = 0
            self.episode_obstacle_penalty[0] = 0
            self.episode_obstacle_approach_penalty[0] = 0
            self.episode_collision_penalty[0] = 0
            self.episode_obstacle_turn_reward[0] = 0
            self.episode_obstacle_pass_reward[0] = 0
            self.episode_obstacle_space_penalty[0] = 0
            self.episode_safe_reward[0] = 0
            self.episode_avoid_reward[0] = 0
            self.episode_stuck_penalty[0] = 0

        return terminated, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if self.obstacle_stage is None:
            is_stage12 = torch.rand(len(env_ids), device=self.device) < 0.2
            self.obstacle_stages[env_ids[is_stage12]] = 12
            self.obstacle_stages[env_ids[~is_stage12]] = 5
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
            obstacle1_pos = mid_pos + direction * (-1.0) + lateral * 0.6
            obstacle2_pos = mid_pos + direction * 1.0 + lateral * (-0.8)
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
            obstacle_state = self.obstacle_pushable1.data.default_root_state[env_ids].clone()
            obstacle_state[:, :3] = pos
            self.obstacle_pushable1.write_root_pose_to_sim(obstacle_state[:, :7], env_ids)
            for wall, y in [(self.obstacle_wallr, -2.0), (self.obstacle_walll, 2.0)]:
                wall_state = wall.data.default_root_state[env_ids].clone()
                wall_state[:, :3] = torch.tensor([3.0, y, 0.25], device=self.device) + self.scene.env_origins[env_ids]
                wall.write_root_pose_to_sim(wall_state[:, :7], env_ids)

        elif self.obstacle_stage == 9:
            # 左右の壁＋障害物3個とPushableObstacle・ランダム位置
            for obstacle in [self.obstacle1, self.obstacle2, self.obstacle_pushable1]:
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

        elif self.obstacle_stage == 10:
            obstacles = [
                self.obstacle_pushable1,
                self.obstacle1,
                self.obstacle_pushable2,
                self.obstacle2,
                self.obstacle_pushable3,
            ]
            ys = [-1.4, -0.7, 0.0, 0.7, 1.4]
            for obstacle, y in zip(obstacles, ys):
                pos = torch.zeros((num_envs, 3), device=self.device)
                pos[:, 0] = 3.0
                pos[:, 1] = y
                pos[:, 2] = 0.25
                obstacle_state = obstacle.data.default_root_state[env_ids].clone()
                obstacle_state[:, :3] = pos + self.scene.env_origins[env_ids]
                obstacle.write_root_pose_to_sim(obstacle_state[:, :7], env_ids)
            for wall, y in [(self.obstacle_wallr, -2.0), (self.obstacle_walll, 2.0)]:
                wall_state = wall.data.default_root_state[env_ids].clone()
                wall_state[:, :3] = torch.tensor([3.0, y, 0.25], device=self.device) + self.scene.env_origins[env_ids]
                wall.write_root_pose_to_sim(wall_state[:, :7], env_ids)

        elif self.obstacle_stage == 11:
            # 横長障害物1個・固定位置
            obstacle_long_state = self.obstacle_long.data.default_root_state[env_ids].clone()
            obstacle_long_state[:, :3] = self.scene.env_origins[env_ids] + torch.tensor([3.0, 0.0, 0.25], device=self.device)
            self.obstacle_long.write_root_pose_to_sim(obstacle_long_state[:, :7], env_ids)
            for wall, y in [(self.obstacle_wallr, -2.0), (self.obstacle_walll, 2.0)]:
                wall_state = wall.data.default_root_state[env_ids].clone()
                wall_state[:, :3] = torch.tensor([3.0, y, 0.25], device=self.device) + self.scene.env_origins[env_ids]
                wall.write_root_pose_to_sim(wall_state[:, :7], env_ids)

        elif self.obstacle_stage == 12:
            # ゴール到達不可
            for wall, y in [(self.obstacle_wallr, -2.0), (self.obstacle_walll, 2.0)]:
                wall_state = wall.data.default_root_state[env_ids].clone()
                wall_state[:, :3] = self.scene.env_origins[env_ids] + torch.tensor(
                    [0.0, y, 0.25], device=self.device
                )
                wall.write_root_pose_to_sim(wall_state[:, :7], env_ids)
            for wall, x in [(self.obstacle_wallf, -2.0), (self.obstacle_wallb, 2.0)]:
                wall_state = wall.data.default_root_state[env_ids].clone()
                wall_state[:, :3] = self.scene.env_origins[env_ids] + torch.tensor(
                    [x, 0.0, 0.25], device=self.device
                )
                wall.write_root_pose_to_sim(wall_state[:, :7], env_ids)

        # =====================
        # 障害物配置
        # =====================
        stage12 = self.obstacle_stages[env_ids] == 12
        stage5 = self.obstacle_stages[env_ids] == 5
        stage12_ids = env_ids[stage12]
        stage5_ids = env_ids[stage5]
        num_stage5 = len(stage5_ids)
        num_stage12 = len(stage12_ids)
        # Stage 5
        if num_stage5 > 0:
            for obstacle in [self.obstacle1, self.obstacle2, self.obstacle3, self.obstacle_long]:
                pos = torch.zeros((num_stage5, 3), device=self.device)
                pos[:, 0] = torch.empty(num_stage5, device=self.device).uniform_(-0.5, 4.5)
                pos[:, 1] = torch.empty(num_stage5, device=self.device).uniform_(-3.5, 3.5)
                pos[:, 2] = 0.25
                obstacle_state = obstacle.data.default_root_state[stage5_ids].clone()
                obstacle_state[:, :3] = pos + self.scene.env_origins[stage5_ids]
                obstacle.write_root_pose_to_sim(obstacle_state[:, :7], stage5_ids)
            # Stage 5では壁を遠ざける
            for wall in [self.obstacle_wallr, self.obstacle_walll, self.obstacle_wallf, self.obstacle_wallb]:
                wall_state = wall.data.default_root_state[stage5_ids].clone()
                wall_state[:, :3] = self.scene.env_origins[stage5_ids] + torch.tensor(
                    [0.0, 0.0, -10.0], device=self.device
                )
                wall.write_root_pose_to_sim(wall_state[:, :7], stage5_ids)
        # Stage 12
        if num_stage12 > 0:
            # Stage 12ではStage 5の障害物を遠ざける
            for obstacle in [self.obstacle1, self.obstacle2, self.obstacle3, self.obstacle_long]:
                obstacle_state = obstacle.data.default_root_state[stage12_ids].clone()
                obstacle_state[:, :3] = self.scene.env_origins[stage12_ids] + torch.tensor(
                    [0.0, 0.0, -10.0], device=self.device
                )
                obstacle.write_root_pose_to_sim(obstacle_state[:, :7], stage12_ids)
            # 左右の壁
            for wall, y in [(self.obstacle_wallr, -2.0), (self.obstacle_walll, 2.0)]:
                wall_state = wall.data.default_root_state[stage12_ids].clone()
                wall_state[:, :3] = self.scene.env_origins[stage12_ids] + torch.tensor(
                    [0.0, y, 0.25], device=self.device
                )
                wall.write_root_pose_to_sim(wall_state[:, :7], stage12_ids)
            # 前後の壁
            for wall, x in [(self.obstacle_wallf, -2.0), (self.obstacle_wallb, 2.0)]:
                wall_state = wall.data.default_root_state[stage12_ids].clone()
                wall_state[:, :3] = self.scene.env_origins[stage12_ids] + torch.tensor(
                    [x, 0.0, 0.25], device=self.device
                )
                wall.write_root_pose_to_sim(wall_state[:, :7], stage12_ids)

        self.obstacle_passed[env_ids] = False
        self.prev_root_pos[env_ids] = self.robot.data.root_pos_w[env_ids]
        self.stuck_count[env_ids] = 0
        self.need_avoid_reward[env_ids] = False
        self.stop_count[env_ids] = 0
        self.safe_stop_done[env_ids] = False
        self.best_goal_dist[env_ids] = 0.0
        self.no_progress_count[env_ids] = 0
        self.prev_goal_valid[env_ids] = False