# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply, quat_apply_inverse

from .humanoid_amp_env_cfg import HumanoidAmpEnvCfg
from .motions import MotionLoader
#追加
from isaaclab.utils.math import quat_apply, euler_xyz_from_quat
from isaaclab.utils.math import yaw_quat, quat_mul, quat_conjugate, quat_rotate_inverse
from tqdm import tqdm
import os
from isaaclab.utils.math import quat_from_euler_xyz
import csv
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import ContactSensor
from pxr import PhysxSchema


class HumanoidAmpEnv(DirectRLEnv):
    cfg: HumanoidAmpEnvCfg

    def __init__(self, cfg: HumanoidAmpEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        #追加
        self.target_yaw = torch.zeros(self.num_envs, device=self.device)
        # action offset and scale
        dof_lower_limits = self.robot.data.soft_joint_pos_limits[0, :, 0]
        dof_upper_limits = self.robot.data.soft_joint_pos_limits[0, :, 1]
        self.action_offset = 0.5 * (dof_upper_limits + dof_lower_limits)
        self.action_scale = dof_upper_limits - dof_lower_limits
        self.left_shin_idx = self.robot.body_names.index("left_shin")
        self.right_shin_idx = self.robot.body_names.index("right_shin")
        self.next_goal_change_step = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.target_v = torch.zeros(self.num_envs, device=self.device)
        self.target_omega = torch.zeros(self.num_envs, device=self.device)
        self.next_command_step = torch.zeros(
            self.num_envs,
            device=self.device,
            dtype=torch.long,
        )
        self.prev_yaw = torch.zeros(
            self.num_envs,
            device=self.device,
        )
        # 膝立ち判定用のカウンタ
        self.knee_down_count = torch.zeros(
            self.num_envs,
            dtype=torch.int32,
            device=self.device,
        )
        
        # load motion
        self._motion_loader = MotionLoader(motion_file=self.cfg.motion_file, device=self.device)

        # DOF and key body indexes
        # 元デフォのhuamnoidリンク名
        key_body_names = ["right_hand", "left_hand", "right_foot", "left_foot"]
        self.ref_body_index = self.robot.data.body_names.index(self.cfg.reference_body)
        self.key_body_indexes = [self.robot.data.body_names.index(name) for name in key_body_names]
        self.motion_dof_indexes = self._motion_loader.get_dof_index(self.robot.data.joint_names)
        self.motion_ref_body_index = self._motion_loader.get_body_index([self.cfg.reference_body])[0]
        self.motion_key_body_indexes = self._motion_loader.get_body_index(key_body_names)
        #追加
        self.waist_body_index = 1
        self.head_body_index = 2
        # 障害物ステージ
        self.obstacle_stage = 10
        #報酬用
        self.direction_reached = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.prev_joint_pos = self.robot.data.joint_pos.clone()
        self.joint_motion_weights = torch.ones(
            self.robot.data.joint_pos.shape[1],
            device=self.device,
        )
        joint_weights = {
            "abdomen_x": 3.0,
            "abdomen_y": 3.0,
            "abdomen_z": 3.0,
            "right_hip_x": 2.5,
            "right_hip_y": 2.5,
            "right_hip_z": 2.5,
            "left_hip_x": 2.5,
            "left_hip_y": 2.5,
            "left_hip_z": 2.5,
            "right_knee": 1.5,
            "left_knee": 1.5,
            "right_ankle_x": 1.0,
            "right_ankle_y": 1.0,
            "right_ankle_z": 1.0,
            "left_ankle_x": 1.0,
            "left_ankle_y": 1.0,
            "left_ankle_z": 1.0,
            "right_shoulder_x": 0.7,
            "right_shoulder_y": 0.7,
            "right_shoulder_z": 0.7,
            "left_shoulder_x": 0.7,
            "left_shoulder_y": 0.7,
            "left_shoulder_z": 0.7,
            "right_elbow": 0.5,
            "left_elbow": 0.5,
        }
        for joint_name, weight in joint_weights.items():
            joint_ids, _ = self.robot.find_joints(joint_name)
            if len(joint_ids) > 0:
                self.joint_motion_weights[joint_ids[0]] = weight
        #ロバスト性向上用
        # Push Recovery
        self.push_interval = 3.0  # 外力を加える間隔 [秒]
        self.push_duration = 0.1  # 外力を加える時間 [秒]
        self.push_force_range = (0.0, 100.0)  # 外力の大きさ [N]
        self.push_timer = torch.zeros(
            self.num_envs, device=self.device
        )
        self.push_duration_timer = torch.zeros(
            self.num_envs, device=self.device
        )
        self.push_force = torch.zeros(
            (self.num_envs, 3), device=self.device
        )

        # reconfigure AMP observation space according to the number of observations and create the buffer
        self.amp_observation_size = self.cfg.num_amp_observations * self.cfg.amp_observation_space
        self.amp_observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.amp_observation_size,))
        self.amp_observation_buffer = torch.zeros(
            (self.num_envs, self.cfg.num_amp_observations, self.cfg.amp_observation_space), device=self.device
        )

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        #障害物配置
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
        # add ground plane
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
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])

        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _update_push_force(self):
        dt = self.step_dt
        # 外力を加えるタイミングを更新
        self.push_timer += dt
        # 外力を加えている時間を更新
        active = self.push_duration_timer > 0
        self.push_duration_timer[active] -= dt
        # 外力終了
        ended = self.push_duration_timer <= 0
        self.push_force[ended] = 0.0
        # 新しい外力を発生させる環境
        trigger = (
            (self.push_timer >= self.push_interval)
            & (self.push_duration_timer <= 0)
        )
        if trigger.any():
            num_trigger = trigger.sum()
            magnitude = torch.empty(
                (self.num_envs,), device=self.device
            ).uniform_(*self.push_force_range)[trigger]
            angle = torch.rand(
                (self.num_envs,), device=self.device
            ) * (2.0 * torch.pi)
            angle = angle[trigger]
            self.push_force[trigger, 0] = magnitude * torch.cos(angle)
            self.push_force[trigger, 1] = magnitude * torch.sin(angle)
            self.push_force[trigger, 2] = 0.0
            self.push_duration_timer[trigger] = self.push_duration
            self.push_timer[trigger] = 0.0
        # 外力を適用
        body_id = self.robot.body_names.index("pelvis")
        force_data = torch.zeros(
            (self.num_envs, self.robot.num_bodies, 3),
            device=self.device,
        )
        torque_data = torch.zeros_like(force_data)
        force_data[:, body_id, :] = self.push_force
        self.robot.root_physx_view.apply_forces_and_torques_at_position(
            force_data=force_data,
            torque_data=torque_data,
            position_data=None,
            indices=torch.arange(
                self.num_envs,
                device=self.device,
                dtype=torch.long,
            ),
            is_global=True,
        )

    def _pre_physics_step(self, actions: torch.Tensor):
        self.use_action_noise = True
        # --- 10秒ごとに目標角速度をランダムに更新する処理 ---
        # interval = int(10.0 / (self.cfg.sim.dt * self.cfg.decimation))
        # mask = self.episode_length_buf >= self.next_command_step
        # num = int(mask.sum().item())
        # if num > 0:
        #     # 速度を 0.0 ～ 1.0 の間でランダム変更する場合
        #     # self.target_v[mask] = (
        #     #     torch.rand(num, device=self.device) * 1.0
        #     # )
        #     # 角速度を -1.0 ～ 1.0 の間でランダム変更
        #     self.target_omega[mask] = (
        #         torch.rand(num, device=self.device) * 2.0 - 1.0
        #     )
        #     self.next_command_step[mask] = (
        #         self.episode_length_buf[mask] + interval
        #     )
        # 目標角速度を積分して目標方向を更新
        dt = self.cfg.sim.dt * self.cfg.decimation
        self.target_yaw = torch.atan2(
            torch.sin(self.target_yaw + self.target_omega * dt),
            torch.cos(self.target_yaw + self.target_omega * dt),
        )
        # ----------------------------------------------------
        if self.use_action_noise:
            if not hasattr(self, "_noise_printed"):
                print("Action noise: ON")
                self._noise_printed = True
            noise_std = 0.05
            noise = torch.randn_like(actions) * noise_std
            actions = actions + noise
        self.actions = torch.clamp(actions, -1.0, 1.0)
        # Push Recovery
        self._update_push_force()

    def _apply_action(self):
        target = self.action_offset + self.action_scale * self.actions
        self.robot.set_joint_position_target(target)

    def _get_observations(self) -> dict:
        #追加
        # -----------------------------------------------------------
        # 1. 基本的なワールドデータの取得
        # -----------------------------------------------------------
        root_pos_w = self.robot.data.body_pos_w[:, self.ref_body_index]
        root_rot_w = self.robot.data.body_quat_w[:, self.ref_body_index]
        lin_vel_w = self.robot.data.body_lin_vel_w[:, self.ref_body_index]
        ang_vel_w = self.robot.data.body_ang_vel_w[:, self.ref_body_index]
        key_pos_w = self.robot.data.body_pos_w[:, self.key_body_indexes]
        _, _, robot_yaw = euler_xyz_from_quat(root_rot_w)
        direction_error = torch.atan2(
            torch.sin(self.target_yaw - robot_yaw),
            torch.cos(self.target_yaw - robot_yaw),
        )
        direction_sin = torch.sin(direction_error)
        direction_cos = torch.cos(direction_error)
        waist_ang_vel = self.robot.data.body_ang_vel_w[:, self.waist_body_index]
        head_ang_vel = self.robot.data.body_ang_vel_w[:, self.head_body_index]
        waist_omega = waist_ang_vel[:, 2:3]
        head_omega = head_ang_vel[:, 2:3]
        # waist_omega = torch.zeros_like(waist_omega) #一旦無効
        # head_omega = torch.zeros_like(head_omega) #一旦無効

        # -----------------------------------------------------------
        # 3. すべてのデータを「Yawをキャンセルしたローカル座標系」に変換
        # -----------------------------------------------------------
        (
            local_root_pos,
            local_root_rot,
            local_lin_vel,
            local_ang_vel,
            local_key_pos,
        ) = localize_observation(
            root_pos_w,
            root_rot_w,
            lin_vel_w,
            ang_vel_w,
            key_pos_w,
        )      
        velocity_error = self.target_v - local_lin_vel[:, 0]
        omega_error = self.target_omega - local_ang_vel[:, 2]
        # omega_error = torch.zeros_like(omega_error) #一旦無効
        policy_obs = compute_policy_obs(
            self.robot.data.joint_pos,
            self.robot.data.joint_vel,
            local_root_pos,
            local_root_rot,
            local_lin_vel,
            local_ang_vel,
            local_key_pos,
            velocity_error,
            omega_error,
            waist_omega,
            head_omega,
            direction_sin,
            direction_cos,
        )


        # -----------------------------------------------------------
        # 4. 各Observationの組み立て
        # -----------------------------------------------------------
        # # world座標系のままのObservation
        # policy_obs = compute_obs0(
        #     self.robot.data.joint_pos,
        #     self.robot.data.joint_vel,
        #     self.robot.data.body_pos_w[:, self.ref_body_index],
        #     self.robot.data.body_quat_w[:, self.ref_body_index],
        #     self.robot.data.body_lin_vel_w[:, self.ref_body_index],
        #     self.robot.data.body_ang_vel_w[:, self.ref_body_index],
        #     self.robot.data.body_pos_w[:, self.key_body_indexes],
        #     heading_sin,
        #     heading_cos,
        # )
        # 【修正】AMP Obs を完全にローカル化
        amp_obs = compute_amp_obs(
            self.robot.data.joint_pos,
            self.robot.data.joint_vel,
            local_root_pos,
            local_root_rot,
            local_lin_vel,
            local_ang_vel,
            local_key_pos,
        )

        #入力変数の形状を確認するためのデバッグ出力
        # if not hasattr(self, "_printed_shape"):
        #     self._printed_shape = True
        #     print("joint_pos      :", self.robot.data.joint_pos.shape)
        #     print("joint_vel      :", self.robot.data.joint_vel.shape)
        #     print("root_pos       :", self.robot.data.body_pos_w[:, self.ref_body_index].shape)
        #     print("root_quat      :", self.robot.data.body_quat_w[:, self.ref_body_index].shape)
        #     print("lin_vel        :", self.robot.data.body_lin_vel_w[:, self.ref_body_index].shape)
        #     print("ang_vel        :", self.robot.data.body_ang_vel_w[:, self.ref_body_index].shape)
        #     print("key_body_pos   :", self.robot.data.body_pos_w[:, self.key_body_indexes].shape)
        #     self._printed_joint_names = True
        #     print("Joint names:")
        #     for i, name in enumerate(self.robot.data.joint_names):
        #         print(f"{i:2d}: {name}")

        # update AMP observation history
        for i in reversed(range(self.cfg.num_amp_observations - 1)):
            self.amp_observation_buffer[:, i + 1] = self.amp_observation_buffer[:, i]
        # build AMP observation
        self.amp_observation_buffer[:, 0] = amp_obs.clone()
        # self.extras = {"amp_obs": self.amp_observation_buffer.view(-1, self.amp_observation_size)}
        #goal_reachedを検出するため、辞書作り直しから追加法へ
        self.extras["amp_obs"] = self.amp_observation_buffer.view(
            -1, self.amp_observation_size
        )

        #ロボットの方向に対する人歩行らしさを判定し、分布を出力
        if self.common_step_counter % 1000 == 0:
                    # evaluate_motion.py と同じフォルダの絶対パスを指定
                    output_dir = "/home/matsuno/IsaacLab/scripts/reinforcement_learning/skrl"
                    output_path = os.path.join(output_dir, "robot_state_latest.pt")
                    
                    # ディレクトリが存在しない場合は自動作成（念のため）
                    os.makedirs(output_dir, exist_ok=True)

                    torch.save(
                        {
                            "joint_pos": self.robot.data.joint_pos.detach().cpu(),
                            "joint_vel": self.robot.data.joint_vel.detach().cpu(),
                            "root_pos": self.robot.data.body_pos_w[:, self.ref_body_index].detach().cpu(),
                            "root_rot": self.robot.data.body_quat_w[:, self.ref_body_index].detach().cpu(),
                            "root_lin_vel": self.robot.data.body_lin_vel_w[:, self.ref_body_index].detach().cpu(),
                            "root_ang_vel": self.robot.data.body_ang_vel_w[:, self.ref_body_index].detach().cpu(),
                            "key_pos": self.robot.data.body_pos_w[:, self.key_body_indexes].detach().cpu(),
                            # 評価用インデックス
                            "motion_ref_body_index": self.motion_ref_body_index,
                            "motion_key_body_indexes": self.motion_key_body_indexes,
                            "motion_dof_indexes": self.motion_dof_indexes,
                        },
                        output_path,  # フルパスを指定して保存
                    )

        # if self.episode_length_buf[0] % 30 == 0:
        #     with open("/tmp/env0_velocity.txt", "w") as f:
        #         f.write(
        #             f"Target v     : {self.target_v[0].item():.3f} m/s\n"
        #             f"Current v    : {local_lin_vel[0, 0].item():.3f} m/s\n"
        #             f"Target omega : {self.target_omega[0].item():.3f} rad/s\n"
        #             f"Current omega: {local_ang_vel[0, 2].item():.3f} rad/s\n"
        #             f"yaw          : {robot_yaw[0].item():.3f} rad\n"
        #         )
        return {"policy": policy_obs}

    def _get_rewards(self) -> torch.Tensor:
        # -----------------------------------------------------------
        # 1. root速度
        # -----------------------------------------------------------
        root_lin_vel = self.robot.data.body_lin_vel_w[:, self.ref_body_index]
        root_ang_vel = self.robot.data.body_ang_vel_w[:, self.ref_body_index]
        # -----------------------------------------------------------
        # 2. 高さ報酬 -----------------------------------------------------------
        root_height = self.robot.data.body_pos_w[:, self.ref_body_index, 2]
        height_reward = torch.exp(
            -5.0 * (root_height - 1.2) ** 2
        )
        # 腰の高さ報酬
        waist_height = self.robot.data.body_pos_w[:, self.waist_body_index, 2]
        waist_height_reward = torch.exp(
            -5.0 * (waist_height - 0.9) ** 2
        )
        # 頭の高さ報酬
        head_height = self.robot.data.body_pos_w[:, self.head_body_index, 2]
        head_height_reward = torch.exp(
            -5.0 * (head_height - 1.4) ** 2
        )
        # -----------------------------------------------------------
        # 3. ロボット前方向速度
        # -----------------------------------------------------------
        root_quat = self.robot.data.body_quat_w[:, self.ref_body_index]
        _, _, yaw = euler_xyz_from_quat(root_quat)
        forward_dir = torch.stack(
            [
                torch.cos(yaw),
                torch.sin(yaw),
            ],
            dim=-1,
        )
        forward_speed = torch.sum(
            root_lin_vel[:, :2] * forward_dir,
            dim=-1,
        )
        # -----------------------------------------------------------
        # 4. 目標速度との誤差
        # -----------------------------------------------------------
        velocity_error = self.target_v - forward_speed
        velocity_reward = torch.exp(
            -5.0 * velocity_error ** 2
        )
        # -----------------------------------------------------------
        # 5. 目標角速度との誤差
        # -----------------------------------------------------------
        angular_velocity = root_ang_vel[:, 2]
        omega_error = self.target_omega - angular_velocity
        torso_omega_reward = torch.exp(
            -5.0 * omega_error ** 2
        )
        waist_ang_vel = self.robot.data.body_ang_vel_w[:, self.waist_body_index]
        waist_omega_error = self.target_omega - waist_ang_vel[:, 2]
        waist_omega_reward = torch.exp(
            -5.0 * waist_omega_error ** 2
        )
        head_ang_vel = self.robot.data.body_ang_vel_w[:, self.head_body_index]
        head_omega_error = self.target_omega - head_ang_vel[:, 2]
        head_omega_reward = torch.exp(
            -5.0 * head_omega_error ** 2
        )
        #逆方向罰則
        angular_velocity = root_ang_vel[:, 2]
        wrong_direction_penalty = torch.where(
            angular_velocity * self.target_omega < 0.0,
            torch.ones_like(angular_velocity),
            torch.zeros_like(angular_velocity),
        )
        self.prev_yaw[:] = yaw
        # 現在方向と目標方向の誤差
        yaw_error = torch.atan2(
            torch.sin(self.target_yaw - yaw),
            torch.cos(self.target_yaw - yaw),
        )
        direction_reward = torch.exp(-2.0 * yaw_error.square())
        reached = direction_reward > 0.9
        # 全環境が目標方向に到達したら、共通の目標方向を変更
        # 全環境の5割以上が目標方向に到達したら、共通の目標方向を変更
        # if reached.float().mean() >= 0.8:
        #     new_angle_deg = torch.randint(
        #         -9, 10, (1,), device=self.device
        #     ).float() * 10.0
        #     self.target_yaw[:] = torch.deg2rad(new_angle_deg)
        #     print(f"Target direction changed: {new_angle_deg.item():.0f} deg")
        #関節変位抑制
        joint_pos = self.robot.data.joint_pos
        delta_q = joint_pos - self.prev_joint_pos
        penalty = torch.sum(self.joint_motion_weights * delta_q.square(), dim=-1)
        self.prev_joint_pos = joint_pos.clone()
        # -----------------------------------------------------------
        # 6. 報酬の合成
        # -----------------------------------------------------------
        reward = (
            + 2.0 * direction_reward       # 目標方向への追従       
            + 1.0 * torso_omega_reward     # 胴体の目標旋回速度への追従            
            + 1.0 * waist_omega_reward     # 腰の目標旋回速度への追従                    
            + 1.0 * head_omega_reward      # 頭部の目標旋回速度への追従
            + 1.0 * velocity_reward        # 目標並進速度への追従  
            # + 0.5 * height_reward          # 指定した高さ(1.0m)の維持
            # + 0.5 * waist_height_reward    # 腰の高さへの追従
            # + 0.5 * head_height_reward     # 頭の高さへの追従
            # - 1.0 * wrong_direction_penalty  # 目標と逆方向に回転している場合のペナルティ
            # - 0.1 * penalty  # 関節変位の抑制
        )   

        # 1000ステップごとにログを出力
        if self.common_step_counter % 1000 == 0:
            tqdm.write(
                f"step={self.common_step_counter} "
                f"height_r={height_reward.mean().item():.3f} "
                f"waist_height_r={waist_height_reward.mean().item():.3f} "
                f"head_height_r={head_height_reward.mean().item():.3f} "
                f"velocity={velocity_reward.mean().item():.3f} "
                f"torso_omega={torso_omega_reward.mean().item():.3f} "
                f"waist_omega={waist_omega_reward.mean().item():.3f} "
                f"head_omega={head_omega_reward.mean().item():.3f} "
                f"direction={direction_reward.mean().item():.3f} "
                f"wrong_direction={wrong_direction_penalty.mean().item():.3f} "
                f"penalty={penalty.mean().item():.3f} "
            )

        # env0の速度・角速度をログに出力
        if self.common_step_counter % 50 == 0:
            with open("/tmp/env0_velocity.log", "a") as f:
                f.write(
                    f"step={self.common_step_counter} "
                    f"target_v={self.target_v[0].item():.3f} "
                    f"forward_v={forward_speed[0].item():.3f} "
                    f"target_yaw={self.target_yaw[0].item():.3f} "
                    f"target_yaw_deg={torch.rad2deg(self.target_yaw[0]).item():.1f}deg "
                    f"direction_error={torch.rad2deg(yaw_error[0]).item():.1f}deg "
                    f"target_omega={self.target_omega[0].item():.3f} "
                    f"root_omega={angular_velocity[0].item():.3f} "
                    f"head_omega={head_ang_vel[0, 2].item():.3f} "
                    f"waist_omega={waist_ang_vel[0, 2].item():.3f}\n"
                )

        #観測をエクセルに出力
        # if self.common_step_counter % 10 == 0:
        #     with open(self.omega_log_path, "a", newline="") as f:
        #         writer = csv.writer(f)
        #         writer.writerow([
        #             self.common_step_counter,
        #             # self.target_v[0].item(),
        #             # forward_speed[0].item(),
        #             self.target_omega[0].item(),
        #             angular_velocity[0].item(),
        #             head_ang_vel[0, 2].item(),
        #             waist_ang_vel[0, 2].item(),
        #             # omega_error[0].item(),
        #             # self.target_yaw[0].item(),
        #             # yaw_error[0].item(),
        #         ])

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        if self.cfg.early_termination:
            died = self.robot.data.body_pos_w[:, self.ref_body_index, 2] < self.cfg.termination_height
        else:
            died = torch.zeros_like(time_out)
        left_shin_h = self.robot.data.body_pos_w[:, self.left_shin_idx, 2]
        right_shin_h = self.robot.data.body_pos_w[:, self.right_shin_idx, 2]
        shin_height_thresh = 0.1
        shin_down = (
            (left_shin_h < shin_height_thresh)
            | (right_shin_h < shin_height_thresh)
        )
        # 膝立ち継続カウント
        self.knee_down_count[shin_down] += 1
        self.knee_down_count[~shin_down] = 0
        # 20ステップ以上続いたら終了
        died |= self.knee_down_count >= 20
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if self.cfg.reset_strategy == "default":
            root_state, joint_pos, joint_vel = self._reset_strategy_default(env_ids)
        elif self.cfg.reset_strategy.startswith("random"):
            start = "start" in self.cfg.reset_strategy
            root_state, joint_pos, joint_vel = self._reset_strategy_random(env_ids, start)
        else:
            raise ValueError(f"Unknown reset strategy: {self.cfg.reset_strategy}")
        
        self.robot.write_root_link_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        _, _, yaw = euler_xyz_from_quat(root_state[:, 3:7])
        self.prev_joint_pos[env_ids] = self.robot.data.joint_pos[env_ids].clone()
        self.prev_yaw[env_ids] = yaw
        self.direction_reached[env_ids] = False
        #ロバスト性向上用
        self.push_timer[env_ids] = 0.0
        self.push_duration_timer[env_ids] = 0.0
        self.push_force[env_ids] = 0.0
        # 初期目標方向をランダムに設定
        # self.target_yaw[env_ids] = torch.deg2rad(
        #     torch.randint(-9, 10, (len(env_ids),), device=self.device).float() * 10.0
        # )
        # リセットされた環境だけ目標方向をランダムに変更（±30度）
        # angle_deg = torch.empty(
        #     len(env_ids), device=self.device
        # ).uniform_(-180.0, 180.0)
        # self.target_yaw[env_ids] = torch.deg2rad(angle_deg)
        # self.target_yaw[env_ids] = torch.deg2rad(
        #     torch.full(
        #         (len(env_ids),),
        #     30.0,  # 目標方向（度）
        #         device=self.device,
        #     )
        # )
        #1_初期目標値の設定
        # self.target_yaw[env_ids] = 0.0
        # self.target_v[env_ids] =0.0      #目標進行速度
        # self.target_omega[env_ids] = 0.0    #目標角速度
        #2_更新間隔(1~2秒)の設定
        min_interval = int(1.0 / (self.cfg.sim.dt * self.cfg.decimation))
        max_interval = int(2.0 / (self.cfg.sim.dt * self.cfg.decimation))
        #2_更新対象の環境を取得
        mask = self.episode_length_buf >= self.next_command_step
        num = int(mask.sum().item())
        #2_目標のランダム変更(1~2秒ごと)
        if num > 0:
            self.target_v[mask] = (
                torch.rand(num, device=self.device) * 0.5 + 0.5
            )
            self.target_omega[mask] = (
                torch.rand(num, device=self.device) * 2.0 - 1.0
            )
            #次回更新タイミングの設定
            self.next_command_step[mask] = (
                self.episode_length_buf[mask]
                + torch.randint(
                    min_interval,
                    max_interval + 1,
                    (num,),
                    device=self.device,
                )
            )
        #3_進行速度：0～1.0 m/sでランダム
        # self.target_v[env_ids] = torch.rand(len(env_ids), device=self.device) * 1.0
        #3_角速度：-1～1 rad/sでランダム
        # self.target_omega[env_ids] = torch.rand(len(env_ids), device=self.device) * 2.0 - 1.0
        
        #4_10秒ごとのランダム変更_prephysicへ
        # interval = int(10.0 / (self.cfg.sim.dt * self.cfg.decimation))
        # self.next_command_step[env_ids] = self.episode_length_buf[env_ids] + interval

        # =====================
        # 障害物配置
        # =====================
        num_envs = len(env_ids)

        if self.obstacle_stage == 0:
            # 障害物なし
            pass

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
    # reset strategies

    def _reset_strategy_default(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        root_state[:, 2] += 0.6
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        return root_state, joint_pos, joint_vel

    def _reset_strategy_random(
        self, env_ids: torch.Tensor, start: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # sample random motion times (or zeros if start is True)
        num_samples = env_ids.shape[0]
        times = np.zeros(num_samples) if start else self._motion_loader.sample_times(num_samples)
        # sample random motions
        (
            dof_positions,
            dof_velocities,
            body_positions,
            body_rotations,
            body_linear_velocities,
            body_angular_velocities,
        ) = self._motion_loader.sample(num_samples=num_samples, times=times)
        
        # H1ロボットの正しい部位名 'pelvis' に修正します
        # motion_torso_index = self._motion_loader.get_body_index(["pelvis"])[0]
        # get root transforms (the humanoid torso)
        motion_torso_index = self._motion_loader.get_body_index(["torso"])[0]
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, 0:3] = body_positions[:, motion_torso_index] + self.scene.env_origins[env_ids]
        root_state[:, 2] += 0.15  # lift the humanoid slightly to avoid collisions with the ground
        root_state[:, 3:7] = body_rotations[:, motion_torso_index]
        root_state[:, 7:10] = body_linear_velocities[:, motion_torso_index]
        root_state[:, 10:13] = body_angular_velocities[:, motion_torso_index]
        # get DOFs state
        dof_pos = dof_positions[:, self.motion_dof_indexes]
        dof_vel = dof_velocities[:, self.motion_dof_indexes]

        # update AMP observation
        amp_observations = self.collect_reference_motions(num_samples, times)
        self.amp_observation_buffer[env_ids] = amp_observations.view(num_samples, self.cfg.num_amp_observations, -1)

        return root_state, dof_pos, dof_vel

    # env methods

    def collect_reference_motions(self, num_samples: int, current_times: np.ndarray | None = None) -> torch.Tensor:
        # sample random motion times (or use the one specified)
        if current_times is None:
            current_times = self._motion_loader.sample_times(num_samples)
        times = (
            np.expand_dims(current_times, axis=-1)
            - self._motion_loader.dt * np.arange(0, self.cfg.num_amp_observations)
        ).flatten()
        # get motions
        (
            dof_positions,
            dof_velocities,
            body_positions,
            body_rotations,
            body_linear_velocities,
            body_angular_velocities,
        ) = self._motion_loader.sample(num_samples=num_samples, times=times)
        
        # compute AMP observation
        (            
            local_root_pos,
            local_root_rot,
            local_lin_vel,
            local_ang_vel,
            local_key_pos,
        ) = localize_observation(
            body_positions[:, self.motion_ref_body_index],
            body_rotations[:, self.motion_ref_body_index],
            body_linear_velocities[:, self.motion_ref_body_index],
            body_angular_velocities[:, self.motion_ref_body_index],
            body_positions[:, self.motion_key_body_indexes],
        )

        amp_observation = compute_amp_obs(
            dof_positions[:, self.motion_dof_indexes],
            dof_velocities[:, self.motion_dof_indexes],
            local_root_pos,
            local_root_rot,
            local_lin_vel,
            local_ang_vel,
            local_key_pos,
        )

        return amp_observation.view(-1, self.amp_observation_size)


@torch.jit.script
def quaternion_to_tangent_and_normal(q: torch.Tensor) -> torch.Tensor:#世界座標でロボットの向きを計算してる
    ref_tangent = torch.zeros_like(q[..., :3])
    ref_normal = torch.zeros_like(q[..., :3])
    ref_tangent[..., 0] = 1
    ref_normal[..., -1] = 1
    tangent = quat_apply(q, ref_tangent)
    normal = quat_apply(q, ref_normal)
    return torch.cat([tangent, normal], dim=len(tangent.shape) - 1)

@torch.jit.script
def localize_footstep(
    foot_pos_w: torch.Tensor,
    root_pos_w: torch.Tensor,
    root_rot_w: torch.Tensor,
) -> torch.Tensor:
    # root基準にする
    rel_pos = foot_pos_w - root_pos_w

    # yawだけ除去
    q_yaw_inv = quat_conjugate(yaw_quat(root_rot_w))

    # local frameへ変換
    foot_pos_local = quat_apply(
        q_yaw_inv,
        rel_pos,
    )

    return foot_pos_local

@torch.jit.script
def localize_observation(
    root_pos: torch.Tensor,
    root_rot: torch.Tensor,
    root_lin_vel: torch.Tensor,
    root_ang_vel: torch.Tensor,
    key_body_pos: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    # RootのYawを打ち消すクォータニオン
    q_yaw_inv = quat_conjugate(yaw_quat(root_rot))
    # Root位置（XYを消して高さだけ残す）
    local_root_pos = root_pos.clone()
    local_root_pos[:, 0:2] = 0.0
    # Root姿勢（Yaw除去）
    local_root_rot = quat_mul(q_yaw_inv, root_rot)
    # Root速度
    local_root_lin_vel = quat_apply_inverse(yaw_quat(root_rot), root_lin_vel)
    local_root_ang_vel = quat_apply_inverse(yaw_quat(root_rot), root_ang_vel)
    # 手足位置
    rel_key_pos = key_body_pos - root_pos.unsqueeze(1)
    num_envs = rel_key_pos.shape[0]
    num_keys = rel_key_pos.shape[1]
    flat_rel_key_pos = rel_key_pos.view(-1, 3)
    repeated_q_yaw_inv = (
        q_yaw_inv.unsqueeze(1)
        .repeat(1, num_keys, 1)
        .view(-1, 4)
    )
    flat_local_key_pos = quat_apply_inverse(
        repeated_q_yaw_inv,
        flat_rel_key_pos,
    )
    local_key_pos = flat_local_key_pos.view(num_envs, num_keys, 3)

    return (
        local_root_pos,
        local_root_rot,
        local_root_lin_vel,
        local_root_ang_vel,
        local_key_pos,
    )

@torch.jit.script
def compute_policy_obs(
    dof_positions: torch.Tensor,
    dof_velocities: torch.Tensor,
    root_positions: torch.Tensor,
    root_rotations: torch.Tensor,
    root_linear_velocities: torch.Tensor,
    root_angular_velocities: torch.Tensor,
    key_body_positions: torch.Tensor,
    #追加
    velocity_error,
    omega_error,
    waist_omega,
    head_omega,
    direction_sin: torch.Tensor,
    direction_cos: torch.Tensor,
) -> torch.Tensor:
    obs = torch.cat(
        (
            dof_positions,
            dof_velocities,
            root_positions[:, 2:3],  # root body height
            quaternion_to_tangent_and_normal(root_rotations),
            root_linear_velocities,
            root_angular_velocities,
            key_body_positions.view(key_body_positions.shape[0], -1),
            #追加
            velocity_error.unsqueeze(-1),
            omega_error.unsqueeze(-1),
            waist_omega,
            head_omega,
            direction_sin.unsqueeze(-1),
            direction_cos.unsqueeze(-1),
        ),
        dim=-1,
    )
    return obs

@torch.jit.script
def compute_amp_obs(
    dof_positions: torch.Tensor,
    dof_velocities: torch.Tensor,
    root_positions: torch.Tensor,
    root_rotations: torch.Tensor,
    root_linear_velocities: torch.Tensor,
    root_angular_velocities: torch.Tensor,
    key_body_positions: torch.Tensor,
) -> torch.Tensor:
    obs = torch.cat(
        (
            dof_positions,
            dof_velocities,
            root_positions[:, 2:3],  # root body height
            quaternion_to_tangent_and_normal(root_rotations),
            root_linear_velocities,
            root_angular_velocities,
            key_body_positions.view(key_body_positions.shape[0], -1),
        ),
        dim=-1,
    )
    return obs

@torch.jit.script
def compute_obs0(
    dof_positions: torch.Tensor,
    dof_velocities: torch.Tensor,
    root_positions: torch.Tensor,
    root_rotations: torch.Tensor,
    root_linear_velocities: torch.Tensor,
    root_angular_velocities: torch.Tensor,
    key_body_positions: torch.Tensor,
    #追加
    heading_sin: torch.Tensor,
    heading_cos: torch.Tensor,
) -> torch.Tensor:
    obs = torch.cat(
        (
            dof_positions,
            dof_velocities,
            root_positions[:, 2:3],  # root body height
            quaternion_to_tangent_and_normal(root_rotations),
            root_linear_velocities,
            root_angular_velocities,
            (key_body_positions - root_positions.unsqueeze(-2)).view(key_body_positions.shape[0], -1),
            #追加
            heading_sin.unsqueeze(-1),
            heading_cos.unsqueeze(-1),
        ),
        dim=-1,
    )
    return obs





