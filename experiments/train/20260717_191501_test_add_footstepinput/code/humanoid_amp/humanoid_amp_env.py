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


class HumanoidAmpEnv(DirectRLEnv):
    cfg: HumanoidAmpEnvCfg

    def __init__(self, cfg: HumanoidAmpEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # --- クラスの初期化（__init__など）に追加 ---------------------
        # # 確認用コード（実行するとターミナルにリンク名一覧が出ます）
        # print("--- H1 Robot Available Body Names ---")
        print(self.robot.data.body_names)
        # print(self.robot.data.joint_names)
        # print(self.robot.data.body_names)
        # print("-------------------------------------")
        
        # import torch
        # from tqdm import tqdm  # 必ずファイルの先頭か、ここでインポートしてください

        # パス設定（絶対パス）初期方向目標の確率分布
        self.prob_file = "/home/matsuno/IsaacLab/scripts/my_evaluate_project/output/yaw_prob_distribution.pt"

        # ログ文字列を一度組み立てる
        log_msg = "\n" + "="*50 + "\n"
        if os.path.exists(self.prob_file):
            log_msg += f"🔥【SUCCESS】確率分布ファイルを検出しました！: {self.prob_file}\n"
            log_msg += f"[Env] Loading target yaw distribution from {self.prob_file}\n"
            
            # データのロードとデバイス転送
            dist_data = torch.load(self.prob_file, map_location=self.device)
            self.yaw_bin_probs = dist_data["probabilities"].to(self.device)
            
            log_msg += f"📊 読み込んだ確率分布: {self.yaw_bin_probs.tolist()}\n"
            self.use_biased_yaw = True
        else:
            log_msg += f"❌【WARNING】確率分布ファイルが見つかりません: {self.prob_file}\n"
            log_msg += "[Env] No distribution file found. Using uniform random yaw.\n"
            log_msg += "🎲 デフォルトの一様ランダム（通常のウォーク）で動きます。\n"
            self.use_biased_yaw = False
        log_msg += "="*50 + "\n"
        # tqdm.write を使ってプログレスバーを壊さずに出力！
        tqdm.write(log_msg)
        # --- クラスの初期化（__init__など）に追加終了 ----------------

        #追加
        self.goal_yaw = torch.zeros(self.num_envs, device=self.device)
        # action offset and scale
        dof_lower_limits = self.robot.data.soft_joint_pos_limits[0, :, 0]
        dof_upper_limits = self.robot.data.soft_joint_pos_limits[0, :, 1]
        self.action_offset = 0.5 * (dof_upper_limits + dof_lower_limits)
        self.action_scale = dof_upper_limits - dof_lower_limits
        self.left_shin_idx = self.robot.body_names.index("left_shin")
        self.right_shin_idx = self.robot.body_names.index("right_shin")

        #footstep_targetsの初期化
        self.footstep_targets = torch.zeros(
            self.num_envs, 4,
            device=self.device
        )
        self.footstep_targets[:,0] = 0.35   # left_x
        self.footstep_targets[:,1] = 0.15   # left_y
        self.footstep_targets[:,2] = 0.35   # right_x
        self.footstep_targets[:,3] = -0.15  # right_y
        self.left_foot_idx = self.robot.body_names.index("left_foot")
        self.right_foot_idx = self.robot.body_names.index("right_foot")
        self.prev_left_contact = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device
        )
        self.prev_right_contact = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device
        )
        # 着地点保存
        self.left_landing_pos = torch.zeros(
            self.num_envs, 2,
            device=self.device,
        )

        self.right_landing_pos = torch.zeros(
            self.num_envs, 2,
            device=self.device,
        )

        # 膝立ち判定用のカウンタ
        self.knee_down_count = torch.zeros(
            self.num_envs,
            dtype=torch.int32,
            device=self.device,
        )

        self.use_action_noise = False
        
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

        # reconfigure AMP observation space according to the number of observations and create the buffer
        self.amp_observation_size = self.cfg.num_amp_observations * self.cfg.amp_observation_space
        self.amp_observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.amp_observation_size,))
        self.amp_observation_buffer = torch.zeros(
            (self.num_envs, self.cfg.num_amp_observations, self.cfg.amp_observation_space), device=self.device
        )

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
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

    def _pre_physics_step(self, actions: torch.Tensor):
        if self.use_action_noise:
            # print("Action noise: ON")
            noise_std = 0.05
            noise = torch.randn_like(actions) * noise_std
            actions = actions + noise
        self.actions = torch.clamp(actions, -1.0, 1.0)

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

        # goal_yaw = torch.zeros_like(robot_yaw)

        heading_error = self.goal_yaw - robot_yaw
        heading_error = torch.atan2(
            torch.sin(heading_error),
            torch.cos(heading_error)
        )
        heading_sin = torch.sin(heading_error)
        heading_cos = torch.cos(heading_error)
        self.heading_error = heading_error

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

        policy_obs = compute_policy_obs(
            self.robot.data.joint_pos,
            self.robot.data.joint_vel,
            local_root_pos,
            local_root_rot,
            local_lin_vel,
            local_ang_vel,
            local_key_pos,
            heading_sin,
            heading_cos,
            self.footstep_targets,
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
            torch.zeros_like(heading_sin),
            torch.ones_like(heading_cos),
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
        self.extras = {"amp_obs": self.amp_observation_buffer.view(-1, self.amp_observation_size)}


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

        return {"policy": policy_obs}

    def _get_rewards(self) -> torch.Tensor:
        # root速度
        root_lin_vel = self.robot.data.body_lin_vel_w[:, self.ref_body_index]
        # 高さ報酬
        root_height = self.robot.data.body_pos_w[:, self.ref_body_index, 2]
        height_reward = torch.exp(
            -5.0 * (root_height - 1.0) ** 2
        )
        # root姿勢からyaw取得
        root_quat = self.robot.data.body_quat_w[:, self.ref_body_index]
        _, _, yaw = euler_xyz_from_quat(root_quat)
        # ロボット前方向
        forward_dir = torch.stack(
            [
                torch.cos(yaw),
                torch.sin(yaw),
            ],
            dim=-1,
        )
        # ロボット前方向速度
        forward_speed = torch.sum(
            root_lin_vel[:, :2] * forward_dir,
            dim=-1,
        )
        target_speed = 0.7
        # 前進速度報酬
        forward_reward = torch.exp(
            -(forward_speed - target_speed) ** 2
        )
        # observationで計算済みのheading_errorを使用
        heading_reward = torch.exp(
            -2.0 * self.heading_error ** 2
        )
        # 方向が合っている時だけ前進報酬を有効化
        forward_reward = forward_reward * heading_reward

        #足の位置誤差-----------------------------------------------------------------------
        left_foot_pos_w = self.robot.data.body_pos_w[:, self.key_body_indexes[3]]
        right_foot_pos_w = self.robot.data.body_pos_w[:, self.key_body_indexes[2]]
        root_pos_w = self.robot.data.body_pos_w[:, self.ref_body_index]
        root_rot_w = self.robot.data.body_quat_w[:, self.ref_body_index]
        left_foot_local = localize_footstep(left_foot_pos_w, root_pos_w,root_rot_w,)
        right_foot_local = localize_footstep(right_foot_pos_w,root_pos_w,root_rot_w,)
        left_contact = left_foot_pos_w[:,2] < 0.05
        right_contact = right_foot_pos_w[:,2] < 0.05
        left_landing = left_contact & (~self.prev_left_contact)
        right_landing = right_contact & (~self.prev_right_contact)
        self.left_landing_pos[left_landing] = left_foot_local[left_landing,:2]
        self.right_landing_pos[right_landing] = right_foot_local[right_landing,:2]

        left_error = torch.norm(
            self.left_landing_pos
            - self.footstep_targets[:,0:2],
            dim=-1,
        )
        right_error = torch.norm(
            self.right_landing_pos
            - self.footstep_targets[:,2:4],
            dim=-1,
        )
        landed = left_landing | right_landing
        footstep_reward = torch.zeros(
            self.num_envs,
            device=self.device
        )
        footstep_reward[left_landing] += torch.exp(
            -10.0 * left_error[left_landing]
        )
        footstep_reward[right_landing] += torch.exp(
            -10.0 * right_error[right_landing]
        )

        # 報酬の合成---------------------------------------------------------------------------
        reward = (
            1.0 * forward_reward
            + 0.5 * height_reward
            + 0.5 * footstep_reward
            # + 0.5 * heading_reward
        )

        self.prev_left_contact = left_contact
        self.prev_right_contact = right_contact

        # 1000ステップごとにログを出力
        if self.common_step_counter % 1000 == 0:
            tqdm.write(
                f"step={self.common_step_counter} "
                f"height={root_height.mean().item():.3f} "
                f"forward={forward_reward.mean().item():.3f} "
                f"height_r={height_reward.mean().item():.3f} "
                f"heading={heading_reward.mean().item():.3f}"
                f"footstep={footstep_reward.mean().item():.3f}"
            )

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
            & (right_shin_h < shin_height_thresh)
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

        self.prev_left_contact[env_ids] = False
        self.prev_right_contact[env_ids] = False
        self.left_landing_pos[env_ids] = 0.0
        self.right_landing_pos[env_ids] = 0.0
        # if self.use_biased_yaw:
        #     # 1. 確率分布に従って、各環境が担当する Bin (0 ~ 9) をサンプリング
        #     # replacement=True で重複を許して環境の数だけ引く
        #     chosen_bins = torch.multinomial(self.yaw_bin_probs, len(env_ids), replacement=True)
            
        #     # 2. 選ばれたBinの左端の角度を計算 (-pi からスタートして 10等分)
        #     bin_width = 2.0 * torch.pi / 10.0
        #     low_angles = -torch.pi + chosen_bins.float() * bin_width
            
        #     # 3. Binの範囲内（low_angles 〜 low_angles + bin_width）で一様ランダムなノイズを加える
        #     # これにより「特定のBinの中のどこか」に綺麗に分散させる
        #     rand_offset = torch.rand(len(env_ids), device=self.device) * bin_width
        #     self.goal_yaw[env_ids] = low_angles + rand_offset
            
        # else:
        #     # ファイルがない場合は従来通りの完全一様ランダム
        #     self.goal_yaw[env_ids] = (
        #         torch.rand(len(env_ids), device=self.device)
        #         * 2.0 * torch.pi
        #         - torch.pi
        #     )
        # self.goal_yaw[env_ids] = (
        #     torch.rand(len(env_ids), device=self.device)
        #     * 2.0 * torch.pi
        #     - torch.pi
        # )#世界座標基準（ワールド基準）の yaw
        #追加 初期の目標の向きをランダムにしている
        # self.goal_yaw[env_ids] = torch.pi*0.55
        # self.goal_yaw[env_ids] = torch.pi / 3

    # reset strategies

    def _reset_strategy_default(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
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
        #追加
        zeros = torch.zeros(num_samples * self.cfg.num_amp_observations, device=self.device)
        ones = torch.ones(num_samples * self.cfg.num_amp_observations, device=self.device)
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
            zeros,
            ones,
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
    heading_sin: torch.Tensor,
    heading_cos: torch.Tensor,
    footstep_targets: torch.Tensor,
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
            heading_sin.unsqueeze(-1),
            heading_cos.unsqueeze(-1),
            footstep_targets,
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
            key_body_positions.view(key_body_positions.shape[0], -1),
            #追加
            heading_sin.unsqueeze(-1),
            heading_cos.unsqueeze(-1),
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





