from __future__ import annotations
import torch
import torch.nn as nn
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import Camera
from isaaclab.sim import GroundPlaneCfg
import isaaclab.sim as sim_utils
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply_inverse, yaw_quat
from isaaclab.sim.spawners import spawn_ground_plane
from .humanoid_amp_env import HumanoidAmpEnv
from .unicycle_hlc_humanoid_env_cfg import UnicycleHLCHumanoid28EnvCfg
from ..unicycle.heightmap_generator import HeightMapGenerator
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import SPHERE_MARKER_CFG
import ast
from .humanoid_amp_env import (
    compute_policy_obs,
    compute_amp_obs,
    localize_observation,
)
import gymnasium as gym
import numpy as np

class LLCPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net_container = nn.Sequential(
            nn.Linear(87, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 28),
        )
        self.log_std_parameter = nn.Parameter(
            torch.full((28,), -2.9)
        )

    def forward(self, obs):
        return self.net_container(obs)

class UnicycleHLCHumanoidEnv(HumanoidAmpEnv):
    cfg: UnicycleHLCHumanoid28EnvCfg
    def __init__(self, cfg, render_mode=None, **kwargs):
        self.obstacle_stage = 0
        self._heightmap_generator = None
        self.pushable_color = None
        super().__init__(cfg, render_mode, **kwargs)
        # HLC observation space
        hlc_policy_space = gym.spaces.Dict({
            "policy_obs": gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(10,),
                dtype=np.float32,
            ),
            "ray_heightmap": gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(1, 64, 64),
                dtype=np.float32,
            ),
        })

        self.single_observation_space = gym.spaces.Dict({
            "policy": hlc_policy_space,
        })

        self.observation_space = gym.spaces.Dict({
            "policy": hlc_policy_space,
        })
        marker_cfg = SPHERE_MARKER_CFG.copy()
        marker_cfg.prim_path = "/Visuals/UnicycleGoalMarker"
        marker_cfg.markers["sphere"].scale = (0.4, 0.4, 0.4)
        marker_cfg.markers["sphere"].visual_material.diffuse_color = (1.0, 0.0, 0.0)
        self.goal_marker = VisualizationMarkers(marker_cfg)
        # front_marker_cfg = SPHERE_MARKER_CFG.copy()
        # front_marker_cfg.prim_path = "/Visuals/UnicycleFrontMarker"
        # front_marker_cfg.markers["sphere"].scale = (0.15, 0.15, 0.15)
        # front_marker_cfg.markers["sphere"].visual_material.diffuse_color = (1.0, 0.0, 0.0)
        # self.front_marker = VisualizationMarkers(front_marker_cfg)        
        self._heightmap_generator_humanoid = HeightMapGenerator(resolution=0.05, map_size=3.2, device=self.device, gui_enabled=False, num_envs=self.num_envs,)
        # self._heightmap_generator = HeightMapGenerator(resolution=0.05, map_size=3.2, device=self.device, gui_enabled=False)

        # LLCポリシーのロード
        self._llc_policy_path = (
            "logs/skrl/humanoid_amp_walk/"
            "2026-11-20_13-39-02_amp_torch/checkpoints/best_agent.pt"
        )
        self._llc_policy = LLCPolicy().to(self.device)
        llc_checkpoint = torch.load(
            self._llc_policy_path,
            map_location=self.device,
        )
        self._llc_policy.load_state_dict(
            llc_checkpoint["policy"],
            strict=True,
        )
        self._llc_policy.eval()
        for param in self._llc_policy.parameters():
            param.requires_grad_(False)
        # LLC Observationの正規化パラメータ
        self.llc_running_mean = (
            llc_checkpoint["observation_preprocessor"]["running_mean"]
            .to(self.device).float()
        )
        self.llc_running_variance = (
            llc_checkpoint["observation_preprocessor"]["running_variance"]
            .to(self.device).float()
        )

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        # self.unicycle = RigidObject(self.cfg.unicycle)
        # self.camera = Camera(self.cfg.camera)
        self.camera_humanoid = Camera(self.cfg.camera_humanoid)
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
        self.scene.articulations["robot"] = self.robot
        # self.scene.rigid_objects["unicycle"] = self.unicycle
        # self.scene.sensors["camera"] = self.camera
        self.scene.sensors["camera_humanoid"] = self.camera_humanoid
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        # self._llc_action = None
        # # 固定速度指令
        # self.target_v = torch.full(
        #     (self.num_envs,), 1.0, device=self.device
        # )
        # self.target_omega = torch.full(
        #     (self.num_envs,), -1.0, device=self.device
        # )
        dt = self.cfg.sim.dt * self.cfg.decimation
        # HLC出力
        actions = torch.clamp(actions, -1.0, 1.0)
        self.target_v = torch.clamp(actions[:, 0], min=0.0)
        self.target_omega = actions[:, 1]
        # 目標角速度を積分して目標方向を更新
        self.target_yaw = torch.atan2(
            torch.sin(self.target_yaw + self.target_omega * dt),
            torch.cos(self.target_yaw + self.target_omega * dt),
        )
        # #semseg各クラスがID割り当てられたときに、pushableのIDを探す
        # if self.pushable_color is None:
        #     semantic_info = self.camera_humanoid.data.info[0]["semantic_segmentation"]["idToLabels"]
        #     self.pushable_color = next(
        #         ast.literal_eval(k) for k, v in semantic_info.items()
        #         if v.get("class") == "pushable"
        #     )

    def _get_observations(self):
        self._update_humanoid_heightmap()
        root_pos_w = self.robot.data.root_pos_w
        root_quat_w = self.robot.data.root_quat_w
        root_lin_vel_w = self.robot.data.root_lin_vel_w
        root_ang_vel_w = self.robot.data.root_ang_vel_w
        _, _, robot_yaw = euler_xyz_from_quat(root_quat_w)
        self.goal_pos_w = (
            self.scene.env_origins[:, :2]
            + torch.tensor([6.0, 0.0], device=self.device)
        )
        # ゴールマーカーの位置を更新
        goal_marker_pos = torch.cat(
            [
                self.goal_pos_w,
                torch.full(
                    (self.num_envs, 1),
                    0.1,
                    device=self.device,
                ),
            ],
            dim=-1,
        )
        self.goal_marker.visualize(translations=goal_marker_pos)
        local_lin_vel = quat_apply_inverse(
            yaw_quat(root_quat_w), root_lin_vel_w
        )
        local_ang_vel = quat_apply_inverse(
            yaw_quat(root_quat_w), root_ang_vel_w
        )
        local_lin_vel[:, 2] = 0.0
        local_ang_vel[:, :2] = 0.0
        goal_vec_w = self.goal_pos_w - root_pos_w[:, :2]
        goal_vec_3d = torch.cat(
            [goal_vec_w, torch.zeros(self.num_envs, 1, device=self.device)],
            dim=-1,
        )
        goal_vec_local = quat_apply_inverse(
            yaw_quat(root_quat_w), goal_vec_3d
        )[:, :2]
        target_yaw = torch.atan2(goal_vec_w[:, 1], goal_vec_w[:, 0])
        heading_error = target_yaw - robot_yaw
        heading_error = torch.atan2(
            torch.sin(heading_error),
            torch.cos(heading_error),
        )
        policy_obs = torch.cat(
            (
                local_lin_vel,
                local_ang_vel,
                torch.sin(heading_error).unsqueeze(-1),
                torch.cos(heading_error).unsqueeze(-1),
                goal_vec_local,
            ),
            dim=-1,
        )
        heightmap = self.humanoid_heightmap.unsqueeze(1)
        hlc_obs = torch.cat(
            [policy_obs, heightmap.flatten(1)],
            dim=-1,
        )
        
        return {
            "policy": {
                "policy_obs": policy_obs,
                "ray_heightmap": heightmap,
            }
        }

    def _apply_action(self):
        llc_obs = self._compute_llc_observation()
        # LLC学習時と同じObservation正規化
        llc_obs = (
            llc_obs - self.llc_running_mean
        ) / torch.sqrt(self.llc_running_variance + 1e-8)
        # LLCの決定論的推論
        with torch.inference_mode():
            llc_action = self._llc_policy(llc_obs)
        target = self.action_offset + self.action_scale * llc_action
        self.robot.set_joint_position_target(target)
    # def _apply_action(self):
    #     # LLC推論は1回だけ
    #     if self._llc_action is None:
    #         llc_obs = self._compute_llc_observation()

    #         # LLC学習時と同じObservation正規化
    #         llc_obs = (
    #             llc_obs - self.llc_running_mean
    #         ) / torch.sqrt(self.llc_running_variance + 1e-8)

    #         with torch.inference_mode():
    #             self._llc_action = self._llc_policy(llc_obs)

    #     # 2回とも同じactionを適用
    #     target = self.action_offset + self.action_scale * self._llc_action
    #     self.robot.set_joint_position_target(target)

    def _compute_llc_observation(self):
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

        return policy_obs

    def _update_humanoid_heightmap(self):
        humanoid_pos_w = self.robot.data.root_pos_w
        humanoid_quat_w = self.robot.data.root_quat_w
        _, _, humanoid_yaw = euler_xyz_from_quat(humanoid_quat_w)
        depth = self.camera_humanoid.data.output["distance_to_image_plane"]
        semantic = self.camera_humanoid.data.output["semantic_segmentation"]
        height_map = self._heightmap_generator_humanoid.generate_from_depth(
            depth,
            self.camera_humanoid,
            humanoid_pos_w,
            humanoid_yaw,
            1.4,  # カメラのワールドZ座標
            semantic,
            self.pushable_color,
        )
        self.humanoid_heightmap = height_map

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        died, time_out = super()._get_dones()
        root_pos_w = self.robot.data.root_pos_w
        goal_dist = torch.norm(self.goal_pos_w - root_pos_w[:, :2], dim=1)
        goal_reached = goal_dist < 0.3
        self.extras["reached_goal"] = goal_reached.clone()
        self.extras["time_out"] = time_out.clone()
        died |= goal_reached
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        super()._reset_idx(env_ids)
        num_envs = len(env_ids)
        if self.obstacle_stage == 0:
            pass
        elif self.obstacle_stage == 1:
            obstacle1_state = self.obstacle1.data.default_root_state[env_ids].clone()
            obstacle1_state[:, :3] = self.scene.env_origins[env_ids] + torch.tensor([2.5, 0.0, 0.25], device=self.device)
            self.obstacle1.write_root_pose_to_sim(obstacle1_state[:, :7], env_ids)
        elif self.obstacle_stage == 5:
            # 障害物4個・ランダム位置
            for obstacle in [self.obstacle1, self.obstacle2, self.obstacle3, self.obstacle_long]:
                pos = torch.zeros((num_envs, 3), device=self.device)
                pos[:, 0] = torch.empty(num_envs, device=self.device).uniform_(0.5, 4.5)
                pos[:, 1] = torch.empty(num_envs, device=self.device).uniform_(-3.5, 3.5)
                pos[:, 2] = 0.25
                obstacle_state = obstacle.data.default_root_state[env_ids].clone()
                obstacle_state[:, :3] = pos + self.scene.env_origins[env_ids]
                obstacle.write_root_pose_to_sim(obstacle_state[:, :7], env_ids)
        elif self.obstacle_stage == 7:
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
                pos[:, 2] = 1.0
                obstacle_state = obstacle.data.default_root_state[env_ids].clone()
                obstacle_state[:, :3] = pos + self.scene.env_origins[env_ids]
                obstacle.write_root_pose_to_sim(obstacle_state[:, :7], env_ids)
            for wall, y in [(self.obstacle_wallr, -2.0), (self.obstacle_walll, 2.0)]:
                wall_state = wall.data.default_root_state[env_ids].clone()
                wall_state[:, :3] = torch.tensor([3.0, y, 1.0], device=self.device) + self.scene.env_origins[env_ids]
                wall.write_root_pose_to_sim(wall_state[:, :7], env_ids)
