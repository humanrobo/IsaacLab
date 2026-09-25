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
from .unicycle_human28_env_cfg import UnicycleHumanoid28EnvCfg
from ..unicycle.heightmap_generator import HeightMapGenerator
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import SPHERE_MARKER_CFG

class UnicyclePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.heightmap_features_container = nn.Sequential(
            nn.Conv2d(1, 16, 5, 2, 0),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 0),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 2, 0),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.net_container = nn.Sequential(
            nn.Linear(1162, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 2),
        )
        self.log_std_parameter = nn.Parameter(torch.full((2,), -1.0))

    def forward(self, ray_heightmap, policy_obs):
        heightmap_features = self.heightmap_features_container(ray_heightmap)
        x = torch.cat([heightmap_features, policy_obs], dim=-1)
        return self.net_container(x)

class UnicycleHumanoid28Env(HumanoidAmpEnv):
    cfg: UnicycleHumanoid28EnvCfg

    def __init__(self, cfg, render_mode=None, **kwargs):
        self.obstacle_stage = 1
        self._unicycle_policy_path = "logs/skrl/unicycle_navigation/2026-09-10_15-04-51_ppo_torch/checkpoints/best_agent.pt"
        self._unicycle_policy = None
        self._heightmap_generator = None
        self._unicycle_action = None
        super().__init__(cfg, render_mode, **kwargs)
        marker_cfg = SPHERE_MARKER_CFG.copy()
        marker_cfg.prim_path = "/Visuals/UnicycleGoalMarker"
        marker_cfg.markers["sphere"].scale = (0.4, 0.4, 0.4)
        marker_cfg.markers["sphere"].visual_material.diffuse_color = (1.0, 0.0, 0.0)
        self.goal_marker = VisualizationMarkers(marker_cfg)
        front_marker_cfg = SPHERE_MARKER_CFG.copy()
        front_marker_cfg.prim_path = "/Visuals/UnicycleFrontMarker"
        front_marker_cfg.markers["sphere"].scale = (0.15, 0.15, 0.15)
        front_marker_cfg.markers["sphere"].visual_material.diffuse_color = (1.0, 0.0, 0.0)
        self.front_marker = VisualizationMarkers(front_marker_cfg)
        self._unicycle_policy = UnicyclePolicy().to(self.device)
        checkpoint = torch.load(self._unicycle_policy_path, map_location=self.device)
        self._unicycle_policy.load_state_dict(checkpoint["policy"], strict=True)
        self._unicycle_policy.eval()
        for param in self._unicycle_policy.parameters():
            param.requires_grad_(False)
        self.unicycle_running_mean = checkpoint["observation_preprocessor"]["running_mean"].to(self.device).float()
        self.unicycle_running_variance = checkpoint["observation_preprocessor"]["running_variance"].to(self.device).float()
        self._heightmap_generator = HeightMapGenerator(resolution=0.05, map_size=3.2, device=self.device, gui_enabled=False)
        self._unicycle_action = torch.zeros(self.num_envs, 2, device=self.device)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.unicycle = RigidObject(self.cfg.unicycle)
        self.camera = Camera(self.cfg.camera)
        if self.obstacle_stage in [1, 2, 3, 4, 5, 7, 9, 10]:
            self.obstacle1 = RigidObject(self.cfg.obstacle1)
            self.scene.rigid_objects["obstacle1"] = self.obstacle1
        if self.obstacle_stage in [2, 5, 7, 9, 10]:
            self.obstacle2 = RigidObject(self.cfg.obstacle2)
            self.scene.rigid_objects["obstacle2"] = self.obstacle2
        if self.obstacle_stage == 5:
            self.obstacle3 = RigidObject(self.cfg.obstacle3)
            self.scene.rigid_objects["obstacle3"] = self.obstacle3
        if self.obstacle_stage in [5, 6]:
            self.obstacle_long = RigidObject(self.cfg.obstacle_long)
            self.scene.rigid_objects["obstacle_long"] = self.obstacle_long
        if self.obstacle_stage in [7, 8, 9, 10]:
            self.obstacle_wallr = RigidObject(self.cfg.obstacle_wallr)
            self.scene.rigid_objects["obstacle_wallr"] = self.obstacle_wallr
        if self.obstacle_stage in [7, 8, 9, 10]:
            self.obstacle_walll = RigidObject(self.cfg.obstacle_walll)
            self.scene.rigid_objects["obstacle_walll"] = self.obstacle_walll
        if self.obstacle_stage in [8, 9, 10]:
            self.obstacle_pushable1 = RigidObject(self.cfg.obstacle_pushable1)
            self.scene.rigid_objects["obstacle_pushable1"] = self.obstacle_pushable1
        if self.obstacle_stage == 10:
            self.obstacle_pushable2 = RigidObject(self.cfg.obstacle_pushable2)
            self.scene.rigid_objects["obstacle_pushable2"] = self.obstacle_pushable2
        if self.obstacle_stage == 10:
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
        self.scene.rigid_objects["unicycle"] = self.unicycle
        self.scene.sensors["camera"] = self.camera
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = torch.clamp(actions, -1.0, 1.0)
        self._compute_unicycle_action()

    def _apply_action(self):
        target = self.action_offset + self.action_scale * self.actions
        self.robot.set_joint_position_target(target)
        v = self._unicycle_action[:, 0]
        omega = self._unicycle_action[:, 1]
        _, _, yaw = euler_xyz_from_quat(self.unicycle.data.root_quat_w)
        vx = v * torch.cos(yaw)
        vy = v * torch.sin(yaw)
        root_velocity = torch.zeros(self.num_envs, 6, device=self.device)
        root_velocity[:, 0] = vx
        root_velocity[:, 1] = vy
        root_velocity[:, 5] = omega
        self.unicycle.write_root_com_velocity_to_sim(root_velocity)
        
    def _compute_unicycle_action(self):
        root_pos_w = self.unicycle.data.root_pos_w
        root_quat_w = self.unicycle.data.root_quat_w
        root_lin_vel_w = self.unicycle.data.root_lin_vel_w
        root_ang_vel_w = self.unicycle.data.root_ang_vel_w
        _, _, robot_yaw = euler_xyz_from_quat(root_quat_w)
        goal_pos_w = self.scene.env_origins[:, :2] + torch.tensor([6.0, 0.0], device=self.device)
        marker_pos_w = torch.cat([goal_pos_w, torch.full((self.num_envs, 1), 0.2, device=self.device)], dim=-1)
        marker_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        self.goal_marker.visualize(marker_pos_w, marker_quat)
        front_offset = 0.8
        front_marker_pos = torch.stack([
            root_pos_w[:, 0] + front_offset * torch.cos(robot_yaw),
            root_pos_w[:, 1] + front_offset * torch.sin(robot_yaw),
            root_pos_w[:, 2] + 0.2,
        ], dim=-1)
        self.front_marker.visualize(front_marker_pos, marker_quat)
        local_lin_vel = quat_apply_inverse(yaw_quat(root_quat_w), root_lin_vel_w)
        local_ang_vel = quat_apply_inverse(yaw_quat(root_quat_w), root_ang_vel_w)
        goal_vec_w = goal_pos_w - root_pos_w[:, :2]
        goal_vec_3d = torch.cat([goal_vec_w, torch.zeros(self.num_envs, 1, device=self.device)], dim=-1)
        goal_vec_local = quat_apply_inverse(yaw_quat(root_quat_w), goal_vec_3d)[:, :2]
        target_yaw = torch.atan2(goal_vec_w[:, 1], goal_vec_w[:, 0])
        heading_error = target_yaw - robot_yaw
        heading_error = torch.atan2(torch.sin(heading_error), torch.cos(heading_error))
        heading_sin = torch.sin(heading_error)
        heading_cos = torch.cos(heading_error)
        policy_obs = torch.cat((
            local_lin_vel,
            local_ang_vel,
            heading_sin.unsqueeze(-1),
            heading_cos.unsqueeze(-1),
            goal_vec_local,
        ), dim=-1)
        depth = self.camera.data.output["distance_to_image_plane"]
        semantic = self.camera.data.output["semantic_segmentation"]
        height_map = self._heightmap_generator.generate_from_depth(
            depth, self.camera, root_pos_w, robot_yaw, semantic, None
        )
        height_map = height_map.unsqueeze(1)
        flat_obs = torch.cat([policy_obs, height_map.flatten(1)], dim=-1)
        assert flat_obs.shape[1] == self.unicycle_running_mean.shape[0], (
            f"Unicycle observation size mismatch: {flat_obs.shape[1]} != {self.unicycle_running_mean.shape[0]}"
        )
        flat_obs = (flat_obs - self.unicycle_running_mean) / torch.sqrt(self.unicycle_running_variance + 1e-8)
        policy_obs_normalized = flat_obs[:, :10]
        height_map_normalized = flat_obs[:, 10:].reshape(self.num_envs, 1, 64, 64)
        with torch.inference_mode():
            self._unicycle_action = self._unicycle_policy(height_map_normalized, policy_obs_normalized)
        self._unicycle_action[:, 0] = torch.clamp(self._unicycle_action[:, 0], -1.0, 1.0)
        self._unicycle_action[:, 1] = torch.clamp(self._unicycle_action[:, 1], -1.0, 1.0)
        self._unicycle_action[:, 0] *= 1.0
        self._unicycle_action[:, 1] *= 2.0
        prediction_time = 0.5
        self.goal_yaw = robot_yaw #+ self._unicycle_action[:, 1] * prediction_time
        self.goal_yaw = torch.atan2(torch.sin(self.goal_yaw), torch.cos(self.goal_yaw))

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