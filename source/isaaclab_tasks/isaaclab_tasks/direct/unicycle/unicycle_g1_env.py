import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import Camera
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from pxr import Gf, UsdGeom, UsdPhysics
from .unicycle_env import UnicycleEnv
from .unicycle_g1_env_cfg import UnicycleG1EnvCfg
import torch
from agile.rl_env.tasks.locomotion.g1.velocity_history_env_cfg import CONTROLLED_JOINT_NAMES
import re
from isaaclab.utils.buffers import CircularBuffer

class UnicycleG1Env(UnicycleEnv):
    cfg: UnicycleG1EnvCfg
    def __init__(self, cfg, render_mode=None, **kwargs):
        self.g1 = None
        self.g1_controlled_joint_names = CONTROLLED_JOINT_NAMES
        self.g1_policy = torch.jit.load(
            "/home/matsuno/WBC-AGILE/agile/data/policy/velocity_g1/unitree_g1_velocity_history.pt",
            map_location="cuda:0"
        )
        self.g1_policy.eval()
        self.g1_velocity_command = None
        self.g1_obs_history = None
        self.g1_last_action = None
        super().__init__(cfg, render_mode, **kwargs)
        self.g1_joint_ids = [
            i for i, name in enumerate(self.g1.joint_names)
            if any(re.fullmatch(pattern, name) for pattern in CONTROLLED_JOINT_NAMES)
        ]
        self.g1_velocity_command = torch.zeros((self.num_envs, 3), device=self.device)
        self.g1_obs_history = CircularBuffer(
            max_len=5,
            batch_size=self.num_envs,
            device=self.device,
        )
        self.g1_last_action = torch.zeros((self.num_envs, 14), device=self.device)
        self.g1_policy_timer = 0.0

    def _setup_scene(self):
        self.robot = RigidObject(self.cfg.robot)
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
        self.g1 = Articulation(self.cfg.g1)
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])
        self.scene.rigid_objects["robot"] = self.robot
        self.scene.sensors["camera"] = self.camera
        self.scene.articulations["g1"] = self.g1
        light_cfg = sim_utils.DomeLightCfg(
            intensity=2000.0,
            color=(0.75, 0.75, 0.75),
        )
        light_cfg.func("/World/Light", light_cfg)

    def _step_g1_policy(self):
        n = self.num_envs
        joint_ids = self.g1_joint_ids
        command = self.g1_velocity_command
        base_ang_vel = self.g1.data.root_ang_vel_b * 0.2
        projected_gravity = self.g1.data.projected_gravity_b
        joint_pos = self.g1.data.joint_pos[:, joint_ids] - self.g1.data.default_joint_pos[:, joint_ids]
        joint_vel = self.g1.data.joint_vel[:, joint_ids] * 0.05
        current_obs = torch.cat([
            command,
            base_ang_vel,
            projected_gravity,
            joint_pos,
            joint_vel,
            self.g1_last_action,
        ], dim=-1)
        self.g1_obs_history.append(current_obs)
        g1_obs = self.g1_obs_history.buffer.reshape(n, -1)
        g1_action = self.g1_policy(g1_obs)
        print("G1 obs:", g1_obs.shape)
        print("G1 action:", g1_action.shape)
        target = self.g1.data.default_joint_pos[:, joint_ids] + 0.5 * g1_action[:, :14]
        self.g1.set_joint_position_target(target, joint_ids=joint_ids)
        self.g1_last_action = g1_action[:, :14].clone()

    def _apply_action(self):
        super()._apply_action()
        v = self.actions[:, 0] * self.action_scale_lin
        omega = self.actions[:, 1] * self.action_scale_ang
        self.g1_velocity_command[:, 0] = 0.0#v
        self.g1_velocity_command[:, 1] = 0.0
        self.g1_velocity_command[:, 2] = 0.0#omega
        self._step_g1_policy()


    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        self.g1.reset(env_ids)
        self.g1_velocity_command[env_ids] = 0.0
        self.g1_obs_history.reset(env_ids)
        self.g1_last_action[env_ids] = 0.0