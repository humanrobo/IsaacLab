from __future__ import annotations
import os
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
from isaaclab_assets import HUMANOID_28_CFG
from .humanoid_amp_env_cfg import HumanoidAmpEnvCfg
import os
from isaaclab.utils import configclass

MOTIONS_DIR = os.path.join(os.path.dirname(__file__), "motions")

@configclass
class UnicycleHumanoid28EnvCfg(HumanoidAmpEnvCfg):
    episode_length_s = 20.0
    decimation = 2
    motion_file = os.path.join(
        MOTIONS_DIR,
        "humanoid_walk.npz"
    )
    robot: ArticulationCfg = HUMANOID_28_CFG.replace(
        prim_path="/World/envs/env_.*/Humanoid"
    ).replace(
        actuators={
            "body": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=None,
                damping=None,
                velocity_limit_sim={".*": 100.0},
            ),
        },
    ).replace(
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(-1.0, 0.0, 0.4),
        )
    )
    unicycle: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Unicycle",
        spawn=sim_utils.CylinderCfg(
            radius=0.25,
            height=0.5,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 0.4, 0.8)
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.25),
        ),
    )
    camera = CameraCfg(
        prim_path="/World/envs/env_.*/Unicycle/Camera",
        update_period=0.0,
        height=64,
        width=64,
        data_types=[
            "rgb",
            "distance_to_image_plane",
            "semantic_segmentation",
        ],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.25, 0.0, 0.0),
            rot=(-0.5, 0.5, -0.5, 0.5),
            convention="ros",
        ),
    )
    unicycle_observation_space = {
        "policy_obs": 10,
        "ray_heightmap": [1, 64, 64],
    }


    obstacle1 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Obstacle1",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 0.5, 0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(100, -0.8, 0.25),
        ),
    )

    obstacle2 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Obstacle2",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 0.5, 0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(2.5, 0.0, 0.25),
        ),
    )

    obstacle3 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Obstacle3",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 0.5, 0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(2.5, 0.8, 0.25),
        ),
    )
    obstacle_long = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Obstaclelong",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 2.0, 0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(2.5, 0.0, 0.25),
        ),
    )
    obstacle_wallr = RigidObjectCfg(
        prim_path="/World/envs/env_.*/ObstacleWallR",
        spawn=sim_utils.CuboidCfg(
            size=(8.0, 0.5, 0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(2.5, 0.0, 0.25),
        ),
    )
    obstacle_walll = RigidObjectCfg(
        prim_path="/World/envs/env_.*/ObstacleWallL",
        spawn=sim_utils.CuboidCfg(
            size=(8.0, 0.5, 0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(2.5, 0.0, 0.25),
        ),
    )
    obstacle_pushable1 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/PushableObstacle1",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 0.4, 0.5),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 0.0, 1.0),
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            semantic_tags=[("class", "pushable")],
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(2.0, 0.0, 0.1)),
    )
    obstacle_pushable2 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/PushableObstacle2",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 0.4, 0.5),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 0.0, 1.0),
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            semantic_tags=[("class", "pushable")],
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(3.0, 0.0, 0.1)),
    )
    obstacle_pushable3 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/PushableObstacle3",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 0.4, 0.5),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 0.0, 1.0),
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            semantic_tags=[("class", "pushable")],
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(4.0, 0.0, 0.1)),
    )

@configclass
class UnicycleHumanoid28WalkEnvCfg(UnicycleHumanoid28EnvCfg):
    motion_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "motions/humanoid_walk.npz"
    )