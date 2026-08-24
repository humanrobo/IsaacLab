# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg

@configclass
class UnicycleEnvCfg(DirectRLEnvCfg):
    """Unicycle environment config (cube base model)."""

    # env
    episode_length_s = 5.0
    decimation = 2

    # spaces (ユニサイクルモデルの入力・出力に合わせて変更)
    # 例: 観測空間 = local_lin_vel(3) + local_ang_vel(3) + height(1) + heading_sin(1) + heading_cos(1) + goal_vec_local(2) = 11次元
    observation_space = 11
    action_space = 2
    state_space = 0

    early_termination = True
    termination_height = 0.2

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
        render_interval=decimation,
        physx=PhysxCfg(
            gpu_found_lost_pairs_capacity=2**23,
            gpu_total_aggregate_pairs_capacity=2**23,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, 
        env_spacing=10.0, 
        replicate_physics=True
    )

    # robot (ヒューマノイドの設定を外し、キューブ等のRigidObjectCfgに置き換え)
# robot
    robot: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 0.5, 0.5),  # キューブのサイズ（例: 0.5m四方）
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.4, 0.8)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.25),
        ),
    )

    obstacle = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Obstacle",
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

    camera = CameraCfg(
        prim_path="/World/envs/env_.*/Robot/Camera",
        update_period=0.1,
        height=240,
        width=320,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.5, 0.0, 0),
            rot=(-0.5, 0.5, -0.5, 0.5),
            convention="ros",
        ),
    )
