# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from dataclasses import MISSING

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from isaaclab.utils import configclass
import isaaclab.sim as sim_utils

from isaaclab_assets import HUMANOID_28_CFG

from isaaclab.sensors import ContactSensor, ContactSensorCfg

MOTIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "motions")


@configclass
class HumanoidAmpEnvCfg(DirectRLEnvCfg):
    """Humanoid AMP environment config (base class)."""

    # env
    episode_length_s = 20.0
    decimation = 2

    # spaces
    observation_space = 87#次元を2+4つ追加
    action_space = 28
    state_space = 0
    num_amp_observations = 2
    amp_observation_space = 81#次元を2+4つ追加

    early_termination = True
    termination_height = 0.5

    motion_file: str = MISSING
    reference_body = "torso"
    reset_strategy = "default"  # default, random, random-start
    """Strategy to be followed when resetting each environment (humanoid's pose and joint states).

    * default: pose and joint states are set to the initial state of the asset.
    * random: pose and joint states are set by sampling motions at random, uniform times.
    * random-start: pose and joint states are set by sampling motion at the start (time zero).
    """

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
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=10.0, replicate_physics=True)

    # robot
    robot: ArticulationCfg = HUMANOID_28_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        spawn=HUMANOID_28_CFG.spawn.replace(
            activate_contact_sensors=True,
        ),
    ).replace(
        actuators={
            "body": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=None,
                damping=None,
                velocity_limit_sim={".*": 100.0},
            ),
        },
    )
    contact_forces = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*_foot",
        history_length=3,
        track_air_time=True,
        update_period=0.0,
    )

    obstacle_pushable1 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/PushableObstacle1",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 0.4, 2.0),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 0.0, 1.0),
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                solver_position_iteration_count=4,),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            semantic_tags=[("class", "pushable")],
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(3.0, 0.0, 2.0)),
    )


@configclass
class HumanoidAmpDanceEnvCfg(HumanoidAmpEnvCfg):
    motion_file = os.path.join(MOTIONS_DIR, "humanoid_dance.npz")


@configclass
class HumanoidAmpRunEnvCfg(HumanoidAmpEnvCfg):
    motion_file = os.path.join(MOTIONS_DIR, "humanoid_run.npz")


@configclass
class HumanoidAmpWalkEnvCfg(HumanoidAmpEnvCfg):
    motion_file = os.path.join(MOTIONS_DIR, "humanoid_walk.npz")
