# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script shows how to use the camera sensor from the Isaac Lab framework.

The camera sensor is created and interfaced through the Omniverse Replicator API. However, instead of using
the simulator or OpenGL convention for the camera, we use the robotics or ROS convention.

.. code-block:: bash

    # Usage with GUI
    ./isaaclab.sh -p scripts/tutorials/04_sensors/run_usd_camera.py --enable_cameras

    # Usage with headless
    ./isaaclab.sh -p scripts/tutorials/04_sensors/run_usd_camera.py --headless --enable_cameras

"""

"""Launch Isaac Sim Simulator first."""

import argparse
import base64

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates how to use the camera sensor.")
parser.add_argument(
    "--draw",
    action="store_true",
    default=False,
    help="Draw the pointcloud from camera at index specified by ``--camera_id``.",
)
parser.add_argument(
    "--save",
    action="store_true",
    default=False,
    help="Save the data from camera at index specified by ``--camera_id``.",
)
parser.add_argument(
    "--camera_id",
    type=int,
    choices={0, 1},
    default=0,
    help=(
        "The camera ID to use for displaying points or saving the camera data. Default is 0."
        " The viewport will always initialize with the perspective of camera 0."
    ),
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import random

import numpy as np
import torch

import omni.replicator.core as rep

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import RAY_CASTER_MARKER_CFG
from isaaclab.sensors.camera import Camera, CameraCfg, camera
from isaaclab.sensors.camera.utils import create_pointcloud_from_depth
from isaaclab.utils import convert_dict_to_backend
import omni.ui as ui
from PIL import Image
import io
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import tempfile
from isaaclab.sensors.camera.utils import create_pointcloud_from_depth
from isaaclab.utils.math import create_rotation_matrix_from_view
import os
from isaaclab.sim import schemas

def define_sensor() -> Camera:
    """Defines the camera sensor to add to the scene."""
    # Setup camera sensor
    # In contrast to the ray-cast camera, we spawn the prim at these locations.
    # This means the camera sensor will be attached to these prims.
    sim_utils.create_prim("/World/Origin_00", "Xform")
    # sim_utils.create_prim("/World/Origin_01", "Xform")
    camera_cfg = CameraCfg(
        prim_path="/World/Origin_00/CameraSensor",  # ← ワイルドカ
        update_period=0,
        height=480,
        width=640,
        data_types=[
            "rgb",
            "distance_to_camera",
            "normals",
            "semantic_segmentation",
            "instance_segmentation_fast",
            "instance_id_segmentation_fast",
        ],
        colorize_semantic_segmentation=True,
        colorize_instance_id_segmentation=True,
        colorize_instance_segmentation=True,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=8.0, 
            focus_distance=400.0, 
            horizontal_aperture=30, 
            clipping_range=(0.1, 1.0e5)
        ),
    )
    # Create camera
    camera = Camera(cfg=camera_cfg)

    return camera


def design_scene() -> dict:
    """Design the scene."""
    # Populate scene
    # -- Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # -- Lights
    cfg = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Create a dictionary for the scene entities
    scene_entities = {}

    # Xform to hold objects
    sim_utils.create_prim("/World/Objects", "Xform")
    # Random objects
    for i in range(3):
        # sample random position
        # position = np.array([
        #     0.0,
        #     0.0,
        #     0.25
        # ])
        position = np.random.rand(3) - np.asarray([0.05, 0.05, -1.0])
        position *= np.asarray([1.5, 1.5, 0.5])
        # sample random color
        color = (random.random(), random.random(), random.random())
        # choose random prim type
        prim_type = random.choice(["Cube", "Cylinder"])
        common_properties = {
            "rigid_props": sim_utils.RigidBodyPropertiesCfg(),
            "mass_props": sim_utils.MassPropertiesCfg(mass=5.0),
            "collision_props": sim_utils.CollisionPropertiesCfg(),
            "visual_material": sim_utils.PreviewSurfaceCfg(diffuse_color=color, metallic=0.5),
            "semantic_tags": [("class", prim_type)],
        }
        if prim_type == "Cube":
            shape_cfg = sim_utils.CuboidCfg(size=(0.25, 0.25, 0.25), **common_properties)
        elif prim_type == "Cylinder":
            shape_cfg = sim_utils.CylinderCfg(radius=0.25, height=0.25, **common_properties)
        # Rigid Object
        obj_cfg = RigidObjectCfg(
            prim_path=f"/World/Objects/Obj_{i:02d}",
            spawn=shape_cfg,
            init_state=RigidObjectCfg.InitialStateCfg(pos=position),
        )
        scene_entities[f"rigid_object{i}"] = RigidObject(cfg=obj_cfg)

    # Sensors
    camera = define_sensor()

    # return the scene information
    scene_entities["camera"] = camera
    return scene_entities

resolution = 0.03
xmin, xmax = -5.0, 5.0
ymin, ymax = -5.0, 5.0

W = int((xmax-xmin)/resolution)
H = int((ymax-ymin)/resolution)

def create_height_map(points_world):

    height_map = np.full((H,W), np.nan)

    for x,y,z in points_world:

        if not np.isfinite(x):
            continue
        if not np.isfinite(y):
            continue
        if not np.isfinite(z):
            continue

        if z < 0.0:
            z = 0.0
        ix = int((x-xmin)/resolution)
        iy = int((y-ymin)/resolution)

        if 0 <= ix < W and 0 <= iy < H:
            # 【修正ポイント】Y軸を反転させて、画像上の上下を直感的な向きに合わせる
            iy_flipped = (H - 1) - iy

            if np.isnan(height_map[iy_flipped, ix]):
                            height_map[iy_flipped, ix] = z
            else:
                height_map[iy_flipped, ix] = max(
                    height_map[iy_flipped, ix],
                    z
                )

    height_map[np.isnan(height_map)] = 0.0

    height_map = gaussian_filter(
        height_map,
        sigma=1.0
    )

    return height_map

def heightmap_to_texture(height_map):

    hmin = height_map.min()
    hmax = height_map.max()

    if hmax - hmin < 1e-6:
        norm = np.zeros_like(height_map)
    else:
        norm = (height_map-hmin)/(hmax-hmin)

    rgb = plt.get_cmap("jet")(norm)

    rgb = (rgb[:,:,:3]*255).astype(np.uint8)

    img = Image.fromarray(rgb)

    buffer = io.BytesIO()
    img.save(
        buffer,
        format="PNG"
    )

    return buffer.getvalue()

def look_at_rotation(camera_pos, target):

    forward = target - camera_pos
    forward = forward / torch.norm(forward)

    world_up = torch.tensor(
        [0.0,0.0,1.0],
        device=camera_pos.device
    )

    right = torch.linalg.cross(
        forward,
        world_up
    )

    right = right / torch.norm(right)

    up = torch.linalg.cross(
        right,
        forward
    )

    R = torch.stack(
        [
            right,
            -up,
            forward
        ],
        dim=1
    )

    return R

def add_camera_marker(height_map, cam_pos):

    ix = int((cam_pos[0] - xmin) / resolution)
    iy = int((cam_pos[1] - ymin) / resolution)

    if 0 <= ix < W and 0 <= iy < H:
        # 【修正ポイント】ヒートマップ本体のY軸反転に合わせて、マーカーのY位置も反転させる
        iy_flipped = (H - 1) - iy

        height_map[
            max(0, iy_flipped - 3) : iy_flipped + 4,
            max(0, ix - 3) : ix + 4
        ] = height_map.max() + 0.2

    return height_map

def run_simulator(sim: sim_utils.SimulationContext, scene_entities: dict):

    window = ui.Window(
        "Height Map",
        width=600,
        height=600
    )
    with window.frame:
        image_widget = ui.Image()

    """Run the simulator."""
    # extract entities for simplified notation
    camera: Camera = scene_entities["camera"]

    # Create replicator writer
    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output", "camera")
    rep_writer = rep.BasicWriter(
        output_dir=output_dir,
        frame_padding=0,
        colorize_instance_id_segmentation=camera.cfg.colorize_instance_id_segmentation,
        colorize_instance_segmentation=camera.cfg.colorize_instance_segmentation,
        colorize_semantic_segmentation=camera.cfg.colorize_semantic_segmentation,
    )

    # Camera positions, targets, orientations
    camera_positions = torch.tensor([[2.5, 2.5, 2.5]], device=sim.device)
    camera_targets = torch.tensor([[0.0, 0.0, 0.0]], device=sim.device)
    camera_pos = camera_positions[0]
    target = camera_targets[0]
    R = look_at_rotation(
        camera_pos,
        target
    )
    # These orientations are in ROS-convention, and will position the cameras to view the origin
    camera_orientations = torch.tensor(  # noqa: F841
        [[-0.1759, 0.3399, 0.8205, -0.4247], [-0.4247, 0.8205, -0.3399, 0.1759]], device=sim.device
    )

    # -- Option-1: Set pose using view
    camera.set_world_poses_from_view(camera_positions, camera_targets)
    camera_index = 0

    # Create the markers for the --draw option outside of is_running() loop
    if sim.has_gui() and args_cli.draw:
        cfg = RAY_CASTER_MARKER_CFG.replace(prim_path="/Visuals/CameraPointCloud")
        cfg.markers["hit"].radius = 0.002
        pc_markers = VisualizationMarkers(cfg)

    # Simulate physics
    count = 0
    while simulation_app.is_running():
        sim.step()
        camera.update(dt=sim.get_physics_dt())
        count += 1
        
        if count % 10 != 0:
            continue

        # 1. 深度から点群への変換・フィルタリング
        depth = camera.data.output["distance_to_camera"][camera_index]
        depth = depth.squeeze(-1)
        
        points_cam = create_pointcloud_from_depth(
            intrinsic_matrix=camera.data.intrinsic_matrices[0],
            depth=depth,
            keep_invalid=False,
            device=sim.device
        )
        mask = torch.isfinite(points_cam).all(dim=1)
        points_cam_isaac = points_cam.clone()
        
        points_world = (
            R @ points_cam_isaac.T
        ).T + camera_pos
        
        mask = torch.isfinite(points_world).all(dim=1)
        points_world = points_world[mask]
        
        bounds_mask = (
            (points_world[:, 0] >= xmin) & (points_world[:, 0] <= xmax) &
            (points_world[:, 1] >= ymin) & (points_world[:, 1] <= ymax) &
            (points_world[:, 2] >= -1.0) & (points_world[:, 2] <= 2.0)
        )
        points_world = points_world[bounds_mask]
        points_world = points_world[torch.isfinite(points_world).all(dim=1)]

        # 2. ヒートマップとマーカーの生成
        height_map = create_height_map(points_world.cpu().numpy())
        height_map = add_camera_marker(height_map, camera_pos.cpu().numpy())

        # 3. 描画・保存処理（10ステップに1回のみ実行）
        rgb = camera.data.output["rgb"][0].cpu().numpy()
        Image.fromarray(rgb).save("/tmp/latest_rgb.png")

        texture = heightmap_to_texture(height_map)
        texture_path = f"/tmp/heightmap_{count}.png"
        
        with open(texture_path, "wb") as f:
            f.write(texture)
            
        image_widget.source_url = texture_path


def main():
    """Main function."""
    # Load simulation context
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    # Design scene
    scene_entities = design_scene()
    # Play simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run simulator
    run_simulator(sim, scene_entities)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
