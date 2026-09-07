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
from isaaclab.utils.math import create_rotation_matrix_from_view, quat_from_euler_xyz
import os
from isaaclab.sim import schemas
import omni.appwindow  # ← ここを追加！
import carb
import carb.input
from isaaclab.utils.math import (
    convert_camera_frame_orientation_convention,
)
from PIL import ImageDraw, ImageFont
from isaaclab.utils.math import matrix_from_quat
from isaaclab.utils.math import unproject_depth
import time
import torch.nn.functional as F
import torch
from isaaclab.utils.math import matrix_from_quat
from heightmap_generator import HeightMapGenerator
from isaaclab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz

def define_sensor() -> Camera:
    """Defines the camera sensor to add to the scene."""
    # Setup camera sensor
    # In contrast to the ray-cast camera, we spawn the prim at these locations.
    # This means the camera sensor will be attached to these prims.
    sim_utils.create_prim("/World/Origin_00", "Xform")
    # sim_utils.create_prim("/World/Origin_01", "Xform")
    camera_cfg = CameraCfg(
        prim_path="/World/Objects/CameraTracker/CameraSensor",  # ← ワイルドカ
        update_period=0,
        height=128,
        width=128,
        data_types=[
            "rgb",
            "distance_to_image_plane",
            "normals",
            "semantic_segmentation",
            "instance_segmentation_fast",
            "instance_id_segmentation_fast",
        ],
        colorize_semantic_segmentation=False,
        colorize_instance_id_segmentation=True,
        colorize_instance_segmentation=True,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=8.0, 
            focus_distance=400.0, 
            horizontal_aperture=30, 
            clipping_range=(0.05, 1.0e2)
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
    for i in range(1):
        # sample random position
        position = np.random.rand(3) - np.asarray([0.05, 0.05, -1.0])
        position *= np.asarray([1.5, 1.5, 0.5])
        # sample random color
        color = (random.random(), random.random(), random.random())
        # choose random prim type
        prim_type = random.choice(["Cube", "Cone", "Cylinder"])
        common_properties = {
            "rigid_props": sim_utils.RigidBodyPropertiesCfg(),
            "mass_props": sim_utils.MassPropertiesCfg(mass=5.0),
            "collision_props": sim_utils.CollisionPropertiesCfg(),
            "visual_material": sim_utils.PreviewSurfaceCfg(diffuse_color=color, metallic=0.5),
            "semantic_tags": [("class", prim_type)],
        }
        if prim_type == "Cube":
            shape_cfg = sim_utils.CuboidCfg(size=(0.25, 0.25, 0.25), **common_properties)
        elif prim_type == "Cone":
            shape_cfg = sim_utils.ConeCfg(radius=0.1, height=0.25, **common_properties)
        elif prim_type == "Cylinder":
            shape_cfg = sim_utils.CylinderCfg(radius=0.25, height=0.25, **common_properties)
        # Rigid Object
        obj_cfg = RigidObjectCfg(
            prim_path=f"/World/Objects/Obj_{i:02d}",
            spawn=shape_cfg,
            init_state=RigidObjectCfg.InitialStateCfg(pos=position),
        )
        scene_entities[f"rigid_object{i}"] = RigidObject(cfg=obj_cfg)

# # --- 【修正】単なるXformではなく、物理オブジェクト（RigidObject）として親を作る ---
#     cam_init_pos = [2.5, 2.5, 2.5]
#     cam_init_quat = [-0.1759, 0.3399, 0.8205, -0.4247] # (x, y, z, w)

#     tracker_obj_cfg = RigidObjectCfg(
#         prim_path="/World/Objects/CameraTracker", # ここを親のパスにする
#         spawn=sim_utils.SphereCfg(
#             radius=0.01, # 小さな球体（視覚的に邪魔なら見えないようにすることも可能）
#             rigid_props=sim_utils.RigidBodyPropertiesCfg(
#                 disable_gravity=True
#             ),
#             mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
#             collision_props=sim_utils.CollisionPropertiesCfg(),
#             visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
#         ),
#         init_state=RigidObjectCfg.InitialStateCfg(
#             pos=cam_init_pos,
#             rot=cam_init_quat,
#         ),
#     )
#     # Python側の辞書に登録するのでKeyErrorが起きなくなる
#     scene_entities["camera_tracker"] = RigidObject(cfg=tracker_obj_cfg)
    sim_utils.create_prim("/World/Objects/CameraTracker", "Xform")
    # Sensors
    camera = define_sensor()
    # return the scene information
    scene_entities["camera"] = camera
    return scene_entities

def create_height_map_torch(
    points_world,
    xmin,
    ymin,
    resolution,
    map_W,
    map_H,
):
    # グリッド座標
    ix = ((points_world[:, 0] - xmin) / resolution).long()
    iy = ((points_world[:, 1] - ymin) / resolution).long()

    valid = (
        (ix >= 0) & (ix < map_W) &
        (iy >= 0) & (iy < map_H)
    )

    ix = ix[valid]
    iy = iy[valid]
    z = points_world[:, 2][valid]

    z = torch.clamp(z, min=0.0)

    # y方向反転
    iy = (map_H - 1) - iy

    linear = iy * map_W + ix

    height_map = torch.full(
        (map_H * map_W,),
        float("-inf"),
        device=points_world.device,
    )

    height_map.scatter_reduce_(
        0,
        linear,
        z,
        reduce="amax",
        include_self=True,
    )

    height_map = height_map.view(map_H, map_W)
    height_map[height_map == float("-inf")] = 0.0

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

def create_gaussian_kernel(kernel_size=5, sigma=1.0, device="cuda"):
    ax = torch.arange(kernel_size, device=device) - kernel_size // 2
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")

    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel /= kernel.sum()

    return kernel.view(1, 1, kernel_size, kernel_size)

def depth_to_world(
    depth: torch.Tensor,
    intrinsic: torch.Tensor,
    cam_pos: torch.Tensor,
    cam_quat_ros: torch.Tensor,
):
    """
    depth      : (img_H, img_W)
    intrinsic  : (3, 3)
    cam_pos    : (3,)
    cam_quat_ros : (4,)  (ROS convention)
    """

    img_H, img_W = depth.shape
    device = depth.device

    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]

    # -----------------------------
    # Pixel coordinates
    # -----------------------------
    u = torch.arange(img_W, device=device)
    v = torch.arange(img_H, device=device)
    uu, vv = torch.meshgrid(u, v, indexing="xy")

    z = depth

    # -----------------------------
    # Camera coordinates (ROS)
    # x : forward
    # y : left
    # z : up
    # -----------------------------
    x = z
    y = -(uu - cx) * z / fx
    z = -(vv - cy) * z / fy

    points_cam = torch.stack([x, y, z], dim=-1)

    # -----------------------------
    # Camera -> World
    # -----------------------------
    R = matrix_from_quat(cam_quat_ros)

    points_world = (
        R @ points_cam.reshape(-1, 3).T
    ).T + cam_pos

    return points_world.reshape(img_H, img_W, 3)

def add_camera_marker(height_map, cam_pos, xmin, ymin, resolution, map_W, map_H):
    ix = int((cam_pos[0] - xmin) / resolution)
    iy = int((cam_pos[1] - ymin) / resolution)    

    if 0 <= ix < map_W and 0 <= iy < map_H:
        iy_flipped = (map_H - 1) - iy
        height_map[
            max(0, iy_flipped - 3) : iy_flipped + 4,
            max(0, ix - 3) : ix + 4
        ] = height_map.max() + 0.2

    return height_map

# --- キーボード入力を管理するヘルパークラス ---
class KeyboardController:
    def __init__(self):
        self.input = carb.input.acquire_input_interface()
        self.appwindow = omni.appwindow.get_default_app_window()
        self.keyboard = self.appwindow.get_keyboard()

    def is_key_pressed(self, char: str) -> bool:
        # キーが押されているかを判定
        carb_key = getattr(carb.input.KeyboardInput, char.upper(), None)
        if carb_key is not None:
            return self.input.get_keyboard_value(self.keyboard, carb_key) > 0.0
        return False

def run_simulator(sim: sim_utils.SimulationContext, scene_entities: dict):
    heightmap_generator = HeightMapGenerator(
        resolution=0.05,
        map_size=3.2,
        gui_enabled=True,
        device=sim.device,
    )
    # HeightMap & Semantic windows
    height_window = ui.Window("Height Map", width=600, height=600)
    with height_window.frame:
        image_widget = ui.Image()
        
    semantic_window = ui.Window("Semantic", width=600, height=600)
    with semantic_window.frame:
        semantic_widget = ui.Image()

    camera: Camera = scene_entities["camera"]

    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output", "camera")
    rep_writer = rep.BasicWriter(
        output_dir=output_dir,
        frame_padding=0,
        colorize_instance_id_segmentation=camera.cfg.colorize_instance_id_segmentation,
        colorize_instance_segmentation=camera.cfg.colorize_instance_segmentation,
        colorize_semantic_segmentation=camera.cfg.colorize_semantic_segmentation,
    )

    gaussian_kernel = create_gaussian_kernel(   
        kernel_size=5,
        sigma=1.0,
        device=sim.device,
    )

    # camera_positions = torch.tensor([[2.5, 2.5, 2.5]], device=sim.device)
    # camera_targets = torch.tensor([[0.0, 0.0, 0.0]], device=sim.device)
    camera_positions = torch.tensor([[2.5, 0.0, 1.0]], device=sim.device)
    camera_targets = torch.tensor([[0.0, 0.0, 1.0]], device=sim.device)
    camera.set_world_poses_from_view(camera_positions, camera_targets)

    camera_index = 0
    keyboard_ctrl = KeyboardController()

    if sim.has_gui() and args_cli.draw:
        cfg = RAY_CASTER_MARKER_CFG.replace(prim_path="/Visuals/CameraPointCloud")
        cfg.markers["hit"].radius = 0.002
        pc_markers = VisualizationMarkers(cfg)

    # マップのサイズ（固定）
    resolution = 0.05
    map_size = 3.2  # マップの一辺の長さ（8m四方）
    map_W = int(map_size / resolution)
    map_H = int(map_size / resolution)

    count = 0
    while simulation_app.is_running():

        # ============================================================
        # while 1周全体 START
        # ============================================================
        print(f"------------------------------------------------------------------------------")
        print(f"Frame {count}")
        torch.cuda.synchronize()
        t_total0 = time.perf_counter()
        
        sim.step()
        sim_dt = sim.get_physics_dt()
        count += 1

        # region Keyboard control for camera position
        speed = 2.0
        yaw_speed = 1.5  # rad/s またはフレームごとの回転量
        vx, vy, vz = 0.0, 0.0, 0.0
        dyaw = 0.0
        if keyboard_ctrl.is_key_pressed("W"):  # Forward
            vx += speed
        if keyboard_ctrl.is_key_pressed("S"):  # Backward
            vx -= speed
        if keyboard_ctrl.is_key_pressed("A"):  # Left
            vy += speed
        if keyboard_ctrl.is_key_pressed("D"):  # Right
            vy -= speed
        if keyboard_ctrl.is_key_pressed("Q"):  # Up
            vz += speed
        if keyboard_ctrl.is_key_pressed("E"):  # Down
            vz -= speed
        # --- Yaw回転用のキー追加 (例: Zキーで左回転、Cキーで右回転) ---
        if keyboard_ctrl.is_key_pressed("Z"):
            dyaw += yaw_speed * sim_dt
        if keyboard_ctrl.is_key_pressed("C"):
            dyaw -= yaw_speed * sim_dt
        # endregion
        if count % 10 != 0:
            continue
        # ============================================================
        # ① カメラ位置更新
        # ============================================================
        t0 = time.perf_counter()
        # region カメラ座標取得
        camera_positions, camera_quats = camera._view.get_world_poses()
        camera_positions[0] += torch.tensor([vx, vy, vz], device=sim.device) * sim_dt        
        # --- Yaw回転の適用 ---
        if dyaw != 0.0:
            # 現在のクォータニオンからオイラー角（Roll, Pitch, Yaw）を取得
            roll, pitch, yaw = euler_xyz_from_quat(camera_quats)
            # Yaw成分のみに回転量を加算
            yaw[0] += dyaw
            # オイラー角からクォータニオンを再構築
            camera_quats = quat_from_euler_xyz(roll, pitch, yaw)
        camera._view.set_world_poses(
            positions=camera_positions,
            orientations=camera_quats,
        )
        camera.update(dt=sim.get_physics_dt())
        camera_quats_ros = convert_camera_frame_orientation_convention(
            camera_quats,
            origin="opengl",
            target="ros",
        )
        _, _, robot_yaw = euler_xyz_from_quat(camera_quats_ros)
        #endregion
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        # --- 【追加】カメラの現在地(X, Y)を中心にした動的範囲の計算 ---
        cam_x = camera_positions[camera_index, 0].item()
        cam_y = camera_positions[camera_index, 1].item()
        xmin = cam_x - map_size / 2.0
        xmax = cam_x + map_size / 2.0
        ymin = cam_y - map_size / 2.0
        ymax = cam_y + map_size / 2.0

        #Get camera data and generate pointcloud
        #region 座標変換

        # ============================================================
        # ② Depth取得
        # ============================================================
        depth = camera.data.output["distance_to_image_plane"]
        
        robot_pos = camera_positions.clone()
        # 今はカメラをロボット扱い
        # yawを固定するなら0
        # robot_yaw = torch.zeros(
        #     camera_positions.shape[0],
        #     device=sim.device,
        # )
        height_map = heightmap_generator.generate_from_depth(
            depth,
            camera,
            robot_pos,
            robot_yaw,
        )
    
        # ============================================================
        # while 1周全体 END
        # ============================================================
        torch.cuda.synchronize()
        t_total1 = time.perf_counter()

        # print(
        #     f"Frame {count} | "
        #     f"camera update: {(t1-t0)*1000:.3f} ms | "
        #     f"depth get: {(t2-t1)*1000:.3f} ms | "
        #     f"pose/quat: {(t3-t2)*1000:.3f} ms | "
        #     f"pixel->camera: {(t4-t3)*1000:.3f} ms | "
        #     f"camera->world: {(t5-t4)*1000:.3f} ms | "
        #     f"valid/bounds: {(t6-t5)*1000:.3f} ms | "
        #     f"semantic get: {(t7-t6)*1000:.3f} ms | "
        #     f"semantic->CPU: {(t8-t7)*1000:.3f} ms | "
        #     f"semantic image: {(t9-t8)*1000:.3f} ms | "
        #     f"heightmap: {(t10-t9)*1000:.3f} ms | "
        #     f"gaussian: {(t11-t10)*1000:.3f} ms | "
        #     f"heightmap->CPU: {(t12-t11)*1000:.3f} ms | "
        #     f"camera marker: {(t13-t12)*1000:.3f} ms | "
        #     f"RGB->CPU: {(t14-t13)*1000:.3f} ms | "
        #     f"RGB PNG: {(t15-t14)*1000:.3f} ms | "
        #     f"texture: {(t16-t15)*1000:.3f} ms | "
        #     f"PNG/UI: {(t17-t16)*1000:.3f} ms | "
        #     f"TOTAL: {(t_total1-t_total0)*1000:.3f} ms | "
        #     f"FPS: {1000.0/(t_total1-t_total0):.2f}"
        # )
        #endregion

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
