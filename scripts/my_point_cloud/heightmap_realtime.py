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
import omni.appwindow  # ← ここを追加！
import carb
import carb.input
from isaaclab.utils.math import (
    convert_camera_frame_orientation_convention,
)
from PIL import ImageDraw, ImageFont
from isaaclab.utils.math import matrix_from_quat
from isaaclab.utils.math import unproject_depth

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
        height=480,
        width=640,
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

resolution = 0.03
xmin, xmax = -4.0, 4.0
ymin, ymax = -4.0, 4.0

W = int((xmax-xmin)/resolution)
H = int((ymax-ymin)/resolution)

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
        position = np.array([
            0.0,
            0.0,
            0.5
        ])
        # position = np.random.rand(3) - np.asarray([0.05, 0.05, -1.0])
        # position *= np.asarray([1.5, 1.5, 0.5])
        # sample random color
        color = (random.random(), random.random(), random.random())
        # choose random prim type
        # prim_type = random.choice(["Cube", "Cylinder"])
        prim_type = "Cylinder"
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

def create_height_map(points_world, labels=None):
    height_map = np.full((H,W), np.nan)
    semantic_map = np.zeros((H,W), dtype=np.int32)
    for i, (x,y,z) in enumerate(points_world):
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
            iy_flipped = (H - 1) - iy #ヒートマップ画像は左上が原点で下向きにy軸だから反転させて、x右y上にする
            # 
            label = 0
            if labels is not None:
                label = labels[i]
            # if label == 2:
            #     print(
            #         "cylinder original z:",
            #         original_z,
            #         "after:",
            #         z
            #     )
            if np.isnan(height_map[iy_flipped, ix]) or z > height_map[iy_flipped, ix]:
                height_map[iy_flipped, ix] = z
                if labels is not None:
                    semantic_map[iy_flipped, ix] = label
            else:
                if z > height_map[iy_flipped, ix]:
                    height_map[iy_flipped, ix] = z
                    semantic_map[iy_flipped, ix] = label
    height_map[np.isnan(height_map)] = 0.0
    return height_map, semantic_map

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

    if torch.abs(torch.dot(forward, world_up)) > 0.99:
        world_up = torch.tensor([0.0, 1.0, 0.0], device=camera_pos.device)

    right = torch.linalg.cross(
        forward,
        world_up
    )

    right = right / torch.norm(right)

    # up = torch.linalg.cross(
    #     right,
    #     forward
    # )
    up = torch.linalg.cross(right, forward)

    R = torch.stack(
        [
            right,
            -up,
            forward,
        ],
        dim=1
    )

    return R

import torch
from isaaclab.utils.math import matrix_from_quat

def depth_to_world(
    depth: torch.Tensor,
    intrinsic: torch.Tensor,
    cam_pos: torch.Tensor,
    cam_quat_ros: torch.Tensor,
):
    """
    depth      : (H, W)
    intrinsic  : (3, 3)
    cam_pos    : (3,)
    cam_quat_ros : (4,)  (ROS convention)
    """

    H, W = depth.shape
    device = depth.device

    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]

    # -----------------------------
    # Pixel coordinates
    # -----------------------------
    u = torch.arange(W, device=device)
    v = torch.arange(H, device=device)
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

    return points_world.reshape(H, W, 3)

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

    camera_positions = torch.tensor([[0.0, 0.0, 5.0]], device=sim.device)
    camera_targets = torch.tensor([[0.0, 0.0, 0.0]], device=sim.device)
    camera.set_world_poses_from_view(camera_positions, camera_targets)

    camera_index = 0
    keyboard_ctrl = KeyboardController()

    if sim.has_gui() and args_cli.draw:
        cfg = RAY_CASTER_MARKER_CFG.replace(prim_path="/Visuals/CameraPointCloud")
        cfg.markers["hit"].radius = 0.002
        pc_markers = VisualizationMarkers(cfg)

    count = 0
    while simulation_app.is_running():
        sim.step()
        sim_dt = sim.get_physics_dt()
        count += 1

        # region Keyboard control for camera position
        speed = 2.0
        vx, vy, vz = 0.0, 0.0, 0.0
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
        # endregion

        # region カメラ座標取得
        camera_positions, camera_quats = camera._view.get_world_poses()
        camera_positions[0] += torch.tensor([vx, vy, vz], device=sim.device) * sim_dt
        camera._view.set_world_poses(
            positions=camera_positions,
            orientations=camera_quats,
        )
        camera.update(dt=sim.get_physics_dt())
        #endregion

        if count % 50 != 0:
            continue

        #region 深度画像Get camera data and generate pointcloud
        depth = camera.data.output["distance_to_image_plane"][camera_index]
        depth = depth.squeeze(-1)

        # Get camera poses & convert convention
        camera_positions, camera_quats_gl = camera._view.get_world_poses()
        cam_pos = camera_positions[camera_index]
        cam_quat_gl = camera_quats_gl[camera_index]
        R = look_at_rotation(
            camera_positions[0],
            camera_targets[0],
        )
        print(R)

        # region 各画素のカメラ座標
        fx = camera.data.intrinsic_matrices[camera_index][0, 0]
        fy = camera.data.intrinsic_matrices[camera_index][1, 1]
        cx = camera.data.intrinsic_matrices[camera_index][0, 2]
        cy = camera.data.intrinsic_matrices[camera_index][1, 2]
        H, W = depth.shape
        u, v = torch.meshgrid(
            torch.arange(W, device=depth.device),
            torch.arange(H, device=depth.device),
            indexing="xy",
        )
        z = depth
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        points_cam = torch.stack([x, y, z], dim=-1)#
        #endregion

        #region カメラ座標からワールド座標への変換
        points_world = (
            R @ points_cam.reshape(-1, 3).T
        ).T + camera_positions[0]
        points_world = points_world.reshape(H, W, 3)
        u = W // 2
        v = H // 2
        print("camera =", points_cam[v, u])
        print("world  =", points_world[v, u])
        #endregion

        camera_quats_ros = convert_camera_frame_orientation_convention(
            cam_quat_gl.unsqueeze(0),
            origin="opengl",
            target="ros",
        )
        cam_quat_ros = camera_quats_ros[0]
        print(cam_pos)
        print(cam_quat_gl)
        print(cam_quat_ros)

        # Generate world pointcloud using Isaac Lab official utility
        # points_world_img = create_pointcloud_from_depth(
        #     intrinsic_matrix=camera.data.intrinsic_matrices[camera_index],
        #     depth=depth,
        #     keep_invalid=True,
        #     position=cam_pos,
        #     orientation=None,
        #     device=sim.device,
        # )
        points_world_img = depth_to_world(
            depth,
            camera.data.intrinsic_matrices[camera_index],
            cam_pos,
            cam_quat_ros,
        )
        print(points_world_img[..., 2].min())
        print(points_world_img[..., 2].max())

        height, width = depth.shape[0], depth.shape[1]
        points_world_img = points_world_img.reshape(height, width, 3)
        
        valid_mask = torch.isfinite(points_world_img).all(dim=-1)
        points_world_img_masked = points_world_img.clone()
        points_world_img_masked[~valid_mask] = 0

        # Filtering pointcloud
        points_world_flat = points_world_img.reshape(-1, 3)
        finite_mask = torch.isfinite(points_world_flat).all(dim=1)
        points_world_flat = points_world_flat[finite_mask]

        bounds_mask = (
            (points_world_flat[:, 0] >= xmin) & (points_world_flat[:, 0] <= xmax) &
            (points_world_flat[:, 1] >= ymin) & (points_world_flat[:, 1] <= ymax) &
            (points_world_flat[:, 2] >= -1.0) & (points_world_flat[:, 2] <= 2.0)
        )
        points_world = points_world_flat[bounds_mask]
        # endregion

        # region Semantic Segmentation
        semantic = camera.data.output["semantic_segmentation"][0].squeeze()
        mask = (semantic == 2)
        print(points_world_img[mask][:, 2].min())
        print(points_world_img[mask][:, 2].max())
        semantic_np = semantic.cpu().numpy()
        
        semantic_info = camera.data.info[0]["semantic_segmentation"]["idToLabels"]
        colors = {
            0: [0, 0, 0],        # BACKGROUND
            1: [80, 80, 80],     # UNLABELLED
            2: [255, 0, 0],      # cylinder
            3: [0, 255, 0],      # cube
        }
        #セマセグカラー画像生成
        semantic_rgb = np.zeros((*semantic_np.shape, 3), dtype=np.uint8)
        for label_id in np.unique(semantic_np):
            semantic_rgb[semantic_np == label_id] = colors.get(
                int(label_id),
                [255, 255, 255]
            )
        semantic_pil = Image.fromarray(semantic_rgb)
        draw = ImageDraw.Draw(semantic_pil)
        #クラスラベル名表示
        for label_id in np.unique(semantic_np):
            if label_id < 2:
                continue
            mask = (semantic_np == label_id) & valid_mask.cpu().numpy()
            ys, xs = np.where(mask)
            if len(xs) == 0:
                continue
            cx = int(xs.mean())
            cy = int(ys.mean())
            name = semantic_info[str(int(label_id))]["class"]
            draw.text((cx, cy), name, fill=(255, 255, 255))

        points_flat = points_world_img[valid_mask]
        labels_flat = semantic[valid_mask]
        #endregion

        print(f"------------------------------------------------------------------------------")
        print(f"Frame {count}")

        #region 中央画素の座標確認
        height, width = depth.shape
        u = width // 2
        v = height // 2
        center_depth = depth[v, u]
        center_world = points_world_img[v, u]
        print(f"pixel = ({u}, {v})")
        print(f"depth = {center_depth:.3f}")
        print(f"world = {center_world}")
        #endregion

        #region --- 【追加】v = 235 の行の u 全部の深度と世界座標の高さを表示 ---
        #cylinderの高さを計算して表示
        cylinder_mask = (labels_flat == 2)
        if cylinder_mask.any():
            cylinder_points = points_flat[cylinder_mask]
            max_cylinder_z = cylinder_points[:, 2].max().item()
            print(f"Cylinder Max Height (Z): {max_cylinder_z:.4f}")
        else:
            print("Cylinder not found in camera view.")

        target_v = 230
        print(f"=== Debug Row v = {target_v} ===")
        # u方向の全幅（通常は width = 640）
        width = depth.shape[1]
        for u in range(width):
            # シリンダー周辺（310〜330あたり）だけをピンポイントで見やすく出力、あるいは全体をスキャン
            if 305 <= u <= 335:
                raw_depth = depth[target_v, u].item()
                label = semantic[target_v, u].item()
                world_pt = points_world_img[target_v, u].cpu().numpy()
                
                print(f"  u={u:3d} | Label: {label} | Depth: {raw_depth:6.3f} | World XYZ: [{world_pt[0]:6.3f}, {world_pt[1]:6.3f}, {world_pt[2]:6.3f}]")
        print(f"==================================")
        #endregion

        # region セマセグ画像一時保存ui表示
        semantic_path = f"/tmp/semantic_{count}.png"
        semantic_pil.save(semantic_path)
        semantic_widget.source_url = semantic_path
        # endregion

        # region HeightMap generation & update
        height_map, semantic_map = create_height_map(
            points_flat.cpu().numpy(),
            labels_flat.cpu().numpy()
        )
        
        height_map[semantic_map == 2] = 0.0
        height_map = gaussian_filter(height_map, sigma=1.0)
        height_map = add_camera_marker(height_map, camera_positions[camera_index].cpu().numpy())
        
        rgb = camera.data.output["rgb"][0].cpu().numpy()
        Image.fromarray(rgb).save("/tmp/latest_rgb.png")
        
        texture = heightmap_to_texture(height_map)
        texture_path = f"/tmp/heightmap_{count}.png"
        with open(texture_path, "wb") as f:
            f.write(texture)
        image_widget.source_url = texture_path
        # endregion

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
