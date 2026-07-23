# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script shows how to use the RayCaster (LiDAR) sensor from the Isaac Lab framework
to generate clean height maps without depth-projection errors.
"""

import argparse
import base64

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates how to use the RayCaster sensor for heightmaps.")
parser.add_argument(
    "--draw",
    action="store_true",
    default=False,
    help="Draw the pointcloud from sensor.",
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

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import RAY_CASTER_MARKER_CFG
# 【変更】Cameraの代わりにRayCaster関連をインポート
from isaaclab.sensors import RayCaster, RayCasterCfg
from isaaclab.sensors.ray_caster import patterns
import omni.ui as ui
from PIL import Image
import io
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import omni.appwindow
import carb
import carb.input
from isaaclab.sensors.ray_caster import patterns

def define_sensor() -> RayCaster:
    """Defines the RayCaster (LiDAR-like) sensor attached to the tracker."""
    ray_caster_cfg = RayCasterCfg(
        prim_path="/World/Objects/CameraTracker",
        update_period=0.0,
        # 少し高めの位置から下に向けてレイを飛ばすオフセット（必要に応じて調整）
        offset=RayCasterCfg.OffsetCfg(pos=(1.0, 1.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
        pattern_cfg=patterns.GridPatternCfg(
            resolution=0.2,       # 解像度を少し粗くしてテスト
            size=[8.0, 8.0]       # スキャン範囲を 8m × 8m に広げる
        ),
        max_distance=100.0,
        mesh_prim_paths=["/World"],
        debug_vis=True,           # ← ここを True にすると、シミュレータ上で「レイの線」が視覚化されるので当たっているか一目で分かります！
    )
    return RayCaster(cfg=ray_caster_cfg)


def design_scene() -> dict:
    """Design the scene."""
    # Populate scene
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    
    cfg = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    scene_entities = {}

    sim_utils.create_prim("/World/Objects", "Xform")
    # Random objects
    for i in range(3):
        position = np.random.rand(3) - np.asarray([0.05, 0.05, -1.0])
        position *= np.asarray([1.5, 1.5, 0.5])
        color = (random.random(), random.random(), random.random())
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
        
        obj_cfg = RigidObjectCfg(
            prim_path=f"/World/Objects/Obj_{i:02d}",
            spawn=shape_cfg,
            init_state=RigidObjectCfg.InitialStateCfg(pos=position),
        )
        scene_entities[f"rigid_object{i}"] = RigidObject(cfg=obj_cfg)

    cam_init_pos = [2.5, 2.5, 2.5]
    cam_init_quat = [-0.1759, 0.3399, 0.8205, -0.4247]

    # ──【修正】CameraTrackerを RigidObject ではなく、単なる Xform プリミティブとして作成する ──
    sim_utils.create_prim(
        prim_path="/World/Objects/CameraTracker",
        prim_type="Xform",
        translation=cam_init_pos,
        orientation=cam_init_quat,
    )
    # scene_entities["camera_tracker"] は物理オブジェクトではないため除外
    
    # Sensors (RayCaster)
    scene_entities["sensor"] = define_sensor()
    return scene_entities

resolution = 0.03
xmin, xmax = -4.0, 4.0
ymin, ymax = -4.0, 4.0

W = int((xmax-xmin)/resolution)
H = int((ymax-ymin)/resolution)

def create_height_map(points_world):
    height_map = np.full((H,W), np.nan)

    for x, y, z in points_world:
        if not np.isfinite(x) or not np.isfinite(y) or not np.isfinite(z):
            continue

        if z < 0.0:
            z = 0.0
            
        ix = int((x-xmin)/resolution)
        iy = int((y-ymin)/resolution)

        if 0 <= ix < W and 0 <= iy < H:
            iy_flipped = (H - 1) - iy

            if np.isnan(height_map[iy_flipped, ix]):
                height_map[iy_flipped, ix] = z
            else:
                height_map[iy_flipped, ix] = max(
                    height_map[iy_flipped, ix],
                    z
                )

    height_map[np.isnan(height_map)] = 0.0
    height_map = gaussian_filter(height_map, sigma=1.0)
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
    img.save(buffer, format="PNG")
    return buffer.getvalue()

def add_camera_marker(height_map, cam_pos):
    ix = int((cam_pos[0] - xmin) / resolution)
    iy = int((cam_pos[1] - ymin) / resolution)

    if 0 <= ix < W and 0 <= iy < H:
        iy_flipped = (H - 1) - iy
        height_map[
            max(0, iy_flipped - 3) : iy_flipped + 4,
            max(0, ix - 3) : ix + 4
        ] = height_map.max() + 0.2

    return height_map

class KeyboardController:
    def __init__(self):
        self.input = carb.input.acquire_input_interface()
        self.appwindow = omni.appwindow.get_default_app_window()
        self.keyboard = self.appwindow.get_keyboard()

    def is_key_pressed(self, char: str) -> bool:
        carb_key = getattr(carb.input.KeyboardInput, char.upper(), None)
        if carb_key is not None:
            return self.input.get_keyboard_value(self.keyboard, carb_key) > 0.0
        return False

def run_simulator(sim: sim_utils.SimulationContext, scene_entities: dict):
    window = ui.Window("Height Map (LiDAR)", width=600, height=600)
    with window.frame:
        image_widget = ui.Image()

    sensor: RayCaster = scene_entities["sensor"]
    keyboard_ctrl = KeyboardController()

    if sim.has_gui() and args_cli.draw:
        cfg = RAY_CASTER_MARKER_CFG.replace(prim_path="/Visuals/PointCloud")
        cfg.markers["hit"].radius = 0.002
        pc_markers = VisualizationMarkers(cfg)

    count = 0
    while simulation_app.is_running():
        sim.step()
        sensor.update(dt=sim.get_physics_dt())
        count += 1
        # tracker_obj = scene_entities["camera_tracker"]

        # キーボード入力
        speed = 2.0
        vx, vy, vz = 0.0, 0.0, 0.0
        if keyboard_ctrl.is_key_pressed("W"): vx += speed
        if keyboard_ctrl.is_key_pressed("S"): vx -= speed
        if keyboard_ctrl.is_key_pressed("A"): vy += speed
        if keyboard_ctrl.is_key_pressed("D"): vy -= speed
        if keyboard_ctrl.is_key_pressed("Q"): vz += speed
        if keyboard_ctrl.is_key_pressed("E"): vz -= speed

        velocity = torch.tensor([[vx, vy, vz, 0.0, 0.0, 0.0]], device=sim.device)
        tracker_obj.write_root_velocity_to_sim(velocity)

        obj_pos = tracker_obj.data.root_pos_w[0]

        if count % 10 != 0:
            continue

        # RayCasterから直接ワールド座標のヒット点群を取得！
        # data.ray_hits_w には [B, N, 3] の形式で障害物との交点座標が直接入っています
        ray_hits = sensor.data.ray_hits_w[0] # shape: (N, 3)

        # マーカーを描画する場合
        if sim.has_gui() and args_cli.draw:
            # pc_markers.visualize(ray_hits.unsqueeze(0))
            pc_markers.visualize(ray_hits)

        # フィルタリング
        valid_mask = torch.isfinite(ray_hits).all(dim=1)
        points_world = ray_hits[valid_mask]

        bounds_mask = (
            (points_world[:, 0] >= xmin) & (points_world[:, 0] <= xmax) &
            (points_world[:, 1] >= ymin) & (points_world[:, 1] <= ymax) &
            (points_world[:, 2] >= -0.5) & (points_world[:, 2] <= 2.0)
        )
        points_world = points_world[bounds_mask]

        # ヒートマップとマーカーの生成
        height_map = create_height_map(points_world.cpu().numpy())
        height_map = add_camera_marker(height_map, obj_pos.cpu().numpy())

        # 描画・更新
        texture = heightmap_to_texture(height_map)
        texture_path = f"/tmp/heightmap_{count}.png"
        
        with open(texture_path, "wb") as f:
            f.write(texture)
            
        image_widget.source_url = texture_path


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    
    scene_entities = design_scene()
    sim.reset()
    print("[INFO]: Setup complete (RayCaster LiDAR mode)...")
    
    try:
        run_simulator(sim, scene_entities)
    except KeyboardInterrupt:
        print("[INFO]: User interrupted simulation.")
    finally:
        # ▼ シミュレーション終了時に必ず安全に通すクリーンアップ処理
        print("[INFO]: Cleaning up simulation...")
        sim.clear()
        # シミュレーションコンテキスト自体を破棄
        if sim:
            del sim


if __name__ == "__main__":
    main()
    # ▼ アプリケーション全体の終了は一番最後に正確に行う
    simulation_app.close()