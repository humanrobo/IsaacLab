# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

import argparse
# ============================================================
# AppLauncher
# ============================================================
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description="Internal-data based height map example.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import io
import os
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import omni.ui as ui
import omni.appwindow
import omni.kit.app
from pxr import UsdGeom
from omni.usd import get_context
import carb
from PIL import Image
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg

# ============================================================
# Height Map
# ============================================================
def create_height_map_torch(points_world, xmin, ymin, resolution, map_W, map_H):
    """
    points_world:
        [N, 3]
        x, y, z = world coordinates
    """
    # World座標 -> grid座標
    ix = ((points_world[:, 0] - xmin) / resolution).long()
    iy = ((points_world[:, 1] - ymin) / resolution).long()
    valid = ((ix >= 0) & (ix < map_W) & (iy >= 0) & (iy < map_H))
    ix = ix[valid]
    iy = iy[valid]
    z = points_world[:, 2][valid]
    # 地面より下は0
    z = torch.clamp(z, min=0.0)
    # 画像表示用にY反転
    iy = (map_H - 1) - iy
    linear = iy * map_W + ix
    height_map = torch.full((map_H * map_W,), float("-inf"), device=points_world.device)
    height_map.scatter_reduce_(0, linear, z, reduce="amax", include_self=True)
    height_map = height_map.view(map_H, map_W)
    height_map[height_map == float("-inf")] = 0.0

    return height_map

# ============================================================
# Cube -> point cloud
# ============================================================
def create_cube_points(center, size, resolution):
    """
    Cubeの上面を点群化する。
    center:
        [3]
        Cube中心のworld座標
    size:
        [3]
        CubeのXYZサイズ
    resolution:
        HeightMapのresolution
    """
    half = size / 2.0
    x = torch.arange(center[0] - half[0], center[0] + half[0] + resolution, resolution, device=center.device)
    y = torch.arange(center[1] - half[1], center[1] + half[1] + resolution, resolution, device=center.device)
    xx, yy = torch.meshgrid(x, y, indexing="xy")
    # Cube上面の高さ
    zz = torch.full_like(xx, center[2] + half[2])
    points = torch.stack([xx, yy, zz], dim=-1)

    return points.reshape(-1, 3)

# ============================================================
# HeightMap -> image
# ============================================================
def heightmap_to_texture(height_map):
    hmin = height_map.min()
    hmax = height_map.max()
    if hmax - hmin < 1e-6:
        norm = np.zeros_like(height_map)
    else:
        norm = (height_map - hmin) / (hmax - hmin)
    rgb = plt.get_cmap("jet")(norm)
    rgb = (rgb[:, :, :3] * 255).astype(np.uint8)
    img = Image.fromarray(rgb)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")

    return buffer.getvalue()

# ============================================================
# Robot marker
# ============================================================
def add_robot_marker(height_map, robot_pos, xmin, ymin, resolution, map_W, map_H):
    """
    HeightMap上にロボット位置を表示する。
    """
    ix = int((robot_pos[0] - xmin) / resolution)
    iy = int((robot_pos[1] - ymin) / resolution)
    if 0 <= ix < map_W and 0 <= iy < map_H:
        iy_flipped = (map_H - 1) - iy
        marker_value = height_map.max() + 0.2
        height_map[max(0, iy_flipped - 3):min(map_H, iy_flipped + 4), max(0, ix - 3):min(map_W, ix + 4)] = marker_value

    return height_map

# ============================================================
# Scene
# ============================================================
def design_scene():
    scene_entities = {}
    # --------------------------------------------------------
    # Ground
    # --------------------------------------------------------
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

    # --------------------------------------------------------
    # Light
    # --------------------------------------------------------
    light_cfg = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    # --------------------------------------------------------
    # Objects
    # --------------------------------------------------------
    sim_utils.create_prim("/World/Objects", "Xform")

    # ========================================================
    # Robot
    # ========================================================
    robot_size = (0.4, 0.4, 0.4)
    robot_cfg = RigidObjectCfg(
        prim_path="/World/Objects/Robot",
        spawn=sim_utils.CuboidCfg(
            size=robot_size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, robot_size[2] / 2.0)),
    )
    scene_entities["robot"] = RigidObject(cfg=robot_cfg)

    # ========================================================
    # Obstacles
    # ========================================================
    for i in range(1):
        # XY位置をランダム
        x = np.random.uniform(1.0, 3.5)
        y = np.random.uniform(-3.0, 3.0)
        size = (np.random.uniform(0.3, 0.7), np.random.uniform(0.3, 0.7), np.random.uniform(0.2, 1.0))
        obstacle_cfg = RigidObjectCfg(
            prim_path=f"/World/Objects/Obstacle_{i:02d}",
            spawn=sim_utils.CuboidCfg(
                size=size,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(x, y, size[2] / 2.0)),
        )
        scene_entities[f"obstacle_{i}"] = RigidObject(cfg=obstacle_cfg)

    return scene_entities

# ============================================================
# Simulator
# ============================================================
def run_simulator(sim: sim_utils.SimulationContext, scene_entities: dict):

    # ============================================================
    # Keyboard
    # ============================================================
    keyboard = carb.input.acquire_input_interface()
    app_window = omni.appwindow.get_default_app_window()
    keyboard_device = app_window.get_keyboard()
    pressed_keys = set()
    def on_keyboard_event(event):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            pressed_keys.add(event.input.name)
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            pressed_keys.discard(event.input.name)
    keyboard.subscribe_to_keyboard_events(keyboard_device, on_keyboard_event)

    # --------------------------------------------------------
    # UI
    # --------------------------------------------------------
    height_window = ui.Window("Height Map", width=600, height=600)
    with height_window.frame:
        image_widget = ui.Image()

    # --------------------------------------------------------
    # HeightMap parameters
    # --------------------------------------------------------
    resolution = 0.1
    map_size = 8.0
    map_W = int(map_size / resolution)
    map_H = int(map_size / resolution)
    print(f"HeightMap size = {map_W} x {map_H}")

    # --------------------------------------------------------
    # Gaussian
    # --------------------------------------------------------
    kernel_size = 5
    sigma = 1.0
    ax = (torch.arange(kernel_size, device=sim.device) - kernel_size // 2)
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    gaussian_kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    gaussian_kernel /= gaussian_kernel.sum()
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)

    # --------------------------------------------------------
    # Main loop
    # --------------------------------------------------------
    count = 0
    # ============================================================
    # 障害物ランダム移動設定
    # ============================================================
    obstacle_move_interval = 100  # 100 simulation stepごとに移動
    # 障害物のサイズ
    obstacle_size = torch.tensor([0.5, 0.5, 0.5], device=sim.device, dtype=torch.float32)
    while simulation_app.is_running():
        # ========================================================
        # KeyboardでRobotを移動
        # ========================================================
        robot = scene_entities["robot"]
        robot_pos = robot.data.root_pos_w[0].clone()
        speed = 0.02
        dx = 0.0
        dy = 0.0
        if "W" in pressed_keys:
            dx += speed
        if "S" in pressed_keys:
            dx -= speed
        if "A" in pressed_keys:
            dy += speed
        if "D" in pressed_keys:
            dy -= speed
        if dx != 0.0 or dy != 0.0:
            new_pos = robot_pos.clone()
            new_pos[0] += dx
            new_pos[1] += dy
            new_pose = torch.tensor([new_pos[0].item(), new_pos[1].item(), 0.2, 1.0, 0.0, 0.0, 0.0], device=sim.device, dtype=torch.float32).unsqueeze(0)
            robot.write_root_pose_to_sim(new_pose)
        sim.step()

        count += 1

        # RigidObjectの内部データを更新
        sim_dt = sim.get_physics_dt()
        for i in range(1):
            scene_entities[f"obstacle_{i}"].update(sim_dt)
        scene_entities["robot"].update(sim_dt)
        # ========================================================
        # 障害物をランダム移動
        # ========================================================
        # if count % obstacle_move_interval == 0:
        #     obstacle = scene_entities["obstacle_0"]
        #     # 新しいXY位置
        #     new_x = np.random.uniform(1.0, 3.5)
        #     new_y = np.random.uniform(-3.0, 3.0)
        #     # 現在の障害物高さ
        #     new_z = obstacle_size[2] / 2.0
        #     # pose [x, y, z, qw, qx, qy, qz]
        #     new_pose = torch.tensor([new_x, new_y, new_z, 1.0, 0.0, 0.0, 0.0], device=sim.device, dtype=torch.float32).unsqueeze(0)
        #     # Isaac Sim上の障害物を移動
        #     obstacle.write_root_pose_to_sim(new_pose)
        #     obstacle.update(sim.get_physics_dt())
        #     print(f">>> Obstacle moved: x={new_x:.3f}, y={new_y:.3f}, z={new_z:.3f}")
        # 毎10stepだけHeightMap更新
        if count % 10 != 0:
            continue
        print("--------------------------------------------------")

        # ====================================================
        # Robot position
        # ====================================================
        robot = scene_entities["robot"]
        robot_pos = robot.data.root_pos_w[0]
        print("robot position =", robot_pos)

        # ====================================================
        # Robot中心のMap範囲
        # ====================================================
        robot_x = robot_pos[0]
        robot_y = robot_pos[1]
        xmin = (robot_x - map_size / 2.0)
        ymin = (robot_y - map_size / 2.0)

        # ====================================================
        # Obstacles -> points
        # ====================================================
        all_points = []
        for i in range(1):
            obstacle = scene_entities[f"obstacle_{i}"]
            obstacle_pos = obstacle.data.root_pos_w[0]
            print(f"Obstacle {i}: x={obstacle_pos[0].item():.3f}, y={obstacle_pos[1].item():.3f}, z={obstacle_pos[2].item():.3f}")
            points = create_cube_points(obstacle_pos, obstacle_size, resolution)
            all_points.append(points)

        # ====================================================
        # 全障害物を結合
        # ====================================================
        points_world = torch.cat(all_points, dim=0)

        # ====================================================
        # HeightMap
        # ====================================================
        height_map = create_height_map_torch(points_world, xmin, ymin, resolution, map_W, map_H)

        # ====================================================
        # Gaussian filter
        # ====================================================
        height_map = F.conv2d(height_map.unsqueeze(0).unsqueeze(0), gaussian_kernel, padding=kernel_size // 2).squeeze(0).squeeze(0)

        # ====================================================
        # GPU -> CPU
        # ====================================================
        height_map_np = height_map.detach().cpu().numpy()

        # ====================================================
        # Robot marker
        # ====================================================
        height_map_np = add_robot_marker(height_map_np, robot_pos.detach().cpu().numpy(), xmin, ymin, resolution, map_W, map_H)

        # ====================================================
        # Texture
        # ====================================================
        texture = heightmap_to_texture(height_map_np)
        texture_path = f"/tmp/internal_heightmap_{count}.png"
        with open(texture_path, "wb") as f:
            f.write(texture)
        print("saved:", texture_path)
        image_widget.source_url = texture_path

        omni.kit.app.get_app().update()

        # ====================================================
        # Debug
        # ====================================================
        print("height min =", height_map_np.min())
        print("height max =", height_map_np.max())
        print("points =", points_world.shape)

# ============================================================
# Main
# ============================================================
def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Isaac SimのGUIカメラ
    # これは「センサー」ではないのでOK
    sim.set_camera_view([5.0, 5.0, 5.0], [0.0, 0.0, 0.0])
    scene_entities = design_scene()
    sim.reset()
    print("[INFO]: Setup complete...")
    run_simulator(sim, scene_entities)

if __name__ == "__main__":

    main()

    simulation_app.close()