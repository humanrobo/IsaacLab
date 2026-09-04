import torch
import torch.nn.functional as F
import io
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import omni.ui as ui
import omni.kit.app
from isaaclab.utils.math import (
    convert_camera_frame_orientation_convention,
    matrix_from_quat,
)

class HeightMapGenerator:
    def __init__(
        self,
        resolution=0.1,
        map_size=8.0,
        kernel_size=5,
        sigma=1.0,
        device="cuda",
        gui_enabled=False,
        gui_update_interval=10,
    ):
        self.resolution = resolution
        self.map_size = map_size
        self.map_W = int(map_size / resolution)
        self.map_H = int(map_size / resolution)
        self.device = device

        # ============================================================
        # Gaussian kernel
        # ============================================================
        ax = (
            torch.arange(
                kernel_size,
                device=device,
                dtype=torch.float32,
            )
            - kernel_size // 2
        )
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        gaussian_kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        gaussian_kernel /= gaussian_kernel.sum()
        self.gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        self.kernel_size = kernel_size
        # ================================================================
        # GUI
        # ================================================================
        self.gui_enabled = gui_enabled
        self.gui_update_interval = gui_update_interval
        self.gui_counter = 0
        self.height_window = None
        self.image_widget = None

        if self.gui_enabled:
            self._create_gui()

    # ================================================================
    # GUI
    # ================================================================
    def _create_gui(self):
        self.height_window = ui.Window("Height Map", width=600, height=650)
        with self.height_window.frame:
            with ui.VStack(spacing=5):
                self.info_label = ui.Label("HeightMap: waiting...")
                self.image_widget = ui.Image(width=550, height=550)

    # ================================================================
    # ロボット前方合わせ
    # ================================================================
    def rotate_to_robot_frame(self, height_maps, robot_yaw):
        N, H, W = height_maps.shape
        y, x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, H, device=height_maps.device),
            torch.linspace(-1.0, 1.0, W, device=height_maps.device),
            indexing="ij",
        )
        x = x.unsqueeze(0).expand(N, -1, -1)
        y = y.unsqueeze(0).expand(N, -1, -1)
        cos_yaw = torch.cos(robot_yaw).view(N, 1, 1)
        sin_yaw = torch.sin(robot_yaw).view(N, 1, 1)
        x_src = cos_yaw * x - sin_yaw * y
        y_src = sin_yaw * x + cos_yaw * y
        grid = torch.stack([x_src, y_src], dim=-1)
        rotated = F.grid_sample(
            height_maps.unsqueeze(1),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        return rotated.squeeze(1)
    
    # ================================================================
    # HeightMap -> colored texture
    # ================================================================
    def heightmap_to_texture(self, height_map):
        # hmin = height_map.min()
        # hmax = height_map.max()
        # if hmax - hmin < 1e-6:
        #     norm = np.zeros_like(height_map)
        # else:
        #     norm = (height_map - hmin) / (hmax - hmin)
        hm = np.clip(height_map, 0.0, 0.5)
        norm = hm / 0.5
        rgb = plt.get_cmap("jet")(norm)
        rgb = (rgb[:, :, :3] * 255).astype(np.uint8)
        img = Image.fromarray(rgb, mode="RGB")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return buffer.getvalue()

    # ================================================================
    # Robot marker
    # ================================================================
    def add_robot_marker(self, height_map):
        ix = self.map_W // 2
        iy = self.map_H // 2
        marker_value = height_map.max() + 0.2
        y0 = max(0, iy - 3)
        y1 = min(self.map_H, iy + 4)
        x0 = max(0, ix - 3)
        x1 = min(self.map_W, ix + 4)
        height_map[y0:y1, x0:x1] = marker_value
        return height_map

    # ================================================================
    # Cube -> points
    # ================================================================
    def create_cube_points(self, center, size):
        half = size / 2.0
        x = torch.arange(
            center[0] - half[0],
            center[0] + half[0] + self.resolution,
            self.resolution,
            device=center.device,
        )
        y = torch.arange(
            center[1] - half[1],
            center[1] + half[1] + self.resolution,
            self.resolution,
            device=center.device,
        )
        xx, yy = torch.meshgrid(x, y, indexing="xy")
        zz = torch.full_like(xx, center[2] + half[2])
        points = torch.stack([xx, yy, zz], dim=-1)
        return points.reshape(-1, 3)

    # ================================================================
    # Point cloud -> HeightMap
    # ================================================================
    def create_height_map(self, points_world, xmin, ymin):
        ix = ((points_world[:, 0] - xmin) / self.resolution).long()
        iy = ((points_world[:, 1] - ymin) / self.resolution).long()

        valid = (ix >= 0) & (ix < self.map_W) & (iy >= 0) & (iy < self.map_H)
        ix = ix[valid]
        iy = iy[valid]
        z = points_world[:, 2][valid]
        z = torch.clamp(z, min=0.0)

        # Y反転
        iy = (self.map_H - 1) - iy
        linear = iy * self.map_W + ix

        height_map = torch.full(
            (self.map_H * self.map_W,),
            float("-inf"),
            device=points_world.device,
        )
        height_map.scatter_reduce_(0, linear, z, reduce="amax", include_self=True)
        height_map = height_map.view(self.map_H, self.map_W)
        height_map[height_map == float("-inf")] = 0.0

        return height_map
    
    # ================================================================
    # GUI update
    # ================================================================
    def update_gui(self, height_map, robot_pos):
        if not self.gui_enabled:
            return
        self.gui_counter += 1
        if self.gui_counter % self.gui_update_interval != 0:
            return
        hm = height_map[0]
        hm_np = hm.detach().cpu().numpy().copy()
        robot_pos_np = robot_pos[0].detach().cpu().numpy()
        # hm_np = self.add_robot_marker(hm_np)
        texture = self.heightmap_to_texture(hm_np)
        texture_path = f"/tmp/heightmap_gui_{self.gui_counter}.png"
        with open(texture_path, "wb") as f:
            f.write(texture)
        self.image_widget.source_url = texture_path
        self.info_label.text = (
            f"HeightMap | env=0 | "
            f"robot=({robot_pos_np[0]:.3f}, {robot_pos_np[1]:.3f}) | "
            f"shape={hm_np.shape} | "
            f"min={hm_np.min():.3f} | "
            f"max={hm_np.max():.3f}"
        )
        omni.kit.app.get_app().update()

    # ================================================================
    # Generate
    # ================================================================
    def generate(self, robot_pos, robot_yaw, obstacle_positions, obstacle_sizes):
        height_maps = []
        for env_id in range(robot_pos.shape[0]):
            xmin = robot_pos[env_id, 0] - self.map_size / 2.0
            ymin = robot_pos[env_id, 1] - self.map_size / 2.0
            all_points = []
            for obstacle_id in range(obstacle_positions.shape[1]):
                points = self.create_cube_points(
                    obstacle_positions[env_id, obstacle_id],
                    obstacle_sizes[env_id, obstacle_id],
                )
                all_points.append(points)
            points = torch.cat(all_points, dim=0)
            height_map = self.create_height_map(points, xmin, ymin)
            height_map = F.conv2d(
                height_map.unsqueeze(0).unsqueeze(0),
                self.gaussian_kernel,
                padding=self.kernel_size // 2,
            ).squeeze(0).squeeze(0)
            height_maps.append(height_map)
        height_maps = torch.stack(height_maps)
        #ロボット回転に追従してヒートマップも回転
        height_maps = self.rotate_to_robot_frame(height_maps, robot_yaw)
        if self.gui_enabled:
            self.update_gui(height_maps, robot_pos)
            
        return height_maps

    # ================================================================
    # Generateカメラで
    # ================================================================
    def generate_from_depth(self, depth, camera, robot_pos, robot_yaw):
        N = depth.shape[0]
        device = depth.device
        depth = depth.squeeze(-1)
        scale = 2
        depth = depth[:, ::scale, ::scale]
        camera_positions, camera_quats_gl = camera._view.get_world_poses()
        camera_quats_ros = convert_camera_frame_orientation_convention(
            camera_quats_gl,
            origin="opengl",
            target="ros",
        )
        height_maps = []
        for env_id in range(N):
            d = depth[env_id]
            cam_pos = camera_positions[env_id]
            cam_quat = camera_quats_ros[env_id]
            fx = camera.data.intrinsic_matrices[env_id][0, 0] / scale
            fy = camera.data.intrinsic_matrices[env_id][1, 1] / scale
            cx = camera.data.intrinsic_matrices[env_id][0, 2] / scale
            cy = camera.data.intrinsic_matrices[env_id][1, 2] / scale
            H, W = d.shape
            u, v = torch.meshgrid(
                torch.arange(W, device=device),
                torch.arange(H, device=device),
                indexing="xy",
            )
            valid = (
                torch.isfinite(d)
                & (d > 0.0)
                & (d < 10.0)
            )
            z = torch.where(valid, d, torch.zeros_like(d))
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            points_cam = torch.stack([x, y, z], dim=-1)
            R = matrix_from_quat(cam_quat)
            points_world = (R @ points_cam.reshape(-1, 3).T).T + cam_pos
            points_world = points_world.reshape(H, W, 3)
            xmin = robot_pos[env_id, 0] - self.map_size / 2.0
            ymin = robot_pos[env_id, 1] - self.map_size / 2.0
            valid_mask = (
                valid
                & torch.isfinite(points_world).all(dim=-1)
                & (points_world[..., 2] > -0.1)
                & (points_world[..., 2] < 3.0)
                & (points_world[..., 0] >= xmin)
                & (points_world[..., 0] < xmin + self.map_size)
                & (points_world[..., 1] >= ymin)
                & (points_world[..., 1] < ymin + self.map_size)
            )
            points_world = points_world[valid_mask]
            if points_world.numel() == 0:
                # print("WARNING: no valid depth points")
                height_map = torch.zeros(
                    (self.map_H, self.map_W),
                    device=device,
                    dtype=torch.float32,
                )
            else:
                # print("point z max:", points_world[:, 2].max().item())
                height_map = self.create_height_map(
                    points_world,
                    xmin,
                    ymin,
                )
                # original_height_map = height_map.clone()
                #周囲3セルに最大高さをいれる
                height_map = F.max_pool2d(
                    height_map.unsqueeze(0).unsqueeze(0),
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ).squeeze(0).squeeze(0)
                height_map = F.conv2d(
                    height_map.unsqueeze(0).unsqueeze(0),
                    self.gaussian_kernel,
                    padding=self.kernel_size // 2,
                ).squeeze(0).squeeze(0)
                # height_map = torch.maximum(height_map, original_height_map)

            height_maps.append(height_map)
        height_maps = torch.stack(height_maps)
        height_maps = self.rotate_to_robot_frame(
            height_maps,
            robot_yaw + torch.pi / 2,
        )
        if self.gui_enabled:
            self.update_gui(height_maps, robot_pos)
            
        return height_maps