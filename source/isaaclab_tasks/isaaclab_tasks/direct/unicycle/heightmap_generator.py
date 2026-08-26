import torch
import torch.nn.functional as F

class HeightMapGenerator:
    def __init__(self, resolution=0.1, map_size=8.0, kernel_size=5, sigma=1.0, device="cuda"):
        self.resolution = resolution
        self.map_size = map_size
        self.map_W = int(map_size / resolution)
        self.map_H = int(map_size / resolution)
        self.device = device

        ax = torch.arange(kernel_size, device=device, dtype=torch.float32) - kernel_size // 2
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        gaussian_kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        gaussian_kernel /= gaussian_kernel.sum()

        self.gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        self.kernel_size = kernel_size

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

    def create_height_map(self, points_world, xmin, ymin):
        ix = ((points_world[:, 0] - xmin) / self.resolution).long()
        iy = ((points_world[:, 1] - ymin) / self.resolution).long()

        valid = (
            (ix >= 0)
            & (ix < self.map_W)
            & (iy >= 0)
            & (iy < self.map_H)
        )

        ix = ix[valid]
        iy = iy[valid]
        z = points_world[:, 2][valid]
        z = torch.clamp(z, min=0.0)

        iy = (self.map_H - 1) - iy
        linear = iy * self.map_W + ix

        height_map = torch.full(
            (self.map_H * self.map_W,),
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

        height_map = height_map.view(self.map_H, self.map_W)
        height_map[height_map == float("-inf")] = 0.0

        return height_map

    def generate(self, robot_pos, obstacle_positions, obstacle_sizes):
        height_maps = []

        for env_id in range(robot_pos.shape[0]):
            xmin = robot_pos[env_id, 0] - self.map_size / 2.0
            ymin = robot_pos[env_id, 1] - self.map_size / 2.0

            points = self.create_cube_points(
                obstacle_positions[env_id],
                obstacle_sizes[env_id],
            )

            height_map = self.create_height_map(
                points,
                xmin,
                ymin,
            )

            height_map = F.conv2d(
                height_map.unsqueeze(0).unsqueeze(0),
                self.gaussian_kernel,
                padding=self.kernel_size // 2,
            ).squeeze(0).squeeze(0)

            height_maps.append(height_map)

        return torch.stack(height_maps)