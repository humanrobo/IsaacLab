import torch
import matplotlib.pyplot as plt
import numpy as np


points = torch.load(
    "points_world.pt"
).numpy()


resolution = 0.05


xmin = -3
xmax = 3
ymin = -3
ymax = 3


W = int((xmax-xmin)/resolution)
H = int((ymax-ymin)/resolution)


height_map = np.full(
    (H,W),
    np.nan
)

height_sum = np.zeros((H,W))
height_count = np.zeros((H,W))


for p in points:

    x,y,z = p

    ix = int((x-xmin)/resolution)
    iy = int((y-ymin)/resolution)

    if (
        0 <= ix < W and
        0 <= iy < H
    ):
        height_sum[iy,ix] += z
        height_count[iy,ix] += 1


# 平均高さ
height_map = np.zeros((H,W))

valid = height_count > 0

height_map[valid] = (
    height_sum[valid]
    /
    height_count[valid]
)


# 未観測領域は地面
height_map[~valid] = 0.0

# height_map[np.isnan(height_map)] = ground_height

plt.figure(figsize=(8,8))

plt.imshow(
    height_map,
    origin="lower",
    extent=[
        xmin,xmax,
        ymin,ymax
    ],
    cmap="jet"
)

# カメラ位置をプロット
camera_pos = np.array([2.5, 2.5, 2.5])

plt.scatter(
    camera_pos[0],
    camera_pos[1],
    c="white",
    s=100,
    marker="x",
    label="camera"
)

# 原点（ロボット位置確認用）
plt.scatter(
    0,
    0,
    c="black",
    s=50,
    marker="o",
    label="origin"
)

plt.colorbar(
    label="Height [m]"
)

plt.xlabel("world X")
plt.ylabel("world Y")

plt.legend()
plt.axis("equal")

plt.show()