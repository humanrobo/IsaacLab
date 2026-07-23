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


for p in points:

    x,y,z = p

    ix = int((x-xmin)/resolution)
    iy = int((y-ymin)/resolution)

    if (
        0 <= ix < W and
        0 <= iy < H
    ):
        if np.isnan(height_map[iy,ix]):
            height_map[iy,ix] = z
        else:
            height_map[iy,ix] = max(
                height_map[iy,ix],
                z
            )
# 未観測領域を地面高さにする
ground_height = 0.0

height_map[np.isnan(height_map)] = ground_height

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