import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter


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


# -------------------------
# max height map
# -------------------------
height_max = np.full(
    (H,W),
    np.nan
)


# -------------------------
# mean height map用
# -------------------------
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

        # max
        if np.isnan(height_max[iy,ix]):
            height_max[iy,ix] = z
        else:
            height_max[iy,ix] = max(
                height_max[iy,ix],
                z
            )

        # mean
        height_sum[iy,ix] += z
        height_count[iy,ix] += 1



# -------------------------
# mean
# -------------------------
height_mean = np.zeros((H,W))

valid = height_count > 0

height_mean[valid] = (
    height_sum[valid]
    /
    height_count[valid]
)


# -------------------------
# unknown = ground
# -------------------------
height_max[np.isnan(height_max)] = 0.0
height_mean[~valid] = 0.0



# -------------------------
# gaussian
# -------------------------
sigma = 1.0

height_max_smooth = gaussian_filter(
    height_max,
    sigma=sigma
)

height_mean_smooth = gaussian_filter(
    height_mean,
    sigma=sigma
)



# -------------------------
# plot
# -------------------------
fig, ax = plt.subplots(
    1,2,
    figsize=(14,6)
)


im0 = ax[0].imshow(
    height_mean_smooth,
    origin="lower",
    extent=[
        xmin,xmax,
        ymin,ymax
    ],
    cmap="jet"
)

ax[0].set_title("Mean height")


im1 = ax[1].imshow(
    height_max_smooth,
    origin="lower",
    extent=[
        xmin,xmax,
        ymin,ymax
    ],
    cmap="jet"
)

ax[1].set_title("Max height")


# camera position
camera_pos = np.array(
    [2.5,2.5,2.5]
)


for a in ax:

    a.scatter(
        camera_pos[0],
        camera_pos[1],
        c="white",
        s=100,
        marker="x",
        label="camera"
    )

    a.scatter(
        0,
        0,
        c="black",
        s=50,
        marker="o"
    )

    a.set_xlabel("world X")
    a.set_ylabel("world Y")
    a.axis("equal")
    a.legend()


fig.colorbar(
    im0,
    ax=ax[0],
    label="height [m]"
)

fig.colorbar(
    im1,
    ax=ax[1],
    label="height [m]"
)


plt.show()