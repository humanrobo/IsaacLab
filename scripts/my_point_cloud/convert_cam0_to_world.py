import torch
import matplotlib.pyplot as plt


# -----------------------------
# load pointcloud
# -----------------------------
points_cam = torch.load(
    "pointcloud_cam0.pt"
).float()

print("camera points:")
print(points_cam.shape)
print(points_cam[:5])


# -----------------------------
# camera pose
# -----------------------------
camera_pos = torch.tensor(
    [2.5, 2.5, 2.5]
)

target = torch.tensor(
    [0.0, 0.0, 0.0]
)


# -----------------------------
# look at rotation
# camera -> world
# -----------------------------
def look_at_rotation(camera_pos, target):

    forward = target - camera_pos
    forward = forward / torch.norm(forward)

    world_up = torch.tensor(
        [0.0, 0.0, 1.0]
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


R = look_at_rotation(
    camera_pos,
    target
)


# -----------------------------
# camera -> world
# -----------------------------
points_world = (
    points_cam @ R.T
    + camera_pos
)


print("\nworld points:")
print(points_world[:5])

print("\nrange:")
print("x:",
      points_world[:,0].min(),
      points_world[:,0].max())

print("y:",
      points_world[:,1].min(),
      points_world[:,1].max())

print("z:",
      points_world[:,2].min(),
      points_world[:,2].max())

torch.save(
    points_world,
    "points_world.pt"
)
# -----------------------------
# XY plot
# -----------------------------
plt.figure(figsize=(8,8))

plt.scatter(
    points_world[:,0].numpy(),
    points_world[:,1].numpy(),
    s=0.2
)

plt.axis("equal")
plt.xlabel("world X")
plt.ylabel("world Y")
plt.grid()

plt.show()