import torch
import open3d as o3d

# 点群を読み込み
pointcloud = torch.load("pointcloud.pt")

# GPUならCPUへ
if pointcloud.is_cuda:
    pointcloud = pointcloud.cpu()

points = pointcloud.numpy()

print("Point cloud shape:", points.shape)
print("Min:", points.min(axis=0))
print("Max:", points.max(axis=0))

# Open3Dへ
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# 座標軸
axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)

o3d.visualization.draw_geometries(
    [pcd, axis],
    window_name="Point Cloud",
)