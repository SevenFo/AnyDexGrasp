# File: Inference/utils/camera.py
import time
import numpy as np
import open3d as o3d


def get_depth(existing_shm_depth):
    """Reads depth data from shared memory."""
    time.sleep(0.1)
    depths = np.copy(
        np.ndarray((720, 1280), dtype=np.uint16, buffer=existing_shm_depth.buf)
    )
    return depths


def get_point_cloud(depths, colors, config):
    """Generates a point cloud from depth and color images."""
    intrinsics = config["cam_intrinsics"]
    fx, fy, cx, cy = (
        intrinsics["fx"],
        intrinsics["fy"],
        intrinsics["cx"],
        intrinsics["cy"],
    )
    s = 1000.0

    xmap, ymap = np.arange(depths.shape[1]), np.arange(depths.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)

    points_z = depths / s
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z

    z_min, z_max = config["point_cloud_mask"]
    mask = (points_z > z_min) & (points_z < z_max)

    points = np.stack([points_x, points_y, points_z], axis=-1)
    points = points[mask].astype(np.float32)
    colors = colors[mask].astype(np.float32)

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)

    return points, cloud


def create_table_pointcloud(
    width=0.5, height=0.005, depth=0.4, dx=-0.25, dy=-0.15, dz=-0.55, grid_size=0.005
):
    """Creates a point cloud representation of a table."""
    xmap = np.linspace(0, width, int(width / grid_size))
    ymap = np.linspace(0, depth, int(depth / grid_size))
    zmap = np.linspace(0, height, int(height / grid_size))
    xmap, ymap, zmap = np.meshgrid(xmap, ymap, zmap, indexing="xy")
    xmap += dx
    ymap += dy
    zmap += dz
    points = np.stack([xmap, -ymap, -zmap], axis=-1).reshape([-1, 3])
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    return points
