# utils/data_processing.py
import numpy as np
import open3d as o3d
import cv2
from multiprocessing import shared_memory
from .config import CAMERA_INTRINSICS

def load_depth_image(file_path):
    """从文件加载深度图"""
    return cv2.imread(file_path, cv2.IMREAD_ANYDEPTH)

def get_depth_from_shared_memory(shm_name='realsense_depth'):
    """从共享内存中读取深度图"""
    existing_shm_depth = shared_memory.SharedMemory(name=shm_name)
    depth_image = np.copy(np.ndarray((720, 1280), dtype=np.uint16, buffer=existing_shm_depth.buf))
    existing_shm_depth.close()
    return depth_image

def get_color_from_shared_memory(shm_name='realsense_color'):
    """从共享内存中读取彩色图"""
    existing_shm_color = shared_memory.SharedMemory(name=shm_name)
    color_image = np.copy(np.ndarray((720, 1280, 3), dtype=np.float32, buffer=existing_shm_color.buf))
    existing_shm_color.close()
    return color_image

def create_point_cloud_from_depth(depth_image, z_range=(0.2, 1.0)):
    """根据深度图和相机内参创建点云"""
    fx, fy = CAMERA_INTRINSICS['fx'], CAMERA_INTRINSICS['fy']
    cx, cy = CAMERA_INTRINSICS['cx'], CAMERA_INTRINSICS['cy']
    s = CAMERA_INTRINSICS['s']

    h, w = depth_image.shape
    xmap, ymap = np.arange(w), np.arange(h)
    xmap, ymap = np.meshgrid(xmap, ymap)

    points_z = depth_image / s
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z

    mask = (points_z > z_range[0]) & (points_z < z_range[1])
    points = np.stack([points_x, points_y, points_z], axis=-1)
    return points[mask].astype(np.float32)

def augment_data(flip=False):
    """对点云进行随机数据增强"""
    # Flipping
    flip_mat = np.eye(4)
    if flip:
        flip_mat[0, 0] = -1

    # Rotation
    rot_angle = (np.random.random() * np.pi / 3) - np.pi / 6  # -30 ~ +30 degree
    c, s = np.cos(rot_angle), np.sin(rot_angle)
    rot_mat = np.array([[c, -s, 0, 0],
                        [s, c, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])

    # Translation
    offset_x = np.random.random() * 0.1 - 0.05
    offset_y = np.random.random() * 0.1 - 0.05
    trans_mat = np.array([[1, 0, 0, offset_x],
                          [0, 1, 0, offset_y],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]])
    
    aug_mat = np.dot(trans_mat, np.dot(rot_mat, flip_mat)).astype(np.float32)
    return aug_mat