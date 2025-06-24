import os
import numpy as np
import open3d as o3d
import cv2
import torch
from models.minkowski_graspnet_single_point import MinkowskiGraspNetMultifingerType1Inference

def load_meshes_pointcloud(path, voxel_grid_size):
    meshes_pcls = {}
    meshes_pcl_path = os.path.join(path, 'meshes/source_pointclouds/voxel_size_' + str(int(voxel_grid_size * 1000)))
    for type_name in os.listdir(meshes_pcl_path):
        type_path = os.path.join(meshes_pcl_path, type_name)
        for name in os.listdir(type_path):
            width = name[:-4]
            name_path = os.path.join(type_path, name)
            meshes_pcl = o3d.io.read_point_cloud(name_path)
            meshes_pcls[f"{type_name}_{width}"] = meshes_pcl
    return meshes_pcls

def get_inspire_model(inspire_models_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    inspire_models = {}
    for model_type in os.listdir(inspire_models_path):
        inspire_model_path = os.path.join(inspire_models_path, model_type)
        inspire_models[model_type] = {}
        for model_class in os.listdir(inspire_model_path):
            model_class_type = os.path.join(inspire_model_path, model_class)
            for model in os.listdir(model_class_type):
                inspire_model_type_path = os.path.join(model_class_type, model)
                inspire_model = MinkowskiGraspNetMultifingerType1Inference(input_num=int(model_type))
                inspire_net = torch.load(inspire_model_type_path)
                inspire_model.load_state_dict(inspire_net.state_dict())
                inspire_model.to(device)
                inspire_model.eval()
                inspire_models[model_type][model_class] = inspire_model
    return inspire_models

def load_point_cloud_from_images(depth_path, color_path=None):
    depths = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
    if depths is None:
        raise ValueError("Failed to load depth image")
    
    color_image = None
    if color_path and os.path.exists(color_path):
        color_image = cv2.imread(color_path)
        if color_image is not None:
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    
    fx, fy = 919.835, 919.61
    cx, cy = 631.119, 363.884
    s = 1000.0

    xmap, ymap = np.arange(depths.shape[1]), np.arange(depths.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)

    points_z = depths / s
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z

    mask = (points_z > 0.35) & (points_z < 0.68)   
    points = np.stack([points_x, points_y, points_z], axis=-1)
    points = points[mask].astype(np.float32)

    cloud = None
    if color_image is not None:
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        if color_image.dtype == np.uint8:
            colors = color_image[mask].astype(np.float32) / 255.0 
        else:
            colors = color_image[mask].astype(np.float32)
        cloud.colors = o3d.utility.Vector3dVector(colors)
    
    return points, cloud

def load_point_cloud_from_file(point_cloud_path):
    pcd = o3d.io.read_point_cloud(point_cloud_path)
    points = np.asarray(pcd.points)
    cloud = pcd
    return points, cloud
