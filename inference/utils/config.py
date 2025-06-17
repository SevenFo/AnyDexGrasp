# utils/config.py
import numpy as np
from ur_toolbox.robot.Allegro.Allegro_grasp import AllegroGraspGroup, grasp_types as allegro_grasp_types
from ur_toolbox.robot.DH3.DH3_grasp import DH3GraspGroup, grasp_types as dh3_grasp_types
from ur_toolbox.robot.Inspire.InspireHandR_grasp import InspireHandRGraspGroup, grasp_types as inspire_grasp_types

# --- 通用配置 ---
BATCH_SIZE = 1
POINTCLOUD_AUGMENT_NUM = 10
DEBUG_VISUALIZATION = False # 是否进行Open3D可视化

# --- 相机内参 (以其中一个为例，实际使用时应确认) ---
CAMERA_INTRINSICS = {
    'fx': 919.835, 'fy': 919.61,
    'cx': 631.119, 'cy': 363.884,
    's': 1000.0
}

# --- GraspNet V1 模型配置 ---
GRASPNET_CHECKPOINT_PATH = 'path/to/your/graspnet_checkpoint.tar' # <<-- 需要您提供正确的路径
USE_GRASPNET_V2 = False
HALF_VIEWS = False


# --- 各机械手特定配置 ---
GRIPPER_CONFIGS = {
    'allegro': {
        'model_path': 'logs/model/allegro_model/allegro_obj140',
        'mesh_json_path': 'generate_mesh_and_pointcloud/allegro_urdf',
        'save_info_path': 'logs/data/allegro/allegro_test/model_obj140trials',
        'max_width': 0.11,
        'min_width': 0.04,
        'num_depth': 4,
        'num_types': 10,
        'voxel_grid': 0.003,
        'default_depth': 0.00,
        'random_grasp': False,
        'grasp_group_class': AllegroGraspGroup,
        'grasp_types_dict': allegro_grasp_types,
        'workspace_bounds': np.array([[-0.2, 0.2], [-0.20, 0.07], [0.2, 0.65]]), # [xmin, xmax], [ymin, ymax], [zmin, zmax]
        'point_cloud_z_range': (0.2, 0.65),
    },
    'dh3': {
        'model_path': 'logs/model/dh3_model/obj140',
        'mesh_json_path': 'generate_mesh_and_pointcloud/dh3_urdf',
        'save_info_path': 'logs/data/dh3/dh3_test/obj140',
        'max_width': 0.099,
        'min_width': 0.045,
        'num_depth': 4,
        'num_types': 4,
        'voxel_grid': 0.003,
        'default_depth': 0.00,
        'random_grasp': False,
        'grasp_group_class': DH3GraspGroup,
        'grasp_types_dict': dh3_grasp_types,
        'workspace_bounds': np.array([[-0.25, 0.25], [-0.205, 0.03], [0.15, 0.72]]),
        'point_cloud_z_range': (0.15, 0.72),
    },
    'inspire': {
        'model_path': 'logs/model/inspire_model/final_single_point/obj140',
        'mesh_json_path': 'generate_mesh_and_pointcloud/inspire_urdf',
        'save_info_path': 'logs/data/inspire/inspire_test/obj140',
        'max_width': 0.1,
        'min_width': 0.01,
        'num_depth': 4,
        'num_types': 8,
        'voxel_grid': 0.003,
        'default_depth': 0.00,
        'random_grasp': True,
        'grasp_group_class': InspireHandRGraspGroup,
        'grasp_types_dict': inspire_grasp_types,
        'workspace_bounds': np.array([[-0.25, 0.25], [-0.205, 0.03], [0.35, 0.68]]),
        'point_cloud_z_range': (0.35, 0.68),
    }
}