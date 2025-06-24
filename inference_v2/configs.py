# File: Inference/configs.py


# 导入不同机械手的特定GraspGroup类和类型定义
from ur_toolbox.robot.Allegro.Allegro_grasp import (
    AllegroGraspGroup,
    grasp_types as allegro_grasp_types,
)
from ur_toolbox.robot.DH3.DH3_grasp import DH3GraspGroup, grasp_types as dh3_grasp_types
from ur_toolbox.robot.Inspire.InspireHandR_grasp import (
    InspireHandRGraspGroup,
    grasp_types as inspire_grasp_types,
)
from models.minkowski_graspnet_single_point import (
    MinkowskiGraspNetMultifingerType1Inference,
)


# 为 Allegro 和 Inspire 定义特定的选择函数
def select_grasp_type_allegro(allegro_gg, num_types):
    select_gg_types = [[] for _ in range(num_types)]
    for idx, score in enumerate(allegro_gg.scores):
        grasp_type = allegro_gg.grasp_types[idx]
        max_num = 50
        if grasp_type in [4, 6]:
            max_num = 5
        if len(select_gg_types[int(grasp_type) - 1]) < max_num:
            select_gg_types[int(grasp_type) - 1].append(idx)
    gg_type_id = []
    for gg_type in select_gg_types:
        gg_type_id += gg_type
    return gg_type_id


def select_grasp_type_inspire(inspire_gg, num_types):
    select_gg_types = [[] for _ in range(num_types)]
    for idx, score in enumerate(inspire_gg.scores):
        grasp_type = inspire_gg.grasp_types[idx]
        max_num = 50
        if grasp_type in [1]:
            max_num = 10
        if len(select_gg_types[int(grasp_type)-1]) < max_num:
            select_gg_types[int(grasp_type)-1].append(idx)
    gg_type_id = []
    for gg_type in select_gg_types:
        gg_type_id += gg_type
    return gg_type_id


GRIPPER_CONFIGS = {
    "allegro": {
        "name": "allegro",
        "gripper_class": AllegroGraspGroup,
        "grasp_types": allegro_grasp_types,
        "model_class": MinkowskiGraspNetMultifingerType1Inference,
        "model_path": "logs/model/allegro_model/allegro_obj140",
        "save_path": "logs/data/allegro/allegro_test/model_obj140trials",
        "mesh_json_path": "generate_mesh_and_pointcloud/allegro_urdf",
        "gripper_port": "192.168.1.29",
        "robot_payload": [2.7, (0, 0, 0.12)],
        "robot_tcp": (0, 0, 0.0, 0, 0, 0),
        "max_width": 0.11,
        "min_width": 0.04,
        "num_depth": 4,
        "num_type": 10,
        "voxel_grid": 0.003,
        "default_depth": 0.00,
        "random_grasp": False,
        "point_cloud_mask": (0.2, 0.65),  # (z_min, z_max)
        "workspace_mask": (-0.2, 0.2, -0.20, 0.07),  # (x_min, x_max, y_min, y_max)
        "cam_intrinsics": {"fx": 913.232, "fy": 912.452, "cx": 628.847, "cy": 350.771},
        "score_thresh": 0.7,
        "approach_dist": 0.08,
        "flip_logic": "y_x < 0",
        "has_z_flip": True,
        "select_type_func": select_grasp_type_allegro,
        "collision_thresh": 1,
        "adjust_gripper_centers": True,
    },
    "dh3": {
        "name": "dh3",
        "gripper_class": DH3GraspGroup,
        "grasp_types": dh3_grasp_types,
        "model_class": MinkowskiGraspNetMultifingerType1Inference,
        "model_path": "logs/model/dh3_model/obj140",
        "save_path": "logs/data/dh3/dh3_test/obj140",
        "mesh_json_path": "generate_mesh_and_pointcloud/dh3_urdf",
        "gripper_port": "192.168.1.29",  # Placeholder, DH3 might have a different setup
        "robot_payload": [3.8, (0, 0, 0.12)],
        "robot_tcp": (0, 0, 0.0, 0, 0, 0),
        "max_width": 0.099,
        "min_width": 0.045,
        "num_depth": 4,
        "num_type": 4,
        "voxel_grid": 0.003,
        "default_depth": 0.00,
        "random_grasp": False,
        "point_cloud_mask": (0.15, 0.72),
        "workspace_mask": (-0.25, 0.25, -0.205, 0.03),
        "cam_intrinsics": {"fx": 919.835, "fy": 919.61, "cx": 631.119, "cy": 363.884},
        "score_thresh": 0.85,
        "approach_dist": 0.05,
        "flip_logic": "y_x > 0",
        "has_z_flip": False,
        "select_type_func": None,  # DH3 doesn't have this special selection
        "collision_thresh": 0,
        "adjust_gripper_centers": False,
    },
    "inspire": {
        "name": "inspire",
        "gripper_class": InspireHandRGraspGroup,
        "grasp_types": inspire_grasp_types,
        "model_class": MinkowskiGraspNetMultifingerType1Inference,
        "model_path": "/data/shiqi/AnyDexGrasp/logs/model/inspire_model/obj140",
        "save_path": "/data/shiqi/AnyDexGrasp/logs/data/inspire/inspire_test/obj140",
        "mesh_json_path": "/data/shiqi/AnyDexGrasp/generate_mesh_and_pointcloud/inspire_urdf",
        "gripper_port": "/dev/ttyUSB1",
        "robot_payload": [1.3, (0, 0, 0.09)],
        "robot_tcp": (0, 0, 0.044, 0, 0, 0),
        "max_width": 0.1,
        "min_width": 0.01,
        "num_depth": 4,
        "num_type": 8,
        "voxel_grid": 0.003,
        "default_depth": 0.00,
        "random_grasp": False,
        "point_cloud_mask": (-0.78-0.3, -0.78), #(0.78, 0.78+0.3), #(0.35, 0.68),
        "workspace_mask": (0.65, 1.13,-0.4,0.4), #(0.65, 1.13,-0.4,0.5), #(-0.25, 0.25, -0.205, 0.03),
        "cam_intrinsics": {"fx": 919.835, "fy": 919.61, "cx": 631.119, "cy": 363.884},
        "score_thresh": 0.85,
        "approach_dist": 0.06,
        "flip_logic": "y_x < 0_inspire",
        "has_z_flip": False,
        "select_type_func": select_grasp_type_inspire,
        "collision_thresh": 0,
        "adjust_gripper_centers": True,
    },
}
