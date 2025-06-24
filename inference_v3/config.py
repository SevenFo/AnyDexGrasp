import argparse

def get_argparse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', default='/data/shiqi/AnyDexGrasp/logs/model/checkpoint.tar.18', help='Model checkpoint path')
    parser.add_argument('--inspire_model_path', default='logs/model/inspire_model/obj140', help='inspire model checkpoint path')
    parser.add_argument('--inspire_mesh_json_path', default='generate_mesh_and_pointcloud/inspire_urdf', help='InspireHandR meshes and json path')
    parser.add_argument('--depth_image', default='depth_2.png', help='Path to pre-recorded depth image (PNG)')
    parser.add_argument('--color_image', default='rgb_2.png', help='Path to pre-recorded color image (PNG)')
    parser.add_argument('--point_cloud', type=str, default="filtered.ply", help="point_cloud path (npy format) for no_robot mode")
    parser.add_argument('--use_graspnet_v2', action='store_true', help='Whether to use graspnet v2 format')
    parser.add_argument('--half_views', action='store_true', help='Use only half views in network.')
    
    cfgs = parser.parse_args()
    
    # Constants
    cfgs.MAX_GRASP_WIDTH = 0.1
    cfgs.MIN_GRASP_WIDTH = 0.01
    cfgs.BATCH_SIZE = 1
    cfgs.DEBUG = True
    cfgs.CALIB = False
    cfgs.GRIPPER_TOTAL_LEN = 0.155
    cfgs.FLANGE_TOTAL_LEN = 0.055
    cfgs.INSPIREHANDR_DEFAULT_DEPTH = 0.000
    cfgs.NUM_OF_INSPIRE_DEPTH = 4
    cfgs.NUM_OF_INSPIRE_TYPE = 8
    cfgs.INSPIREHANDR_VOXElGRID = 0.003
    cfgs.POINTCLOUD_AUGMENT_NUM = 10
    cfgs.RANDOM_GRASP = False
    
    return cfgs

def get_config():
    # 转换为字典
    config_dict = {
        "checkpoint_path":'/data/shiqi/AnyDexGrasp/logs/model/checkpoint.tar.18',
        "inspire_model_path": '/data/shiqi/AnyDexGrasp/logs/model/inspire_model/obj140',
        "inspire_mesh_json_path": '/data/shiqi/AnyDexGrasp/generate_mesh_and_pointcloud/inspire_urdf',
        "use_graspnet_v2": True,
        "half_views": False
    }

    # 添加常量到字典
    config_dict.update({
        'MAX_GRASP_WIDTH': 0.1,
        'MIN_GRASP_WIDTH': 0.01,
        'BATCH_SIZE': 1,
        'DEBUG': True,
        'CALIB': False,
        'GRIPPER_TOTAL_LEN': 0.155,
        'FLANGE_TOTAL_LEN': 0.055,
        'INSPIREHANDR_DEFAULT_DEPTH': 0.000,
        'NUM_OF_INSPIRE_DEPTH': 4,
        'NUM_OF_INSPIRE_TYPE': 8,
        'INSPIREHANDR_VOXElGRID': 0.003,
        'POINTCLOUD_AUGMENT_NUM': 10,
        'RANDOM_GRASP': False,
        "visualize_final_grasps": True
    })
    from types import SimpleNamespace
    return SimpleNamespace(**config_dict)