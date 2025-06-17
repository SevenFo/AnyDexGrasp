# run_inference.py
import argparse
import time
import numpy as np
import open3d as o3d
from graspnetAPI import GraspGroup

from utils.config import GRIPPER_CONFIGS, GRASPNET_CHECKPOINT_PATH, POINTCLOUD_AUGMENT_NUM, USE_GRASPNET_V2, HALF_VIEWS, DEBUG_VISUALIZATION
from utils.data_processing import get_depth_from_shared_memory, create_point_cloud_from_depth, augment_data
from utils.model_loader import get_graspnet_model, get_multifinger_model
from utils.grasp_processing import run_graspnet_on_point_cloud, run_collision_detection
from inference.allegro_inference import AllegroInference
# from inference.dh3_inference import DH3Inference # <-- 类似地导入
# from inference.inspire_inference import InspireInference # <-- 类似地导入

INFERENCE_CLASSES = {
    'allegro': AllegroInference,
    # 'dh3': DH3Inference,
    # 'inspire': InspireInference,
}

def main(args):
    gripper_name = args.gripper
    if gripper_name not in INFERENCE_CLASSES:
        raise ValueError(f"Gripper '{gripper_name}' not supported.")

    # 1. 加载配置和模型
    print(f"Loading models for {gripper_name}...")
    cfg = GRIPPER_CONFIGS[gripper_name]
    graspnet_model = get_graspnet_model(GRASPNET_CHECKPOINT_PATH, USE_GRASPNET_V2, HALF_VIEWS)
    multifinger_models = get_multifinger_model(cfg['model_path'])
    
    inference_handler = INFERENCE_CLASSES[gripper_name](cfg, graspnet_model, multifinger_models)

    # 2. 获取和处理点云
    print("Capturing and processing scene...")
    # 假设共享内存已启动并有数据
    depth = get_depth_from_shared_memory() 
    scene_points = create_point_cloud_from_depth(depth, cfg['point_cloud_z_range'])

    # 3. 运行GraspNet (带数据增强)
    all_gg = []
    all_gf = []
    scene_points_downsampled = None
    
    print(f"Running GraspNet with {POINTCLOUD_AUGMENT_NUM} augmentations...")
    for i in range(POINTCLOUD_AUGMENT_NUM):
        flip = i % 2 != 0
        aug_mat = augment_data(flip=flip)
        gg, gf, points_down = run_graspnet_on_point_cloud(graspnet_model, scene_points, cfg, aug_mat, flip)
        
        if gg is not None:
            all_gg.append(gg)
            all_gf.append(gf)
            if scene_points_downsampled is None:
                scene_points_downsampled = points_down

    if not all_gg:
        print("No grasps detected by GraspNet.")
        return

    ggarray = np.vstack([g.cpu().numpy() for g in all_gg])
    grasp_features = np.vstack([f.cpu().numpy() for f in all_gf])
    
    # 4. 运行特定机械手的推理流程
    print("Running multi-finger grasp prediction...")
    
    # 4.1 翻转和排序
    ggarray, if_flip = inference_handler.flip_ggarray(ggarray)
    grasp_features = np.c_[grasp_features, if_flip]
    
    # 可选：过滤掉翻转过的姿态（如 Inspire 的情况）
    # if gripper_name == 'inspire':
    #     ggarray = ggarray[~if_flip]
    #     grasp_features = grasp_features[~if_flip]

    sort_indices = ggarray[:, 0].argsort()[::-1][:2000] # Top 2000
    ggarray = ggarray[sort_indices]
    grasp_features = grasp_features[sort_indices]

    # 4.2 第二阶段推理
    mf_depths, mf_types, mf_scores, ggarray, grasp_features = \
        inference_handler.predict_multifinger_grasps(ggarray, grasp_features)

    # 4.3 创建抓取组对象
    two_finger_gg = GraspGroup(ggarray)
    GraspGroupClass = cfg['grasp_group_class']
    multi_finger_gg = GraspGroupClass()
    multi_finger_gg.from_graspgroup(two_finger_gg, mf_types, cfg['mesh_json_path'])
    multi_finger_gg.scores = mf_scores
    multi_finger_gg.depths += mf_depths + cfg['default_depth']

    # 4.4 后处理和筛选
    multi_finger_gg, two_finger_gg = \
        inference_handler.process_and_filter_grasps(multi_finger_gg, two_finger_gg)

    if multi_finger_gg is None or len(multi_finger_gg) == 0:
        print("No valid multi-finger grasps after filtering.")
        return
        
    # 5. 碰撞检测
    print("Running collision detection...")
    multi_finger_gg_col, two_finger_gg_col, _, _ = run_collision_detection(
        multi_finger_gg, two_finger_gg, scene_points_downsampled, cfg
    )

    if len(multi_finger_gg_col) == 0:
        print("No grasps remaining after collision detection.")
        return

    # 6. 选择最终结果
    # 这里我们简单地选择分数最高的
    best_grasp_idx = np.argmax(multi_finger_gg_col.scores)
    final_mf_grasp = multi_finger_gg_col[best_grasp_idx]
    final_tf_grasp = two_finger_gg_col[best_grasp_idx]
    
    print("\n--- Final Best Grasp Found ---")
    print(f"Gripper Type: {gripper_name}")
    print(f"Grasp Type ID: {final_mf_grasp.grasp_type}")
    print(f"Score: {final_mf_grasp.score:.4f}")
    print(f"Translation: {final_mf_grasp.translation}")
    print(f"Rotation Matrix:\n{final_mf_grasp.rotation_matrix}")

    # 你现在拥有了最终的抓取对象 `final_mf_grasp` 和 `final_tf_grasp`
    # 可以将它们传递给你的机器人控制模块
    
    if DEBUG_VISUALIZATION:
        scene_pcd = o3d.geometry.PointCloud()
        scene_pcd.points = o3d.utility.Vector3dVector(scene_points)
        gripper_mesh = final_mf_grasp.load_mesh(cfg['mesh_json_path'], final_tf_grasp)
        gripper_mesh.paint_uniform_color([1, 0, 0])
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        o3d.visualization.draw_geometries([scene_pcd, gripper_mesh, coord_frame])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gripper', type=str, required=True, choices=['allegro', 'dh3', 'inspire'],
                        help='Name of the gripper to use for inference.')
    args = parser.parse_args()
    
    main(args)