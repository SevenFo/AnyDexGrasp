# utils/grasp_processing.py
import copy
import torch
import numpy as np
import open3d as o3d
import MinkowskiEngine as ME
import os
from adg_utils.np_utils import transform_point_cloud
from adg_utils.pt_utils import batch_viewpoint_params_to_matrix
from adg_utils.collision_detector import ModelFreeCollisionDetectorMultifinger

def parse_graspnet_predictions(end_points, max_width, batch_size=1):
    """从GraspNet的输出中解析抓取姿态和特征"""
    # (这部分代码与原脚本中的 parse_preds 基本一致)
    # ...
    # 为了简洁，此处省略了与原脚本完全相同的代码实现
    # 注意: 原始代码中的MAX_GRASP_WIDTH需要作为参数`max_width`传入
    before_generator = end_points['before_generator']
    point_features = end_points['point_features']
    coords = end_points['sinput'].C
    objectness_pred = end_points['stage1_objectness_pred']
    objectness_mask = torch.argmax(objectness_pred, dim=1).bool()
    seed_xyz = end_points['stage2_seed_xyz']
    seed_inds = end_points['stage2_seed_inds']
    grasp_view_xyz = end_points['stage2_view_xyz']
    grasp_view_inds = end_points['stage2_view_inds']
    grasp_view_scores = end_points['stage2_view_scores']
    grasp_scores = end_points['stage3_grasp_scores']
    grasp_features_two_finger = end_points['stage3_grasp_features'].view(grasp_scores.size()[0], grasp_scores.size()[1], -1)
    grasp_widths = max_width * end_points['stage3_normalized_grasp_widths']
    grasp_widths[grasp_widths > max_width] = max_width

    grasp_preds = []
    grasp_features = []
    # for i in range(batch_size):
    #     cloud_mask_i = (coords[:, 0] == i)
    #     seed_inds_i = seed_inds[i]
    #     objectness_mask_i = objectness_mask[cloud_mask_i][seed_inds_i]
    #     if not objectness_mask_i.any():
    #         continue
    #     # 为了简洁，后续的详细解析逻辑省略，与原脚本相同
    #     # ...
    # # 假设解析完成，返回 grasp_preds 和 grasp_features
    # # return grasp_preds, grasp_features
    # # 为了使代码可运行，我将完整复制此函数
    for i in range(batch_size):
        cloud_mask_i = (coords[:, 0] == i)
        seed_inds_i = seed_inds[i]
        objectness_mask_i = objectness_mask[cloud_mask_i][seed_inds_i]
        if not objectness_mask_i.any(): continue

        seed_xyz_i = seed_xyz[i]
        point_features_i = point_features[i]
        seed_inds_i_ = seed_inds[i]
        before_generator_i = before_generator[i]
        grasp_view_xyz_i = grasp_view_xyz[i]
        grasp_view_inds_i = grasp_view_inds[i]
        grasp_view_scores_i = grasp_view_scores[i]
        grasp_scores_i = grasp_scores[i]
        grasp_widths_i = grasp_widths[i]
        
        Ns, A, D = grasp_scores_i.size()
        grasp_features_two_finger_i = grasp_features_two_finger[i]
        grasp_scores_i_A_D = copy.deepcopy(grasp_scores_i).view(Ns, -1)
        grasp_scores_i = torch.minimum(grasp_scores_i[:,:24,:], grasp_scores_i[:,24:,:])
        seed_inds_i_ = seed_inds_i_.view(Ns, -1)
        grasp_view_inds_i = grasp_view_inds_i.view(Ns, -1)
        grasp_view_scores_i = grasp_view_scores_i.view(Ns, -1)
        grasp_scores_i, grasp_angles_class_i = torch.max(grasp_scores_i, dim=1)
        grasp_angles_i = (grasp_angles_class_i.float() - 12) / 24 * np.pi
        grasp_angles_class_i = grasp_angles_class_i.unsqueeze(1)
        grasp_widths_pos_i = torch.gather(grasp_widths_i, 1, grasp_angles_class_i).squeeze(1)
        grasp_widths_neg_i = torch.gather(grasp_widths_i, 1, grasp_angles_class_i + 24).squeeze(1)
        grasp_scores_i, grasp_depths_class_i = torch.max(grasp_scores_i, dim=1, keepdims=True)
        grasp_depths_i = (grasp_depths_class_i.float() + 1) * 0.01
        grasp_depths_i -= 0.01
        grasp_depths_i[grasp_depths_class_i == 0] = 0.005
        grasp_angles_i = torch.gather(grasp_angles_i, 1, grasp_depths_class_i)
        grasp_widths_pos_i = torch.gather(grasp_widths_pos_i, 1, grasp_depths_class_i)
        grasp_widths_neg_i = torch.gather(grasp_widths_neg_i, 1, grasp_depths_class_i)
        rotation_matrices_i = batch_viewpoint_params_to_matrix(-grasp_view_xyz_i, grasp_angles_i.squeeze(1))
        grasp_widths_i = grasp_widths_pos_i + grasp_widths_neg_i
        rotation_matrices_i = rotation_matrices_i.view(Ns, 9)
        grasp_preds.append(torch.cat([grasp_scores_i, grasp_widths_i, grasp_depths_i, rotation_matrices_i, seed_xyz_i], axis=1))
        grasp_features.append(torch.cat([grasp_scores_i_A_D, grasp_features_two_finger_i, before_generator_i, point_features_i, grasp_view_inds_i, grasp_view_scores_i, seed_inds_i_, grasp_angles_i*24/np.pi+12, grasp_depths_i], axis=1))
        
    return grasp_preds, grasp_features


def run_graspnet_on_point_cloud(net, points, gripper_config, augment_mat=np.eye(4), flip=False, voxel_size=0.005):
    """在（增强后的）点云上运行GraspNet，返回初步的抓取姿态"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Transform and quantize
    points_transformed = transform_point_cloud(points, augment_mat).astype(np.float32)
    points_tensor = torch.from_numpy(points_transformed)
    coords = np.ascontiguousarray(points_tensor / voxel_size, dtype=int)
    _, idxs = ME.utils.sparse_quantize(coords, return_index=True)
    
    # Create Minkowski Sparse Tensor
    sinput = ME.SparseTensor(
        features=points_tensor[idxs],
        coordinates=ME.utils.sparse_collate([coords[idxs]], [torch.zeros(len(idxs), 1)])[0],
        device=device
    )

    # Inference
    end_points = {'sinput': sinput, 'point_clouds': [sinput.F]}
    with torch.no_grad():
        end_points = net(end_points)
        preds, grasp_features = parse_graspnet_predictions(end_points, max_width=gripper_config['max_width'])
        if not preds:
            return None, None, None

    preds = preds[0]
    grasp_features = grasp_features[0]

    # Transform back
    if flip: augment_mat[0, 0] = -augment_mat[0, 0]
    inv_augment_mat = torch.from_numpy(np.linalg.inv(augment_mat)).float().to(device)
    rotation = inv_augment_mat[:3, :3].unsqueeze(0).repeat(preds.size(0), 1, 1)
    translation = inv_augment_mat[:3, 3].unsqueeze(0).repeat(preds.size(0), 1)

    preds[:, 12:15] = torch.bmm(rotation, preds[:, 12:15].unsqueeze(-1)).squeeze(-1) + translation
    pose_rotation = torch.bmm(rotation, preds[:, 3:12].view(-1, 3, 3))
    if flip:
        # TODO Note: This flip logic is different for different grippers in the original files.
        # This generic version assumes Y and Z axes are flipped for the pose.
        # This may need to be handled in the inference script if logic differs.
        # pose_rotation[:, :, 1] = -pose_rotation[:, :, 1]
        # pose_rotation[:, :, 2] = -pose_rotation[:, :, 2]
        preds[:, 12] = -preds[:, 12]
        pose_rotation[:, 0, :] = -pose_rotation[:, 0, :]
        pose_rotation[:, :, 1] = -pose_rotation[:, :, 1]
    preds[:, 3:12] = pose_rotation.view(-1, 9)

    # TODO Filter
    # different grasper should set different threshold
    score_mask = preds[:, 9] > 0.9  # Use a general high score threshold
    width_mask = (preds[:, 1] < gripper_config['max_width']) & (preds[:, 1] > gripper_config['min_width'])
    ws = gripper_config['workspace_bounds']
    workspace_mask = (preds[:, 12] > ws[0, 0]) & (preds[:, 12] < ws[0, 1]) & \
                     (preds[:, 13] > ws[1, 0]) & (preds[:, 13] < ws[1, 1])
    
    final_mask = score_mask & width_mask & workspace_mask
    if not final_mask.any():
        return None, None, None

    preds = preds[final_mask]
    grasp_features = grasp_features[final_mask]

    heights = 0.03 * torch.ones([preds.shape[0], 1]).cuda()
    object_ids = -1 * torch.ones([preds.shape[0], 1]).cuda()
    ggarray = torch.cat([preds[:, 0:2], heights, preds[:, 2:15], preds[:, 15:16], object_ids], axis=-1)
    
    return ggarray, grasp_features, sinput.F[::4] # Downsample for collision checking

def load_gripper_mesh_pcds(mesh_path, voxel_grid):
    """加载用于碰撞检测的机械手网格点云"""
    pcds = {}
    pcd_path = os.path.join(mesh_path, 'meshes', 'source_pointclouds', f'voxel_size_{int(voxel_grid * 1000)}')
    # Inspire Hand has a different structure
    if 'inspire' in mesh_path:
        pcd_path = os.path.join(mesh_path, 'source_pointclouds', f'voxel_size_{int(voxel_grid * 1000)}')

    for type_folder in os.listdir(pcd_path):
        type_path = os.path.join(pcd_path, type_folder)
        if not os.path.isdir(type_path): continue
        for pcd_file in os.listdir(type_path):
            width = pcd_file[:-4]
            full_path = os.path.join(type_path, pcd_file)
            pcds[f'{type_folder}_{width}'] = o3d.io.read_point_cloud(full_path)
    return pcds

def run_collision_detection(multi_finger_gg, two_finger_gg, scene_points, gripper_config):
    """运行无模型碰撞检测"""
    mfcdetector = ModelFreeCollisionDetectorMultifinger(scene_points.cpu().numpy(), voxel_size=0.001)
    
    mesh_pcds = load_gripper_mesh_pcds(gripper_config['mesh_json_path'], gripper_config['voxel_grid'])
    
    # The 'detect' method parameters may vary slightly. This is a generalized call.
    # TODO The original code for Allegro and DH3/Inspire had different 'adjust_gripper_centers' etc.
    adjust_centers = 'inspire' not in gripper_config['mesh_json_path']
    collision_thresh = 1 if 'allegro' in gripper_config['mesh_json_path'] else 0
    approach_dist = 0.08 if 'allegro' in gripper_config['mesh_json_path'] else 0.05
    
    multi_finger_gg_col, two_finger_gg_col, empty_mask, min_width_idx = mfcdetector.detect(
        multi_finger_gg, two_finger_gg,
        gripper_config['mesh_json_path'], mesh_pcds,
        min_grasp_width=gripper_config['min_width'],
        VoxelGrid=gripper_config['voxel_grid'],
        approach_dist=approach_dist,
        collision_thresh=collision_thresh,
        adjust_gripper_centers=adjust_centers
    )
    
    return multi_finger_gg_col, two_finger_gg_col, empty_mask, min_width_idx