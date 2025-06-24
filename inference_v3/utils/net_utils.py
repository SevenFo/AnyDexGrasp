import torch
import MinkowskiEngine as ME
import copy
from models.minkowski_graspnet_single_point import MinkowskiGraspNet, MinkowskiGraspNetMultifingerType1Inference
import sys
import numpy as np
from models import minkowski_graspnet
sys.modules['minkowski_graspnet'] = minkowski_graspnet
from adg_utils.pt_utils import batch_viewpoint_params_to_matrix

def get_net(checkpoint_path, use_v2=False, half_views=False):
    if use_v2:
        net = MinkowskiGraspNet(num_depth=5, num_seed=2048, is_training=False, half_views=half_views)
    else:
        net = MinkowskiGraspNet(num_depth=4, num_seed=2048, is_training=False, half_views=half_views)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    return net

def parse_preds(end_points, max_grasp_width, min_grasp_width, use_v2=False):
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
    grasp_widths = max_grasp_width * end_points['stage3_normalized_grasp_widths']
    grasp_widths[grasp_widths > max_grasp_width] = max_grasp_width

    grasp_preds = []
    grasp_features = []
    grasp_vdistance_list = []
    batch_size = grasp_scores.size(0)
    
    for i in range(batch_size):
        cloud_mask_i = (coords[:, 0] == i)
        seed_inds_i = seed_inds[i]
        objectness_mask_i = objectness_mask[cloud_mask_i][seed_inds_i]

        if not objectness_mask_i.any():
            continue

        seed_xyz_i = seed_xyz[i]
        point_features_i = point_features[i]
        seed_inds_i = seed_inds_i
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
        seed_inds_i = seed_inds_i.view(Ns, -1)
        grasp_view_inds_i = grasp_view_inds_i.view(Ns, -1)
        grasp_view_scores_i = grasp_view_scores_i.view(Ns, -1)

        grasp_scores_i, grasp_angles_class_i = torch.max(grasp_scores_i, dim=1)
        grasp_angles_i = (grasp_angles_class_i.float()-12) / 24 * np.pi

        grasp_angles_class_i = grasp_angles_class_i.unsqueeze(1)
        grasp_widths_pos_i = torch.gather(grasp_widths_i, 1, grasp_angles_class_i).squeeze(1)
        grasp_widths_neg_i = torch.gather(grasp_widths_i, 1, grasp_angles_class_i+24).squeeze(1)

        grasp_scores_i, grasp_depths_class_i = torch.max(grasp_scores_i, dim=1, keepdims=True)
        grasp_depths_i = (grasp_depths_class_i.float() + 1) * 0.01
        grasp_depths_i -= 0.01
        grasp_depths_i[grasp_depths_class_i==0] = 0.005
        
        grasp_angles_i = torch.gather(grasp_angles_i, 1, grasp_depths_class_i)
        grasp_widths_pos_i = torch.gather(grasp_widths_pos_i, 1, grasp_depths_class_i)
        grasp_widths_neg_i = torch.gather(grasp_widths_neg_i, 1, grasp_depths_class_i)

        rotation_matrices_i = batch_viewpoint_params_to_matrix(-grasp_view_xyz_i, grasp_angles_i.squeeze(1))
        grasp_widths_i = grasp_widths_pos_i + grasp_widths_neg_i
        rotation_matrices_i = rotation_matrices_i.view(Ns, 9)

        grasp_preds.append(torch.cat([grasp_scores_i, grasp_widths_i, grasp_depths_i, rotation_matrices_i, seed_xyz_i],axis=1))
        grasp_features.append(torch.cat([grasp_scores_i_A_D, grasp_features_two_finger_i, before_generator_i, point_features_i, grasp_view_inds_i, grasp_view_scores_i, seed_inds_i, grasp_angles_i*24/np.pi+12, grasp_depths_i], axis=1))
        
    return grasp_preds, grasp_features
