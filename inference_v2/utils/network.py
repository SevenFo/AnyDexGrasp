# File: Inference/utils/network.py
import copy
import torch
import os
import numpy as np
import MinkowskiEngine as ME
from collections import OrderedDict

from models.minkowski_graspnet_single_point import MinkowskiGraspNet
from adg_utils.pt_utils import batch_viewpoint_params_to_matrix
from adg_utils.np_utils import transform_point_cloud

BATCH_SIZE = 1  # Constant from original files


def get_net(checkpoint_path, cfgs):
    """Loads the base GraspNet model."""
    if cfgs.use_graspnet_v2:
        net = MinkowskiGraspNet(
            num_depth=5, num_seed=2048, is_training=False, half_views=cfgs.half_views
        )
    else:
        net = MinkowskiGraspNet(
            num_depth=4, num_seed=2048, is_training=False, half_views=cfgs.half_views
        )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint["model_state_dict"])
    return net


def parse_preds(end_points, max_width):
    ## load preds
    MAX_GRASP_WIDTH = max_width
    before_generator = end_points['before_generator']  # (B, Ns, 256)
    point_features = end_points['point_features']  # (B, Ns, 512)
    coords = end_points['sinput'].C  # (\Sigma Ni, 4)
    objectness_pred = end_points['stage1_objectness_pred']  # (Sigma Ni, 2)
    objectness_mask = torch.argmax(objectness_pred, dim=1).bool()  # (\Sigma Ni,)
    seed_xyz = end_points['stage2_seed_xyz']  # (B, Ns, 3)
    seed_inds = end_points['stage2_seed_inds']  # (B, Ns)
    grasp_view_xyz = end_points['stage2_view_xyz']  # (B, Ns, 3)
    grasp_view_inds = end_points['stage2_view_inds']
    grasp_view_scores = end_points['stage2_view_scores']
    grasp_scores = end_points['stage3_grasp_scores']  # (B, Ns, A, D)
    grasp_features_two_finger = end_points['stage3_grasp_features'].view(grasp_scores.size()[0], grasp_scores.size()[1], -1) # (B, Ns, 3 + C)
    grasp_widths = MAX_GRASP_WIDTH * end_points['stage3_normalized_grasp_widths']  # (B, Ns, A, D)
    grasp_widths[grasp_widths > MAX_GRASP_WIDTH] = MAX_GRASP_WIDTH

    grasp_preds = []
    grasp_features = []
    grasp_vdistance_list = []
    for i in range(BATCH_SIZE):
        
        cloud_mask_i = (coords[:, 0] == i)
        seed_inds_i = seed_inds[i]
        objectness_mask_i = objectness_mask[cloud_mask_i][seed_inds_i]  # (Ns,)

        if objectness_mask_i.any() == False:
            continue

        seed_xyz_i = seed_xyz[i] # [objectness_mask_i]  # (Ns', 3)
        point_features_i = point_features[i] # [objectness_mask_i]
        
        seed_inds_i = seed_inds_i # [objectness_mask_i]
        before_generator_i = before_generator[i] # [objectness_mask_i]
        grasp_view_xyz_i = grasp_view_xyz[i] # [objectness_mask_i]  # (Ns', 3)
        grasp_view_inds_i = grasp_view_inds[i] # [objectness_mask_i]
        grasp_view_scores_i = grasp_view_scores[i] # [objectness_mask_i]
        grasp_scores_i = grasp_scores[i] # [objectness_mask_i]  # (Ns', A, D)
        grasp_widths_i = grasp_widths[i] # [objectness_mask_i] # (Ns', A, D)
        
        Ns, A, D = grasp_scores_i.size()
        grasp_features_two_finger_i = grasp_features_two_finger[i] # [objectness_mask_i] # (Ns', 3 + C)
        grasp_scores_i_A_D = copy.deepcopy(grasp_scores_i).view(Ns, -1)

        grasp_scores_i = torch.minimum(grasp_scores_i[:,:24,:], grasp_scores_i[:,24:,:])
        seed_inds_i = seed_inds_i.view(Ns, -1)
        grasp_view_inds_i = grasp_view_inds_i.view(Ns, -1)
        grasp_view_scores_i = grasp_view_scores_i.view(Ns, -1)

        grasp_scores_i, grasp_angles_class_i = torch.max(grasp_scores_i, dim=1) # (Ns', D), (Ns', D)
        grasp_angles_i = (grasp_angles_class_i.float()-12) / 24 * np.pi  # (Ns', topk, D)

        # grasp width & vdistance
        grasp_angles_class_i = grasp_angles_class_i.unsqueeze(1) # (Ns', 1, D)
        grasp_widths_pos_i = torch.gather(grasp_widths_i, 1, grasp_angles_class_i).squeeze(1) # (Ns', D)
        grasp_widths_neg_i = torch.gather(grasp_widths_i, 1, grasp_angles_class_i+24).squeeze(1) # (Ns', D)

        ## slice preds by grasp score/depth
        # grasp score & depth
        grasp_scores_i, grasp_depths_class_i = torch.max(grasp_scores_i, dim=1, keepdims=True) # (Ns', 1), (Ns', 1)
        grasp_depths_i = (grasp_depths_class_i.float() + 1) * 0.01  # (Ns'*topk, 1)

        grasp_depths_i -= 0.01
        grasp_depths_i[grasp_depths_class_i==0] = 0.005
        # grasp angle & width & vdistance
        grasp_angles_i = torch.gather(grasp_angles_i, 1, grasp_depths_class_i) # (Ns', 1)
        grasp_widths_pos_i = torch.gather(grasp_widths_pos_i, 1, grasp_depths_class_i) # (Ns', 1)
        grasp_widths_neg_i = torch.gather(grasp_widths_neg_i, 1, grasp_depths_class_i) # (Ns', 1)

        # convert to rotation matrix
        rotation_matrices_i = batch_viewpoint_params_to_matrix(-grasp_view_xyz_i, grasp_angles_i.squeeze(1))

        # # adjust gripper centers
        grasp_widths_i = grasp_widths_pos_i + grasp_widths_neg_i
        rotation_matrices_i = rotation_matrices_i.view(Ns, 9)

        # merge preds
        grasp_preds.append(torch.cat([grasp_scores_i, grasp_widths_i, grasp_depths_i, rotation_matrices_i, seed_xyz_i],axis=1))  # (Ns, 15)
        grasp_features.append(torch.cat([grasp_scores_i_A_D, grasp_features_two_finger_i, before_generator_i, point_features_i, grasp_view_inds_i, grasp_view_scores_i, seed_inds_i, grasp_angles_i*24/np.pi+12, grasp_depths_i], axis=1)) # (Ns'*3, A, D)
        
    return grasp_preds, grasp_features

def parse_preds_d(end_points, max_width):
    """Parses raw network output into grasp predictions and features."""
    coords = end_points["sinput"].C
    seed_inds = end_points["stage2_seed_inds"]
    seed_xyz = end_points["stage2_seed_xyz"]
    grasp_scores = end_points["stage3_grasp_scores"]
    grasp_widths = max_width * end_points["stage3_normalized_grasp_widths"]
    grasp_widths[grasp_widths > max_width] = max_width
    objectness_pred = end_points['stage1_objectness_pred']  # (Sigma Ni, 2)
    objectness_mask = torch.argmax(objectness_pred, dim=1).bool()  # (\Sigma Ni,)

    grasp_preds, grasp_features = [], []
    for i in range(BATCH_SIZE):
        # cloud_mask_i = (coords[:, 0] == i)
        # seed_inds_i = seed_inds[i]
        # objectness_mask_i = objectness_mask[cloud_mask_i][seed_inds_i]  # (Ns,)
        # if objectness_mask_i.any() == False:
        #     continue
        seed_xyz_i = seed_xyz[i]
        grasp_scores_i = end_points["stage3_grasp_scores"][i]
        grasp_widths_i = grasp_widths[i]
        grasp_view_xyz_i = end_points["stage2_view_xyz"][i]

        Ns, A, D = grasp_scores_i.size()

        # Symmetric scores
        grasp_scores_i_sym = torch.minimum(
            grasp_scores_i[:, :24, :], grasp_scores_i[:, 24:, :]
        )

        grasp_scores_i, grasp_angles_class_i = torch.max(grasp_scores_i_sym, dim=1)
        grasp_angles_i = (grasp_angles_class_i.float() - 12) / 24 * np.pi

        grasp_angles_class_i = grasp_angles_class_i.unsqueeze(1)
        grasp_widths_pos_i = torch.gather(
            grasp_widths_i, 1, grasp_angles_class_i
        ).squeeze(1)
        grasp_widths_neg_i = torch.gather(
            grasp_widths_i, 1, grasp_angles_class_i + 24
        ).squeeze(1)

        grasp_scores_i, grasp_depths_class_i = torch.max(
            grasp_scores_i, dim=1, keepdims=True
        )
        grasp_depths_i = (grasp_depths_class_i.float() + 1) * 0.01
        grasp_depths_i[grasp_depths_class_i == 0] = 0.005  # Special case for depth 0
        grasp_depths_i -= 0.01

        grasp_angles_i = torch.gather(grasp_angles_i, 1, grasp_depths_class_i)
        grasp_widths_pos_i = torch.gather(grasp_widths_pos_i, 1, grasp_depths_class_i)
        grasp_widths_neg_i = torch.gather(grasp_widths_neg_i, 1, grasp_depths_class_i)
        grasp_widths_i = grasp_widths_pos_i + grasp_widths_neg_i

        rotation_matrices_i = batch_viewpoint_params_to_matrix(
            -grasp_view_xyz_i, grasp_angles_i.squeeze(1)
        ).view(Ns, 9)

        grasp_preds.append(
            torch.cat(
                [
                    grasp_scores_i,
                    grasp_widths_i,
                    grasp_depths_i,
                    rotation_matrices_i,
                    seed_xyz_i,
                ],
                axis=1,
            )
        )

        # Assemble full feature vector
        grasp_scores_i_A_D = copy.deepcopy(end_points["stage3_grasp_scores"][i]).view(
            Ns, -1
        )
        grasp_features_two_finger_i = end_points["stage3_grasp_features"].view(
            grasp_scores.size()[0], grasp_scores.size()[1], -1
        )[i]
        before_generator_i = end_points["before_generator"][i]
        point_features_i = end_points["point_features"][i]
        grasp_view_inds_i = end_points["stage2_view_inds"][i].view(Ns, -1)
        grasp_view_scores_i = end_points["stage2_view_scores"][i].view(Ns, -1)
        seed_inds_i_flat = seed_inds[i].view(Ns, -1)

        grasp_features.append(
            torch.cat(
                [
                    grasp_scores_i_A_D,
                    grasp_features_two_finger_i,
                    before_generator_i,
                    point_features_i,
                    grasp_view_inds_i,
                    grasp_view_scores_i,
                    seed_inds_i_flat,
                    grasp_angles_i * 24 / np.pi + 12,
                    grasp_depths_i,
                ],
                axis=1,
            )
        )

    return grasp_preds, grasp_features


def predict_grasps(
    net, points, config, augment_mat=np.eye(4), flip=False, voxel_size=0.005
):
    """Performs a single grasp prediction pass on a point cloud."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    points_transformed = transform_point_cloud(points, augment_mat).astype(np.float32)
    points_tensor = torch.from_numpy(points_transformed)

    coords = np.ascontiguousarray(points_transformed / voxel_size, dtype=int)
    _, idxs = ME.utils.sparse_quantize(coords, return_index=True)
    coords, points_quantized = coords[idxs], points_tensor[idxs]

    coords_batch, points_batch = ME.utils.sparse_collate([coords], [points_quantized])
    sinput = ME.SparseTensor(points_batch, coords_batch, device=device)

    end_points = {"sinput": sinput, "point_clouds": [sinput.F]}
    with torch.no_grad():
        end_points = net(end_points)
        preds, grasp_features = parse_preds(end_points, config["max_width"])
        if not preds:
            return None, None, points_quantized.cuda(), None

    preds = preds[0]

    if flip:
        augment_mat[:, 0] = -augment_mat[:, 0]

    augment_mat_tensor = torch.tensor(copy.deepcopy(np.linalg.inv(augment_mat).astype(np.float32)), device=device)
    rotation = augment_mat_tensor[:3, :3].reshape((-1)).repeat((preds.size()[0], 1)).view((preds.size()[0], 3, 3))
    translation = augment_mat_tensor[:3, 3]
    
    preds[:,12:15] = torch.matmul(rotation, preds[:,12:15].view((-1, 3, 1))).view(-1, 3) + translation
    pose_rotation = torch.matmul(rotation, preds[:,3:12].view((-1, 3, 3)))

    if flip:
        preds[:, 12] = -preds[:, 12]
        pose_rotation[:, 0, :] = -pose_rotation[:, 0, :]
        pose_rotation[:, :, 1] = -pose_rotation[:, :, 1]

    preds[:, 3:12] = pose_rotation.view(-1, 9)

    # Filtering
    mask_quality = (
        (preds[:, 9] > 0.92)
        & (preds[:, 1] < config["max_width"])
        & (preds[:, 1] > config["min_width"])
    )
    x_min, x_max, y_min, y_max = config["workspace_mask"]
    mask_workspace = (
        (preds[:, 12] > x_min)
        & (preds[:, 12] < x_max)
        & (preds[:, 13] > y_min)
        & (preds[:, 13] < y_max)
    )

    final_mask = mask_quality & mask_workspace
    preds = preds[final_mask]
    grasp_features = grasp_features[0][final_mask]

    if len(preds) == 0:
        return None, None, points_quantized.cuda(), None

    heights = 0.03 * torch.ones([preds.shape[0], 1], device=device)
    object_ids = -1 * torch.ones([preds.shape[0], 1], device=device)
    ggarray = torch.cat(
        [preds[:, 0:2], heights, preds[:, 2:15], preds[:, 15:16], object_ids], axis=-1
    )

    return ggarray, grasp_features, points_quantized.cuda(), [sinput]


def get_gripper_model(config):
    """Loads the gripper-specific prediction models."""
    import sys
    from models import minkowski_graspnet
    sys.modules['minkowski_graspnet'] = minkowski_graspnet
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gripper_models = {}
    model_base_path = config["model_path"]
    for model_type in os.listdir(model_base_path):
        gripper_models[model_type] = {}
        type_path = os.path.join(model_base_path, model_type)
        for model_class in os.listdir(type_path):
            class_path = os.path.join(type_path, model_class)
            models = []
            for model_file in os.listdir(class_path):
                model_path = os.path.join(class_path, model_file)
                model = config["model_class"](input_num=int(model_type))
                model.load_state_dict(torch.load(model_path).state_dict())
                model.to(device)
                model.eval()
                models.append(model)
            assert len(models) == 1, f"len(models):{len(models)} != 1"
            gripper_models[model_type][model_class] = models[0]
    return gripper_models


def predict_multi_finger_grasp(
    gripper_models, grasp_features_dic, ggarray, grasp_features, config
):
    """Predicts multi-finger grasp type and depth from two-finger features."""
    if config["random_grasp"]:
        num_grasps = len(ggarray)
        gripper_types = np.random.randint(0, config["num_type"], (num_grasps,))
        gripper_depths = np.random.randint(0, config["num_depth"], (num_grasps,))
        if config["name"] == "inspire":  # Special case for Inspire
            gripper_depths[gripper_types == 5] += 2
        gripper_depths = gripper_depths * 0.01
        scores = ggarray[:, 0]
        return gripper_depths, gripper_types + 1, scores, ggarray, grasp_features

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for k, v in grasp_features_dic.items():
        if k in ["point_id", "sinput"]:
            continue
        grasp_features_dic[k] = torch.from_numpy(v).to(device)

    all_scores = []
    num_depth, num_type = config["num_depth"], config["num_type"]

    for model_type, models_by_class in gripper_models.items():
        if model_type == "240":  # Skip as '480' is the final one used
            continue
        elif model_type == "480":
            print('use final model: ', model_type)
            model_input = grasp_features_dic["grasp_preds_features"]
        
        models_by_class_sorted = OrderedDict(sorted(models_by_class.items(), key = lambda t : int(t[0])))
        for model_class, sub_models in models_by_class_sorted.items():
            # class_scores = torch.tensor(0, device=device)
            # for sub_model in sub_models:
            #     with torch.no_grad():
            #         pred, _ = sub_model(model_input)
            #         pred = pred.view(
            #             pred.shape[0], -1
            #         )  # (B, 5 * num_depth) for one class
            #     class_scores += pred

            # class_scores /= len(sub_models)
            # HARD CODE
            with torch.no_grad():
                grasp_pred, _ = sub_models(model_input)
                grasp_pred = grasp_pred.view(grasp_pred.shape[0],-1) # (B, num_depth)
            # Select scores based on two-finger depth
            two_finger_depths = grasp_features_dic["grasp_depths"].view(-1, 1)  # (B, 1) for boardcasting
            base_indices = torch.arange(num_depth, device=device).unsqueeze(
                0
            )  # (1, num_depth)
            select_indices = (
                two_finger_depths * num_depth + base_indices
            )  # (B, num_depth)

            selected_scores = grasp_pred.gather(1, select_indices.long())
            all_scores.append(selected_scores)

    final_scores = torch.cat(all_scores, dim=1).view(-1)

    top_k_count = min(3000, final_scores.size(0)) # HARD CODE
    scores, indices = final_scores.topk(top_k_count)

    pose_indices = (indices / (num_depth * num_type)).long()

    ggarray_out = torch.from_numpy(ggarray).to(device)[pose_indices].cpu().numpy()
    grasp_features_out = (
        torch.from_numpy(grasp_features).to(device)[pose_indices].cpu().numpy()
    )

    type_indices = ((indices % (num_depth * num_type)) / num_depth).int()
    depth_indices = ((indices % (num_depth * num_type)) % num_depth).int()

    if config["name"] == "inspire":
        depth_indices[type_indices == 5] += 2

    gripper_depths = depth_indices.cpu().numpy() * 0.01
    gripper_types = type_indices.cpu().numpy() + 1

    return (
        gripper_depths,
        gripper_types,
        scores.detach().cpu().numpy(),
        ggarray_out,
        grasp_features_out,
    ) 
