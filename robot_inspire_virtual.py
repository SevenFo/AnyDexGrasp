import os
os.environ['DISPLAY'] = '109.105.4.86:0.0'
import copy
import os
import time
import argparse
import torch
import numpy as np
import sys
import cv2
import types
from tqdm import tqdm
import open3d as o3d
import MinkowskiEngine as ME
from graspnetAPI import GraspGroup
from collections import OrderedDict

from models.minkowski_graspnet_single_point import MinkowskiGraspNet, MinkowskiGraspNetMultifingerType1Inference
from adg_utils.np_utils import transform_point_cloud
from adg_utils.pt_utils import batch_viewpoint_params_to_matrix
from adg_utils.collision_detector import ModelFreeCollisionDetectorMultifinger
from ur_toolbox.robot.Inspire.InspireHandR_grasp import InspireHandRGraspGroup

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
parser.add_argument('--inspire_model_path', default='logs/model/inspire_model/obj140', help='inspire model checkpoint path')
parser.add_argument('--inspire_mesh_json_path', default='generate_mesh_and_pointcloud/inspire_urdf', help='InspireHandR meshes and json path')
parser.add_argument('--depth_image', default=None, help='Path to pre-recorded depth image (PNG)')
parser.add_argument('--color_image', default=None, help='Path to pre-recorded color image (PNG)')
parser.add_argument('--use_graspnet_v2', action='store_true', help='Whether to use graspnet v2 format')
parser.add_argument('--half_views', action='store_true', help='Use only half views in network.')
cfgs = parser.parse_args()

MAX_GRASP_WIDTH = 0.1
MIN_GRASP_WIDTH = 0.01
BATCH_SIZE = 1
DEBUG = True  # Always enable visualization
CALIB = False
GRIPPER_TOTAL_LEN = 0.155
FLANGE_TOTAL_LEN = 0.055
INSPIREHANDR_DEFAULT_DEPTH = 0.000
NUM_OF_INSPIRE_DEPTH = 4
NUM_OF_INSPIRE_TYPE = 8
INSPIREHANDR_VOXElGRID = 0.003
POINTCLOUD_AUGMENT_NUM = 10
RANDOM_GRASP = False

def parse_preds(end_points, use_v2=False):
    ## load preds
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

def get_net(checkpoint_path, use_v2=False):
    if use_v2:
        net = MinkowskiGraspNet(num_depth=5, num_seed=2048, is_training=False, half_views=cfgs.half_views)
    else:
        net = MinkowskiGraspNet(num_depth=4, num_seed=2048, is_training=False, half_views=cfgs.half_views)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    return net

def flip_ggarray(ggarray):
    ggarray_rotations = ggarray[:, 4:13].reshape((-1, 3, 3))
    tcp_x_axis_on_base_frame = ggarray_rotations[:, 1, 1]
    if_flip = [False for _ in range(len(ggarray))]
    for ids, y_x in enumerate(tcp_x_axis_on_base_frame):
        if y_x < 0:
            ggarray_rotations[ids, :3, 0:2] = -ggarray_rotations[ids, :3, 0:2]
            if_flip[ids] = True
    ggarray[:, 4:13] = ggarray_rotations.reshape((-1, 9))
    return ggarray, if_flip

def get_grasp_features(grasp_features_array):
    grasp_features = dict()
    grasp_features['grasp_angles'] = int(grasp_features_array[-3]+0.1)
    grasp_features['grasp_depths'] = int(grasp_features_array[-2]*100+0.1)
    grasp_features['stage3_grasp_scores'] = grasp_features_array[:240].tolist()
    grasp_features['grasp_preds_features'] = grasp_features_array[240:240+480].tolist()
    grasp_features['stage3_grasp_features'] = grasp_features_array[240+480:240+480+512].tolist()
    grasp_features['before_generator'] = grasp_features_array[240+480+512:240+480+512+512].tolist()
    grasp_features['point_features'] = grasp_features_array[240+480+512+512:240+480+512+512+512].tolist()
    grasp_features['point_id'] = int(grasp_features_array[-4])
    grasp_features['if_flip'] = bool(int(grasp_features_array[-1]))
    grasp_features['view_inds'] = bool(int(grasp_features_array[-6]))
    grasp_features['view_score'] = bool(int(grasp_features_array[-5]))
    return grasp_features

def get_graspgroup_features(grasp_features_array, sinput):
    grasp_features = dict()
    grasp_features['grasp_angles'] = (grasp_features_array[:, -3]+0.1).astype(int)
    grasp_features['grasp_depths'] = (grasp_features_array[:, -2]*100+0.1).astype(int)
    grasp_features['stage3_grasp_scores'] = grasp_features_array[:, :240]
    grasp_features['grasp_preds_features'] = grasp_features_array[:, 240:240+480]
    grasp_features['stage3_grasp_features'] = grasp_features_array[:, 240+480:240+480+512]
    grasp_features['before_generator'] = grasp_features_array[:, 240+480+512:240+480+512+512]
    grasp_features['point_features'] = grasp_features_array[:, 240+480+512+512:240+480+512+512+512]
    grasp_features['point_id'] = (grasp_features_array[:, -4]+0.1).astype(int)
    grasp_features['if_flip'] = grasp_features_array[:, -1]
    grasp_features['view_inds'] = (grasp_features_array[:, -6]+0.1).astype(int)
    grasp_features['view_score'] = grasp_features_array[:, -5]
    grasp_features['sinput'] = sinput

    grasp_preds_features_rot = np.zeros(grasp_features['grasp_preds_features'].shape, dtype = np.float32)
    new_type = grasp_features['grasp_angles']
    for idx, if_flip in enumerate(grasp_features['if_flip']):
        if if_flip:
            first_half_scores = copy.deepcopy(grasp_features['grasp_preds_features'][idx, :120])
            last_half_scores = copy.deepcopy(grasp_features['grasp_preds_features'][idx, 120:240])
            first_half_widths = copy.deepcopy(grasp_features['grasp_preds_features'][idx, 240:360])
            last_half_widths = copy.deepcopy(grasp_features['grasp_preds_features'][idx, 360:480])
            grasp_features['grasp_preds_features'][idx, :120] = last_half_scores
            grasp_features['grasp_preds_features'][idx, 120:240] = first_half_scores
            grasp_features['grasp_preds_features'][idx, 240:360] = last_half_widths
            grasp_features['grasp_preds_features'][idx, 360:480] = first_half_widths
        grasp_preds_features_rot[idx, :240-new_type[idx]*5] = grasp_features['grasp_preds_features'][idx, new_type[idx]*5:240]
        grasp_preds_features_rot[idx, 240-new_type[idx]*5:240] = grasp_features['grasp_preds_features'][idx, 0:new_type[idx]*5]
        grasp_preds_features_rot[idx, 240:480-new_type[idx]*5] = grasp_features['grasp_preds_features'][idx, 240+new_type[idx]*5:480]
        grasp_preds_features_rot[idx, 480-new_type[idx]*5:480] = grasp_features['grasp_preds_features'][idx, 240:240+new_type[idx]*5] 
    grasp_features['grasp_preds_features'] = grasp_preds_features_rot

    return grasp_features

def load_meshes_pointcloud(path):
    meshes_pcls = dict()
    meshes_pcl_path = os.path.join(path, 'meshes/source_pointclouds/voxel_size_' + str(int(INSPIREHANDR_VOXElGRID * 1000)))
    for type in os.listdir(meshes_pcl_path):
        type_path = os.path.join(meshes_pcl_path, type)
        for name in os.listdir(type_path):
            width = name[:-4]
            name_path = os.path.join(type_path, name)
            meshes_pcl = o3d.io.read_point_cloud(name_path)
            meshes_pcls[type + '_' + width] = meshes_pcl
    return meshes_pcls

def get_inspire_model(inspire_models_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    import sys
    from models import minkowski_graspnet
    sys.modules['minkowski_graspnet'] = minkowski_graspnet
    inspire_models = dict()
    for model_type in os.listdir(inspire_models_path):
        inspire_model_path = os.path.join(inspire_models_path, model_type)
        inspire_models[model_type] = dict()
        for model_class in os.listdir(inspire_model_path):
            model_classs_type = os.path.join(inspire_model_path, model_class)
            for model in os.listdir(model_classs_type):
                inspire_model_type_path = os.path.join(model_classs_type, model)
                inspire_model = MinkowskiGraspNetMultifingerType1Inference(input_num=int(model_type))
                inspire_net = torch.load(inspire_model_type_path)
                inspire_model.load_state_dict(inspire_net.state_dict())
                inspire_model.to(device)
                inspire_model.eval()
                inspire_models[model_type][model_class] = inspire_model
    return inspire_models

def get_inspire_depth_type(inspire_models, grasp_features_dic, ggarray, grasp_features):
    if RANDOM_GRASP:
        inspire_type = np.random.randint(0, NUM_OF_INSPIRE_TYPE, (len(ggarray),))
        inspire_depth = (np.random.randint(0, 4, (len(ggarray),)))
        inspire_depth[inspire_type==5] = inspire_depth[inspire_type==5] + 2
        inspire_depth = inspire_depth * 0.01
        scores = ggarray[:, 0] 
        return inspire_depth, inspire_type, scores, ggarray, grasp_features
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for k, v in grasp_features_dic.items():
        if k == 'point_id' or k == 'sinput':
            continue
        grasp_features_dic[k] = torch.tensor(copy.deepcopy(v), device=device)

    inspire_depth_type_scores = []
    for model_type, inspire_model in inspire_models.items():
        if model_type == '240':
            continue
            model_input = torch.cat([grasp_features_dic["stage3_grasp_scores"]], dim=1)
        elif model_type == '480':
            print('use final model: ', model_type)
            model_input = torch.cat([grasp_features_dic["grasp_preds_features"]], dim=1)
        

        inspire_model_sorted = OrderedDict(sorted(inspire_model.items(), key = lambda t : int(t[0])))
        for model_class, sub_inspire_model in inspire_model_sorted.items():
            with torch.no_grad():
                grasp_pred, _ = sub_inspire_model(model_input) # (B, 1， NUM_OF_TWO_FINGER_DEPTH*NUM_OF_INSPIRE_DEPTH)
                grasp_pred = grasp_pred.view(grasp_pred.shape[0], 5*NUM_OF_INSPIRE_DEPTH)
            two_fingers_depth = grasp_features_dic['grasp_depths'] # (B, )
            base = torch.tensor(np.array([[i for i in range(NUM_OF_INSPIRE_DEPTH)]
                                  for _ in range(grasp_pred.size()[0])]), device=device) # (B, NUM_OF_INSPIRE_DEPTH)
            select_index = (two_fingers_depth).view(-1, 1) * NUM_OF_INSPIRE_DEPTH + base
            inspire_depth_type_scores.append(grasp_pred.gather(1, select_index))
    inspire_depth_type_scores = torch.cat(inspire_depth_type_scores, axis=1).view(-1) # (B, NUM_OF_INSPIRE_DEPTH*NUM_OF_INSPIRE_TYPE)
    scores, index = inspire_depth_type_scores.topk(min(3000, inspire_depth_type_scores.size()[0]))
    pose_index = (index / (NUM_OF_INSPIRE_DEPTH * NUM_OF_INSPIRE_TYPE)).long()
    ggarray = torch.tensor(copy.deepcopy(ggarray), device=device)[pose_index]
    grasp_features = torch.tensor(copy.deepcopy(grasp_features), device=device)[pose_index]
    inspire_depth = ((index % (NUM_OF_INSPIRE_DEPTH * NUM_OF_INSPIRE_TYPE)) % NUM_OF_INSPIRE_DEPTH).int()
    inspire_type = ((index % (NUM_OF_INSPIRE_DEPTH * NUM_OF_INSPIRE_TYPE)) / NUM_OF_INSPIRE_DEPTH).int()

    inspire_depth[inspire_type==5] = inspire_depth[inspire_type==5] + 2
    inspire_depth = inspire_depth * 0.01

    return inspire_depth.cpu().numpy(), inspire_type.cpu().numpy() + 1, scores.detach().cpu().numpy(), \
                ggarray.cpu().numpy(), grasp_features.cpu().numpy()

def augment_data(flip=False):
    flip_mat = np.identity(4)
    # Flipping along the YZ plane
    if flip:
        flip_mat = np.array([[-1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]])

    # Rotation along up-axis/Z-axis
    rot_angle = (np.random.random() * np.pi / 3) - np.pi / 6  # -30 ~ +30 degree
    c, s = np.cos(rot_angle), np.sin(rot_angle)
    rot_mat = np.array([[c, -s, 0, 0],
                        [s, c, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])

    # Translation along X/Y/Z-axis
    offset_x = np.random.random() * 0.1 - 0.05  # -0.05 ~ 0.05
    offset_y = np.random.random() * 0.1 - 0.05  # -0.05 ~ 0.05
    trans_mat = np.array([[1, 0, 0, offset_x],
                          [0, 1, 0, offset_y],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]])

    aug_mat = np.dot(trans_mat, np.dot(rot_mat, flip_mat).astype(np.float32)).astype(np.float32)
    return aug_mat

def get_grasp(net, depths, color_image=None, augment_mat=np.eye(4), flip=False, voxel_size=0.005):
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

    if color_image is not None:
        if color_image.dtype == np.uint8:
            colors = color_image[mask].astype(np.float32) / 255.0 
        else:
            colors = color_image[mask].astype(np.float32)

    cloud = None
    if color_image is not None:
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        cloud.colors = o3d.utility.Vector3dVector(colors)

    points = transform_point_cloud(points, augment_mat).astype(np.float32)
    points = torch.from_numpy(points)
    coords = np.ascontiguousarray(points / voxel_size, dtype=int)
    # Upd Note. API change.
    _, idxs = ME.utils.sparse_quantize(coords, return_index=True)
    coords = coords[idxs]
    points = points[idxs]
    coords_batch, points_batch = ME.utils.sparse_collate([coords], [points])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sinput = ME.SparseTensor(points_batch, coords_batch, device=device)

    end_points = {'sinput': sinput, 'point_clouds': [sinput.F]}
    with torch.no_grad():
        end_points = net(end_points)
        preds, grasp_features = parse_preds(end_points, use_v2=cfgs.use_graspnet_v2)
        if len(preds) == 0:
            print('No grasp detected')
            return None, cloud, points.cuda(), None, None
        else:
            preds = preds[0]
    # filter
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
    preds[:, 3:12] = pose_rotation.view((-1, 9))

    mask = (preds[:,9] > 0.92) & (preds[:,1] < MAX_GRASP_WIDTH) & (preds[:,1] > MIN_GRASP_WIDTH)
    workspace_mask = (preds[:,12] > -0.25) & (preds[:,12] < 0.25) & (preds[:,13] > -0.205) & (preds[:,13] < 0.03)
    preds = preds[workspace_mask & mask]
    grasp_features = grasp_features[0][workspace_mask & mask]
    if len(preds) == 0:
        print('No grasp detected after masking')
        return None, cloud, points.cuda(), None, None

    points = points.cuda()
    heights = 0.03 * torch.ones([preds.shape[0], 1]).cuda()
    object_ids = -1 * torch.ones([preds.shape[0], 1]).cuda()
    ggarray = torch.cat([preds[:, 0:2], heights, preds[:, 2:15], preds[:, 15:16], object_ids], axis=-1)

    return ggarray, cloud, points, grasp_features, [sinput]

def get_ggarray_features(net, depths, color_image=None):
    augment_mat1 = np.eye(4)
    augment_mats = []
    
    for i in range(POINTCLOUD_AUGMENT_NUM):
        if i % 2 == 0:
            augment_mat = augment_data()
        else:
            augment_mat = augment_data(flip=True)
        augment_mats.append(augment_mat)

    # 第一次调用 - 获取点云和抓取
    ggarray, cloud, points_down, grasp_features, sinput = get_grasp(
        net, depths, color_image, augment_mat=augment_mat1
    )
    
    # 保存第一次调用的点云（带颜色）
    colored_cloud = cloud
    
    # 处理后续增强迭代
    for i in range(POINTCLOUD_AUGMENT_NUM):
        if i % 2 == 0:
            ggarray2, _, _, grasp_features2, sinput2 = get_grasp(
                net, depths, color_image, augment_mat=augment_mats[i]
            )
        else:
            ggarray2, _, _, grasp_features2, sinput2 = get_grasp(
                net, depths, color_image, augment_mat=augment_mats[i], flip=True
            )
            
        if ggarray2 is None:
            continue
            
        if ggarray is None:
            ggarray = ggarray2
            grasp_features = grasp_features2
            sinput = sinput2
        else:
            sinput.append(sinput2[0])
            ggarray = torch.cat([ggarray, ggarray2], axis=0)
            grasp_features = torch.cat([grasp_features, grasp_features2], axis=0)
    
    # 确保点云不为空
    if colored_cloud is None:
        # 如果没有点云，创建一个空点云
        colored_cloud = o3d.geometry.PointCloud()
    
    return ggarray, colored_cloud, points_down, grasp_features, sinput


def select_grasp_type(inspire_gg):
    select_gg_types = [[] for _ in range(1, NUM_OF_INSPIRE_TYPE+1)]
    for idx, score in enumerate(inspire_gg.scores):
        grasp_type = inspire_gg.grasp_types[idx]
        if grasp_type in [1]:
            max_num = 10
        else:
            max_num = 50
        if len(select_gg_types[int(grasp_type)-1]) < max_num:
            select_gg_types[int(grasp_type)-1].append(idx)
    gg_type_id = []
    for gg_type in select_gg_types:
        gg_type_id += gg_type
    return gg_type_id

def visualize_grasps(scene_cloud, pre_collision_gg, post_collision_gg, inspire_pre_gg, inspire_post_gg):
    """
    Visualize point cloud with pre-collision and post-collision grasps
    """
    # Create coordinate frame
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    
    # Create visualization objects
    vis_objects = [scene_cloud, frame]
    
    # Add pre-collision grasps (green)
    for i in range(min(10, len(pre_collision_gg))):
        grasp_mesh = inspire_pre_gg[i].load_mesh(cfgs.inspire_mesh_json_path, pre_collision_gg[i])
        grasp_mesh.paint_uniform_color([0, 1, 0])  # Green for pre-collision
        vis_objects.append(grasp_mesh)
    
    # Add post-collision grasps (red)
    for i in range(min(10, len(post_collision_gg))):
        grasp_mesh = inspire_post_gg[i].load_mesh(cfgs.inspire_mesh_json_path, post_collision_gg[i])
        grasp_mesh.paint_uniform_color([1, 0, 0])  # Red for post-collision
        vis_objects.append(grasp_mesh)
    
    # Visualize
    o3d.visualization.draw_geometries(vis_objects)

def visualize_grasp_proposals(scene_cloud, gg_array, inspire_gg_array, title="Grasp Proposals"):
    """
    Visualize grasp proposals with different colors for different grasp types
    """
    # Create coordinate frame
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    
    # Create visualization objects
    vis_objects = [scene_cloud, frame]
    
    # Color map for different grasp types
    color_map = {
        1: [1, 0, 0],   # Red
        2: [0, 1, 0],   # Green
        3: [0, 0, 1],   # Blue
        4: [1, 1, 0],   # Yellow
        5: [1, 0, 1],   # Magenta
        6: [0, 1, 1],   # Cyan
        7: [0.5, 0.5, 0], # Olive
        8: [0.5, 0, 0.5]  # Purple
    }
    
    # Add grasps with color coding by type
    for i in range(min(80, len(gg_array))):
        grasp_type = int(inspire_gg_array.grasp_types[i])
        color = color_map.get(grasp_type, [0.5, 0.5, 0.5])  # Default to gray
        
        grasp_mesh = inspire_gg_array[i].load_mesh(cfgs.inspire_mesh_json_path, gg_array[i])
        grasp_mesh.paint_uniform_color(color)
        vis_objects.append(grasp_mesh)
    
    # Visualize
    o3d.visualization.draw_geometries(vis_objects, window_name=title)

def process_and_visualize():
    # Load network
    net = get_net(cfgs.checkpoint_path, use_v2=cfgs.use_graspnet_v2)
    
    # Load InspireHandR models
    inspire_models = get_inspire_model(cfgs.inspire_model_path)
    
    # Load meshes for collision detection
    meshes_pcls = load_meshes_pointcloud(cfgs.inspire_mesh_json_path)
    
    # Load depth and color images
    if cfgs.depth_image is None or not os.path.exists(cfgs.depth_image):
        print("Error: Depth image path is invalid")
        return
    
    depths = cv2.imread(cfgs.depth_image, cv2.IMREAD_ANYDEPTH)
    if depths is None:
        print("Error: Failed to load depth image")
        return
    
    color_image = None
    if cfgs.color_image and os.path.exists(cfgs.color_image):
        color_image = cv2.imread(cfgs.color_image)
        if color_image is not None:
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    
    # Process depth image
    t1 = time.time()
    ggarray, cloud, points_down, grasp_features, sinput = get_ggarray_features(net, depths, color_image)
    t3 = time.time()
    print(f'Net Time:{t3 - t1}')
    
    if ggarray is None:
        print('No grasp detected')
        return
    
    # Process grasps
    ggarray = ggarray.cpu().numpy()
    grasp_features = grasp_features.cpu().numpy()
    two_fingers_source_grasp_features = copy.deepcopy(grasp_features)
    
    ggarray, if_flip = flip_ggarray(ggarray)
    grasp_features = np.c_[grasp_features, if_flip]
    ggarray = ggarray[~np.array(if_flip)]
    grasp_features = grasp_features[~np.array(if_flip)]
    
    source_index = ggarray[:, 0].argsort()
    ggarray = ggarray[source_index][::-1][:1000]
    grasp_features = grasp_features[source_index][::-1][:1000]
    
    grasp_features_dic = get_graspgroup_features(grasp_features, sinput)
    
    inspire_depth, inspire_type, scores, ggarray, grasp_features = \
        get_inspire_depth_type(inspire_models, grasp_features_dic, ggarray, grasp_features=grasp_features)
    
    if not RANDOM_GRASP:
        score_thresh = 0.85
        mask = (scores > score_thresh)
        ggarray = ggarray[mask]
        grasp_features = grasp_features[mask]
        inspire_depth = inspire_depth[mask]
        inspire_type = inspire_type[mask]
        scores = scores[mask]
    
    if len(ggarray) == 0:
        print('There is no grasp that score greater than 0.9 ')
        return
    
    # Create grasp groups
    two_fingers_ggarray = GraspGroup(ggarray)
    two_fingers_ggarray_object_ids_source = ggarray[:, 16]
    two_fingers_ggarray_source = copy.deepcopy(two_fingers_ggarray)
    
    InspireHandR_ggarray = InspireHandRGraspGroup() 
    InspireHandR_ggarray.set_grasp_min_width(MIN_GRASP_WIDTH)
    InspireHandR_ggarray.from_graspgroup(two_fingers_ggarray, inspire_type, cfgs.inspire_mesh_json_path)
    InspireHandR_ggarray.object_ids = two_fingers_ggarray_object_ids_source
    InspireHandR_ggarray.scores = scores
    InspireHandR_ggarray.depths = InspireHandR_ggarray.depths + inspire_depth + INSPIREHANDR_DEFAULT_DEPTH
    InspireHandR_ggarray_source = copy.deepcopy(InspireHandR_ggarray)
    
    # Filter by z-axis
    index_filter_by_z_axis = InspireHandR_ggarray.filter_grasp_group_by_z_axis(0.4)
    two_fingers_ggarray = two_fingers_ggarray[index_filter_by_z_axis]
    two_fingers_ggarray_object_ids = two_fingers_ggarray_object_ids_source[index_filter_by_z_axis]
    grasp_features = grasp_features[index_filter_by_z_axis]
    
    if len(InspireHandR_ggarray) == 0:
        print('No grasp detected after filter')
        return
    
    # Select by grasp type
    index_type = select_grasp_type(InspireHandR_ggarray)
    InspireHandR_ggarray = InspireHandR_ggarray[index_type]
    two_fingers_ggarray = two_fingers_ggarray[index_type]
    grasp_features = grasp_features[index_type]
    two_fingers_ggarray_object_ids = two_fingers_ggarray_object_ids[index_type]
    
    # Sort by score
    index_score = np.argsort(InspireHandR_ggarray.scores)[::-1]
    InspireHandR_ggarray = InspireHandR_ggarray[index_score]
    two_fingers_ggarray = two_fingers_ggarray[index_score]
    grasp_features = grasp_features[index_score]
    two_fingers_ggarray_object_ids = two_fingers_ggarray_object_ids[index_score]
    
    # Collision detection
    approach_distance = 0.06
    mfcdetector = ModelFreeCollisionDetectorMultifinger(points_down.cpu().numpy(), voxel_size=0.001)
    InspireHandR_ggarray_post, two_fingers_ggarray_post, empty_mask, min_width_index = mfcdetector.detect(
        InspireHandR_ggarray, two_fingers_ggarray,
        cfgs.inspire_mesh_json_path, meshes_pcls, min_grasp_width=MIN_GRASP_WIDTH,
        VoxelGrid=INSPIREHANDR_VOXElGRID, DEBUG=False, approach_dist=approach_distance,
        collision_thresh=0, adjust_gripper_centers=True
    )
    
    # Filter collision-free grasps
    InspireHandR_ggarray_post = InspireHandR_ggarray_post[empty_mask]
    two_fingers_ggarray_post = two_fingers_ggarray_post[empty_mask]
    
    if len(InspireHandR_ggarray_post) == 0:
        print('No Grasp detected after collision detection!')
        return
    
    # Sort post-collision grasps
    index_score_post = np.argsort(InspireHandR_ggarray_post.scores)[::-1][:80]
    InspireHandR_ggarray_post = InspireHandR_ggarray_post[index_score_post]
    two_fingers_ggarray_post = two_fingers_ggarray_post[index_score_post]
    
    # Visualize results
    print("Visualizing pre-collision grasps...")
    # index_score_post = np.argsort(InspireHandR_ggarray.scores)[::-1][:20]
    # InspireHandR_ggarray = InspireHandR_ggarray[index_score_post]
    # two_fingers_ggarray = two_fingers_ggarray[index_score_post]
    visualize_grasp_proposals(cloud, two_fingers_ggarray, InspireHandR_ggarray, "Pre-Collision Grasp Proposals")
    
    print("Visualizing post-collision grasps...")
    visualize_grasp_proposals(cloud, two_fingers_ggarray_post, InspireHandR_ggarray_post, "Post-Collision Grasp Proposals")
    
    # print("Visualizing comparison...")
    # visualize_grasps(cloud, two_fingers_ggarray, two_fingers_ggarray_post, 
    #                 InspireHandR_ggarray, InspireHandR_ggarray_post)

if __name__ == '__main__':
    t0 = time.time()
    process_and_visualize()
    tn = time.time()
    print(f'Total processing time:{tn - t0}')
