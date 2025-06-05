import copy
import json
import os
import sys
import pdb
import time
import datetime
import random
import argparse
import torch
import numpy as np
import cv2
import open3d as o3d

from graspnetAPI import GraspGroup

from multiprocessing import shared_memory
from collections import OrderedDict
import matplotlib.pyplot as plt
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from ur_toolbox.robot import UR_Camera_Gripper
from ur_toolbox.robot.Allegro.Allegro_grasp import AllegroGraspGroup, grasp_types
from utils.np_utils import transform_point_cloud
from utils.pt_utils import batch_viewpoint_params_to_matrix
from utils.collision_detector import ModelFreeCollisionDetectorMultifinger
import queue
from itertools import count
from threading import Thread
from queue import Queue
import multiprocessing as mp
import MinkowskiEngine as ME
from models.minkowski_graspnet_single_point import MinkowskiGraspNet, MinkowskiGraspNetMultifingerType1Inference

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
parser.add_argument('--allegro_model_path', default='logs/model/allegro_model/allegro_obj140', help='allegro model checkpoint path')
parser.add_argument('--Allegro_mesh_json_path', default='generate_mesh_and_pointcloud/allegro_urdf', help='Allegro meshes and json path')
parser.add_argument('--use_graspnet_v2', action='store_true', help='Whether to use graspnet v2 format')
parser.add_argument('--half_views', action='store_true', help='Use only half views in network.')
parser.add_argument('--depth_image', default=None, help='Path to pre-recorded depth image (PNG)')
parser.add_argument('--color_image', default=None, help='Path to pre-recorded color image (PNG)')
cfgs = parser.parse_args()

MAX_GRASP_WIDTH = 0.11
MIN_GRASP_WIDTH = 0.04
BATCH_SIZE = 1
DEBUG = True
NUM_OF_ALLEGRO_DEPTH = 4
NUM_OF_ALLEGRO_TYPE = 10
Allegro_VOXElGRID = 0.003
POINTCLOUD_AUGMENT_NUM = 10
DEFAULT_DEPTH = 0.00
RANDOM = False

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
            ggarray_rotations[ids, :3, 1:3] = -ggarray_rotations[ids, :3, 1:3]
            if_flip[ids] = True
    return ggarray, if_flip

def flip_z_ggarray(ggarray, allegro_types):
    ggarray_rotations = ggarray[:, 4:13].reshape((-1, 3, 3))
    tcp_z_axis_on_base_frame = ggarray_rotations[:, 0, 2]
    tcp_x_axis_on_base_frame = ggarray_rotations[:, 1, 1]
    if_flip = [False for _ in range(len(ggarray))]
    for ids, y_z in enumerate(tcp_z_axis_on_base_frame):
        if y_z < 0 and allegro_types[ids] in [9]:
            ggarray_rotations[ids, :3, 1:3] = -ggarray_rotations[ids, :3, 1:3]

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

def get_robot(robot_ip="192.168.2.102", use_rt=False, robot_debug=True, gripper_type='robotiq',
              gripper_port='192.168.1.29', global_cam=False):
    robot = UR_Camera_Gripper(robot_ip, use_rt, camera=None, robot_debug=robot_debug, gripper_type=gripper_type,
                              gripper_port=gripper_port, global_cam=global_cam)
    robot.set_tcp((0, 0, 0.0, 0, 0, 0))
    robot.set_payload(2.7, (0, 0, 0.12))
    return robot

def get_depth(existing_shm_depth):
    time.sleep(0.1)
    depths_1 = np.copy(np.ndarray((720, 1280), dtype=np.uint16, buffer=existing_shm_depth.buf))
    return depths_1

def save_grasp_information(two_fingers_ggarray, two_fingers_ggarray_object_ids, Allegro_ggarray, 
                            two_fingers_ggarray_source, Allegro_ggarray_source, two_fingers_ggarray_object_ids_source,
                            two_fingers_grasp_used, Allegro_grasp_used, grasp_features_used, 
                            mat_pose, colors_saved, depths_saved, grasp_features, two_fingers_source_grasp_features,
                            before_collision, after_collision, Allegro_grasp_type):
    save_path = cfgs.save_information_path

    timeStamp = datetime.datetime.now().timestamp()
    timeArray = time.localtime(timeStamp)
    otherStyleTime = time.strftime("%Y-%m-%d_%H-%M-%S", timeArray)
    save_path = os.path.join(save_path, grasp_types[str(int(Allegro_grasp_type))]['name'], otherStyleTime)

    information = OrderedDict()
    two_fingers_ggarray_proposals = []
    Allegro_ggarray_proposals = []
    for idx, tfg in enumerate(two_fingers_ggarray):
        two_fingers_ggarray_proposals.append(
            [float(tfg.score), float(tfg.width), float(tfg.height), float(tfg.depth)] +
            np.array(tfg.rotation_matrix).reshape((-1)).tolist() + list(tfg.translation.tolist()) +
            [float(two_fingers_ggarray_object_ids[idx])])
        Allegro_ggarray_proposals.append(list(Allegro_ggarray[idx].get_array_grasp()))


    two_fingers_ggarray_source_saved = []
    Allegro_ggarray_source_saved = []
    for idx, tfg in enumerate(two_fingers_ggarray_source):
        two_fingers_ggarray_source_saved.append(
            [float(tfg.score), float(tfg.width), float(tfg.height), float(tfg.depth)] +
            np.array(tfg.rotation_matrix).reshape((-1)).tolist() + list(tfg.translation.tolist()) +
            [float(two_fingers_ggarray_object_ids_source[idx])])
        Allegro_ggarray_source_saved.append(list(Allegro_ggarray_source[idx].get_array_grasp()))

    tfg = two_fingers_grasp_used
    two_fingers_array = [float(tfg.score), float(tfg.width), float(tfg.height), float(tfg.depth)] + \
                        np.array(tfg.rotation_matrix).reshape((-1)).tolist() + \
                        list(tfg.translation.reshape(-1).tolist()) + \
                        [float(Allegro_grasp_used.object_id)]
    grasp_features_used = get_grasp_features(grasp_features_used)
    restart = False
    print('Is the grasping successful? press 1 successfully, press 2 failed, restart grasping and press 3, exit press 4\n')
    if_success = input('The result is: ')
    while True:
        if if_success == '1':
            information['result'] = True
            break
        elif if_success == '2':
            information['result'] = False
            break
        elif if_success == '3':
                restart = True
                break
        elif if_success == '4':
            exit()
        else:
            if_success = input('Re-enter the result: ')
    if restart:
        return
    information['two_fingers_pose'] = list(two_fingers_array)
    information['Allegro_pose'] = list(Allegro_grasp_used.get_array_grasp())
    information['two_fingers_pose_angle_type'] = grasp_features_used['grasp_angles']
    information['two_fingers_pose_depth_type'] = grasp_features_used['grasp_depths']

    information['two_fingers_pose_AD'] = grasp_features_used['stage3_grasp_scores']
    information['two_fingers_pose_features_grasp_preds'] = grasp_features_used['grasp_preds_features']
    information['two_fingers_pose_features'] = grasp_features_used['stage3_grasp_features']
    information['two_fingers_pose_features_before_generator'] = grasp_features_used['before_generator']
    information['point_features'] = grasp_features_used['point_features']
    
    information['point_id'] = grasp_features_used['point_id']
    information['if_flip'] = grasp_features_used['if_flip']
    information['Allegro_pose_finger_type'] = int(Allegro_grasp_used.grasp_type + 0.1)
    information['Allegro_pose_depth_type'] = int(Allegro_grasp_used.depth*100 + 0.1) - grasp_features_used['grasp_depths']
    information['base_2_tcp1'] = np.array(mat_pose[0]).tolist()
    information['base_2_tcp1_backup'] = np.array(mat_pose[1]).tolist()
    information['tcp_2_gripper'] = np.array(mat_pose[2]).tolist()
    information['base_2_TwoFingersGripper_pose'] = np.array(mat_pose[3]).tolist()
    information['tcp_2_camera'] = np.array(mat_pose[4]).tolist()
    information['base_2_tcp_ready'] = np.array(mat_pose[5]).tolist()

    information['camera_internal'] = [[631.119, 363.884], [919.835, 919.61]]

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    color_path = os.path.join(save_path, 'color.png')
    depth_path = os.path.join(save_path, 'depth.png')
    from PIL import Image
    cv2.imwrite(color_path, (cv2.cvtColor(colors_saved, cv2.COLOR_RGB2BGR) * 255.0).astype(np.float32))
    cv2.imwrite(depth_path, depths_saved)
    json_path = os.path.join(save_path, 'information.json')

    json_file = json.dumps(information, indent=4)
    with open(json_path, 'w') as handle:
        handle.write(json_file)
    print('Saved successfully')

def get_grasp(net, depths, existing_shm_color, augment_mat=np.eye(4), flip=False, voxel_size=0.005):
    fx, fy = 913.232, 912.452
    cx, cy = 628.847, 350.771
    s = 1000.0

    xmap, ymap = np.arange(depths.shape[1]), np.arange(depths.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)

    points_z = depths / s
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z

    mask = (points_z > 0.2) & (points_z < 0.65) # 23.4.26
    points = np.stack([points_x, points_y, points_z], axis=-1)
    points = points[mask].astype(np.float32)

    # 修改颜色处理部分
    if DEBUG and existing_shm_color is not None:
        colors = np.copy(np.ndarray((720, 1280, 3), dtype=np.float32, buffer=existing_shm_color.buf))
        colors = colors[mask].astype(np.float32)
    else:
        # 创建虚拟颜色数据（白色）
        colors = np.ones((points.shape[0], 3), dtype=np.float32)
    
    cloud = None
    if DEBUG:
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
            return None, cloud, points.cuda(), None, [sinput]
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

    mask = (preds[:,9] > 0.93) & (preds[:,1] < MAX_GRASP_WIDTH) & (preds[:,1] > MIN_GRASP_WIDTH)
    workspace_mask = (preds[:,12] > -0.2) & (preds[:,12] < 0.2) & (preds[:,13] > -0.20) & (preds[:,13] < 0.07) 

    preds = preds[workspace_mask & mask]
    grasp_features = grasp_features[0][workspace_mask & mask]
    if len(preds) == 0:
        print('No grasp detected after masking')
        return None, cloud, points.cuda(), None, [sinput]

    points = points.cuda()
    heights = 0.03 * torch.ones([preds.shape[0], 1]).cuda()
    object_ids = -1 * torch.ones([preds.shape[0], 1]).cuda()
    ggarray = torch.cat([preds[:, 0:2], heights, preds[:, 2:15], preds[:, 15:16], object_ids], axis=-1)

    return ggarray, cloud, points, grasp_features, [sinput]

def load_meshes_pointcloud(path):
    meshes_pcls = dict()
    meshes_pcl_path = os.path.join(path, 'meshes/source_pointclouds/voxel_size_' + str(int(Allegro_VOXElGRID * 1000)))
    for type in os.listdir(meshes_pcl_path):
        type_path = os.path.join(meshes_pcl_path, type)
        for name in os.listdir(type_path):
            width = name[:-4]
            name_path = os.path.join(type_path, name)
            meshes_pcl = o3d.io.read_point_cloud(name_path)
            meshes_pcls[type + '_' + width] = meshes_pcl
    return meshes_pcls

def create_tale_pointcloud(width=0.5, height=0.005, depth=0.4, dx=-0.25, dy=-0.15, dz=-0.55, grid_size=0.005):
    xmap = np.linspace(0, width, int(width/grid_size))
    ymap = np.linspace(0, depth, int(depth/grid_size))
    zmap = np.linspace(0, height, int(height/grid_size))
    xmap, ymap, zmap = np.meshgrid(xmap, ymap, zmap, indexing='xy')
    xmap += dx
    ymap += dy
    zmap += dz
    points = np.stack([xmap, -ymap, -zmap], axis=-1)
    points = points.reshape([-1, 3])
    cloud = o3d.geometry.PointCloud()
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.01)
    cloud.points = o3d.utility.Vector3dVector(points)
    return points

def get_allegro_model(allegro_models_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    allegro_models = dict()
    for model_type in os.listdir(allegro_models_path):
        allegro_model_path = os.path.join(allegro_models_path, model_type)
        for model_class in os.listdir(allegro_model_path):
            allegro_models[model_type] = dict()
            model_classs_type = os.path.join(allegro_model_path, model_class)
            for model in os.listdir(model_classs_type):
                allegro_model_type_path = os.path.join(model_classs_type, model)
                allegro_model = MinkowskiGraspNetMultifingerType1Inference(input_num=int(model_type))
                allegro_net = torch.load(allegro_model_type_path)
                allegro_model.load_state_dict(allegro_net.state_dict())
                allegro_model.to(device)
                allegro_model.eval()
                if model_class not in allegro_models.keys():
                    allegro_models[model_type][model_class] = [allegro_model]
                else:
                    allegro_models[model_type][model_class].append(allegro_model)
    return allegro_models

def get_allegro_depth_type(allegro_models, grasp_features_dic, ggarray, grasp_features):
    if RANDOM:
        allegro_depth = (np.random.randint(0, NUM_OF_ALLEGRO_DEPTH, (len(ggarray),))) * 0.01
        allegro_type = (np.random.randint(0, NUM_OF_ALLEGRO_TYPE, (len(ggarray),)))
        scores = ggarray[:, 0] 
        return allegro_depth, allegro_type, scores, ggarray, grasp_features
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for k, v in grasp_features_dic.items():
        if k == 'point_id' or k == 'sinput':
            continue
        grasp_features_dic[k] = torch.tensor(copy.deepcopy(v), device=device)
    allegro_depth_type_scores = []
    for model_type, allegro_model in allegro_models.items():
        if model_type == '240':
            continue
            model_input = torch.cat([grasp_features_dic["stage3_grasp_scores"]], dim=1)
        elif model_type == '480':
            print('use final model: ', model_type)
            model_input = torch.cat([grasp_features_dic["grasp_preds_features"]], dim=1)

        allegro_model_sorted = OrderedDict(sorted(allegro_model.items(), key = lambda t : t[0]))
        for model_class, sub_allegro_model in allegro_model_sorted.items():
            allegro_depth_types = torch.tensor(0, device=device)
            for sub_sub_allegro_model in sub_allegro_model:
                with torch.no_grad():
                    grasp_pred, _ = sub_sub_allegro_model(model_input) # (B, 1， NUM_OF_TWO_FINGER_DEPTH*NUM_OF_ALLEGRO_DEPTH)
                    grasp_pred = grasp_pred.view(grasp_pred.shape[0], 5*NUM_OF_ALLEGRO_DEPTH)
                two_fingers_depth = grasp_features_dic['grasp_depths'] # (B, )
                base = torch.tensor(np.array([[i for i in range(NUM_OF_ALLEGRO_DEPTH)]
                                    for _ in range(grasp_pred.size()[0])]), device=device) # (B, NUM_OF_ALLEGRO_DEPTH)
                select_index = (two_fingers_depth).view(-1, 1) * NUM_OF_ALLEGRO_DEPTH + base
                grasp_pred = grasp_pred.gather(1, select_index)
                allegro_depth_types = allegro_depth_types + grasp_pred
            allegro_depth_types = (allegro_depth_types / len(sub_allegro_model)).squeeze(1)
            allegro_depth_type_scores.append(allegro_depth_types)
                
    allegro_depth_type_scores = torch.cat(allegro_depth_type_scores, axis=1).view(-1) # (B, NUM_OF_ALLEGRO_DEPTH*NUM_OF_ALLEGRO_TYPE)
    
    scores, index = allegro_depth_type_scores.topk(min(3000, allegro_depth_type_scores.size()[0]))
    pose_index = (index / (NUM_OF_ALLEGRO_DEPTH * NUM_OF_ALLEGRO_TYPE)).long()
    ggarray = torch.tensor(copy.deepcopy(ggarray), device=device)[pose_index]
    grasp_features = torch.tensor(copy.deepcopy(grasp_features), device=device)[pose_index]
    allegro_depth = ((index % (NUM_OF_ALLEGRO_DEPTH * NUM_OF_ALLEGRO_TYPE)) % NUM_OF_ALLEGRO_DEPTH).int() * 0.01
    allegro_type = ((index % (NUM_OF_ALLEGRO_DEPTH * NUM_OF_ALLEGRO_TYPE)) / NUM_OF_ALLEGRO_DEPTH).int()

    return allegro_depth.cpu().numpy(), allegro_type.cpu().numpy() + 1, scores.detach().cpu().numpy(), \
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
    # offset_z = np.random.random() * 0.3 - 0.1  # -0.1 ~ 0.2
    trans_mat = np.array([[1, 0, 0, offset_x],
                          [0, 1, 0, offset_y],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]])

    aug_mat = np.dot(trans_mat, np.dot(rot_mat, flip_mat).astype(np.float32)).astype(np.float32)
    return aug_mat

def get_ggarray_features(existing_shm_depth, existing_shm_color, net):
    depths = get_depth(existing_shm_depth)
    augment_mat1 = np.eye(4)
    augment_mats = []
    for i in range(POINTCLOUD_AUGMENT_NUM):
        if i % 2 == 0:
            augment_mat = augment_data()
        else:
            augment_mat = augment_data(flip=True)
        augment_mats.append(augment_mat)

    ggarray, cloud, points_down, grasp_features, sinput = get_grasp(net, depths, existing_shm_color, augment_mat=augment_mat1)
    for i in range(POINTCLOUD_AUGMENT_NUM):
        if i % 2 == 0:
            ggarray2, _, _, grasp_features2, sinput2 = get_grasp(net, depths, existing_shm_color, augment_mat=augment_mats[i])
        else:
            ggarray2, _, _, grasp_features2, sinput2 = get_grasp(net, depths, existing_shm_color, augment_mat=augment_mats[i], flip=True)
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
    return ggarray, cloud, points_down, grasp_features, sinput

def select_grasp_type(allegro_gg):
    select_gg_types = [[] for _ in range(NUM_OF_ALLEGRO_TYPE)]
    for idx, score in enumerate(allegro_gg.scores):
        grasp_type = allegro_gg.grasp_types[idx]
        max_num = 50
        if grasp_type in [4, 6]:
            max_num = 5

        if len(select_gg_types[int(grasp_type)-1]) < max_num:
            select_gg_types[int(grasp_type)-1].append(idx)
    print('select grasp type shape: ', [len(i) for i in select_gg_types])
    gg_type_id = []
    for gg_type in select_gg_types:
        gg_type_id += gg_type
    return gg_type_id

def visualize_grasp(two_fingers_grasp, allegro_grasp, cloud, points_down):
    """可视化抓取姿势和点云"""
    allegro_pose = allegro_grasp.load_mesh(cfgs.Allegro_mesh_json_path, two_fingers_grasp)
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
    sphere = o3d.geometry.TriangleMesh.create_sphere(0.002, 20).translate([0, 0, 0.490])
    
    # 创建点云可视化
    scene_cloud = o3d.geometry.PointCloud()
    scene_cloud.points = o3d.utility.Vector3dVector(points_down.cpu().numpy())
    scene_cloud.paint_uniform_color([0.5, 0.5, 0.5])
    
    # 创建两指抓取可视化
    two_finger_vis = two_fingers_grasp.to_open3d_geometry()
    two_finger_vis.paint_uniform_color([0, 0, 1])  # 蓝色
    
    # Allegro手抓取可视化
    allegro_pose.paint_uniform_color([1, 0, 0])  # 红色
    
    # 显示所有元素
    o3d.visualization.draw_geometries([
        scene_cloud, 
        frame, 
        sphere,
        two_finger_vis,
        allegro_pose
    ])

def simulate_depth_image():
    """生成模拟深度图"""
    depth_img = np.zeros((720, 1280), dtype=np.uint16)*1000
    # 在图像中心添加一个立方体
    center_x, center_y = 640, 360
    size = 100
    depth_img[center_y-int(size//5):center_y+int(size//5), center_x-size*5:center_x+size*5] = 500  # 500mm深度
    
    # 添加随机噪声
    noise = np.random.randint(-20, 20, (720, 1280))
    depth_img = depth_img + noise
    depth_img = np.clip(depth_img, 0, 1100).astype(np.uint16)
    return depth_img

def load_depth_image(file_path):
    """从文件加载深度图"""
    return cv2.imread(file_path, cv2.IMREAD_ANYDEPTH)

def visualize_multiple_grasps(two_fingers_ggarray, Allegro_ggarray, cloud, points_down, max_visualize=20):
    """可视化多个抓取姿势和点云"""
    # 创建点云可视化
    scene_cloud = o3d.geometry.PointCloud()
    scene_cloud.points = o3d.utility.Vector3dVector(points_down)
    scene_cloud.paint_uniform_color([0.5, 0.5, 0.5])
    
    # 创建坐标系
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
    
    # 创建抓取姿势可视化
    geometries = [scene_cloud, frame]
    
    # 限制可视化数量
    num_visualize = min(max_visualize, len(two_fingers_ggarray))
    pre_collision_indices = np.argsort(Allegro_ggarray.scores)[::-1][:num_visualize]
    pre_collision_geometries = []

    # 为不同抓取类型创建颜色映射
    type_colors = {
        1: [1, 0, 0],   # 红色 - power
        2: [0, 1, 0],   # 绿色 - pinch
        3: [0, 0, 1],   # 蓝色 - lateral
        4: [1, 1, 0],   # 黄色 - precision
        5: [1, 0, 1],   # 紫色 - tripod
        6: [0, 1, 1],   # 青色 - hook
        7: [0.5, 0.5, 0], # 橄榄色 - spherical
        8: [0.5, 0, 0.5], # 紫红色 - cylindrical
        9: [0, 0.5, 0.5], # 蓝绿色 - palmar
        10: [0.3, 0.3, 0.3] # 灰色 - other
    }
    # 添加点云
    scene_cloud = o3d.geometry.PointCloud()
    scene_cloud.points = o3d.utility.Vector3dVector(points_down)
    scene_cloud.paint_uniform_color([0.5, 0.5, 0.5])
    pre_collision_geometries.append(scene_cloud)

    # 添加坐标系
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
    pre_collision_geometries.append(frame)
    print(f"Visualizing {num_visualize} candidate grasps before collision detection")
    
    print(pre_collision_indices)
    for i, idx in enumerate(pre_collision_indices):
        # 两指抓取可视化
        idx = idx.item()
        two_finger_vis = two_fingers_ggarray[idx].to_open3d_geometry()
        two_finger_vis.paint_uniform_color([0, 0, 1])  # 蓝色
        
        # Allegro手抓取可视化
        allegro_pose = Allegro_ggarray[idx].load_mesh(
            cfgs.Allegro_mesh_json_path, 
            two_fingers_ggarray[idx]
        )
        grasp_type = int(Allegro_ggarray[idx].grasp_type)
        color = type_colors.get(grasp_type, [0.3, 0.3, 0.3])  # 默认为灰色
        allegro_pose.paint_uniform_color(color)
        
        pre_collision_geometries.append(two_finger_vis)
        pre_collision_geometries.append(allegro_pose)
        
        # 打印抓取信息
        print(f"\nGrasp #{i+1} (Pre-Collision):")
        print(f"  Score: {Allegro_ggarray[idx].score:.4f}")
        print(f"  Type: {grasp_types[str(int(Allegro_ggarray[idx].grasp_type))]['name']}")
        print(f"  Position: {Allegro_ggarray[idx].translation}")

            
    # 显示所有元素
    o3d.visualization.draw_geometries(pre_collision_geometries)

def generate_grasp_poses():
    net = get_net(cfgs.checkpoint_path, use_v2=cfgs.use_graspnet_v2)
    allegro_models = get_allegro_model(cfgs.allegro_model_path)
    meshes_pcls = load_meshes_pointcloud(cfgs.Allegro_mesh_json_path)
    
    # 获取深度图
    if cfgs.depth_image:
        depths = load_depth_image(cfgs.depth_image)
    else:
        depths = simulate_depth_image()
    
    # 生成点云
    fx, fy = 913.232, 912.452
    cx, cy = 628.847, 350.771
    s = 1000.0
    
    xmap, ymap = np.arange(depths.shape[1]), np.arange(depths.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)
    
    points_z = depths / s
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z
    
    mask = (points_z > 0.2) & (points_z < 0.65)
    points = np.stack([points_x, points_y, points_z], axis=-1)
    points = points[mask].astype(np.float32)
    
    # 生成点云
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    
    # 获取抓取姿势
    augment_mat1 = np.eye(4)
    ggarray, _, points_down, grasp_features, sinput = get_grasp(
        net, depths, None, augment_mat=augment_mat1
    )
    
    if ggarray is None:
        print("No grasps detected!")
        return
    
    # 处理抓取姿势
    ggarray = ggarray.cpu().numpy()
    grasp_features = grasp_features.cpu().numpy()
    ggarray, if_flip = flip_ggarray(ggarray)
    grasp_features = np.c_[grasp_features, if_flip]
    
    source_index = ggarray[:, 0].argsort()
    ggarray = ggarray[source_index][::-1][:2000]
    grasp_features = grasp_features[source_index][::-1][:2000]
    
    grasp_features_dic = get_graspgroup_features(grasp_features, sinput)
    allegro_depths, allegro_types, scores, ggarray, grasp_features = \
        get_allegro_depth_type(allegro_models, grasp_features_dic, ggarray, grasp_features=grasp_features)
    
    if not RANDOM:
        score_thresh = 0.7
        mask = (scores > score_thresh)
        ggarray = ggarray[mask]
        grasp_features = grasp_features[mask]
        allegro_depths = allegro_depths[mask]
        allegro_types = allegro_types[mask]
        scores = scores[mask]
    
    ggarray, if_flip = flip_z_ggarray(ggarray, allegro_types)
    
    if len(ggarray) == 0:
        print('No grasps meet the criteria')
        return
    
    # 创建抓取组
    two_fingers_ggarray = GraspGroup(ggarray)
    Allegro_ggarray = AllegroGraspGroup() 
    Allegro_ggarray.from_graspgroup(two_fingers_ggarray, allegro_types, cfgs.Allegro_mesh_json_path)
    Allegro_ggarray.scores = scores
    Allegro_ggarray.depths = Allegro_ggarray.depths + allegro_depths + DEFAULT_DEPTH
    
    # 在碰撞检测前添加可视化
    print("\n===== Visualizing candidate grasps before collision detection =====")
    visualize_multiple_grasps(
        two_fingers_ggarray, 
        Allegro_ggarray, 
        cloud, 
        points_down.cpu().numpy(),
        max_visualize=10  # 只可视化前10个候选抓取
    )
    
    # 碰撞检测（调整参数）
    approach_distance = 0.08
    mfcdetector = ModelFreeCollisionDetectorMultifinger(
        points_down.cpu().numpy(), 
        voxel_size=0.001
    )
    
    # 调整碰撞检测参数
    Allegro_ggarray, two_fingers_ggarray, empty_mask, _ = mfcdetector.detect(
        Allegro_ggarray, 
        two_fingers_ggarray,
        cfgs.Allegro_mesh_json_path, 
        meshes_pcls, 
        min_grasp_width=MIN_GRASP_WIDTH,
        VoxelGrid=Allegro_VOXElGRID, 
        DEBUG=False,  # 开启调试模式以获取更多信息
        approach_dist=approach_distance,
        collision_thresh=0.5,  # 降低碰撞阈值（原为1）
        adjust_gripper_centers=True,
    )
    
    Allegro_ggarray = Allegro_ggarray[empty_mask]
    two_fingers_ggarray = two_fingers_ggarray[empty_mask]
    
    if len(Allegro_ggarray) == 0:
        print('No grasps after collision detection')
        return
    
    # 选择最佳抓取
    index_score = np.argsort(Allegro_ggarray.scores)[::-1][0:10]
    Allegro_ggarray = Allegro_ggarray[index_score]
    two_fingers_ggarray = two_fingers_ggarray[index_score]
    
    # 可视化最佳抓取
    best_idx = 0  # 选择分数最高的抓取
    visualize_grasp(
        two_fingers_ggarray[best_idx],
        Allegro_ggarray[best_idx],
        cloud,
        points_down
    )
    
    # 打印抓取信息
    print("\n===== Best Grasp Info =====")
    print(f"Score: {Allegro_ggarray[best_idx].score:.4f}")
    print(f"Type: {grasp_types[str(int(Allegro_ggarray[best_idx].grasp_type))]['name']}")
    print(f"Position: {Allegro_ggarray[best_idx].translation}")
    print(f"Rotation Matrix:\n{Allegro_ggarray[best_idx].rotation_matrix}")
    print(f"Width: {Allegro_ggarray[best_idx].width:.4f}m")
    print(f"Depth: {Allegro_ggarray[best_idx].depth:.4f}m")
    print(f"Angle: {np.rad2deg(Allegro_ggarray[best_idx].angle)}°")

if __name__ == '__main__':
    import os
    os.environ['DISPLAY'] = '109.105.4.86:0.0'
    generate_grasp_poses()
