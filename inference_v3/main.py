import os
os.environ['DISPLAY'] = '109.105.4.86:0.0'
import copy
import time
import numpy as np
import torch
import random
import open3d as o3d
import MinkowskiEngine as ME
from tqdm import tqdm
from adg_utils.np_utils import transform_point_cloud
from adg_utils.collision_detector import ModelFreeCollisionDetectorMultifinger
from ur_toolbox.robot.Inspire.InspireHandR_grasp import InspireHandRGraspGroup
from graspnetAPI import GraspGroup

from inference_v3.config import get_config
from inference_v3.utils.net_utils import get_net, parse_preds
from inference_v3.utils.data_utils import load_meshes_pointcloud, get_inspire_model, load_point_cloud_from_images, load_point_cloud_from_file
from inference_v3.utils.grasp_utils import flip_ggarray, get_graspgroup_features, augment_data, select_grasp_type, get_inspire_depth_type
from inference_v3.utils.visualization_utils import visualize_grasps, visualize_grasp_proposals

cfgs = get_config()

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)

def get_grasp(net, points, cloud, augment_mat=np.eye(4), flip=False, voxel_size=0.005):
    points = transform_point_cloud(points, augment_mat).astype(np.float32)
    points = torch.from_numpy(points)
    coords = np.ascontiguousarray(points / voxel_size, dtype=int)
    _, idxs = ME.utils.sparse_quantize(coords, return_index=True)
    coords = coords[idxs]
    points = points[idxs]
    coords_batch, points_batch = ME.utils.sparse_collate([coords], [points])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sinput = ME.SparseTensor(points_batch, coords_batch, device=device)

    end_points = {'sinput': sinput, 'point_clouds': [sinput.F]}
    with torch.no_grad():
        end_points = net(end_points)
        preds, grasp_features = parse_preds(end_points, cfgs.MAX_GRASP_WIDTH, cfgs.MIN_GRASP_WIDTH, use_v2=cfgs.use_graspnet_v2)
        if len(preds) == 0:
            print('No grasp detected')
            return None, cloud, points.cuda(), None, None
        else:
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
    preds[:, 3:12] = pose_rotation.view((-1, 9))

    mask = (preds[:,9] > 0.92) & (preds[:,1] < cfgs.MAX_GRASP_WIDTH) & (preds[:,1] > cfgs.MIN_GRASP_WIDTH)
    workspace_mask = (preds[:,12] > -0.25) & (preds[:,12] < 0.25) & (preds[:,13] > -0.205) & (preds[:,13] < 0.03) | (torch.ones_like(preds[:,13],dtype=torch.bool, device=preds.device))
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

def get_ggarray_features(net, points, cloud):
    augment_mat1 = np.eye(4)
    augment_mats = []
    
    for i in range(cfgs.POINTCLOUD_AUGMENT_NUM):
        if i % 2 == 0:
            augment_mat = augment_data()
        else:
            augment_mat = augment_data(flip=True)
        augment_mats.append(augment_mat)

    ggarray, cloud, points_down, grasp_features, sinput = get_grasp(
        net, points, cloud, augment_mat=augment_mat1
    )
    
    colored_cloud = cloud
    
    for i in range(cfgs.POINTCLOUD_AUGMENT_NUM):
        if i % 2 == 0:
            ggarray2, _, _, grasp_features2, sinput2 = get_grasp(
                net, points, cloud, augment_mat=augment_mats[i]
            )
        else:
            ggarray2, _, _, grasp_features2, sinput2 = get_grasp(
                net, points, cloud, augment_mat=augment_mats[i], flip=True
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
    
    if colored_cloud is None:
        colored_cloud = o3d.geometry.PointCloud()
    
    return ggarray, colored_cloud, points_down, grasp_features, sinput

def process_and_visualize():
    net = get_net(cfgs.checkpoint_path, use_v2=cfgs.use_graspnet_v2, half_views=cfgs.half_views)
    inspire_models = get_inspire_model(cfgs.inspire_model_path)
    meshes_pcls = load_meshes_pointcloud(cfgs.inspire_mesh_json_path, cfgs.INSPIREHANDR_VOXElGRID)
    
    pcd = None
    depths = None
    color_image = None
    if cfgs.point_cloud and os.path.exists(cfgs.point_cloud):
        points, cloud = load_point_cloud_from_file(cfgs.point_cloud)
    else:
        points, cloud = load_point_cloud_from_images(cfgs.depth_image, cfgs.color_image)
    
    t1 = time.time()
    ggarray, cloud, points_down, grasp_features, sinput = get_ggarray_features(net, points, cloud)
    t3 = time.time()
    print(f'Net Time:{t3 - t1}')
    
    if ggarray is None:
        print('No grasp detected')
        return
    
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
        get_inspire_depth_type(inspire_models, grasp_features_dic, ggarray, grasp_features=grasp_features, 
                              num_inspire_depth=cfgs.NUM_OF_INSPIRE_DEPTH, num_inspire_type=cfgs.NUM_OF_INSPIRE_TYPE)
    
    if not cfgs.RANDOM_GRASP:
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
    
    two_fingers_ggarray = GraspGroup(ggarray)
    two_fingers_ggarray_object_ids_source = ggarray[:, 16]
    two_fingers_ggarray_source = copy.deepcopy(two_fingers_ggarray)
    
    InspireHandR_ggarray = InspireHandRGraspGroup() 
    InspireHandR_ggarray.set_grasp_min_width(cfgs.MIN_GRASP_WIDTH)
    InspireHandR_ggarray.from_graspgroup(two_fingers_ggarray, inspire_type, cfgs.inspire_mesh_json_path)
    InspireHandR_ggarray.object_ids = two_fingers_ggarray_object_ids_source
    InspireHandR_ggarray.scores = scores
    InspireHandR_ggarray.depths = InspireHandR_ggarray.depths + inspire_depth + cfgs.INSPIREHANDR_DEFAULT_DEPTH
    InspireHandR_ggarray_source = copy.deepcopy(InspireHandR_ggarray)
    two_fingers_ggarray_object_ids = two_fingers_ggarray_object_ids_source
    
    if len(InspireHandR_ggarray) == 0:
        print('No grasp detected after filter')
        return
    
    index_type = select_grasp_type(InspireHandR_ggarray, cfgs.NUM_OF_INSPIRE_TYPE)
    InspireHandR_ggarray = InspireHandR_ggarray[index_type]
    two_fingers_ggarray = two_fingers_ggarray[index_type]
    grasp_features = grasp_features[index_type]
    two_fingers_ggarray_object_ids = two_fingers_ggarray_object_ids[index_type]
    
    index_score = np.argsort(InspireHandR_ggarray.scores)[::-1]
    InspireHandR_ggarray = InspireHandR_ggarray[index_score]
    two_fingers_ggarray = two_fingers_ggarray[index_score]
    grasp_features = grasp_features[index_score]
    two_fingers_ggarray_object_ids = two_fingers_ggarray_object_ids[index_score]
    
    approach_distance = 0.06
    mfcdetector = ModelFreeCollisionDetectorMultifinger(points_down.cpu().numpy(), voxel_size=0.001)
    InspireHandR_ggarray_post, two_fingers_ggarray_post, empty_mask, min_width_index = mfcdetector.detect(
        InspireHandR_ggarray, two_fingers_ggarray,
        cfgs.inspire_mesh_json_path, meshes_pcls, min_grasp_width=cfgs.MIN_GRASP_WIDTH,
        VoxelGrid=cfgs.INSPIREHANDR_VOXElGRID, DEBUG=False, approach_dist=approach_distance,
        collision_thresh=0, adjust_gripper_centers=True
    )
    
    InspireHandR_ggarray_post = InspireHandR_ggarray_post[empty_mask]
    two_fingers_ggarray_post = two_fingers_ggarray_post[empty_mask]
    
    if len(InspireHandR_ggarray_post) == 0:
        print('No Grasp detected after collision detection!')
        return
    
    index_score_post = np.argsort(InspireHandR_ggarray_post.scores)[::-1][:80]
    InspireHandR_ggarray_post = InspireHandR_ggarray_post[index_score_post]
    two_fingers_ggarray_post = two_fingers_ggarray_post[index_score_post]
    
    print("Visualizing pre-collision grasps...")
    visualize_grasp_proposals(cloud, two_fingers_ggarray, InspireHandR_ggarray, cfgs, "Pre-Collision Grasp Proposals")
    
    print("Visualizing post-collision grasps...")
    visualize_grasp_proposals(cloud, two_fingers_ggarray_post, InspireHandR_ggarray_post, cfgs, "Post-Collision Grasp Proposals")

if __name__ == '__main__':
    t0 = time.time()
    process_and_visualize()
    tn = time.time()
    print(f'Total processing time:{tn - t0}')
