import copy
import numpy as np
import torch
from adg_utils.pt_utils import batch_viewpoint_params_to_matrix
from graspnetAPI import GraspGroup
from collections import OrderedDict
from ur_toolbox.robot.Inspire.InspireHandR_grasp import InspireHandRGraspGroup
from adg_utils.collision_detector import ModelFreeCollisionDetectorMultifinger

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
    grasp_features = {}
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
    grasp_features = {}
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

    grasp_preds_features_rot = np.zeros(grasp_features['grasp_preds_features'].shape, dtype=np.float32)
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

def augment_data(flip=False):
    flip_mat = np.identity(4)
    if flip:
        flip_mat = np.array([[-1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]])

    rot_angle = (np.random.random() * np.pi / 3) - np.pi / 6
    c, s = np.cos(rot_angle), np.sin(rot_angle)
    rot_mat = np.array([[c, -s, 0, 0],
                        [s, c, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])

    offset_x = np.random.random() * 0.1 - 0.05
    offset_y = np.random.random() * 0.1 - 0.05
    trans_mat = np.array([[1, 0, 0, offset_x],
                          [0, 1, 0, offset_y],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]])

    aug_mat = np.dot(trans_mat, np.dot(rot_mat, flip_mat).astype(np.float32)).astype(np.float32)
    return aug_mat

def get_inspire_depth_type(inspire_models, grasp_features_dic, ggarray, grasp_features, num_inspire_depth, num_inspire_type):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for k, v in grasp_features_dic.items():
        if k in ['point_id', 'sinput']:
            continue
        grasp_features_dic[k] = torch.tensor(copy.deepcopy(v), device=device)

    inspire_depth_type_scores = []
    for model_type, inspire_model in inspire_models.items():
        if model_type == '240':
            model_input = torch.cat([grasp_features_dic["stage3_grasp_scores"]], dim=1)
        elif model_type == '480':
            model_input = torch.cat([grasp_features_dic["grasp_preds_features"]], dim=1)
        
        inspire_model_sorted = OrderedDict(sorted(inspire_model.items(), key=lambda t: int(t[0])))
        for model_class, sub_inspire_model in inspire_model_sorted.items():
            with torch.no_grad():
                grasp_pred, _ = sub_inspire_model(model_input)
                grasp_pred = grasp_pred.view(grasp_pred.shape[0], 5*num_inspire_depth)
            two_fingers_depth = grasp_features_dic['grasp_depths']
            base = torch.tensor(np.array([[i for i in range(num_inspire_depth)]
                                  for _ in range(grasp_pred.size()[0])]), device=device)
            select_index = (two_fingers_depth).view(-1, 1) * num_inspire_depth + base
            inspire_depth_type_scores.append(grasp_pred.gather(1, select_index))
            
    inspire_depth_type_scores = torch.cat(inspire_depth_type_scores, axis=1).view(-1)
    scores, index = inspire_depth_type_scores.topk(min(3000, inspire_depth_type_scores.size()[0]))
    pose_index = (index / (num_inspire_depth * num_inspire_type)).long()
    ggarray = torch.tensor(copy.deepcopy(ggarray), device=device)[pose_index]
    grasp_features = torch.tensor(copy.deepcopy(grasp_features), device=device)[pose_index]
    inspire_depth = ((index % (num_inspire_depth * num_inspire_type)) % num_inspire_depth).int()
    inspire_type = ((index % (num_inspire_depth * num_inspire_type)) / num_inspire_depth).int()

    inspire_depth[inspire_type==5] = inspire_depth[inspire_type==5] + 2
    inspire_depth = inspire_depth * 0.01

    return inspire_depth.cpu().numpy(), inspire_type.cpu().numpy() + 1, scores.detach().cpu().numpy(), \
                ggarray.cpu().numpy(), grasp_features.cpu().numpy()

def select_grasp_type(inspire_gg, num_inspire_type):
    select_gg_types = [[] for _ in range(1, num_inspire_type+1)]
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
