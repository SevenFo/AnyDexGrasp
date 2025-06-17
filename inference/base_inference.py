# inference/base_inference.py
import abc
import torch
import numpy as np
from collections import OrderedDict

class BaseInference(abc.ABC):
    def __init__(self, gripper_config, graspnet_model, multifinger_models):
        self.cfg = gripper_config
        self.graspnet_model = graspnet_model
        self.multifinger_models = multifinger_models
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def get_graspgroup_features(self, grasp_features_array):
        """解析GraspNet特征为字典格式，用于多指手模型输入"""
        features = {
            'grasp_angles': (grasp_features_array[:, -3] + 0.1).astype(int),
            'grasp_depths': (grasp_features_array[:, -2] * 100 + 0.1).astype(int),
            'stage3_grasp_scores': grasp_features_array[:, :240],
            'grasp_preds_features': grasp_features_array[:, 240:240 + 480],
            'if_flip': grasp_features_array[:, -1]
        }
        
        # This rotation logic is common across all scripts.
        grasp_preds_features_rot = np.zeros_like(features['grasp_preds_features'])
        new_type = features['grasp_angles']
        for idx, if_flip in enumerate(features['if_flip']):
            current_preds = features['grasp_preds_features'][idx].copy()
            if if_flip:
                current_preds[:120], current_preds[120:240] = current_preds[120:240], current_preds[:120]
                current_preds[240:360], current_preds[360:480] = current_preds[360:480], current_preds[240:360]
            
            angle_offset = new_type[idx] * 5
            grasp_preds_features_rot[idx, :240] = np.roll(current_preds[:240], angle_offset)
            grasp_preds_features_rot[idx, 240:] = np.roll(current_preds[240:], angle_offset)

        features['grasp_preds_features'] = grasp_preds_features_rot
        return features

    @abc.abstractmethod
    def predict_multifinger_grasps(self, ggarray, grasp_features):
        """
        核心方法：使用第二阶段模型预测多指抓取。
        每个子类必须实现此方法。
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def process_and_filter_grasps(self, multi_finger_gg, two_finger_gg, grasp_features, **kwargs):
        """
        核心方法：对多指抓取进行后处理和筛选。
        """
        raise NotImplementedError