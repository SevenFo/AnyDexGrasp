# utils/model_loader.py
import os
import torch
import sys
from models.minkowski_graspnet_single_point import MinkowskiGraspNet, MinkowskiGraspNetMultifingerType1Inference
from models import minkowski_graspnet
sys.modules['minkowski_graspnet'] = minkowski_graspnet

def get_graspnet_model(checkpoint_path, use_v2=True, half_views=False):
    """加载GraspNet基础模型"""
    num_depth = 5 if use_v2 else 4
    net = MinkowskiGraspNet(num_depth=num_depth, num_seed=2048, is_training=False, half_views=half_views)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()
    return net

def get_multifinger_model(models_path):
    """加载多指手型预测模型"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gripper_models = {}
    for model_type in os.listdir(models_path):
        model_type_path = os.path.join(models_path, model_type)
        if not os.path.isdir(model_type_path): continue
        
        gripper_models[model_type] = {}
        for model_class in os.listdir(model_type_path):
            model_class_path = os.path.join(model_type_path, model_class)
            if not os.path.isdir(model_class_path): continue

            gripper_models[model_type][model_class] = []
            for model_file in os.listdir(model_class_path):
                if not model_file.endswith('.pth'): continue # 假设模型文件以 .pt 结尾
                
                model_full_path = os.path.join(model_class_path, model_file)
                model = MinkowskiGraspNetMultifingerType1Inference(input_num=int(model_type))
                model_state = torch.load(model_full_path)
                
                # 兼容不同保存方式
                if 'state_dict' in dir(model_state):
                    model.load_state_dict(model_state.state_dict())
                else:
                    model.load_state_dict(model_state)

                model.to(device)
                model.eval()
                gripper_models[model_type][model_class].append(model)
                
    return gripper_models