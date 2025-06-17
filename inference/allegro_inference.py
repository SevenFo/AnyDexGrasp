# inference/allegro_inference.py
import torch
import copy
import numpy as np
from collections import OrderedDict
from graspnetAPI import GraspGroup
from .base_inference import BaseInference

class AllegroInference(BaseInference):
    def flip_ggarray(self, ggarray):
        """Allegro-specific grasp flipping logic."""
        rotations = ggarray[:, 4:13].reshape((-1, 3, 3))
        if_flip = rotations[:, 1, 1] < 0
        rotations[if_flip, :, 1:3] *= -1
        ggarray[:, 4:13] = rotations.reshape((-1, 9))
        return ggarray, if_flip

    def predict_multifinger_grasps(self, ggarray, grasp_features):
        """Predicts multi-finger grasps for the Allegro hand."""
        if self.cfg['random_grasp']:
            depths = np.random.randint(0, self.cfg['num_depth'], len(ggarray)) * 0.01
            types = np.random.randint(0, self.cfg['num_types'], len(ggarray)) + 1
            scores = ggarray[:, 0]
            return depths, types, scores, ggarray, grasp_features

        features_dic = self.get_graspgroup_features(grasp_features)
        
        # Use only the feature required by the model
        model_input_features = torch.from_numpy(features_dic["grasp_preds_features"]).to(self.device)
        
        all_scores = []
        # Sort models by class ('type_1', 'type_2', etc.)
        sorted_models = OrderedDict(sorted(self.multifinger_models['480'].items(), key=lambda t: t[0]))
        
        for model_class, sub_models in sorted_models.items():
            class_preds = torch.tensor(0, device=self.device)
            for model in sub_models:
                with torch.no_grad():
                    pred, _ = model(model_input_features)
                    pred = pred.view(pred.shape[0], 5 * self.cfg['num_depth'])
                class_preds += pred
            
            class_preds /= len(sub_models)
            
            two_fingers_depth_idx = torch.from_numpy(features_dic['grasp_depths']).long().to(self.device)
            base = torch.arange(self.cfg['num_depth'], device=self.device).repeat(len(ggarray), 1)
            select_index = two_fingers_depth_idx.view(-1, 1) * self.cfg['num_depth'] + base
            
            all_scores.append(class_preds.gather(1, select_index))

        all_scores = torch.cat(all_scores, dim=1).view(-1)
        
        scores, indices = all_scores.topk(min(3000, len(all_scores)))
        pose_indices = (indices / (self.cfg['num_depth'] * self.cfg['num_types'])).long()
        
        ggarray_out = ggarray[pose_indices.cpu().numpy()]
        grasp_features_out = grasp_features[pose_indices.cpu().numpy()]
        
        type_indices = (indices % (self.cfg['num_depth'] * self.cfg['num_types']))
        depths = (type_indices % self.cfg['num_depth']).int().cpu().numpy() * 0.01
        types = (type_indices / self.cfg['num_depth']).int().cpu().numpy() + 1
        
        return depths, types, scores.cpu().numpy(), ggarray_out, grasp_features_out

    def process_and_filter_grasps(self, multi_finger_gg, two_finger_gg, grasp_features, **kwargs):
        """Post-process and select the best grasp for Allegro."""
        # Custom filtering for Allegro
        score_thresh = 0.7
        mask = multi_finger_gg.scores > score_thresh
        multi_finger_gg = multi_finger_gg[mask]
        
        if len(multi_finger_gg) == 0:
            return None, None
            
        # Select top-k from different grasp types
        select_indices = []
        type_counts = {i: 0 for i in range(1, self.cfg['num_types'] + 1)}
        
        for i in range(len(multi_finger_gg)):
            gtype = multi_finger_gg[i].grasp_type
            max_num = 5 if gtype in [4, 6] else 50 # Example of type-specific limit
            if type_counts[gtype] < max_num:
                select_indices.append(i)
                type_counts[gtype] += 1
        
        multi_finger_gg = multi_finger_gg[select_indices]
        # Keep other arrays in sync
        two_finger_gg = two_finger_gg[mask][select_indices]
        
        return multi_finger_gg, two_finger_gg