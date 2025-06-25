import numpy as np
import math
import warnings
from adg_utils import np_utils
from graspnetAPI import GraspGroup

class GraspGroupEnhance(GraspGroup):
    def filter_grasp_group_by_grasp_direction(self, theta_min=+math.pi/2, theta_max=+math.pi, direction=None):
        # 设置目标方向（默认为基础坐标系+x轴）
        direction = np.array([1.0, 0.0, 0.0]) if direction is None else np.asarray(direction)
        
        try:
            # 计算抓取方向(x轴)与目标方向的夹角
            angles = np_utils.compute_angle_between_axes(
                self.rotation_matrices,
                axis_index=0,  # 指定使用x轴
                target_direction=direction
            )
            # import pdb
            # pdb.set_trace()
        except ValueError as e:
            warnings.warn(f"{e} Using fallback direction [1.0, 0.0, 0.0]")
            angles = np_utils.compute_angle_between_axes(
                self.rotation_matrices,
                axis_index=0,
                target_direction=np.array([1.0, 0.0, 0.0])
            )
        
        # 应用角度过滤
        mask = (angles >= theta_min) & (angles <= theta_max)
        filtered_grasp_array = self.grasp_group_array[mask]
        return self.__class__(filtered_grasp_array), mask