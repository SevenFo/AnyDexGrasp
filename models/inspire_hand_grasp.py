import numpy as np
import math
import warnings
from adg_utils import np_utils
from ur_toolbox.robot.Inspire.InspireHandR_grasp import InspireHandRGraspGroup, InspireHandRGrasp

class InspireHandRGraspGroupEnhance(InspireHandRGraspGroup):
    def filter_grasp_group_by_grasp_direction(self, theta_min=0, theta_max=math.pi/2, direction=None):
        # 设置目标方向（默认为基础坐标系+x轴）
        direction = np.array([1.0, 0.0, 0.0]) if direction is None else np.asarray(direction)
        
        try:
            # 计算抓取方向(x轴)与目标方向的夹角
            angles = np_utils.compute_angle_between_axes(
                self.rotation_matrices,
                axis_index=0,  # 指定使用x轴
                target_direction=direction
            )
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
    
    
# # 1. 计算抓取面法线方向(z轴)是否垂直
# surface_normals = rotations[:, :, 2]
# vertical_dir = np.array([0, 0, 1])
# verticality = compute_angle_between_axes(rotations, 2, vertical_dir)

# # 2. 检查抓取侧向方向(y轴)是否指向特定方向
# side_dir = np.array([0, 1, 0])
# side_alignment = compute_angle_between_axes(rotations, 1, side_dir)

# # 3. 组合多个方向约束
# horizontal_grasps_mask = (
#     (compute_angle_between_axes(rotations, 0, [0, 1, 0]) < np.pi/6) & 
#     (compute_angle_between_axes(rotations, 2, [0, 0, 1]) < np.pi/8)
# )