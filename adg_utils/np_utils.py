import numpy as np

def transform_point_cloud(cloud, transform, format='4x4'):
    if not (format == '3x3' or format == '4x4' or format == '3x4'):
        raise ValueError('Unknown transformation format, only support \'3x3\' or \'4x4\' or \'3x4\'.')
    if format == '3x3':
        cloud_transformed = np.dot(transform, cloud.T).T
    elif format == '4x4' or format == '3x4':
        ones = np.ones(cloud.shape[0])[:, np.newaxis]
        cloud_ = np.concatenate([cloud, ones], axis=1)
        cloud_transformed = np.dot(transform, cloud_.T).T
        cloud_transformed = cloud_transformed[:, :3]
    return cloud_transformed

def compute_point_dists(A, B):
    A_ = A[:, np.newaxis, :]
    B_ = B[np.newaxis, :, :]
    dists = np.linalg.norm(A_-B_, axis=-1)
    return dists

def remove_invisible_grasp_points(cloud, grasp_points, pose):
    grasp_points_trans = transform_point_cloud(grasp_points, pose)
    dists = compute_point_dists(cloud, grasp_points_trans)
    visible_idxs = dists.argmin(axis=-1)
    visible_idxs = np.unique(visible_idxs)
    return visible_idxs

def create_point_cloud_from_depth_image(depth, camera, organized=True):
    assert(depth.shape[0] == camera.height and depth.shape[1] == camera.width)
    xmap = np.arange(camera.width)
    ymap = np.arange(camera.height)
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depth / camera.scale
    points_x = (xmap - camera.cx) * points_z / camera.fx
    points_y = (ymap - camera.cy) * points_z / camera.fy
    cloud = np.stack([points_x, points_y, points_z], axis=-1)
    if not organized:
        cloud = cloud.reshape([-1, 3])
    return cloud

def get_workspace_mask(cloud, seg, trans=None, organized=True, outlier=0):
    if organized:
        h, w, _ = cloud.shape
        cloud = cloud.reshape([h*w, 3])
        seg = seg.reshape(h*w)
    if trans is not None:
        cloud = transform_point_cloud(cloud, trans)
    foreground = cloud[seg>0]
    xmin, ymin, zmin = foreground.min(axis=0)
    xmax, ymax, zmax = foreground.max(axis=0)
    mask_x = ((cloud[:,0] > xmin-outlier) & (cloud[:,0] < xmax+outlier))
    mask_y = ((cloud[:,1] > ymin-outlier) & (cloud[:,1] < ymax+outlier))
    mask_z = ((cloud[:,2] > zmin-outlier) & (cloud[:,2] < zmax+outlier))
    workspace_mask = (mask_x & mask_y & mask_z)
    if organized:
        workspace_mask = workspace_mask.reshape([h, w])
    return workspace_mask

def generate_views(N, phi=(np.sqrt(5)-1)/2, center=np.zeros(3, dtype=np.float32), R=1):
    idxs = np.arange(N, dtype=np.float32)
    Z = (2 * idxs + 1) / N - 1
    X = np.sqrt(1 - Z**2) * np.cos(2 * idxs * np.pi * phi)
    Y = np.sqrt(1 - Z**2) * np.sin(2 * idxs * np.pi * phi)
    views = np.stack([X,Y,Z], axis=1)
    views = R * np.array(views) + center
    return views

def batch_viewpoint_params_to_matrix(batch_towards, batch_angle):
    axis_x = batch_towards
    ones = np.ones(axis_x.shape[0], dtype=axis_x.dtype)
    zeros = np.zeros(axis_x.shape[0], dtype=axis_x.dtype)
    axis_y = np.stack([-axis_x[:,1], axis_x[:,0], zeros], axis=-1)
    mask_y = (np.linalg.norm(axis_y, axis=-1) == 0)
    axis_y[mask_y] = np.array([0, 1, 0])
    axis_x = axis_x / np.linalg.norm(axis_x, axis=-1, keepdims=True)
    axis_y = axis_y / np.linalg.norm(axis_y, axis=-1, keepdims=True)
    axis_z = np.cross(axis_x, axis_y)
    sin = np.sin(batch_angle)
    cos = np.cos(batch_angle)
    R1 = np.stack([ones, zeros, zeros, zeros, cos, -sin, zeros, sin, cos], axis=-1)
    R1 = R1.reshape([-1,3,3])
    R2 = np.stack([axis_x, axis_y, axis_z], axis=-1)
    matrix = np.matmul(R2, R1)
    return matrix.astype(np.float32)

def compute_angle_between_axes(rot_matrices, axis_index, target_direction):
    """
    计算旋转矩阵的指定局部轴与目标方向向量之间的夹角（弧度）

    参数:
        rot_matrices (np.ndarray): 旋转矩阵数组，形状为(N, 3, 3)
        axis_index (int): 要比较的局部轴索引 (0=x, 1=y, 2=z)
        target_direction (np.ndarray): 目标方向向量，形状为(3,)
    
    返回:
        angles (np.ndarray): 每个旋转矩阵对应的角度误差（弧度），形状为(N,)
    
    异常:
        ValueError: 如果目标方向是零向量或axis_index无效
    """
    # 验证轴索引
    if axis_index not in [0, 1, 2]:
        raise ValueError("axis_index must be 0 (x), 1 (y), or 2 (z)")
    
    # 提取指定局部轴
    local_axes = rot_matrices[:, :, axis_index]  # (N, 3)

    # 归一化目标方向
    target_norm = np.linalg.norm(target_direction)
    if target_norm < 1e-6:
        raise ValueError("Target direction vector is near zero!")
    target_dir_norm = target_direction / target_norm

    # 计算点积 (cosθ值)
    dots = np.sum(local_axes * target_dir_norm, axis=1)  # (N,)

    # 防止点积超出[-1,1]范围然后计算角度
    clipped_dots = np.clip(dots, -1.0, 1.0)
    angles = np.arccos(clipped_dots)

    return angles