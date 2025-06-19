# File: Inference/utils/data_processing.py
import numpy as np
import copy


def transform_point_cloud(cloud, transform):
    """Apply a 4x4 transform to a point cloud."""
    if cloud.shape[1] != 3:
        raise ValueError("Point cloud must have 3 columns (X, Y, Z)")
    return (np.dot(transform[:3, :3], cloud.T) + transform[:3, 3:4]).T


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


def get_grasp_features(grasp_features_array):
    """Parses a flat feature array into a structured dictionary."""
    grasp_features = {
        "grasp_angles": int(grasp_features_array[-3] + 0.1),
        "grasp_depths": int(grasp_features_array[-2] * 100 + 0.1),
        "stage3_grasp_scores": grasp_features_array[:240].tolist(),
        "grasp_preds_features": grasp_features_array[240:720].tolist(),
        "stage3_grasp_features": grasp_features_array[720:1232].tolist(),
        "before_generator": grasp_features_array[1232:1744].tolist(),
        "point_features": grasp_features_array[1744:2256].tolist(),
        "point_id": int(grasp_features_array[-4]),
        "if_flip": bool(int(grasp_features_array[-1])),
        "view_inds": int(grasp_features_array[-6]),
        "view_score": grasp_features_array[-5],
    }
    return grasp_features


def get_graspgroup_features(grasp_features_array, sinput):
    """Parses a batch of flat feature arrays and performs rotation normalization."""
    num_grasps = grasp_features_array.shape[0]
    grasp_features = {
        "grasp_angles": (grasp_features_array[:, -3] + 0.1).astype(int),
        "grasp_depths": (grasp_features_array[:, -2] * 100 + 0.1).astype(int),
        "stage3_grasp_scores": grasp_features_array[:, :240],
        "grasp_preds_features": grasp_features_array[:, 240:720],
        "stage3_grasp_features": grasp_features_array[:, 720:1232],
        "before_generator": grasp_features_array[:, 1232:1744],
        "point_features": grasp_features_array[:, 1744:2256],
        "point_id": (grasp_features_array[:, -4] + 0.1).astype(int),
        "if_flip": grasp_features_array[:, -1],
        "view_inds": (grasp_features_array[:, -6] + 0.1).astype(int),
        "view_score": grasp_features_array[:, -5],
        "sinput": sinput,
    }

    preds_features = grasp_features["grasp_preds_features"]
    grasp_preds_features_rot = np.zeros_like(preds_features)
    new_type = grasp_features["grasp_angles"]

    for idx, if_flip in enumerate(grasp_features["if_flip"]):
        current_preds = copy.deepcopy(preds_features[idx, :])
        if if_flip:
            scores = current_preds[:240].reshape(2, 120)
            widths = current_preds[240:].reshape(2, 120)
            current_preds[:120], current_preds[120:240] = scores[1], scores[0]
            current_preds[240:360], current_preds[360:480] = widths[1], widths[0]

        # Rotate features based on grasp angle
        angle_offset = new_type[idx] * 5
        grasp_preds_features_rot[idx, :240] = np.roll(
            current_preds[:240], -angle_offset
        )
        grasp_preds_features_rot[idx, 240:480] = np.roll(
            current_preds[240:480], -angle_offset
        )

    grasp_features["grasp_preds_features"] = grasp_preds_features_rot
    return grasp_features
