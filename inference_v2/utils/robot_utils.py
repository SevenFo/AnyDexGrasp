# File: Inference/utils/robot_utils.py
import os
import json
import datetime
import cv2
import numpy as np
import open3d as o3d
from collections import OrderedDict
from ur_toolbox.robot import UR_Camera_Gripper


def get_robot(cfgs, config):
    """Initializes and configures the robot and gripper."""
    robot = UR_Camera_Gripper(
        cfgs.robot_ip,
        use_rt=False,
        camera=None,
        robot_debug=True,
        gripper_type=config["name"].capitalize(),
        gripper_port=config["gripper_port"],
        global_cam=cfgs.global_camera,
    )
    robot.set_tcp(config["robot_tcp"])
    robot.set_payload(*config["robot_payload"])
    return robot


def load_gripper_meshes(config):
    """Loads pre-computed point clouds of the gripper meshes."""
    meshes_pcls = {}
    path = config["mesh_json_path"]
    voxel_size_str = str(int(config["voxel_grid"] * 1000))
    pcl_path = os.path.join(
        path, f"meshes/source_pointclouds/voxel_size_{voxel_size_str}"
    )
    if not os.path.exists(pcl_path):  # Inspire has a different structure
        pcl_path = os.path.join(path, f"source_pointclouds/voxel_size_{voxel_size_str}")

    for type_dir in os.listdir(pcl_path):
        type_path = os.path.join(pcl_path, type_dir)
        for name in os.listdir(type_path):
            width = name[:-4]
            name_path = os.path.join(type_path, name)
            meshes_pcl = o3d.io.read_point_cloud(name_path)
            meshes_pcls[f"{type_dir}_{width}"] = meshes_pcl
    return meshes_pcls


def flip_ggarray(ggarray, flip_logic):
    """Flips grasp poses to ensure a consistent orientation."""
    rotations = ggarray[:, 4:13].reshape((-1, 3, 3))
    y_axis_x_comp = rotations[:, 1, 1]
    if_flip = np.zeros(len(ggarray), dtype=bool)

    if flip_logic == "y_x < 0":
        indices_to_flip = np.where(y_axis_x_comp < 0)[0]
        rotations[indices_to_flip, :, 1:3] *= -1
    elif flip_logic == "y_x > 0":
        indices_to_flip = np.where(y_axis_x_comp > 0)[0]
        rotations[indices_to_flip, :, 1:3] *= -1
    elif flip_logic == "y_x < 0_inspire":
        indices_to_flip = np.where(y_axis_x_comp < 0)[0]
        rotations[indices_to_flip, :, 0:2] *= -1  # Inspire flips x and y axes

    if_flip[indices_to_flip] = True
    ggarray[:, 4:13] = rotations.reshape((-1, 9))
    return ggarray, if_flip


def flip_z_ggarray(ggarray, allegro_types):
    """Special z-axis flip for Allegro hand."""
    rotations = ggarray[:, 4:13].reshape((-1, 3, 3))
    z_axis_on_base_frame = rotations[:, 0, 2]

    for ids, z_val in enumerate(z_axis_on_base_frame):
        if z_val < 0 and allegro_types[ids] == 9:
            rotations[ids, :, 1:3] *= -1

    ggarray[:, 4:13] = rotations.reshape((-1, 9))
    return ggarray


def save_grasp_information(data_dict):
    """Saves all information about the grasp attempt to a JSON file and images."""
    config = data_dict["config"]
    gripper_grasp_used = data_dict["gripper_grasp_used"]

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    grasp_type_name = config["grasp_types"][str(int(gripper_grasp_used.grasp_type))][
        "name"
    ]
    save_path = os.path.join(config["save_path"], grasp_type_name, timestamp)
    os.makedirs(save_path, exist_ok=True)

    # User input for success/failure
    while True:
        response = input("Grasp successful? (1=yes, 2=no, 3=restart, 4=exit): ")
        if response in ["1", "2", "3", "4"]:
            break
        print("Invalid input. Please enter 1, 2, 3, or 4.")

    if response == "3":
        return "restart"
    if response == "4":
        exit()

    info = OrderedDict()
    info["result"] = response == "1"

    tfg = data_dict["two_fingers_grasp_used"]
    info["two_fingers_pose"] = (
        [float(tfg.score), float(tfg.width), float(tfg.height), float(tfg.depth)]
        + tfg.rotation_matrix.flatten().tolist()
        + tfg.translation.flatten().tolist()
        + [float(gripper_grasp_used.object_id)]
    )

    info["gripper_pose"] = gripper_grasp_used.get_array_grasp().tolist()
    info["gripper_name"] = config["name"]

    grasp_features_used = get_grasp_features(data_dict["grasp_features_used"])
    info.update(grasp_features_used)

    info["gripper_pose_finger_type"] = int(gripper_grasp_used.grasp_type)
    info["gripper_pose_depth_type_offset"] = (
        int(gripper_grasp_used.depth * 100) - grasp_features_used["grasp_depths"]
    )

    mat_pose = data_dict["mat_pose"]
    info["base_2_tcp1"] = np.array(mat_pose[0]).tolist()
    info["base_2_tcp1_backup"] = np.array(mat_pose[1]).tolist()
    info["tcp_2_gripper"] = np.array(mat_pose[2]).tolist()
    info["base_2_TwoFingersGripper_pose"] = np.array(mat_pose[3]).tolist()
    info["tcp_2_camera"] = np.array(mat_pose[4]).tolist()
    info["base_2_tcp_ready"] = np.array(mat_pose[5]).tolist()

    cam_intrinsics = config["cam_intrinsics"]
    info["camera_internal"] = [
        [cam_intrinsics["cx"], cam_intrinsics["cy"]],
        [cam_intrinsics["fx"], cam_intrinsics["fy"]],
    ]

    # Save data
    cv2.imwrite(
        os.path.join(save_path, "color.png"),
        cv2.cvtColor(data_dict["colors_saved"] * 255.0, cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(os.path.join(save_path, "depth.png"), data_dict["depths_saved"])

    with open(os.path.join(save_path, "information.json"), "w") as f:
        json.dump(info, f, indent=4)

    print(f"Saved grasp information to {save_path}")
    return "saved"
