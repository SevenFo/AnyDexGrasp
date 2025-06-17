# File: Inference/main.py
import os
import sys
import time
import argparse
import random
import torch
import numpy as np
import open3d as o3d
from multiprocessing import shared_memory

# Add project directories to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, "..")
sys.path.append(os.path.join(ROOT_DIR, "models"))
sys.path.append(os.path.join(ROOT_DIR, "dataset"))
sys.path.append(os.path.join(ROOT_DIR, "utils"))

from graspnetAPI import GraspGroup
from collision_detector import ModelFreeCollisionDetectorMultifinger

# Import from refactored utils
from configs import GRIPPER_CONFIGS
from utils.camera import get_depth, get_point_cloud
from utils.network import (
    get_net,
    predict_grasps,
    get_gripper_model,
    predict_multi_finger_grasp,
)
from utils.data_processing import augment_data, get_graspgroup_features
from utils.robot_utils import (
    get_robot,
    load_gripper_meshes,
    flip_ggarray,
    flip_z_ggarray,
    save_grasp_information,
)

# Constants
POINTCLOUD_AUGMENT_NUM = 10
DEBUG = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gripper",
        required=True,
        choices=GRIPPER_CONFIGS.keys(),
        help="Specify the gripper to use.",
    )
    parser.add_argument(
        "--checkpoint_path", required=True, help="GraspNet model checkpoint path"
    )
    parser.add_argument("--robot_ip", required=True, help="Robot IP address")
    parser.add_argument(
        "--use_graspnet_v2",
        action="store_true",
        help="Whether to use graspnet v2 format",
    )
    parser.add_argument(
        "--half_views", action="store_true", help="Use only half views in network."
    )
    parser.add_argument(
        "--global_camera",
        action="store_true",
        help="Use settings for a global camera setup.",
    )
    return parser.parse_args()


def get_all_grasp_proposals(net, existing_shm_depth, existing_shm_color, config):
    """Augments point cloud and aggregates grasp proposals."""
    depths = get_depth(existing_shm_depth)
    colors = np.copy(
        np.ndarray((720, 1280, 3), dtype=np.float32, buffer=existing_shm_color.buf)
    )
    points, cloud = get_point_cloud(depths, colors, config)

    all_gg, all_grasp_features, all_sinput = None, None, []

    # Generate augmentations
    augment_mats = [np.eye(4)] + [
        augment_data(flip=(i % 2 != 0)) for i in range(POINTCLOUD_AUGMENT_NUM)
    ]

    for i, mat in enumerate(augment_mats):
        flip = i > 0 and i % 2 != 0
        gg, grasp_features, points_down, sinput = predict_grasps(
            net, points, config, augment_mat=mat, flip=flip
        )

        if gg is None:
            continue

        if all_gg is None:
            all_gg, all_grasp_features = gg, grasp_features
        else:
            all_gg = torch.cat([all_gg, gg], axis=0)
            all_grasp_features = torch.cat([all_grasp_features, grasp_features], axis=0)

        if sinput:
            all_sinput.extend(sinput)

    return all_gg, cloud, points_down, all_grasp_features, all_sinput, depths, colors


def robot_grasp_loop(cfgs, config):
    """Main robot grasping loop."""
    # Initialization
    net = get_net(cfgs.checkpoint_path, cfgs)
    robot = get_robot(cfgs, config)
    gripper_models = get_gripper_model(config)
    gripper_meshes = load_gripper_meshes(config)

    existing_shm_color = shared_memory.SharedMemory(name="realsense_color")
    existing_shm_depth = shared_memory.SharedMemory(name="realsense_depth")

    # Initial robot movement
    v, a = 0.07, 0.07
    if cfgs.global_camera:
        robot.movej(robot.throwj2, acc=a * 2, vel=v * 3)
    else:
        robot.movel(robot.ready_pose(), acc=a * 2, vel=v * 3)

    while True:
        loop_start_time = time.time()

        # 1. Get grasp proposals
        if not cfgs.global_camera:
            robot.movel(robot.ready_pose(), acc=a * 10, vel=v * 10, wait=True)
            time.sleep(0.5)
            if config["name"] == "dh3":
                robot.gripper_home()
            time.sleep(0.7)

        (
            ggarray,
            cloud,
            points_down,
            grasp_features,
            sinput,
            depths_saved,
            colors_saved,
        ) = get_all_grasp_proposals(net, existing_shm_depth, existing_shm_color, config)

        net_time = time.time() - loop_start_time
        print(f"Grasp Prediction Time: {net_time:.2f}s")

        if ggarray is None:
            print("No grasps detected. Retrying...")
            if DEBUG and cloud:
                o3d.visualization.draw_geometries(
                    [cloud, o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)]
                )
            continue

        # 2. Process proposals: convert to numpy, flip, sort
        ggarray = ggarray.cpu().numpy()
        grasp_features = grasp_features.cpu().numpy()

        ggarray, if_flip = flip_ggarray(ggarray, config["flip_logic"])
        grasp_features = np.c_[grasp_features, if_flip]

        # For Inspire, remove flipped grasps as the hand is not symmetric
        if config["name"] == "inspire":
            valid_indices = ~if_flip
            ggarray = ggarray[valid_indices]
            grasp_features = grasp_features[valid_indices]

        sorted_indices = ggarray[:, 0].argsort()[::-1][:2000]
        ggarray = ggarray[sorted_indices]
        grasp_features = grasp_features[sorted_indices]

        # 3. Predict multi-finger grasp type
        grasp_features_dic = get_graspgroup_features(grasp_features, sinput)
        gripper_depths, gripper_types, scores, ggarray, grasp_features = (
            predict_multi_finger_grasp(
                gripper_models, grasp_features_dic, ggarray, grasp_features, config
            )
        )

        mask = scores > config["score_thresh"]
        ggarray, grasp_features, gripper_depths, gripper_types, scores = (
            ggarray[mask],
            grasp_features[mask],
            gripper_depths[mask],
            gripper_types[mask],
            scores[mask],
        )

        if config.get("has_z_flip", False):
            ggarray = flip_z_ggarray(ggarray, gripper_types)

        if len(ggarray) == 0:
            print(
                f"No grasps passed score threshold {config['score_thresh']}. Retrying..."
            )
            continue

        # 4. Create GraspGroups and filter
        two_fingers_gg = GraspGroup(ggarray)
        gripper_gg = config["gripper_class"]()
        if config["name"] == "inspire":
            gripper_gg.set_grasp_min_width(config["min_width"])

        gripper_gg.from_graspgroup(
            two_fingers_gg, gripper_types, config["mesh_json_path"]
        )
        gripper_gg.scores = scores
        gripper_gg.depths += gripper_depths + config["default_depth"]

        if config.get("select_type_func"):
            selected_indices = config["select_type_func"](
                gripper_gg, config["num_type"]
            )
            gripper_gg = gripper_gg[selected_indices]
            two_fingers_gg = two_fingers_gg[selected_indices]
            grasp_features = grasp_features[selected_indices]

        if len(gripper_gg) == 0:
            print("No grasps left after type selection. Retrying...")
            continue

        # 5. Collision Detection
        mfcdetector = ModelFreeCollisionDetectorMultifinger(
            points_down.cpu().numpy(), voxel_size=0.001
        )
        coll_gg, coll_tf_gg, empty_mask, width_mask = mfcdetector.detect(
            gripper_gg,
            two_fingers_gg,
            config["mesh_json_path"],
            gripper_meshes,
            min_grasp_width=config["min_width"],
            VoxelGrid=config["voxel_grid"],
            approach_dist=config["approach_dist"],
            collision_thresh=config["collision_thresh"],
            adjust_gripper_centers=config["adjust_gripper_centers"],
        )

        final_indices = np.where(empty_mask)[0]
        if len(final_indices) == 0:
            print("No grasps left after collision detection. Retrying...")
            if DEBUG and cloud:
                o3d.visualization.draw_geometries(
                    [cloud] + gripper_gg.to_open3d_geometry_list()
                )
            continue

        gripper_gg_final = coll_gg[final_indices]
        two_fingers_gg_final = coll_tf_gg[final_indices]
        grasp_features_final = grasp_features[width_mask][final_indices]

        # 6. Select and Execute Grasp
        # Sort by score and pick one randomly from top 10
        sorted_indices = np.argsort(gripper_gg_final.scores)[::-1]
        top_indices = sorted_indices[: min(10, len(sorted_indices))]
        chosen_idx = random.choice(top_indices)

        gripper_grasp_used = gripper_gg_final[chosen_idx]
        two_fingers_grasp_used = two_fingers_gg_final[chosen_idx]
        grasp_features_used = grasp_features_final[chosen_idx]

        print(f"\n--- Executing Grasp for {config['name'].upper()} ---")
        print(
            f"Type: {gripper_grasp_used.grasp_type}, Score: {gripper_grasp_used.score:.4f}"
        )
        print(
            f"Width: {gripper_grasp_used.width:.4f}, Depth: {gripper_grasp_used.depth:.4f}"
        )

        if DEBUG:
            gripper_mesh = gripper_grasp_used.load_mesh(
                config["mesh_json_path"], two_fingers_grasp_used
            )
            o3d.visualization.draw_geometries(
                [
                    cloud,
                    gripper_mesh,
                    o3d.geometry.TriangleMesh.create_coordinate_frame(0.1),
                ]
            )

        # Execute
        gripper_time = 0.8
        robot.open_gripper(gripper_grasp_used.angle, sleep_time=gripper_time)
        mat_pose = robot.grasp_and_throw(
            gripper_grasp_used,
            two_fingers_grasp_used,
            cloud,
            config["mesh_json_path"],
            acc=a * 2,
            vel=v * 3,
            approach_dist=config["approach_dist"],
            execute_grasp=True,
            use_ready_pose=True,
            gripper_time=gripper_time,
        )
        while robot.is_program_running():
            pass

        # 7. Save Information
        save_data = {
            "config": config,
            "gripper_grasp_used": gripper_grasp_used,
            "two_fingers_grasp_used": two_fingers_grasp_used,
            "grasp_features_used": grasp_features_used,
            "mat_pose": mat_pose,
            "colors_saved": colors_saved,
            "depths_saved": depths_saved,
        }
        result = save_grasp_information(save_data)
        if result == "restart":
            continue

        exec_time = time.time() - loop_start_time
        mpph = 3600 / exec_time
        print(f"Execution Time: {exec_time:.2f}s | MPPH: {mpph:.2f}")
        print("----------------------------------------\n")


if __name__ == "__main__":
    args = parse_args()
    config = GRIPPER_CONFIGS[args.gripper]

    # Update config with args
    config["model_path"] = getattr(
        args, f"{config['name']}_model_path", config["model_path"]
    )
    config["save_path"] = getattr(
        args, f"{config['name']}_save_path", config["save_path"]
    )
    config["mesh_json_path"] = getattr(
        args, f"{config['name']}_mesh_json_path", config["mesh_json_path"]
    )

    start_time = time.time()
    try:
        robot_grasp_loop(args, config)
    except KeyboardInterrupt:
        print("Program interrupted by user.")
    finally:
        print(f"Total run time: {time.time() - start_time:.2f} seconds.")
