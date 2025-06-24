# File: Inference/main.py
import os
import sys
import time
import argparse
import random
import torch
import numpy as np
import open3d as o3d
import cv2
from multiprocessing import shared_memory
# 设置随机种子
def set_seed(seed):
    # 设置 NumPy 的随机种子
    np.random.seed(seed)
    
    # 设置 Python 内置 random 模块的随机种子
    random.seed(seed)
    
    # 设置 PyTorch 的随机种子
    torch.manual_seed(seed)
    
    # 如果使用 CUDA（GPU），设置相关种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 多 GPU 情况
        torch.backends.cudnn.deterministic = True  # 确保卷积操作确定性
        torch.backends.cudnn.benchmark = False     # 关闭优化基准（保证可复现性）

# 使用示例（设置种子为42）
set_seed(42)
# # Add project directories to sys.path
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ROOT_DIR = os.path.join(BASE_DIR, "..")
# sys.path.append(os.path.join(ROOT_DIR, "models"))
# sys.path.append(os.path.join(ROOT_DIR, "dataset"))
# sys.path.append(os.path.join(ROOT_DIR, "utils"))

from graspnetAPI import GraspGroup
from adg_utils.collision_detector import ModelFreeCollisionDetectorMultifinger

# Import from refactored utils
from .configs import GRIPPER_CONFIGS
from .utils.camera import get_depth, get_point_cloud
from .utils.network import (
    get_net,
    predict_grasps,
    get_gripper_model,
    predict_multi_finger_grasp,
)
from .utils.visualization import visualize_grasp_proposals
from .utils.data_processing import augment_data, get_graspgroup_features
from .utils.robot_utils import (
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
        "--checkpoint_path", default='/data/shiqi/AnyDexGrasp/logs/model/checkpoint.tar.18', help="GraspNet model checkpoint path"
    )
    parser.add_argument("--robot_ip", default='0.0.0.0', help="Robot IP address")
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
    # 添加无机器人模式参数
    parser.add_argument(
        "--no_robot", action="store_true",
        help="Run without physical robot, using mock robot and file inputs"
    )
    # 添加文件输入参数
    parser.add_argument(
        "--depth_image", type=str, default="depth.png",
        help="Depth file path (npy format) for no_robot mode"
    )
    parser.add_argument(
        "--color_image", type=str, default="rgb.png",
        help="Color file path (npy format) for no_robot mode"
    )
    parser.add_argument(
        "--point_cloud", type=str, default="filtered.ply",
        help="point_cloud path (npy format) for no_robot mode"
    )
    return parser.parse_args()


def get_all_grasp_proposals(net, cfgs, config):
    """Augments point cloud and aggregates grasp proposals."""
    pcd = None
    depths = None
    colors = None
    if cfgs.no_robot:
        if cfgs.point_cloud and os.path.exists(cfgs.point_cloud):
            pcd = o3d.io.read_point_cloud(cfgs.point_cloud)
        else:  
            # 从文件加载深度和彩色图
            depths = cv2.imread(cfgs.depth_image, cv2.IMREAD_ANYDEPTH)
            if depths is None:
                print("Error: Failed to load depth image")
                return
            
            colors = None
            if cfgs.color_image and os.path.exists(cfgs.color_image):
                colors = cv2.imread(cfgs.color_image)
                if colors is not None:
                    colors = cv2.cvtColor(colors, cv2.COLOR_BGR2RGB)
            print("Get rgb and depth")
    else:
        # 从共享内存获取
        existing_shm_depth = shared_memory.SharedMemory(name="realsense_depth")
        existing_shm_color = shared_memory.SharedMemory(name="realsense_color")
        depths = get_depth(existing_shm_depth)
        colors = np.copy(np.ndarray((720, 1280, 3), dtype=np.float32, buffer=existing_shm_color.buf))
    
    if pcd is None:
        points, cloud = get_point_cloud(depths, colors, config)
    else:
        points, cloud = np.asarray(pcd.points), pcd
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
        ) = get_all_grasp_proposals(net, cfgs, config)

        net_time = time.time() - loop_start_time
        print(f"Grasp Prediction Time: {net_time:.2f}s")

        if DEBUG and cloud:
            # 检查点云是否为空
            if not cloud.has_points():
                print("Cloud has no points!")
            else:
                # 打印点云的范围
                print("Cloud bounding box:", cloud.get_axis_aligned_bounding_box())
                # 打印点云的中心
                print("Cloud center:", cloud.get_center())
                # 尝试给点云上色（如果点云没有颜色属性）
                # 注意：如果点云已经有颜色，我们可以跳过这一步
                if not cloud.has_colors():
                    # 将点云染成红色
                    cloud.paint_uniform_color([1, 0, 0])
                    print("Painted cloud red.")

            # 绘制
            o3d.visualization.draw_geometries(
                [cloud, o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)]
            )
        if ggarray is None:
            print("No grasps detected. Retrying...")
            continue

        # 2. Process proposals: convert to numpy, flip, sort
        ggarray = ggarray.cpu().numpy()
        grasp_features = grasp_features.cpu().numpy()

        ggarray, if_flip = flip_ggarray(ggarray, config["flip_logic"])
        grasp_features = np.c_[grasp_features, if_flip]

        # For Inspire, remove flipped grasps as the hand is not symmetric
        if config["name"] == "inspire":
            valid_indices = ~np.array(if_flip)
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
        gripper_gg_top = gripper_gg_final[top_indices]
        two_fingers_gg_top = two_fingers_gg_final[top_indices]

        # 可视化top10抓取
        print("Visualizing top 10 grasps...")
        visualize_grasp_proposals(
            cloud, 
            two_fingers_gg_top, 
            gripper_gg_top, 
            config,
            "Top 10 Grasp Proposals"
        )
        
        chosen_idx = random.choice(top_indices).item()
        print(f"random chose grasp index:{chosen_idx} from top 10 {top_indices}")
        gripper_grasp_used = gripper_gg_final[chosen_idx]
        two_fingers_grasp_used = two_fingers_gg_final[chosen_idx]
        grasp_features_used = grasp_features_final[chosen_idx]

        mode = "Visualization" if cfgs.no_robot else "Execution"
        print(f"\n--- {mode} for {config['name'].upper()} ---")
        print(f"Type: {gripper_grasp_used.grasp_type}, Score: {gripper_grasp_used.score:.4f}")
        print(f"Width: {gripper_grasp_used.width:.4f}, Depth: {gripper_grasp_used.depth:.4f}")
        print((f"Angles: [little:{gripper_grasp_used.angle[0]}, ring:{gripper_grasp_used.angle[1]},\n",
                f"mid:{gripper_grasp_used.angle[2]}, index:{gripper_grasp_used.angle[3]},\n" 
                f"thumb_bending:{gripper_grasp_used.angle[4]}, thumb_rotation:{gripper_grasp_used.angle[5]}]"))
        if cfgs.no_robot:
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
        if not cfgs.no_robot:
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
        
        if cfgs.no_robot:
            user_input = input("Grasp visualized. Continue? (y/n): ")
            if user_input.lower() != 'y':
                print("Exiting no_robot mode...")
                break

if __name__ == "__main__":
    import os
    os.environ['DISPLAY'] = '109.105.4.86:0.0'
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

    # 检查no_robot模式下的文件存在性
    if args.no_robot:
        if not os.path.exists(args.depth_image):
            raise FileNotFoundError(f"Depth file not found: {args.depth_image}")
        if not os.path.exists(args.color_image):
            import warnings
            # warnings.warn(f"Color file not found: {args.color_image}")
            raise FileNotFoundError(f"Color file not found: {args.color_image}")

    start_time = time.time()
    try:
        robot_grasp_loop(args, config)
    except KeyboardInterrupt:
        print("Program interrupted by user.")
    finally:
        print(f"Total run time: {time.time() - start_time:.2f} seconds.")
