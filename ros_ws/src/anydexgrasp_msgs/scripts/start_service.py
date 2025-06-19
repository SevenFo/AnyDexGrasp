#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Provides a service to convert point clouds to Inspire hand grasp poses
"""
from typing import Union
import rospy
import numpy as np
import sys
import torch
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from std_srvs.srv import Trigger, TriggerResponse
import sensor_msgs.point_cloud2 as pc2
from scipy.spatial.transform import Rotation as R

# Setup AnyDexGrasp paths
from anydexgrasp_config import setup_anydexgrasp_paths, get_default_model_paths

anydexgrasp_root = setup_anydexgrasp_paths()
default_paths = get_default_model_paths(anydexgrasp_root)

# Import from refactored utils
from inference_v2.configs import GRIPPER_CONFIGS
from inference_v2.utils.network import (
    get_net,
    predict_grasps,
    get_gripper_model,
    predict_multi_finger_grasp,
)
from inference_v2.utils.data_processing import get_graspgroup_features, augment_data
from inference_v2.utils.robot_utils import flip_ggarray,load_gripper_meshes,flip_z_ggarray
from inference_v2.main import get_all_grasp_proposals
from ur_toolbox.robot.Inspire.InspireHandR_grasp import InspireHandRGraspGroup,InspireHandRGrasp
from graspnetAPI import GraspGroup, Grasp
from adg_utils.collision_detector import ModelFreeCollisionDetectorMultifinger
# Custom service messages (需要先定义这些服务消息类型)

try:
    from anydexgrasp_msgs.srv import GraspPlanning, GraspPlanningResponse
    from anydexgrasp_msgs.msg import GraspPose

    CUSTOM_MSGS_AVAILABLE = True
except ImportError:
    rospy.logwarn("Custom message types not found. Using standard ROS messages.")
    CUSTOM_MSGS_AVAILABLE = False


class InspireGraspPlanningService:
    """ROS service for Inspire hand grasp planning from point clouds"""

    def __init__(self):
        rospy.init_node("inspire_grasp_planning_service", anonymous=True)

        # 加载配置
        self.config = GRIPPER_CONFIGS["inspire"]

        # 获取参数
        self.checkpoint_path = rospy.get_param(
            "~checkpoint_path",
            "logs/model/inspire_model/final_single_point/obj140/checkpoint.tar",
        )
        self.use_graspnet_v2 = rospy.get_param("~use_graspnet_v2", True)
        self.half_views = rospy.get_param("~half_views", False)
        self.max_grasps = rospy.get_param("~max_grasps", 10)
        self.score_threshold = rospy.get_param("~score_threshold", 0.85)

        rospy.loginfo("Initializing Inspire Grasp Planning Service...")

        # 初始化模型
        self._initialize_models()

        # 创建服务
        self.grasp_service = rospy.Service(
            "~plan_grasps", self._get_service_type(), self._handle_grasp_planning
        )

        # 创建发布者用于可视化
        self.grasp_poses_pub = rospy.Publisher(
            "~grasp_poses", PoseStamped, queue_size=10
        )

        rospy.loginfo("Inspire Grasp Planning Service initialized successfully!")
        rospy.loginfo(f"Service available at: {self.grasp_service.resolved_name}")

    def _get_service_type(self):
        """获取服务类型，如果自定义消息不可用则使用Trigger"""
        try:
            return GraspPlanning
        except NameError:
            rospy.logwarn("Using Trigger service as fallback")
            return Trigger

    def _initialize_models(self):
        """初始化神经网络模型"""
        try:
            # 初始化基础GraspNet模型
            rospy.loginfo("Loading base GraspNet model...")

            # 创建配置对象
            class Args:
                def __init__(self):
                    self.use_graspnet_v2 = self.use_graspnet_v2
                    self.half_views = self.half_views

            args = Args()
            self.net = get_net(self.checkpoint_path, args)
            rospy.loginfo("Base GraspNet model loaded successfully")

            # 初始化多指抓取模型
            rospy.loginfo("Loading gripper-specific models...")
            self.gripper_models = get_gripper_model(self.config)
            self.gripper_meshes = load_gripper_meshes(self.config)
            rospy.loginfo("Gripper-specific models loaded successfully")

            # 初始化碰撞检测器 (will be created per request)
            self.collision_detector = None

        except Exception as e:
            rospy.logerr(f"Failed to initialize models: {e}")
            raise

    def _handle_grasp_planning(self, req):
        """处理抓取规划请求"""
        try:
            rospy.loginfo("Received grasp planning request")

            # 从请求中提取点云
            if hasattr(req, "pointcloud"):
                points = self._extract_points_from_pointcloud2(req.pointcloud)
            else:
                # 如果使用Trigger服务，从topic获取点云
                rospy.loginfo(
                    "Waiting for point cloud on /camera/depth/color/points..."
                )
                pointcloud_msg = rospy.wait_for_message(
                    "/camera/depth/color/points", PointCloud2, timeout=5.0
                )
                points = self._extract_points_from_pointcloud2(pointcloud_msg)

            rospy.loginfo(f"Received point cloud with {len(points)} points")

            # 执行抓取规划
            grasp_poses = self._plan_grasps(points)

            # 创建响应
            if hasattr(req, "pointcloud"):  # 自定义服务
                response = GraspPlanningResponse()
                response.success = len(grasp_poses) > 0
                response.message = f"Generated {len(grasp_poses)} grasp poses"
                response.grasp_poses = grasp_poses
            else:  # Trigger服务
                response = TriggerResponse()
                response.success = len(grasp_poses) > 0
                response.message = f"Generated {len(grasp_poses)} grasp poses"

                # 发布抓取姿态用于可视化
                for i, grasp_pose in enumerate(grasp_poses):
                    pose_stamped = PoseStamped()
                    pose_stamped.header.stamp = rospy.Time.now()
                    pose_stamped.header.frame_id = "camera_link"
                    pose_stamped.pose = grasp_pose
                    self.grasp_poses_pub.publish(pose_stamped)
                    rospy.sleep(0.1)  # 延迟发布以便可视化

            rospy.loginfo(f"Successfully generated {len(grasp_poses)} grasp poses")
            return response

        except Exception as e:
            rospy.logerr(f"Error in grasp planning: {e}")

            if hasattr(req, "pointcloud"):
                response = GraspPlanningResponse()
                response.success = False
                response.message = f"Error: {str(e)}"
                response.grasp_poses = []
            else:
                response = TriggerResponse()
                response.success = False
                response.message = f"Error: {str(e)}"

            return response

    def _extract_points_from_pointcloud2(self, pointcloud_msg):
        """从PointCloud2消息中提取点云数据"""
        points_list = []

        for point in pc2.read_points(
            pointcloud_msg, field_names=("x", "y", "z"), skip_nans=True
        ):
            points_list.append([point[0], point[1], point[2]])

        if len(points_list) == 0:
            raise ValueError("No valid points found in point cloud")

        points = np.array(points_list, dtype=np.float32)

        # 应用工作空间过滤
        x_min, x_max, y_min, y_max = self.config["workspace_mask"]
        z_min, z_max = self.config["point_cloud_mask"]

        mask = (
            (points[:, 0] > x_min)
            & (points[:, 0] < x_max)
            & (points[:, 1] > y_min)
            & (points[:, 1] < y_max)
            & (points[:, 2] > z_min)
            & (points[:, 2] < z_max)
        )

        filtered_points = points[mask]

        if len(filtered_points) == 0:
            raise ValueError("No points remain after workspace filtering")

        return filtered_points

    def _plan_grasps(self, points):
        """执行抓取规划"""
        # 1. 预测二指抓取
        rospy.loginfo("Predicting two-finger grasps...")
        POINTCLOUD_AUGMENT_NUM = 10
        all_gg, all_grasp_features, all_sinput = None, None, []
        # Generate augmentations
        augment_mats = [np.eye(4)] + [
            augment_data(flip=(i % 2 != 0)) for i in range(POINTCLOUD_AUGMENT_NUM)
        ]

        for i, mat in enumerate(augment_mats):
            flip = i > 0 and i % 2 != 0
            gg, grasp_features, points_down, sinput = predict_grasps(
                self.net, points, self.config, augment_mat=mat, flip=flip
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

        if all_gg is None:
            rospy.logwarn("No two-finger grasps detected")
            return []
        rospy.loginfo(f"Generated {len(all_gg)} two-finger grasps")
        ggarray = all_gg
        grasp_features = all_grasp_features
        sinput = all_sinput
        # 2. 处理抓取姿态
        ggarray = ggarray.cpu().numpy()
        grasp_features = grasp_features.cpu().numpy()

        # 翻转处理
        ggarray, if_flip = flip_ggarray(ggarray, self.config["flip_logic"])
        grasp_features = np.c_[grasp_features, if_flip]

        # 对于Inspire手，移除翻转的抓取（因为手不对称）
        if config["name"] == "inspire":
            valid_indices = ~np.array(if_flip)
            ggarray = ggarray[valid_indices]
            grasp_features = grasp_features[valid_indices]

        # 按分数排序并限制数量
        sorted_indices = ggarray[:, 0].argsort()[::-1][:2000]
        ggarray = ggarray[sorted_indices]
        grasp_features = grasp_features[sorted_indices]

        # 3. 预测多指抓取类型
        rospy.loginfo("Predicting multi-finger grasp types...")
        grasp_features_dic = get_graspgroup_features(grasp_features, sinput)
        gripper_depths, gripper_types, scores, ggarray, grasp_features = (
            predict_multi_finger_grasp(
                self.gripper_models,
                grasp_features_dic,
                ggarray,
                grasp_features,
                self.config,
            )
        )
        rospy.loginfo(f"Generated {len(scores)} multi-finger grasp")

        # 过滤低分抓取
        rospy.loginfo(f"Filtering multi-finger grasp by score_threshold {self.score_threshold}")
        mask = scores > self.score_threshold
        ggarray, grasp_features, gripper_depths, gripper_types, scores = (
            ggarray[mask],
            grasp_features[mask],
            gripper_depths[mask],
            gripper_types[mask],
            scores[mask],
        )
        rospy.loginfo(f"Remain {len(ggarray)} multi-finger grasp")
        if self.config.get("has_z_flip", False): # for allegro
            ggarray = flip_z_ggarray(ggarray, gripper_types)
        if len(ggarray) == 0:
            rospy.logwarn(f"No grasps passed score threshold {self.score_threshold}")
            return []

        # 4. 创建GraspGroup并处理
        two_fingers_gg = GraspGroup(ggarray)
        gripper_gg = self.config["gripper_class"]()
        if self.config["name"] == "inspire":
            gripper_gg.set_grasp_min_width(self.config["min_width"])

        gripper_gg.from_graspgroup(
            two_fingers_gg, gripper_types, self.config["mesh_json_path"]
        )
        gripper_gg.scores = scores
        gripper_gg.depths += gripper_depths + self.config["default_depth"]
        
        # 类型选择
        if self.config.get("select_type_func"):
            selected_indices = self.config["select_type_func"](
                gripper_gg, self.config["num_type"]
            )
            gripper_gg = gripper_gg[selected_indices]
            two_fingers_gg = two_fingers_gg[selected_indices]
            grasp_features = grasp_features[selected_indices]

        if len(gripper_gg) == 0:
            print("No grasps left after type selection. Retrying...")
            return []
        rospy.loginfo(f"Remain {len(gripper_gg)} grasps after type selection")
        
        # 5. 碰撞检测
        rospy.loginfo("Performing collision detection...")
        self.collision_detector = ModelFreeCollisionDetectorMultifinger(
            points_down.cpu().numpy(), voxel_size=0.001
        )

        coll_gg, coll_tf_gg, empty_mask, width_mask = self.collision_detector.detect(
            gripper_gg,
            two_fingers_gg,
            self.config["mesh_json_path"],
            self.gripper_meshes,
            min_grasp_width=self.config["min_width"],
            VoxelGrid=self.config["voxel_grid"],
            approach_dist=self.config["approach_dist"],
            collision_thresh=self.config["collision_thresh"],
            adjust_gripper_centers=self.config["adjust_gripper_centers"],
        )

        final_indices = np.where(empty_mask)[0]
        if len(final_indices) == 0:
            rospy.logwarn("No grasps left after collision detection")
            return []
        rospy.loginfo(f"{len(final_indices)} left after collision detection")

        gripper_gg_final:InspireHandRGraspGroup = coll_gg[final_indices]

        # 6. 选择最佳抓取并转换为ROS消息
        sorted_indices = np.argsort(gripper_gg_final.scores)[::-1]
        top_indices = sorted_indices[: min(self.max_grasps, len(sorted_indices))]

        grasp_poses = []
        for idx in top_indices:
            grasp = gripper_gg_final[idx]
            pose = self._convert_grasp_to_pose(grasp)
            grasp_poses.append(pose)

        rospy.loginfo(
            f"Successfully generated {len(grasp_poses)} collision-free grasps"
        )
        return grasp_poses

    def _convert_grasp_to_pose(self, grasp:Union[InspireHandRGrasp,Grasp]) -> GraspPose:
        """将抓取转换为ROS Pose消息"""
        pose = Pose()
        grasp_pose = GraspPose()

        # 位置
        pose.position = Point(
            x=float(grasp.translation[0]),
            y=float(grasp.translation[1]),
            z=float(grasp.translation[2]),
        )

        # 旋转矩阵转四元数
        rotation_matrix = grasp.rotation_matrix.reshape(3, 3)
        r = R.from_matrix(rotation_matrix)
        quat = r.as_quat()  # [x, y, z, w]

        pose.orientation = Quaternion(
            x=float(quat[0]), y=float(quat[1]), z=float(quat[2]), w=float(quat[3])
        )
        grasp_pose.pose = pose
        grasp_pose.score = grasp.score
        grasp_pose.angles = grasp.angle.tolist()
        return grasp_pose

    def run(self):
        """运行服务"""
        rospy.loginfo("Inspire Grasp Planning Service is running...")
        rospy.loginfo("Waiting for grasp planning requests...")
        rospy.spin()


if __name__ == "__main__":
    try:
        service = InspireGraspPlanningService()
        service.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Inspire Grasp Planning Service interrupted")
    except Exception as e:
        rospy.logerr(f"Failed to start service: {e}")
        sys.exit(1)
