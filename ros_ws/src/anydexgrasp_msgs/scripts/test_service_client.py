#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
增强版的ROS Service客户端测试脚本
用于测试inspire手抓取规划服务
- 从深度话题获取深度图像
- 使用相机内参转换为点云
- 将点云变换到base_link坐标系
- 发送变换后的点云给服务端
"""

import rospy
import numpy as np
from std_srvs.srv import Trigger, TriggerResponse
from sensor_msgs.msg import PointCloud2, CameraInfo, Image
from geometry_msgs.msg import Point, PoseStamped, Pose
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge
import tf2_ros
import tf2_geometry_msgs
import tf2_sensor_msgs
import sys
import cv2
import open3d as o3d

from inference_v3.utils.robot_utils import GalaxeaRobot

# 尝试导入自定义服务消息
try:
    from anydexgrasp_msgs.srv import GraspPlanning, GraspPlanningRequest, GraspPlanningResponse
    CUSTOM_MSGS_AVAILABLE = True
except ImportError:
    rospy.logwarn("Custom message types not found. Using standard Trigger service.")
    CUSTOM_MSGS_AVAILABLE = False

class DepthToPointCloudConverter:
    """将深度图像转换为点云的辅助类"""
    
    def __init__(self):
        self.bridge = CvBridge()
        self.camera_info = None
        self.camera_matrix = None
        self.distortion_model = None
        self.distortion_params = None
        
    def set_camera_info(self, camera_info_msg):
        """设置相机内参"""
        self.camera_info = camera_info_msg
        self.camera_matrix = np.array(camera_info_msg.K).reshape(3, 3)
        self.distortion_model = camera_info_msg.distortion_model
        self.distortion_params = np.array(camera_info_msg.D)
        rospy.loginfo("Camera info received and processed")

    def depth_to_pointcloud(self, depth_image_msg,rgb_image_msg):
        """将深度图像转换为点云"""
        if self.camera_matrix is None:
            rospy.logerr("Camera info not available!")
            return None, None
            
        # 将ROS图像消息转换为OpenCV格式
        depth_image = self.bridge.imgmsg_to_cv2(depth_image_msg, desired_encoding="passthrough")
        cv2.imwrite("/data/shiqi/AnyDexGrasp/depth.png", depth_image)
        rgb_image = self.bridge.imgmsg_to_cv2(rgb_image_msg, desired_encoding="passthrough")
        cv2.imwrite("/data/shiqi/AnyDexGrasp/rgb.png", rgb_image)
        

        # 获取图像尺寸
        height, width = depth_image.shape
        
        # 生成点云
        points = []
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]
        
        # 遍历每个像素生成3D点
        for v in range(height):
            for u in range(width):
                depth = depth_image[v, u]
                if depth > 0:  # 忽略无效深度
                    # 转换为3D坐标
                    x = (u - cx) * depth / fx
                    y = (v - cy) * depth / fy
                    z = depth
                    points.append([x, y, z])
        
        rospy.loginfo(f"Generated {len(points)} points from depth image")
        
        # 创建PointCloud2消息
        header = depth_image_msg.header
        pc2_msg = pc2.create_cloud_xyz32(header, points)
        return pc2_msg, np.array(points)

class GraspPlanningClient:
    """抓取规划客户端"""
    
    def __init__(self):
        rospy.init_node("enhanced_grasp_planning_client")
        
        # 参数配置
        self.depth_topic = rospy.get_param("~depth_topic", "/hdas/camera_head/depth/depth_registered")
        self.rgb_topic = rospy.get_param("~rgb_topic", "/hdas/camera_head/rgb/image_rect_color")
        self.pointcloud_topic = rospy.get_param("~pointcloud_topic", "/hdas/camera_head/point_cloud/cloud_registered")
        self.camera_info_topic = rospy.get_param("~camera_info_topic", "/hdas/camera_head/depth/camera_info")
        self.target_frame = rospy.get_param("~target_frame", "base_link_frd")
        self.source_frame = rospy.get_param("~source_frame", "zed_frame")
        self.service_name = rospy.get_param("~service_name", "/inspire_grasp_planning_service/plan_grasps")
        self.grasp_poses_pub = rospy.Publisher(
            "/move_base_simple/goal", PoseStamped, queue_size=10
        )
        # 初始化组件
        self.converter = DepthToPointCloudConverter()
        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(secs=10))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.robot = GalaxeaRobot()
        
        self.camera_info:CameraInfo = CameraInfo()
        # 获取相机内参
        rospy.loginfo(f"Waiting for camera info on {self.camera_info_topic}")
        camera_info = rospy.wait_for_message(self.camera_info_topic, CameraInfo, timeout=5.0)
        self.converter.set_camera_info(camera_info)
        
        # 等待服务可用
        rospy.loginfo(f"Waiting for service {self.service_name}...")
        rospy.wait_for_service(self.service_name)
        
        # 创建服务代理
        if CUSTOM_MSGS_AVAILABLE:
            self.grasp_service = rospy.ServiceProxy(self.service_name, GraspPlanning)
        else:
            self.grasp_service = rospy.ServiceProxy(self.service_name, Trigger)
        
        rospy.loginfo("Service connected")
    
    def get_transform(self, target_frame, source_frame):
        """获取坐标变换"""
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame, 
                source_frame, 
                rospy.Time(0), 
                rospy.Duration(secs=5)
            )
            return transform
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, 
                tf2_ros.ExtrapolationException) as e:
            rospy.logerr(f"TF lookup failed: {e}")
            return None
    
    def transform_pointcloud(self, pointcloud, transform):
        """变换点云坐标系"""
        try:
            transformed_pc = tf2_sensor_msgs.do_transform_cloud(pointcloud, transform)
            rospy.loginfo(f"Pointcloud transformed from {pointcloud.header.frame_id} to {self.target_frame}")
            return transformed_pc
        except Exception as e:
            rospy.logerr(f"Pointcloud transformation failed: {e}")
            return None
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
        x_min, x_max, y_min, y_max = (0.57,  1.0,-0.2,0.2)
        z_min, z_max = (-0.6-1, -0.6)

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
            # 创建Open3D点云对象并保存
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            
            # 保存原始点云
            error_path = '/data/shiqi/AnyDexGrasp/error_points.ply'
            o3d.io.write_point_cloud(error_path, pcd)
            
            rospy.logwarn(f"x_min:{min(points[:,0])}, x_max:{max(points[:,0])}, "
                        f"y_min:{min(points[:,1])}, y_max:{max(points[:,1])}, "
                        f"z_min:{min(points[:,2])}, z_max:{max(points[:,2])}")
            raise ValueError("No points remain after workspace filtering")

        # 可选：将结果转换为Open3D点云对象
        filtered_pcd = o3d.geometry.PointCloud()
        filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
        blue_colors = np.full((len(filtered_points), 3), [0.0, 0.0, 1.0], dtype=np.float32)
        filtered_pcd.colors = o3d.utility.Vector3dVector(blue_colors)

        # 可选：保存过滤后的点云
        o3d.io.write_point_cloud('/data/shiqi/AnyDexGrasp/filtered.ply', filtered_pcd)
        return filtered_points
    def transform_pose(self, pose:PoseStamped, transform) -> PoseStamped:
        try:
            transformed_pose = tf2_geometry_msgs.do_transform_pose(pose, transform)
            rospy.loginfo(f"Pose transformed from {pose.header.frame_id} to {self.target_frame}")
            return transformed_pose
        except Exception as e:
            rospy.logerr(f"Pointcloud transformation failed: {e}")
            return None
    def run(self):
        """主运行逻辑"""
        # rospy.loginfo(f"Waiting for depth on {self.depth_topic}")
        depth = rospy.wait_for_message(self.depth_topic, Image, timeout=5)
        rgb = rospy.wait_for_message(self.rgb_topic, Image, timeout=5)
        # _ = self.converter.depth_to_pointcloud(depth,rgb)
        # if point is None or pointcloud is None:
        #     rospy.logerr("Get PointCloud Error")
        #     return 
        # np.savetxt("/data/shiqi/AnyDexGrasp/point_from_depth.txt", point)
        rospy.loginfo(f"Waiting for pointcloud on {self.pointcloud_topic}...")
        pointcloud:PointCloud2 = rospy.wait_for_message(self.pointcloud_topic, PointCloud2, timeout=5.0)
        rospy.loginfo(f"Received pointcloud with {len(pointcloud.data)} Bytes points")
        
        # 获取坐标变换
        rospy.loginfo(f"Looking up transform from {self.source_frame} to {self.target_frame}...")
        transform = self.get_transform(self.target_frame, self.source_frame)
        if transform is None:
            rospy.logerr("Transform not available, using original pointcloud")
        else:
            # 变换点云坐标系
            pointcloud = self.transform_pointcloud(pointcloud, transform)
            self._extract_points_from_pointcloud2(pointcloud_msg=pointcloud)
            if pointcloud is None:
                rospy.logerr("Pointcloud transformation failed, using original")
        
        # 调用抓取规划服务
        rospy.loginfo("Calling grasp planning service...")
        try:
            if CUSTOM_MSGS_AVAILABLE:
                # 使用自定义服务消息
                request = GraspPlanningRequest()
                request.pointcloud = pointcloud
                response = self.grasp_service(request)
            else:
                # 使用Trigger服务（服务端会自行获取点云）
                response = self.grasp_service()
            
            if response.success:
                rospy.loginfo(f"Service call successful: {response.message}")
                # rospy.loginfo(f"Trans grasp result to base_link...")
                # rospy.loginfo(f"Looking up transform from {self.source_frame} to {self.target_frame}...")
                # transform = self.get_transform(self.target_frame, self.source_frame)
                # if transform is None:
                #     rospy.logerr("Transform not available, using original pointcloud")
                # else:
                #     # 变换点云坐标系
                #     for grasp in response.grasp_poses:
                #         pose_stamped = PoseStamped()
                #         pose_stamped.header = grasp.header
                #         pose_stamped.pose=  grasp.pose
                #         pose_base_link = self.transform_pose(pose_stamped, transform)
                #         if pose_base_link is None:
                #             rospy.logerr("Pose transformation failed, using original")
                #             continue
                #         grasp.pose = pose_base_link

                if CUSTOM_MSGS_AVAILABLE:
                    rospy.loginfo(f"Received {len(response.grasp_poses)} grasp poses")
                    for i, grasp in enumerate(response.grasp_poses):
                        # 打印每个抓取姿态的详细信息
                        rospy.loginfo(f"\nGrasp {i+1}:")
                        rospy.loginfo(f"  Header: [frame_id: {grasp.header.frame_id}, timestamp: {grasp.header.stamp}]")
                        rospy.loginfo(f"  Score: {grasp.score:.3f}")
                        rospy.loginfo(f"  Position: [x: {grasp.pose.position.x:.3f}, y: {grasp.pose.position.y:.3f}, z: {grasp.pose.position.z:.3f}]")
                        rospy.loginfo(f"  Orientation: [x: {grasp.pose.orientation.x:.3f}, y: {grasp.pose.orientation.y:.3f}, z: {grasp.pose.orientation.z:.3f}, w: {grasp.pose.orientation.w:.3f}]")
                        rospy.loginfo(f"  Width: {grasp.width:.3f}m")
                        rospy.loginfo(f"  Depth: {grasp.depth:.3f}m")
                        rospy.loginfo(f"  Grasp Type: {grasp.grasp_type}")
                        rospy.loginfo(f"  Gripper: {grasp.gripper_name}")
                        rospy.loginfo(f"  Joint Angles: {list(grasp.angles)}")
                        pose_stamped = PoseStamped()
                        pose_stamped.pose = grasp.pose
                        pose_stamped.header.frame_id = grasp.header.frame_id
                        pose_stamped.header.stamp = rospy.Time.now()
                        trans = self.get_transform("torso_link4", "base_link_frd")
                        pose_stamped = self.transform_pose(pose_stamped, trans)
                        pose = pose_stamped.pose
                        rospy.loginfo(f"  TransPosition: [x: {pose.position.x:.3f}, y: {pose.position.y:.3f}, z: {pose.position.z:.3f}]")
                        rospy.loginfo(f"  TransPosition: [x: {pose.orientation.x:.3f}, y: {pose.orientation.y:.3f}, z: {pose.orientation.z:.3f}], w: {pose.orientation.w:.3f}]")
                        confirm = input("Press Y to exected")
                        self.grasp_poses_pub.publish(pose_stamped)
                        rospy.sleep(0.1)
                        if confirm.lower() == 'y':
                            import pdb
                            pdb.set_trace()
                            self.robot.move_ee(pose)
                        break
            else:
                rospy.logerr(f"Service call failed: {response.message}")
                
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")

if __name__ == "__main__":
    try:
        client = GraspPlanningClient()
        client.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Client interrupted")
    except Exception as e:
        rospy.logerr(f"Client failed: {e}")
        sys.exit(1)
