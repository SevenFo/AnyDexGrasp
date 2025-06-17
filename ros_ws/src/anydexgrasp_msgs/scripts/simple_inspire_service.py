#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
简化版Inspire手抓取服务
用于快速测试和演示
"""

import rospy
from std_srvs.srv import Trigger, TriggerResponse
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
import random
import time


class SimpleInspireGraspService:
    """简化版的Inspire手抓取服务，用于演示"""

    def __init__(self):
        rospy.init_node("simple_inspire_grasp_service")

        # 创建服务
        self.grasp_service = rospy.Service(
            "/inspire_grasp_planning_service/plan_grasps",
            Trigger,
            self.handle_grasp_planning,
        )

        # 创建发布者
        self.grasp_poses_pub = rospy.Publisher(
            "/inspire_grasp_planning_service/grasp_poses", PoseStamped, queue_size=10
        )

        rospy.loginfo("Simple Inspire Grasp Service started!")
        rospy.loginfo("Service: /inspire_grasp_planning_service/plan_grasps")
        rospy.loginfo("Publisher: /inspire_grasp_planning_service/grasp_poses")

    def handle_grasp_planning(self, req):
        """处理抓取规划请求 - 简化版本"""
        try:
            rospy.loginfo("Received grasp planning request (simplified)")

            # 模拟处理时间
            time.sleep(1.0)

            # 生成一些示例抓取姿态
            num_grasps = random.randint(3, 8)
            grasp_poses = self.generate_sample_grasps(num_grasps)

            # 发布抓取姿态
            for i, pose in enumerate(grasp_poses):
                pose_stamped = PoseStamped()
                pose_stamped.header.stamp = rospy.Time.now()
                pose_stamped.header.frame_id = "camera_link"
                pose_stamped.pose = pose
                self.grasp_poses_pub.publish(pose_stamped)
                time.sleep(0.2)

            response = TriggerResponse()
            response.success = True
            response.message = f"Generated {num_grasps} sample grasp poses"

            rospy.loginfo(f"Successfully generated {num_grasps} sample grasps")
            return response

        except Exception as e:
            rospy.logerr(f"Error in grasp planning: {e}")
            response = TriggerResponse()
            response.success = False
            response.message = f"Error: {str(e)}"
            return response

    def generate_sample_grasps(self, num_grasps):
        """生成示例抓取姿态"""
        poses = []

        for i in range(num_grasps):
            pose = Pose()

            # 随机位置 (在相机前方0.5米左右)
            pose.position = Point(
                x=random.uniform(-0.1, 0.1),
                y=random.uniform(-0.1, 0.1),
                z=random.uniform(0.4, 0.6),
            )

            # 随机旋转 (简单的四元数)
            pose.orientation = Quaternion(
                x=random.uniform(-0.1, 0.1),
                y=random.uniform(-0.1, 0.1),
                z=random.uniform(-0.1, 0.1),
                w=random.uniform(0.9, 1.0),
            )

            poses.append(pose)

        return poses

    def run(self):
        """运行服务"""
        rospy.loginfo("Simple Inspire Grasp Service is running...")
        rospy.spin()


if __name__ == "__main__":
    try:
        service = SimpleInspireGraspService()
        service.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Service interrupted")
    except Exception as e:
        rospy.logerr(f"Failed to start service: {e}")
