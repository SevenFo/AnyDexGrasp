#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
简化版的ROS Service客户端测试脚本
用于测试inspire手抓取规划服务
"""

import rospy
from std_srvs.srv import Trigger


def test_service():
    """测试抓取规划服务"""
    rospy.init_node("test_grasp_service")

    # 等待服务可用
    service_name = "/inspire_grasp_planning_service/plan_grasps"
    rospy.loginfo(f"Waiting for service {service_name}...")
    rospy.wait_for_service(service_name)

    try:
        # 创建服务代理
        plan_grasp = rospy.ServiceProxy(service_name, Trigger)

        rospy.loginfo("Calling grasp planning service...")
        response = plan_grasp()

        if response.success:
            rospy.loginfo(f"Service call successful: {response.message}")
        else:
            rospy.logerr(f"Service call failed: {response.message}")

    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")


if __name__ == "__main__":
    try:
        test_service()
    except rospy.ROSInterruptException:
        pass
