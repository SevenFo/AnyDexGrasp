import rospy
import tf
from geometry_msgs.msg import PoseStamped,Pose

class GalaxeaRobot:
    def __init__(self) -> None:
        if not rospy.core.is_initialized():
            rospy.init_node('GlaaxaeaRobot') # 初始化ROS节点
        self.left_arm_ee_pub = rospy.Publisher('/motion_target/target_pose_arm_left', PoseStamped, queue_size=1,latch=True)
        self.right_arm_ee_pub = rospy.Publisher('/motion_target/target_pose_arm_right', PoseStamped, queue_size=1,latch=True)


    def move_ee(self, pose: Pose):
        for _ in range(10):
            target_pose = PoseStamped()
            target_pose.pose = pose
            target_pose.header.stamp = rospy.Time.now()
            self.right_arm_ee_pub.publish(target_pose)
