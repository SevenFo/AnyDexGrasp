# msg/GraspPose.msg  
# 抓取姿态消息定义
std_msgs/Header header             # 时间戳和坐标系
geometry_msgs/Pose pose           # 抓取姿态 (位置+旋转)
float32 score                     # 抓取分数
float32 width                     # 抓取宽度
float32 depth                     # 抓取深度
int32 grasp_type                  # 抓取类型
string gripper_name               # 机械手名称
int32[] angles