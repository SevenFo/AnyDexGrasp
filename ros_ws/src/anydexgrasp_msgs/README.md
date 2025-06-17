# anydexgrasp_msgs

ROS package for AnyDexGrasp custom message definitions and services.

## 包含内容

### 消息类型 (msg/)
- `GraspPose.msg` - 抓取姿态消息定义

### 服务类型 (srv/)
- `GraspPlanning.srv` - 抓取规划服务定义

### 脚本 (scripts/)
- `start_service.py` - 完整功能的Inspire手抓取规划服务
- `simple_inspire_service.py` - 简化演示版服务
- `test_service_client.py` - 测试客户端

### Launch文件 (launch/)
- `inspire_grasp_service.launch` - 启动抓取规划服务

### RViz配置 (rviz/)
- `grasp_planning.rviz` - 用于可视化的RViz配置

## 编译和安装

```bash
# 进入工作空间
cd /home/ps/Projects/AnyDexGrasp/ros_ws

# 编译
catkin_make

# 或使用catkin_tools
catkin build anydexgrasp_msgs

# 添加到环境变量
source devel/setup.bash
```

## 使用方法

### 1. 启动简化演示服务
```bash
roslaunch anydexgrasp_msgs inspire_grasp_service.launch simple_mode:=true
```

### 2. 启动完整功能服务
```bash
roslaunch anydexgrasp_msgs inspire_grasp_service.launch simple_mode:=false
```

### 3. 测试服务
```bash
# 在另一个终端
rosrun anydexgrasp_msgs test_service_client.py
```

### 4. 使用命令行调用服务
```bash
# 调用抓取规划服务
rosservice call /inspire_grasp_planning_service/plan_grasps "{}"
```

## 依赖

- ROS Noetic
- std_msgs
- geometry_msgs
- sensor_msgs
- message_generation
- message_runtime

## 话题和服务

### 服务
- `/inspire_grasp_planning_service/plan_grasps` - 抓取规划服务

### 发布的话题
- `/inspire_grasp_planning_service/grasp_poses` - 抓取姿态

### 订阅的话题
- `/camera/depth/color/points` - 输入点云

## 注意事项

1. 完整功能服务需要训练好的模型文件
2. 确保AnyDexGrasp项目的其他依赖已正确安装
3. 简化服务仅用于演示，生成的是随机抓取姿态
