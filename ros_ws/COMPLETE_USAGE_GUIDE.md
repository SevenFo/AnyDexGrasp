# AnyDexGrasp ROS1 包完整使用指南

## 📁 包结构

```
ros_ws/
├── build_package.sh              # 自动构建脚本
└── src/
    └── anydexgrasp_msgs/
        ├── package.xml           # ROS包元数据
        ├── CMakeLists.txt        # 构建配置
        ├── README.md             # 包说明文档
        ├── msg/
        │   └── GraspPose.msg     # 抓取姿态消息定义
        ├── srv/
        │   └── GraspPlanning.srv # 抓取规划服务定义
        ├── scripts/
        │   ├── anydexgrasp_config.py      # 路径配置管理
        │   ├── start_service.py           # 完整功能服务
        │   ├── simple_inspire_service.py  # 简化演示服务
        │   └── test_service_client.py     # 测试客户端
        ├── launch/
        │   └── inspire_grasp_service.launch # 启动文件
        └── rviz/
            └── grasp_planning.rviz         # RViz可视化配置
```

## 🚀 快速开始

### 1. 编译包
```bash
cd /home/ps/Projects/AnyDexGrasp/ros_ws
./build_package.sh
```

### 2. 启动简化演示服务
```bash
# 终端1: 启动ROS核心
roscore

# 终端2: 启动简化服务
source devel/setup.bash
roslaunch anydexgrasp_msgs inspire_grasp_service.launch simple_mode:=true

# 终端3: 测试服务
source devel/setup.bash
rosrun anydexgrasp_msgs test_service_client.py
```

### 3. 启动完整功能服务（需要训练好的模型）
```bash
# 确保模型文件存在
export ANYDEXGRASP_ROOT="/home/ps/Projects/AnyDexGrasp"

# 启动完整服务
roslaunch anydexgrasp_msgs inspire_grasp_service.launch simple_mode:=false
```

## 📋 详细功能说明

### 自定义消息类型

**GraspPose.msg**
```
std_msgs/Header header      # 时间戳和坐标系
geometry_msgs/Pose pose     # 抓取姿态 (位置+旋转)
float32 score              # 抓取分数
float32 width              # 抓取宽度
float32 depth              # 抓取深度
int32 grasp_type           # 抓取类型
string gripper_name        # 机械手名称
```

**GraspPlanning.srv**
```
# 请求
sensor_msgs/PointCloud2 pointcloud  # 输入点云
float32 score_threshold            # 分数阈值
int32 max_grasps                   # 最大抓取数量
---
# 响应
bool success                       # 是否成功
string message                     # 响应消息
GraspPose[] grasp_poses           # 抓取姿态数组
```

### 脚本功能

1. **start_service.py** - 完整功能服务
   - 使用训练好的深度学习模型
   - 完整的点云处理和抓取预测
   - 碰撞检测和姿态优化

2. **simple_inspire_service.py** - 简化演示服务
   - 不需要训练好的模型
   - 生成随机示例抓取姿态
   - 用于快速测试和演示

3. **test_service_client.py** - 测试客户端
   - 调用抓取规划服务
   - 验证服务功能

4. **anydexgrasp_config.py** - 配置管理
   - 自动设置项目路径
   - 处理环境变量

## 🔧 配置和参数

### 环境变量
```bash
export ANYDEXGRASP_ROOT="/home/ps/Projects/AnyDexGrasp"  # 项目根目录
export ROS_WORKSPACE="/home/ps/Projects/AnyDexGrasp/ros_ws"  # ROS工作空间
```

### Launch文件参数
```bash
# 模型文件路径
checkpoint_path:="path/to/checkpoint.tar"

# 最大抓取数量
max_grasps:=10

# 分数阈值
score_threshold:=0.85

# 是否使用GraspNet v2
use_graspnet_v2:=false

# 是否使用简化模式
simple_mode:=true
```

### 话题和服务

**服务:**
- `/inspire_grasp_planning_service/plan_grasps` - 抓取规划

**发布话题:**
- `/inspire_grasp_planning_service/grasp_poses` - 抓取姿态

**订阅话题:**
- `/camera/depth/color/points` - 输入点云

## 🔍 调试和故障排除

### 常见问题

1. **编译错误**
   ```bash
   # 检查依赖
   rosdep check anydexgrasp_msgs
   
   # 安装缺失依赖
   rosdep install --from-paths src --ignore-src -r -y
   ```

2. **模型文件找不到**
   ```bash
   # 检查路径
   ls /home/ps/Projects/AnyDexGrasp/logs/model/inspire_model/
   
   # 设置环境变量
   export ANYDEXGRASP_ROOT="/path/to/your/anydexgrasp"
   ```

3. **Python导入错误**
   ```bash
   # 确保路径正确设置
   export PYTHONPATH="${PYTHONPATH}:/home/ps/Projects/AnyDexGrasp"
   ```

### 日志和调试

**启用详细日志:**
```bash
export ROSCONSOLE_CONFIG_FILE=/path/to/custom_rosconsole.conf
```

**检查服务状态:**
```bash
# 查看可用服务
rosservice list | grep grasp

# 查看服务类型
rosservice type /inspire_grasp_planning_service/plan_grasps

# 手动调用服务
rosservice call /inspire_grasp_planning_service/plan_grasps "{}"
```

**查看话题:**
```bash
# 监控话题
rostopic echo /inspire_grasp_planning_service/grasp_poses

# 查看话题信息
rostopic info /camera/depth/color/points
```

## 📊 性能和优化

### 推荐硬件配置
- GPU: NVIDIA RTX 3060 或更高
- 内存: 16GB 或更多
- CUDA: 11.0 或更高版本

### 性能调优
1. **GPU加速**: 确保CUDA正确安装
2. **内存优化**: 调整batch size
3. **点云降采样**: 减少输入点云大小

## 🎯 使用示例

### Python客户端示例
```python
#!/usr/bin/env python
import rospy
from std_srvs.srv import Trigger

def call_grasp_service():
    rospy.init_node('grasp_client')
    rospy.wait_for_service('/inspire_grasp_planning_service/plan_grasps')
    
    try:
        plan_grasp = rospy.ServiceProxy('/inspire_grasp_planning_service/plan_grasps', Trigger)
        response = plan_grasp()
        
        if response.success:
            print(f"Success: {response.message}")
        else:
            print(f"Failed: {response.message}")
            
    except rospy.ServiceException as e:
        print(f"Service call failed: {e}")

if __name__ == '__main__':
    call_grasp_service()
```

### 命令行使用
```bash
# 调用服务
rosservice call /inspire_grasp_planning_service/plan_grasps "{}"

# 监控抓取姿态
rostopic echo /inspire_grasp_planning_service/grasp_poses
```

## 📝 开发和扩展

### 添加新的机械手支持
1. 在 `configs.py` 中添加新配置
2. 创建对应的服务脚本
3. 更新launch文件
4. 添加相应的测试

### 自定义消息类型
1. 在 `msg/` 目录添加新的 `.msg` 文件
2. 在 `CMakeLists.txt` 中注册
3. 重新编译包

## 📄 许可证和贡献

请遵循AnyDexGrasp项目的许可证要求。欢迎提交issue和pull request。

## 📞 支持

如有问题，请：
1. 查看本文档的故障排除部分
2. 在AnyDexGrasp项目仓库提交issue
3. 检查ROS日志: `rqt_console`
