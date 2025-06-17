#!/bin/bash

# AnyDexGrasp ROS 包构建脚本

echo "=== AnyDexGrasp ROS Package Build Script ==="

# 检查是否在正确的目录
if [ ! -f "src/anydexgrasp_msgs/package.xml" ]; then
    echo "Error: Please run this script from the workspace root (ros_ws)"
    echo "Expected structure: ros_ws/src/anydexgrasp_msgs/"
    exit 1
fi

echo "✓ Workspace structure verified"

# 检查ROS环境
if [ -z "$ROS_DISTRO" ]; then
    echo "Error: ROS environment not sourced"
    echo "Please run: source /opt/ros/noetic/setup.bash"
    exit 1
fi

echo "✓ ROS $ROS_DISTRO environment loaded"

# 检查依赖
echo "Checking dependencies..."
MISSING_DEPS=""

for dep in "geometry_msgs" "sensor_msgs" "std_msgs"; do
    if ! rospack find $dep >/dev/null 2>&1; then
        MISSING_DEPS="$MISSING_DEPS $dep"
    fi
done

if [ -n "$MISSING_DEPS" ]; then
    echo "Error: Missing ROS dependencies:$MISSING_DEPS"
    echo "Please install them using: sudo apt-get install ros-$ROS_DISTRO-<package-name>"
    exit 1
fi

echo "✓ All dependencies found"

# 清理之前的构建
echo "Cleaning previous build..."
if [ -d "build" ]; then
    rm -rf build/
fi
if [ -d "devel" ]; then
    rm -rf devel/
fi

# 构建包
echo "Building anydexgrasp_msgs package..."
catkin_make

if [ $? -eq 0 ]; then
    echo "✓ Build successful!"
    
    # 源化环境
    echo "Sourcing workspace setup..."
    source devel/setup.bash
    
    # 验证包
    echo "Verifying package installation..."
    if rospack find anydexgrasp_msgs >/dev/null 2>&1; then
        echo "✓ Package anydexgrasp_msgs found"
        
        # 检查消息和服务
        if rosmsg show anydexgrasp_msgs/GraspPose >/dev/null 2>&1; then
            echo "✓ GraspPose message available"
        else
            echo "⚠ Warning: GraspPose message not found"
        fi
        
        if rossrv show anydexgrasp_msgs/GraspPlanning >/dev/null 2>&1; then
            echo "✓ GraspPlanning service available" 
        else
            echo "⚠ Warning: GraspPlanning service not found"
        fi
        
        echo ""
        echo "=== Build Complete ==="
        echo "To use the package:"
        echo "1. Source the workspace: source devel/setup.bash"
        echo "2. Run simple service: roslaunch anydexgrasp_msgs inspire_grasp_service.launch simple_mode:=true"
        echo "3. Test service: rosrun anydexgrasp_msgs test_service_client.py"
        
    else
        echo "✗ Error: Package not found after build"
        exit 1
    fi
else
    echo "✗ Build failed!"
    exit 1
fi
