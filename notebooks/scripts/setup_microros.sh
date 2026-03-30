#!/bin/bash
# setup_microros.sh -- Check / install micro-ROS agent (run outside conda)
set +e

# Strip conda from environment
unset CONDA_DEFAULT_ENV CONDA_PREFIX CONDA_EXE
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v conda | tr '\n' ':')
source /opt/ros/humble/setup.bash

MROS_WS="$HOME/microros_ws"

echo "=== Checking micro-ROS agent ==="

# Check system install
if ros2 pkg list 2>/dev/null | grep -q micro_ros_agent; then
    echo "micro-ROS agent found (system)"
    ros2 run micro_ros_agent micro_ros_agent --help 2>&1 | head -3
    exit 0
fi

# Check local workspace
if [ -f "$MROS_WS/install/local_setup.bash" ]; then
    source "$MROS_WS/install/local_setup.bash"
    if ros2 pkg list 2>/dev/null | grep -q micro_ros_agent; then
        echo "micro-ROS agent found ($MROS_WS)"
        ros2 run micro_ros_agent micro_ros_agent --help 2>&1 | head -3
        exit 0
    fi
    echo "Workspace exists but agent not built. Rebuilding..."
fi

echo "micro-ROS agent not found. Building from source..."
echo "(This takes ~1-2 minutes on first run)"
echo ""

rosdep update 2>&1 | tail -1

mkdir -p "$MROS_WS/src"
cd "$MROS_WS"

if [ ! -d src/micro_ros_setup ]; then
    git clone -b humble \
        https://github.com/micro-ROS/micro_ros_setup.git \
        src/micro_ros_setup
fi

colcon build --symlink-install --packages-select micro_ros_setup 2>&1 | tail -3
source install/local_setup.bash

ros2 run micro_ros_setup create_agent_ws.sh 2>&1 | tail -5
ros2 run micro_ros_setup build_agent.sh 2>&1 | tail -5
source install/local_setup.bash

echo ""
if ros2 run micro_ros_agent micro_ros_agent --help 2>&1 | head -1 | grep -q Usage; then
    echo "=== micro-ROS agent installed successfully ==="
else
    echo "=== BUILD FAILED — check logs in $MROS_WS/log/ ==="
    exit 1
fi
