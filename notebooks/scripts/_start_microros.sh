#!/bin/bash
# Shared helper: start micro-ROS agent in the background (outside conda).
# Source this or call it before launching ROS nodes that need the Teensy.
#
# Usage: source _start_microros.sh [PORT]
#   or:  bash _start_microros.sh [PORT]
#
# Sets MICROROS_PID if launched successfully.
set +e

TEENSY_PORT="${1:-/dev/ttyACM0}"
WS="/home/boxbunny/Desktop/doomsday_integration/boxing_robot_ws"

# Check if Teensy is plugged in
if [ ! -e "$TEENSY_PORT" ]; then
    echo "[micro-ROS] Teensy not found at $TEENSY_PORT — skipping agent"
    export MICROROS_PID=""
    return 0 2>/dev/null || exit 0
fi

# Check if agent is already running
if pgrep -f "micro_ros_agent.*serial" > /dev/null 2>&1; then
    echo "[micro-ROS] Agent already running"
    export MICROROS_PID=$(pgrep -f "micro_ros_agent.*serial" | head -1)
    return 0 2>/dev/null || exit 0
fi

echo "[micro-ROS] Starting agent on $TEENSY_PORT ..."

# Write temp launcher that strips conda
_AGENT_TMP="$WS/notebooks/scripts/_microros_agent.sh"
cat > "$_AGENT_TMP" << 'EOF'
#!/bin/bash
set +e
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v conda | tr '\n' ':')
unset CONDA_DEFAULT_ENV CONDA_PREFIX CONDA_EXE CONDA_PYTHON_EXE
source /opt/ros/humble/setup.bash
[ -f "$HOME/microros_ws/install/local_setup.bash" ] && source "$HOME/microros_ws/install/local_setup.bash"
ros2 run micro_ros_agent micro_ros_agent serial --dev "$1" -b 115200
EOF
chmod +x "$_AGENT_TMP"

bash "$_AGENT_TMP" "$TEENSY_PORT" &
export MICROROS_PID=$!
sleep 2
echo "[micro-ROS] Agent PID: $MICROROS_PID"
