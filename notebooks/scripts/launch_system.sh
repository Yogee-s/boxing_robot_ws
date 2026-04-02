#!/usr/bin/env bash
# =============================================================================
# BoxBunny Full System Launcher
# =============================================================================
# Launches everything needed for the complete robot system:
#   1. micro-ROS agent (outside conda, in a separate terminal)
#   2. RealSense camera driver
#   3. All BoxBunny ROS nodes (cv_node, imu_node, punch_processor, etc.)
#   4. GUI
#   5. IMU Simulator (dev mode only -- mirrors real hardware)
#
# Usage:
#   bash notebooks/scripts/launch_system.sh          # full mode (no simulator)
#   bash notebooks/scripts/launch_system.sh --dev    # dev mode (+ IMU simulator)
#
# Both modes launch the Teensy via micro-ROS. The --dev flag additionally
# opens the IMU simulator window which mirrors real pad strikes and lets
# you inject simulated ones.
#
# Press Ctrl+C or notebook STOP to shut down.
# =============================================================================
set +e

WS="/home/boxbunny/Desktop/doomsday_integration/boxing_robot_ws"
cd "$WS"

DEV_MODE=false
TEENSY_PORT="${TEENSY_PORT:-/dev/ttyACM0}"

if [[ "$1" == "--dev" ]]; then
    DEV_MODE=true
fi

cleanup() {
    echo ""
    echo "=== Stopping BoxBunny ==="
    pkill -f "micro_ros_agent.*serial" 2>/dev/null
    pkill -f "realsense2_camera" 2>/dev/null
    pkill -f "imu_simulator.py" 2>/dev/null
    kill -- -$LAUNCH_PID 2>/dev/null
    sleep 1
    kill -9 -- -$LAUNCH_PID 2>/dev/null
    pkill -9 -f 'imu_simulator.py' 2>/dev/null
    pkill -9 -f 'gui_main' 2>/dev/null
    pkill -9 -f 'ros2.launch' 2>/dev/null
    pkill -9 -f 'micro_ros_agent' 2>/dev/null
    fuser -k 8080/tcp 2>/dev/null
    echo "All processes stopped."
}
trap cleanup EXIT INT TERM

# ── Step 1: micro-ROS Agent (Teensy communication) ──────────────────────────
echo "=== Starting micro-ROS Agent ==="
echo "  Port: $TEENSY_PORT"

# Write agent launcher script (strips conda from PATH)
AGENT_SCRIPT="$WS/notebooks/scripts/_microros_agent.sh"
cat > "$AGENT_SCRIPT" << 'AGENTEOF'
#!/bin/bash
set +e
# Strip conda from PATH to avoid conflicts with micro-ROS
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v conda | tr '\n' ':')
unset CONDA_DEFAULT_ENV CONDA_PREFIX CONDA_EXE CONDA_PYTHON_EXE

source /opt/ros/humble/setup.bash
[ -f "$HOME/microros_ws/install/local_setup.bash" ] && source "$HOME/microros_ws/install/local_setup.bash"

TEENSY_PORT="${1:-/dev/ttyACM0}"
echo "micro-ROS agent starting on $TEENSY_PORT ..."
ros2 run micro_ros_agent micro_ros_agent serial --dev "$TEENSY_PORT" -b 115200
AGENTEOF
chmod +x "$AGENT_SCRIPT"

# Launch in separate terminal if available, else background
if command -v gnome-terminal &>/dev/null; then
    gnome-terminal --title="micro-ROS Agent" -- \
        bash "$AGENT_SCRIPT" "$TEENSY_PORT" &
else
    bash "$AGENT_SCRIPT" "$TEENSY_PORT" &
fi
sleep 3
echo "  micro-ROS agent started"
echo ""

# ── Step 2: RealSense Camera Driver ─────────────────────────────────────────
echo "=== Starting RealSense Camera ==="
ros2 launch realsense2_camera rs_launch.py \
    depth_module.depth_profile:=848x480x30 \
    rgb_camera.color_profile:=960x540x30 \
    align_depth.enable:=true \
    enable_gyro:=false \
    enable_accel:=false &
RS_PID=$!
sleep 3
echo "  RealSense started (PID: $RS_PID)"
echo ""

# ── Step 3: BoxBunny ROS Nodes + GUI ────────────────────────────────────────
source /opt/ros/humble/setup.bash
source "$WS/install/setup.bash"

echo "=== Launching BoxBunny ROS Nodes ==="
setsid ros2 launch boxbunny_core boxbunny_full.launch.py &
LAUNCH_PID=$!
sleep 5

# ── Step 4: IMU Simulator (dev mode only) ────────────────────────────────────
if [ "$DEV_MODE" = true ]; then
    echo ""
    echo "=== Starting IMU Simulator (mirrors real hardware) ==="
    python3 "$WS/tools/imu_simulator.py" &
    SIM_PID=$!
    echo "  IMU simulator PID: $SIM_PID"
fi

echo ""
echo "=== Active ROS Nodes ==="
ros2 node list 2>/dev/null || echo "(nodes still starting...)"
echo ""
echo "=== Active Topics ==="
ros2 topic list 2>/dev/null | head -25 || echo "(topics not ready)"
echo ""
if [ "$DEV_MODE" = true ]; then
    echo "=== RUNNING (DEV MODE + hardware) — Press STOP to shut down ==="
else
    echo "=== RUNNING (FULL MODE) — Press STOP to shut down ==="
fi

wait $LAUNCH_PID
