#!/usr/bin/env bash
# =============================================================================
# BoxBunny Full System Launcher
# =============================================================================
# Launches everything needed for the complete robot system:
#   1. micro-ROS agent (Teensy serial bridge)
#   2. V4 Arm Control GUI (motor arming, calibration, IMU strike detection)
#   3. RealSense camera driver
#   4. All BoxBunny ROS nodes (cv_node, imu_node, punch_processor, etc.)
#   5. BoxBunny GUI
#   6. IMU Simulator (dev mode only — mirrors real hardware)
#
# Usage:
#   bash notebooks/scripts/launch_system.sh          # full mode (no simulator)
#   bash notebooks/scripts/launch_system.sh --dev    # dev mode (+ IMU simulator)
#
# Requires calibration (cell 3b) — checks config files before starting.
# Press Ctrl+C or notebook STOP to shut down.
# =============================================================================
set +e

WS="/home/boxbunny/Desktop/doomsday_integration/boxing_robot_ws"
cd "$WS"

DEV_MODE=false
TEENSY_PORT="${TEENSY_PORT:-/dev/ttyACM0}"
DATA_DIR="$WS/Boxing_Arm_Control/ros2_ws/unified_v4/data"
export DISPLAY="${DISPLAY:-:0}"

if [[ "$1" == "--dev" ]]; then
    DEV_MODE=true
fi

# ── Check calibration files ─────────────────────────────────────────────────
echo "=== Checking Calibration Status ==="
CAL_OK=true

if [ -f "$DATA_DIR/arm_config.yaml" ]; then
    echo "  Cal:   OK"
else
    echo "  Cal:   MISSING — run cell 3b (Arm Calibration) first"
    CAL_OK=false
fi

if [ -f "$DATA_DIR/strike_library.json" ] || [ -f "$DATA_DIR/strike_library_V1.json" ]; then
    echo "  Lib:   OK"
else
    echo "  Lib:   MISSING — run cell 3b (Arm Calibration) first"
    CAL_OK=false
fi

if [ -f "$DATA_DIR/ros_slots.json" ]; then
    echo "  ROS:   OK"
else
    echo "  ROS:   MISSING — run cell 3b (Arm Calibration) first"
    CAL_OK=false
fi

if [ "$CAL_OK" = false ]; then
    echo ""
    echo "ERROR: Calibration incomplete. Run cell 3b first."
    exit 1
fi

echo ""
echo "All configs present — V4 GUI will auto-activate ROS Control on startup."
echo ""

cleanup() {
    echo ""
    echo "=== Stopping BoxBunny ==="
    pkill -f "micro_ros_agent.*serial" 2>/dev/null
    pkill -f "unified_GUI_V4.py" 2>/dev/null
    pkill -f "realsense2_camera" 2>/dev/null
    pkill -f "imu_simulator.py" 2>/dev/null
    kill -- -$LAUNCH_PID 2>/dev/null
    sleep 1
    kill -9 -- -$LAUNCH_PID 2>/dev/null
    pkill -9 -f 'imu_simulator.py' 2>/dev/null
    pkill -9 -f 'gui_main' 2>/dev/null
    pkill -9 -f 'ros2.launch' 2>/dev/null
    pkill -9 -f 'micro_ros_agent' 2>/dev/null
    pkill -9 -f 'unified_GUI_V4' 2>/dev/null
    fuser -k 8080/tcp 2>/dev/null
    echo "All processes stopped."
}
trap cleanup EXIT INT TERM

# ── Step 1: micro-ROS agent ─────────────────────────────────────────────────
source "$WS/notebooks/scripts/_start_microros.sh" "$TEENSY_PORT"

# ── Step 2: V4 Arm Control GUI (outside conda) ──────────────────────────────
echo ""
echo "=== Starting V4 Arm Control GUI (outside conda) ==="

_V4_SCRIPT="$WS/notebooks/scripts/_v4_gui_launcher.sh"
cat > "$_V4_SCRIPT" << 'V4EOF'
#!/bin/bash
set +e
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v conda | tr '\n' ':')
unset CONDA_DEFAULT_ENV CONDA_PREFIX CONDA_EXE CONDA_PYTHON_EXE
export DISPLAY="${DISPLAY:-:0}"

source /opt/ros/humble/setup.bash
[ -f "$HOME/microros_ws/install/local_setup.bash" ] && source "$HOME/microros_ws/install/local_setup.bash"

WS="/home/boxbunny/Desktop/doomsday_integration/boxing_robot_ws"
cd "$WS/Boxing_Arm_Control/ros2_ws/unified_v4"
python3 unified_GUI_V4.py
V4EOF
chmod +x "$_V4_SCRIPT"

if command -v gnome-terminal &>/dev/null; then
    gnome-terminal --title="V4 Arm Control GUI" -- bash "$_V4_SCRIPT" &
else
    bash "$_V4_SCRIPT" &
fi
V4_PID=$!
echo "  V4 GUI launching (PID: $V4_PID)"
echo "  Waiting for GUI to load and auto-activate ROS Control..."
sleep 5

# ── Step 3: RealSense Camera Driver ─────────────────────────────────────────
echo ""
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

# ── Step 4: BoxBunny ROS Nodes + GUI ────────────────────────────────────────
echo ""
source /opt/ros/humble/setup.bash
source "$WS/install/setup.bash"

echo "=== Launching BoxBunny ROS Nodes ==="
setsid ros2 launch boxbunny_core boxbunny_full.launch.py &
LAUNCH_PID=$!
sleep 5

# ── Step 5: IMU Simulator (dev mode only) ────────────────────────────────────
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
    echo "=== RUNNING (DEV MODE) — Press STOP to shut down ==="
else
    echo "=== RUNNING (FULL MODE) — Press STOP to shut down ==="
fi

wait $LAUNCH_PID
