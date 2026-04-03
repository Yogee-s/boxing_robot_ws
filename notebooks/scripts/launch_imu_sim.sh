#!/usr/bin/env bash
# =============================================================================
# IMU Simulator + V4 Arm Control GUI + micro-ROS Agent
# =============================================================================
# Launches all three components needed for the arms to respond to simulator
# punch commands:
#   1. micro-ROS agent (Teensy serial bridge)
#   2. V4 Arm Control GUI (motor control + strike execution) — outside conda
#   3. IMU Simulator (punch buttons, pad mirroring)
#
# The V4 GUI must have all 3 config files to auto-activate ROS Control:
#   - data/arm_config.yaml
#   - data/strike_library.json (or strike_library_V1.json)
#   - data/ros_slots.json
#
# Run calibration first (cell 3b) if any are missing.
# =============================================================================
set +e

WS="/home/boxbunny/Desktop/doomsday_integration/boxing_robot_ws"
cd "$WS"

TEENSY_PORT="${TEENSY_PORT:-/dev/ttyACM0}"
DATA_DIR="$WS/Boxing_Arm_Control/ros2_ws/unified_v4/data"
export DISPLAY="${DISPLAY:-:0}"

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
    echo "The V4 GUI will not accept ROS commands without all 3 config files."
    exit 1
fi

echo ""
echo "All configs present — V4 GUI will auto-activate ROS Control on startup."
echo ""

cleanup() {
    echo ""
    echo "=== Stopping ==="
    [ -n "$SIM_PID" ] && kill $SIM_PID 2>/dev/null
    [ -n "$V4_PID" ] && kill $V4_PID 2>/dev/null
    pkill -f "micro_ros_agent.*serial" 2>/dev/null
    kill -- -$$ 2>/dev/null
    sleep 0.5
    pkill -9 -f "imu_simulator.py" 2>/dev/null
    pkill -9 -f "unified_GUI_V4.py" 2>/dev/null
    pkill -9 -f "micro_ros_agent" 2>/dev/null
    kill -9 -- -$$ 2>/dev/null
    echo "Done."
}
trap cleanup EXIT INT TERM

# ── Step 1: micro-ROS agent ─────────────────────────────────────────────────
source "$WS/notebooks/scripts/_start_microros.sh" "$TEENSY_PORT"

# ── Step 2: V4 Arm Control GUI (outside conda) ──────────────────────────────
echo ""
echo "=== Starting V4 Arm Control GUI (outside conda) ==="

# Write a launcher that strips conda and runs the V4 GUI
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

# ── Step 3: IMU Simulator ───────────────────────────────────────────────────
echo ""
echo "=== Starting IMU Simulator ==="
source /opt/ros/humble/setup.bash
source "$WS/install/setup.bash"

echo ""
echo "  Punch buttons on the simulator will:"
echo "    → Send /robot/strike_command to the V4 GUI"
echo "    → V4 GUI executes the FSM strike (windup → apex → snap-back)"
echo "    → Motors move via micro-ROS"
echo ""
echo "  Real pad strikes on the robot flash on the simulator automatically."
echo "  Close the simulator window to stop."
echo ""

python3 "$WS/tools/imu_simulator.py" 2>&1
