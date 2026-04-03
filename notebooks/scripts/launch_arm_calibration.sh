#!/usr/bin/env bash
# =============================================================================
# Boxing Arm Calibration Launcher
# =============================================================================
# Launches the micro-ROS agent and the V4 Arm Control GUI for calibration.
# Both must run OUTSIDE conda (system ROS 2 + PyQt5).
#
# This is a one-time setup step. Once calibration is complete, the data is
# saved to data/ and auto-loaded on subsequent launches:
#   - arm_config.yaml       (motor offsets, direction signs, pitch limits)
#   - strike_library.json   (strike windup/apex waypoints)
#   - ros_slots.json        (ROS slot assignments for front-end commands)
#
# Usage:  bash notebooks/scripts/launch_arm_calibration.sh [TEENSY_PORT]
# =============================================================================
set +e

WS="/home/boxbunny/Desktop/doomsday_integration/boxing_robot_ws"
TEENSY_PORT="${1:-/dev/ttyACM0}"
export DISPLAY="${DISPLAY:-:0}"

# Strip conda from PATH — both micro-ROS agent and V4 GUI need system Python
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v conda | tr '\n' ':')
unset CONDA_DEFAULT_ENV CONDA_PREFIX CONDA_EXE CONDA_PYTHON_EXE

source /opt/ros/humble/setup.bash
[ -f "$HOME/microros_ws/install/local_setup.bash" ] && source "$HOME/microros_ws/install/local_setup.bash"
[ -f "$WS/install/setup.bash" ] && source "$WS/install/setup.bash"

cleanup() {
    echo ""
    echo "=== Stopping Calibration ==="
    [ -n "$AGENT_PID" ] && kill $AGENT_PID 2>/dev/null
    sleep 0.5
    pkill -f "micro_ros_agent.*serial" 2>/dev/null
    echo "Done."
}
trap cleanup EXIT INT TERM

# ── Check Teensy ─────────────────────────────────────────────────────────────
if [ ! -e "$TEENSY_PORT" ]; then
    echo "WARNING: Teensy not found at $TEENSY_PORT"
    echo "The GUI will start but motors won't respond until Teensy is connected."
    echo ""
fi

# ── Start micro-ROS agent ───────────────────────────────────────────────────
echo "=== Starting micro-ROS Agent on $TEENSY_PORT ==="
ros2 run micro_ros_agent micro_ros_agent serial --dev "$TEENSY_PORT" -b 115200 &
AGENT_PID=$!
sleep 3

# ── Check data directory status ─────────────────────────────────────────────
DATA_DIR="$WS/Boxing_Arm_Control/ros2_ws/unified_v4/data"
echo ""
echo "=== Current Calibration Status ==="
[ -f "$DATA_DIR/arm_config.yaml" ]      && echo "  Cal:   FOUND" || echo "  Cal:   MISSING (need to calibrate)"
[ -f "$DATA_DIR/strike_library.json" -o -f "$DATA_DIR/strike_library_V1.json" ] \
                                         && echo "  Lib:   FOUND" || echo "  Lib:   MISSING (need to create strikes)"
[ -f "$DATA_DIR/ros_slots.json" ]        && echo "  ROS:   FOUND" || echo "  ROS:   MISSING (need to assign slots)"
echo ""

# ── Launch V4 Arm Control GUI ───────────────────────────────────────────────
echo "=== Starting Arm Control GUI (V4) ==="
echo ""
echo "  Calibration Workflow:"
echo "    1. Calibration & Twin tab  →  Zero All Here  →  Run Pitch Scan"
echo "    2. Strike Library tab      →  Place windup/apex  →  Save"
echo "    3. ROS Control tab         →  Assign slots 1-6  →  Save Slots"
echo ""
echo "  When all three are saved, ROS Control auto-activates on next launch."
echo "  Close the GUI window when done."
echo ""

cd "$WS/Boxing_Arm_Control/ros2_ws/unified_v4"
python3 unified_GUI_V4.py 2>&1

echo ""
echo "=== Calibration GUI Closed ==="
echo ""
echo "Post-calibration status:"
[ -f "$DATA_DIR/arm_config.yaml" ]      && echo "  Cal:   OK" || echo "  Cal:   MISSING"
[ -f "$DATA_DIR/strike_library.json" ]  && echo "  Lib:   OK" || echo "  Lib:   MISSING"
[ -f "$DATA_DIR/ros_slots.json" ]        && echo "  ROS:   OK" || echo "  ROS:   MISSING"
