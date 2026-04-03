#!/usr/bin/env bash
# =============================================================================
# BoxBunny GUI + IMU Simulator + V4 Arm Control GUI + micro-ROS + ROS Nodes
# =============================================================================
# Launches the full dev stack:
#   1. micro-ROS agent (Teensy serial bridge)
#   2. V4 Arm Control GUI (motor control, IMU strike detection)
#   3. Core ROS nodes (imu_node, punch_processor, session_manager, etc.)
#   4. BoxBunny touchscreen GUI
#   5. IMU Simulator (mirrors real hardware, commands robot arms)
#
# Pad strikes flow: Teensy → V4 GUI → /robot/strike_detected → imu_node →
# /boxbunny/imu/pad/impact → punch_processor → /boxbunny/punch/confirmed →
# BoxBunny GUI (punch counter updates) + IMU Simulator (pad flashes)
#
# Requires calibration (cell 3b) — checks config files before starting.
# Press Ctrl+C or notebook STOP to close everything.
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
    exit 1
fi

echo ""
echo "All configs present."
echo ""

source /opt/ros/humble/setup.bash
source "$WS/install/setup.bash"

cleanup() {
    echo ""
    echo "=== Closing everything ==="
    pkill -f "micro_ros_agent.*serial" 2>/dev/null
    pkill -f "unified_GUI_V4.py" 2>/dev/null
    pkill -f "imu_simulator.py" 2>/dev/null
    # Kill the ROS nodes we started
    [ -n "$IMU_NODE_PID" ] && kill $IMU_NODE_PID 2>/dev/null
    [ -n "$PUNCH_PID" ] && kill $PUNCH_PID 2>/dev/null
    [ -n "$SESSION_PID" ] && kill $SESSION_PID 2>/dev/null
    [ -n "$DRILL_PID" ] && kill $DRILL_PID 2>/dev/null
    [ -n "$SPARRING_PID" ] && kill $SPARRING_PID 2>/dev/null
    [ -n "$ROBOT_PID" ] && kill $ROBOT_PID 2>/dev/null
    [ -n "$GUI_PID" ] && kill $GUI_PID 2>/dev/null
    [ -n "$SIM_PID" ] && kill $SIM_PID 2>/dev/null
    sleep 0.5
    pkill -9 -f "imu_node" 2>/dev/null
    pkill -9 -f "punch_processor" 2>/dev/null
    pkill -9 -f "session_manager" 2>/dev/null
    pkill -9 -f "gui_main" 2>/dev/null
    pkill -9 -f "imu_simulator" 2>/dev/null
    pkill -9 -f "unified_GUI_V4" 2>/dev/null
    pkill -9 -f "micro_ros_agent" 2>/dev/null
    echo "Done."
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
echo "  V4 GUI launching..."
sleep 5

# ── Step 3: Core BoxBunny ROS nodes ─────────────────────────────────────────
echo ""
echo "=== Starting BoxBunny ROS Nodes ==="

ros2 run boxbunny_core imu_node &
IMU_NODE_PID=$!
echo "  imu_node PID: $IMU_NODE_PID"

ros2 run boxbunny_core punch_processor &
PUNCH_PID=$!
echo "  punch_processor PID: $PUNCH_PID"

ros2 run boxbunny_core session_manager &
SESSION_PID=$!
echo "  session_manager PID: $SESSION_PID"

ros2 run boxbunny_core drill_manager &
DRILL_PID=$!
echo "  drill_manager PID: $DRILL_PID"

ros2 run boxbunny_core sparring_engine &
SPARRING_PID=$!
echo "  sparring_engine PID: $SPARRING_PID"

ros2 run boxbunny_core robot_node &
ROBOT_PID=$!
echo "  robot_node PID: $ROBOT_PID"

sleep 2

# ── Step 4: BoxBunny GUI ────────────────────────────────────────────────────
echo ""
echo "=== Starting BoxBunny GUI ==="

export QT_QPA_PLATFORM=xcb
export QT_QPA_PLATFORM_PLUGIN_PATH=$(python3 -c "import PySide6; print(PySide6.__path__[0])")/Qt/plugins/platforms
unset QT_PLUGIN_PATH

python3 -c "
import sys, os, signal
sys.path.insert(0, 'src/boxbunny_gui')
os.environ.pop('QT_PLUGIN_PATH', None)
from boxbunny_gui.app import BoxBunnyApp
app = BoxBunnyApp()
def _shutdown(sig, frame):
    try: app._window.close()
    except: pass
    sys.exit(0)
signal.signal(signal.SIGINT, _shutdown)
signal.signal(signal.SIGTERM, _shutdown)
app.run()
" 2>&1 &
GUI_PID=$!
echo "  BoxBunny GUI PID: $GUI_PID"

sleep 2

# ── Step 5: IMU Simulator ───────────────────────────────────────────────────
echo ""
echo "=== Starting IMU Simulator ==="
echo "  Punch buttons → command robot arms via V4 GUI"
echo "  Real pad strikes → flash on simulator + process through BoxBunny nodes"
echo ""

python3 "$WS/tools/imu_simulator.py" 2>&1 &
SIM_PID=$!
echo "  IMU Simulator PID: $SIM_PID"

echo ""
echo "=== All running — Press STOP to close ==="

# Wait for GUI to exit (primary window)
wait $GUI_PID
