#!/bin/bash
# =============================================================================
# CV + IMU Fusion Test
# =============================================================================
# Launches the ORIGINAL run.py (Voxel Live Inference) and the ORIGINAL
# IMU Simulator side by side. Same exact GUIs you already know, just
# launched together with the Teensy hardware connected.
#
# Left window:  run.py — CV action prediction (Tkinter, no video, max FPS)
# Right window: IMU Simulator — pad strikes, Teensy live data, punch buttons
#
# Also launches: micro-ROS agent, V4 Arm Control GUI, imu_node, punch_processor.
#
# Requires: conda boxing_ai, Teensy connected, calibration (A2), RealSense.
# =============================================================================
set +e

WS="/home/boxbunny/Desktop/doomsday_integration/boxing_robot_ws"
cd "$WS"

TEENSY_PORT="${1:-/dev/ttyACM0}"
DATA_DIR="$WS/Boxing_Arm_Control/ros2_ws/unified_v4/data"
export DISPLAY="${DISPLAY:-:0}"

# ── Check calibration ───────────────────────────────────────────────────────
echo "=== Checking Calibration Status ==="
[ -f "$DATA_DIR/arm_config.yaml" ] && echo "  Cal: OK" || echo "  Cal: MISSING"
([ -f "$DATA_DIR/strike_library.json" ] || [ -f "$DATA_DIR/strike_library_V1.json" ]) && echo "  Lib: OK" || echo "  Lib: MISSING"
[ -f "$DATA_DIR/ros_slots.json" ] && echo "  ROS: OK" || echo "  ROS: MISSING"
echo ""

# ── Cleanup ─────────────────────────────────────────────────────────────────
cleanup() {
    echo ""
    echo "=== Stopping all processes ==="
    # Kill by name — covers all windows including gnome-terminal children
    pkill -f "run_with_ros" 2>/dev/null
    pkill -f "fusion_monitor" 2>/dev/null
    pkill -f "imu_simulator.py" 2>/dev/null
    pkill -f "run.py" 2>/dev/null
    pkill -f "live_voxelflow" 2>/dev/null
    pkill -f "unified_GUI_V4.py" 2>/dev/null
    pkill -f "micro_ros_agent.*serial" 2>/dev/null
    pkill -f "imu_node" 2>/dev/null
    pkill -f "punch_processor" 2>/dev/null
    pkill -f "cv_node" 2>/dev/null
    pkill -f "_v4_gui_launcher" 2>/dev/null
    # Kill entire process group
    kill -- -$$ 2>/dev/null
    sleep 1
    # Force kill anything still alive
    pkill -9 -f "run_with_ros" 2>/dev/null
    pkill -9 -f "run.py" 2>/dev/null
    pkill -9 -f "live_voxelflow" 2>/dev/null
    pkill -9 -f "imu_simulator" 2>/dev/null
    pkill -9 -f "fusion_monitor" 2>/dev/null
    pkill -9 -f "unified_GUI_V4" 2>/dev/null
    pkill -9 -f "micro_ros_agent" 2>/dev/null
    pkill -9 -f "imu_node" 2>/dev/null
    pkill -9 -f "punch_processor" 2>/dev/null
    kill -9 -- -$$ 2>/dev/null
    # Release camera if still held
    fuser -k /dev/video* 2>/dev/null
    echo "Done."
}
trap cleanup EXIT INT TERM

# ── Step 1: micro-ROS agent ─────────────────────────────────────────────────
source "$WS/notebooks/scripts/_start_microros.sh" "$TEENSY_PORT"

# ── Step 2: V4 Arm Control GUI ──────────────────────────────────────────────
echo ""
echo "=== Starting V4 Arm Control GUI ==="
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

# ── Step 3: Activate conda (for run_with_ros.py which needs torch) ──────────
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate boxing_ai 2>/dev/null || true
source /opt/ros/humble/setup.bash
source "$WS/install/setup.bash"

# ── Step 4: ROS nodes ───────────────────────────────────────────────────────
echo ""
echo "=== Starting ROS Nodes ==="

ros2 run boxbunny_core imu_node &
echo "  imu_node started"
ros2 run boxbunny_core punch_processor &
echo "  punch_processor started"
sleep 3

# Switch imu_node to TRAINING mode (publish 3 times to ensure delivery)
echo "  Switching IMU to TRAINING mode..."
for i in 1 2 3; do
    ros2 topic pub --once /boxbunny/session/state boxbunny_msgs/msg/SessionState \
        "{state: 'active', mode: 'training', username: 'test'}" 2>/dev/null
    sleep 0.5
done
echo "  TRAINING mode set"

# ── Step 5: IMU Simulator ───────────────────────────────────────────────────
echo ""
echo "=== Starting IMU Simulator ==="
python3 "$WS/tools/imu_simulator.py" &
SIM_PID=$!
echo "  IMU Simulator PID: $SIM_PID"
sleep 2

# ── Step 6: Fusion Monitor ──────────────────────────────────────────────────
echo ""
echo "=== Starting Fusion Monitor ==="
python3 "$WS/notebooks/scripts/fusion_monitor.py" &
FUSION_PID=$!
echo "  Fusion Monitor PID: $FUSION_PID"
sleep 1

# ── Step 7: CV Inference (run.py + ROS publisher) ───────────────────────────
echo ""
echo "=== Starting CV Inference (with ROS publishing) ==="
echo "  Publishes predictions to /boxbunny/cv/detection"
echo "  punch_processor fuses with IMU events"
echo "  Close the CV window to stop everything."
echo ""

cd "$WS/action_prediction"
python3 "$WS/notebooks/scripts/run_with_ros.py" 2>&1

echo ""
echo "=== Test Complete ==="
