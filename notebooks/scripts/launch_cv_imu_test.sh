#!/bin/bash
# launch_cv_imu_test.sh -- Launches CV + IMU fusion test
#
# Spawns a separate terminal for micro-ROS agent + IMU bridge (no conda),
# then runs the CV fusion test in the current shell (conda).
#
# Usage:  bash notebooks/scripts/launch_cv_imu_test.sh [TEENSY_PORT] [BAUD]
set +e

WS="/home/boxbunny/Desktop/doomsday_integration/boxing_robot_ws"
cd "$WS"

TEENSY_PORT="${1:-/dev/ttyACM0}"
TEENSY_BAUD="${2:-115200}"

# ── Cleanup on exit ──
SPAWNED_PIDS=""
cleanup() {
    echo ""
    echo "=== Stopping CV + IMU Fusion Test ==="
    for pid in $SPAWNED_PIDS; do
        kill "$pid" 2>/dev/null
    done
    sleep 1
    for pid in $SPAWNED_PIDS; do
        kill -9 "$pid" 2>/dev/null
    done
    pkill -f "imu_udp_bridge" 2>/dev/null
    pkill -f "micro_ros_agent.*serial" 2>/dev/null
    echo "All processes stopped."
}
trap cleanup EXIT INT TERM

# ── Terminal 1: micro-ROS agent + IMU bridge (NO conda) ──
echo "=== Starting micro-ROS agent + IMU bridge (no conda) ==="

IMU_SCRIPT=$(mktemp /tmp/boxbunny_imu_XXXX.sh)
cat > "$IMU_SCRIPT" << INNEREOF
#!/bin/bash
set +e
# Strip conda from PATH
export PATH=\$(echo "\$PATH" | tr ':' '\n' | grep -v conda | tr '\n' ':')
unset CONDA_DEFAULT_ENV CONDA_PREFIX CONDA_EXE CONDA_PYTHON_EXE

source /opt/ros/humble/setup.bash
[ -f "\$HOME/microros_ws/install/local_setup.bash" ] && source "\$HOME/microros_ws/install/local_setup.bash"
[ -f "$WS/install/setup.bash" ] && source "$WS/install/setup.bash"

echo "[IMU] micro-ROS agent on $TEENSY_PORT @ $TEENSY_BAUD"
ros2 run micro_ros_agent micro_ros_agent serial --dev "$TEENSY_PORT" -b "$TEENSY_BAUD" &
AGENT_PID=\$!
sleep 3

echo "[IMU] Starting UDP bridge..."
python3 "$WS/notebooks/scripts/imu_udp_bridge.py" &
BRIDGE_PID=\$!

echo "[IMU] Running (agent=\$AGENT_PID, bridge=\$BRIDGE_PID). Close this window to stop."
wait \$BRIDGE_PID
kill \$AGENT_PID 2>/dev/null
INNEREOF
chmod +x "$IMU_SCRIPT"

if command -v gnome-terminal &>/dev/null; then
    gnome-terminal --title="BoxBunny IMU Bridge" -- bash "$IMU_SCRIPT" &
elif command -v xterm &>/dev/null; then
    xterm -title "BoxBunny IMU Bridge" -e bash "$IMU_SCRIPT" &
else
    echo "No terminal emulator found. Running IMU bridge in background..."
    bash "$IMU_SCRIPT" &
fi
SPAWNED_PIDS="$!"
sleep 4

# ── Main: CV inference with IMU fusion (conda) ──
echo ""
echo "=== Starting CV + IMU Fusion Test ==="
echo "Press 'q' in the display window to quit."
echo ""

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate boxing_ai 2>/dev/null || true

python3 notebooks/scripts/cv_imu_fusion_test.py \
    --checkpoint action_prediction/model/best_model.pth \
    --pose-weights action_prediction/model/yolo26n-pose.pt \
    --show-video \
    2>&1

echo ""
echo "=== Test Complete ==="
