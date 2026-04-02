#!/usr/bin/env bash
# Launch BoxBunny GUI + IMU Simulator + Teensy (micro-ROS agent).
# Real pad strikes flash on the simulator; simulator punches execute on robot.
# Press Ctrl+C or notebook STOP to close everything.
set +e

WS="/home/boxbunny/Desktop/doomsday_integration/boxing_robot_ws"
cd "$WS"
source /opt/ros/humble/setup.bash && source install/setup.bash

export QT_QPA_PLATFORM=xcb
export QT_QPA_PLATFORM_PLUGIN_PATH=$(python3 -c "import PySide6; print(PySide6.__path__[0])")/Qt/plugins/platforms
unset QT_PLUGIN_PATH

GUI_PID=""
IMU_PID=""

cleanup() {
    echo ""
    echo "=== Closing all windows ==="
    [ -n "$GUI_PID" ] && kill $GUI_PID 2>/dev/null
    [ -n "$IMU_PID" ] && kill $IMU_PID 2>/dev/null
    [ -n "$MICROROS_PID" ] && kill $MICROROS_PID 2>/dev/null
    sleep 0.5
    [ -n "$GUI_PID" ] && kill -9 $GUI_PID 2>/dev/null
    [ -n "$IMU_PID" ] && kill -9 $IMU_PID 2>/dev/null
    pkill -f "micro_ros_agent.*serial" 2>/dev/null
    echo "All windows closed."
}
trap cleanup EXIT INT TERM

# Start micro-ROS agent (connects to Teensy)
source "$WS/notebooks/scripts/_start_microros.sh" "${TEENSY_PORT:-/dev/ttyACM0}"

echo ""
echo "Launching BoxBunny GUI + IMU Simulator (paired with Teensy)..."
echo "  Click IMU pads -> navigate GUI & command robot arms"
echo "  Real pad strikes -> flash on simulator"
echo "  Press STOP (interrupt) to close both."
echo ""

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
echo "GUI started (PID=$GUI_PID)"

sleep 2

python3 tools/imu_simulator.py 2>&1 &
IMU_PID=$!
echo "IMU Simulator started (PID=$IMU_PID)"

echo ""
echo "=== All windows running — Press STOP to close ==="

wait -n $GUI_PID $IMU_PID 2>/dev/null || wait $GUI_PID
