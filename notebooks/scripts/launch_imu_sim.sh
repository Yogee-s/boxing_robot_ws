#!/usr/bin/env bash
# Launch the IMU Simulator paired with the Teensy via micro-ROS.
# Real pad strikes flash on the simulator; simulator punches execute on the robot.
set +e

WS="/home/boxbunny/Desktop/doomsday_integration/boxing_robot_ws"
cd "$WS"
source /opt/ros/humble/setup.bash && source install/setup.bash

cleanup() {
    echo ""
    echo "=== Stopping ==="
    [ -n "$SIM_PID" ] && kill $SIM_PID 2>/dev/null
    [ -n "$MICROROS_PID" ] && kill $MICROROS_PID 2>/dev/null
    sleep 0.5
    pkill -9 -f "imu_simulator.py" 2>/dev/null
    pkill -f "micro_ros_agent.*serial" 2>/dev/null
    echo "Done."
}
trap cleanup EXIT INT TERM

# Start micro-ROS agent (connects to Teensy)
source "$WS/notebooks/scripts/_start_microros.sh" "${TEENSY_PORT:-/dev/ttyACM0}"

echo ""
echo "Launching IMU Simulator (paired with Teensy)..."
echo "  Pad mapping: LEFT=prev, RIGHT=next, CENTRE=enter, HEAD=back"
echo "  Real pad strikes will flash on the simulator"
echo "  Simulator punch buttons will command the robot arms"
echo "  Close the window to stop."
echo ""

python3 tools/imu_simulator.py 2>&1
