#!/usr/bin/env bash
# Run the CV action prediction model with live camera feed.
# Runs run.py from inside the action_prediction folder so all
# default paths resolve correctly. Uses conda boxing_ai for PyTorch.
set +e

WS="/home/boxbunny/Desktop/doomsday_integration/boxing_robot_ws"
export DISPLAY="${DISPLAY:-:0}"

# Kill all children on exit (ensures camera is released)
cleanup() {
    kill -- -$$ 2>/dev/null
    sleep 0.5
    kill -9 -- -$$ 2>/dev/null
}
trap cleanup EXIT INT TERM

# Activate conda for torch/ultralytics/pyrealsense2
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate boxing_ai 2>/dev/null || true

echo "=== CV Model Live Test ==="
echo "Stand 1.5-2m from camera."
echo "Close the window or press Ctrl+C to stop."
echo ""

# Run from inside action_prediction/ so default paths work
cd "$WS/action_prediction"
python3 run.py \
    2>&1 || echo "(D435i camera not connected or models missing)"
