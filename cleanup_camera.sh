#!/bin/bash
# cleanup_camera.sh - Aggressively kill conflicting nodes for boxing robot

echo "Killing conflicting camera nodes..."
# Try nice kill first
pkill -f realsense2_camera_node || true
pkill -f realsense_glove_tracker || true
pkill -f live_infer_rgbd.py || true
pkill -f reaction_drill_manager || true
pkill -f vision_debug_gui.py || true

# Kill launch processes
pkill -f "ros2 launch boxbunny_drills reaction_drill.launch.py" || true
pkill -f "ros2 launch boxbunny_main boxbunny_main.launch.py" || true

sleep 1

# Force kill if still running
echo "Force killing any survivors..."
pkill -9 -f realsense2_camera_node || true
pkill -9 -f realsense_glove_tracker || true
pkill -9 -f live_infer_rgbd.py || true
pkill -9 -f reaction_drill_manager || true
pkill -9 -f vision_debug_gui.py || true

echo "Checking for remaining processes..."
REMAINING=$(ps aux | grep -iE "realsense|live_infer|reaction_drill" | grep -v grep | grep -v cleanup_camera.sh)

if [ -n "$REMAINING" ]; then
    echo "WARNING: Some processes may still be running:"
    echo "$REMAINING"
    echo "You may need to kill key PIDs manually."
else
    echo "Cleanup complete. Camera should be free."
fi
