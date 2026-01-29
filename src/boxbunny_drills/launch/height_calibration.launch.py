"""
Height Calibration Launch File
Launches standalone height calibration GUI with pose estimation + depth.
"""

import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, SetEnvironmentVariable


def generate_launch_description():
    ws_root = "/home/boxbunny/Desktop/doomsday_integration/boxing_robot_ws"
    gui_path = os.path.join(ws_root, "src/boxbunny_vision/boxbunny_vision/height_calibration_gui.py")

    return LaunchDescription([
        # Set LD_PRELOAD for RealSense
        SetEnvironmentVariable(name='LD_PRELOAD', value='/usr/local/lib/librealsense2.so'),
        
        # Height Calibration GUI (Standalone - runs its own camera + pose)
        ExecuteProcess(
            cmd=['python3', gui_path],
            cwd=ws_root,
            output='screen'
        )
    ])
