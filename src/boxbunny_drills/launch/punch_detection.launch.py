
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    # Paths
    ws_root = "/home/boxbunny/Desktop/doomsday_integration/boxing_robot_ws"
    gui_path = os.path.join(ws_root, "src/boxbunny_vision/boxbunny_vision/vision_debug_gui.py")
    
    bringup_share = get_package_share_directory('boxbunny_bringup')
    
    return LaunchDescription([
        # 1. Set LD_PRELOAD for RealSense (Critical for correct version loading)
        SetEnvironmentVariable(name='LD_PRELOAD', value='/usr/local/lib/librealsense2.so'),
        
        # 2. Vision Only (Color Tracking / Glove Detection)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(bringup_share, 'launch', 'vision_only.launch.py')
            )
        ),
        
        # 3. IMU System (Classifier + Calib GUI)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(bringup_share, 'launch', 'imu_only.launch.py')
            )
        ),
        
        # 4. Vision Debug GUI (Mode: Color)
        ExecuteProcess(
            cmd=[
                'python3', gui_path,
                '--mode', 'color'
            ],
            output='screen'
        )
    ])
