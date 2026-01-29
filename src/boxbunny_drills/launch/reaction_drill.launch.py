
import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, SetEnvironmentVariable
from launch_ros.actions import Node

def generate_launch_description():
    # Paths
    ws_root = "/home/boxbunny/Desktop/doomsday_integration/boxing_robot_ws"
    script_path = os.path.join(ws_root, "action_prediction/tools/inference/live_infer_rgbd.py")
    gui_path = os.path.join(ws_root, "src/boxbunny_vision/boxbunny_vision/vision_debug_gui.py")
    
    # Models
    ckpt_path = os.path.join(ws_root, "models/action_prediction_model/best_acc_82.4_epoch_161.pth")
    yolo_path = os.path.join(ws_root, "models/checkpoints/yolo26n.pt")
    config_path = os.path.join(ws_root, "action_prediction/configs/rgbd_boxing_anticipation.py")

    return LaunchDescription([
        # 1. Set LD_PRELOAD for RealSense
        SetEnvironmentVariable(name='LD_PRELOAD', value='/usr/local/lib/librealsense2.so'),
        
        # 2. Reaction Drill Manager
        Node(
            package='boxbunny_drills',
            executable='reaction_drill_manager',
            name='reaction_drill_manager',
            output='screen'
        ),
        
        # 3. Headless Pose AI (Action Prediction)
        ExecuteProcess(
            cmd=[
                'python3', script_path,
                '--model-config', config_path,
                '--model-checkpoint', ckpt_path,
                '--yolo-model', yolo_path,
                '--fps', '30',
                '--rgb-res', '640x480',
                '--depth-res', '640x480',
                '--headless'
                # Default is Pose Only. To enable action: '--enable-action-model'
            ],
            cwd=os.path.join(ws_root, "action_prediction"),
            output='screen'
        ),
        
        # 4. Vision Debug GUI (Mode: Action)
        ExecuteProcess(
            cmd=[
                'python3', gui_path,
                '--mode', 'action'
            ],
            output='screen'
        )
    ])
