"""
BoxBunny Deployment Launch File

This is the main deployment entry point for the boxing robot system.
It supports multiple detection modes and optional experimental features.

Usage:
    # Default (Color Tracking, no IMU)
    ros2 launch boxbunny_bringup boxbunny_deploy.launch.py

    # With Action Model (experimental)
    ros2 launch boxbunny_bringup boxbunny_deploy.launch.py detection_mode:=action

    # With IMU (experimental)
    ros2 launch boxbunny_bringup boxbunny_deploy.launch.py enable_imu:=true

    # Full experimental mode
    ros2 launch boxbunny_bringup boxbunny_deploy.launch.py detection_mode:=action enable_imu:=true

Detection Modes:
    - color: (Default) Uses color tracking for punch detection. Fast, reliable.
    - action: Uses YOLO Pose + Action Model for detection. More accurate but GPU-intensive.

Experimental Features:
    - IMU: Glove-mounted accelerometer for punch force measurement
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, SetEnvironmentVariable, GroupAction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch.conditions import IfCondition, UnlessCondition
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Launch arguments
    detection_mode = LaunchConfiguration("detection_mode")  # 'color' or 'action'
    enable_imu = LaunchConfiguration("enable_imu")
    enable_llm = LaunchConfiguration("enable_llm")
    headless = LaunchConfiguration("headless")

    # Paths
    ws_root = "/home/boxbunny/Desktop/doomsday_integration/boxing_robot_ws"
    action_script = os.path.join(ws_root, "action_prediction/tools/inference/live_infer_rgbd.py")
    gui_path = os.path.join(ws_root, "src/boxbunny_gui/boxbunny_gui/gui_main.py")
    
    # Models for Action mode
    ckpt_path = os.path.join(ws_root, "models/action_prediction_model/best_acc_82.4_epoch_161.pth")
    yolo_path = os.path.join(ws_root, "models/checkpoints/yolo26n.pt")
    config_path = os.path.join(ws_root, "action_prediction/configs/rgbd_boxing_anticipation.py")

    # Config files
    realsense_config = PathJoinSubstitution([FindPackageShare("boxbunny_bringup"), "config", "realsense.yaml"])
    glove_config = PathJoinSubstitution([FindPackageShare("boxbunny_vision"), "config", "glove_tracker.yaml"])
    fusion_config = PathJoinSubstitution([FindPackageShare("boxbunny_fusion"), "config", "fusion.yaml"])
    drill_config = PathJoinSubstitution([FindPackageShare("boxbunny_drills"), "config", "drill.yaml"])
    shadow_config = PathJoinSubstitution([FindPackageShare("boxbunny_drills"), "config", "drill_definitions.yaml"])
    imu_config = PathJoinSubstitution([FindPackageShare("boxbunny_imu"), "config", "imu.yaml"])
    analytics_config = PathJoinSubstitution([FindPackageShare("boxbunny_analytics"), "config", "analytics.yaml"])
    llm_config = PathJoinSubstitution([FindPackageShare("boxbunny_llm"), "config", "llm.yaml"])

    # Condition for color mode
    is_color_mode = PythonExpression(["'", detection_mode, "' == 'color'"])
    is_action_mode = PythonExpression(["'", detection_mode, "' == 'action'"])

    # =========================================================================
    # COLOR TRACKING MODE (Default)
    # Uses live_infer_rgbd.py for camera (same as unit tests) + Color-based glove tracker
    # =========================================================================
    
    # Camera process for color mode - uses live_infer_rgbd.py in pose-only mode (no action model)
    # This publishes camera frames that the glove tracker subscribes to
    color_camera_process = ExecuteProcess(
        condition=IfCondition(is_color_mode),
        cmd=[
            'python3', action_script,
            '--model-config', config_path,
            '--model-checkpoint', ckpt_path,
            '--yolo-model', yolo_path,
            '--fps', '30',
            '--rgb-res', '640x480',
            '--depth-res', '640x480',
            '--headless'
            # Note: No --enable-action-model, so it runs in pose-only mode
        ],
        cwd=os.path.join(ws_root, "action_prediction"),
        output='screen'
    )
    
    color_mode_nodes = GroupAction(
        condition=IfCondition(is_color_mode),
        actions=[
            # Color-based Glove Tracker (subscribes to camera topics from live_infer_rgbd.py)
            Node(
                package="boxbunny_vision",
                executable="realsense_glove_tracker",
                parameters=[glove_config],
                output="screen",
            ),
            # Punch Fusion (combines color tracking + optional IMU)
            Node(
                package="boxbunny_fusion",
                executable="punch_fusion_node",
                parameters=[fusion_config],
                output="screen",
            ),
        ]
    )

    # =========================================================================
    # ACTION MODEL MODE (Experimental)
    # Uses YOLO Pose + Action Prediction model
    # =========================================================================
    action_mode_process = ExecuteProcess(
        condition=IfCondition(is_action_mode),
        cmd=[
            'python3', action_script,
            '--model-config', config_path,
            '--model-checkpoint', ckpt_path,
            '--yolo-model', yolo_path,
            '--fps', '30',
            '--rgb-res', '640x480',
            '--depth-res', '640x480',
            '--headless',
            '--enable-action-model'  # Enable full action prediction
        ],
        cwd=os.path.join(ws_root, "action_prediction"),
        output='screen'
    )

    # =========================================================================
    # COMMON NODES (Both modes)
    # =========================================================================
    
    # Reaction Drill Manager
    drill_manager = Node(
        package="boxbunny_drills",
        executable="reaction_drill_manager",
        parameters=[drill_config],
        output="screen",
    )

    # Shadow Sparring Drill
    shadow_drill = Node(
        package="boxbunny_drills",
        executable="shadow_sparring_drill",
        parameters=[{"drill_config": shadow_config}],
        output="screen",
    )

    # Analytics
    analytics = Node(
        package="boxbunny_analytics",
        executable="punch_stats_node",
        parameters=[analytics_config],
        output="screen",
    )

    # =========================================================================
    # OPTIONAL: IMU (Experimental - disabled by default)
    # =========================================================================
    imu_driver = Node(
        package="boxbunny_imu",
        executable="mpu6050_node",
        parameters=[imu_config],
        output="screen",
        condition=IfCondition(enable_imu),
    )

    imu_classifier = Node(
        package="boxbunny_imu",
        executable="imu_punch_classifier",
        parameters=[imu_config],
        output="screen",
        condition=IfCondition(enable_imu),
    )

    # =========================================================================
    # OPTIONAL: LLM Coach (enabled by default)
    # =========================================================================
    llm_node = Node(
        package="boxbunny_llm",
        executable="trash_talk_node",
        parameters=[llm_config],
        output="screen",
        condition=IfCondition(enable_llm),
    )

    # =========================================================================
    # GUI (runs via conda environment with PySide6 - system Python has Qt issues)
    # =========================================================================
    gui_process = ExecuteProcess(
        condition=UnlessCondition(headless),
        cmd=[
            'conda', 'run', '-n', 'boxing_ai', '--no-capture-output',
            'python', gui_path
        ],
        output='screen'
    )

    return LaunchDescription([
        # Environment setup
        SetEnvironmentVariable(name='LD_PRELOAD', value='/usr/local/lib/librealsense2.so'),
        
        # Launch arguments
        DeclareLaunchArgument(
            "detection_mode",
            default_value="color",
            description="Detection mode: 'color' (default, fast) or 'action' (experimental, uses AI model)"
        ),
        DeclareLaunchArgument(
            "enable_imu",
            default_value="false",
            description="Enable IMU sensor (experimental)"
        ),
        DeclareLaunchArgument(
            "enable_llm",
            default_value="true",
            description="Enable LLM coach for trash talk"
        ),
        DeclareLaunchArgument(
            "headless",
            default_value="false",
            description="Run without GUI (for testing)"
        ),

        # Detection mode: Camera + detection nodes
        color_camera_process,  # Camera for color mode (uses live_infer_rgbd.py)
        color_mode_nodes,      # Glove tracker + fusion for color mode
        action_mode_process,   # Action mode (camera + action model combined)

        # Common nodes
        drill_manager,
        shadow_drill,
        analytics,

        # Optional experimental features
        imu_driver,
        imu_classifier,
        llm_node,

        # GUI
        gui_process,
    ])
