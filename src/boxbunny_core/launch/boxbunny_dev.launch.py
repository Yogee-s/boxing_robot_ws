"""Launch BoxBunny in development mode with IMU simulator."""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    ws_root = Path(__file__).resolve().parents[3]
    config_dir = ws_root / "config"
    launch_dir = Path(__file__).parent

    return LaunchDescription([
        # IMU Simulator instead of real hardware
        ExecuteProcess(
            cmd=["python3", str(ws_root / "tools" / "imu_simulator.py")],
            name="imu_simulator",
            output="screen",
        ),

        # Core nodes (same as full, but CV may use CPU fallback)
        Node(
            package="boxbunny_core",
            executable="imu_node",
            name="imu_node",
            output="screen",
        ),
        Node(
            package="boxbunny_core",
            executable="cv_node",
            name="cv_node",
            output="screen",
            parameters=[{"device": "cpu", "inference_interval": 3}],
        ),
        Node(
            package="boxbunny_core",
            executable="punch_processor",
            name="punch_processor",
            output="screen",
        ),
        Node(
            package="boxbunny_core",
            executable="session_manager",
            name="session_manager",
            output="screen",
        ),
        Node(
            package="boxbunny_core",
            executable="drill_manager",
            name="drill_manager",
            output="screen",
        ),
        Node(
            package="boxbunny_core",
            executable="llm_node",
            name="llm_node",
            output="screen",
            parameters=[{
                "model_path": str(ws_root / "models" / "llm" / "qwen2.5-3b-instruct-q4_k_m.gguf"),
                "fallback_tips_path": str(config_dir / "fallback_tips.json"),
            }],
        ),
        # GUI
        Node(
            package="boxbunny_gui",
            executable="gui_main",
            name="boxbunny_gui",
            output="screen",
        ),
    ])
