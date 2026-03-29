"""Launch BoxBunny ROS nodes without GUI (for headless testing)."""

from pathlib import Path

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    ws_root = Path(__file__).resolve().parents[3]

    return LaunchDescription([
        Node(package="boxbunny_core", executable="imu_node", name="imu_node", output="screen"),
        Node(package="boxbunny_core", executable="cv_node", name="cv_node", output="screen",
             parameters=[{"device": "cpu"}]),
        Node(package="boxbunny_core", executable="punch_processor", name="punch_processor",
             output="screen"),
        Node(package="boxbunny_core", executable="session_manager", name="session_manager",
             output="screen"),
        Node(package="boxbunny_core", executable="drill_manager", name="drill_manager",
             output="screen"),
        Node(package="boxbunny_core", executable="analytics_node", name="analytics_node",
             output="screen"),
        Node(package="boxbunny_core", executable="llm_node", name="llm_node", output="screen",
             parameters=[{
                 "model_path": str(ws_root / "models" / "llm" / "qwen2.5-3b-instruct-q4_k_m.gguf"),
                 "fallback_tips_path": str(ws_root / "config" / "fallback_tips.json"),
             }]),
    ])
