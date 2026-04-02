"""Launch all BoxBunny nodes + GUI + dashboard."""

import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    ws_root = Path(__file__).resolve().parents[3]
    config_dir = ws_root / "config"

    return LaunchDescription([
        # --- Core Processing Nodes ---
        Node(
            package="boxbunny_core",
            executable="imu_node",
            name="imu_node",
            output="screen",
            parameters=[{
                "nav_debounce_ms": 300,
                "nav_global_debounce_ms": 200,
                "mode_transition_ms": 200,
            }],
        ),
        Node(
            package="boxbunny_core",
            executable="cv_node",
            name="cv_node",
            output="screen",
            parameters=[{
                "checkpoint_path": str(ws_root / "action_prediction" / "model" / "best_model.pth"),
                "yolo_model_path": str(ws_root / "action_prediction" / "model" / "yolo26n-pose.pt"),
                "device": "cuda:0",
                "inference_interval": 1,
            }],
        ),
        Node(
            package="boxbunny_core",
            executable="robot_node",
            name="robot_node",
            output="screen",
            parameters=[{
                "punch_sequences_dir": str(ws_root / "data" / "punch_sequences"),
                "heartbeat_hz": 10.0,
            }],
        ),
        Node(
            package="boxbunny_core",
            executable="punch_processor",
            name="punch_processor",
            output="screen",
            parameters=[{
                "fusion_window_ms": 200,
                "defense_window_ms": 500,
            }],
        ),
        Node(
            package="boxbunny_core",
            executable="session_manager",
            name="session_manager",
            output="screen",
            parameters=[{
                "countdown_seconds": 3,
                "autosave_interval_s": 10.0,
            }],
        ),
        Node(
            package="boxbunny_core",
            executable="drill_manager",
            name="drill_manager",
            output="screen",
        ),
        Node(
            package="boxbunny_core",
            executable="sparring_engine",
            name="sparring_engine",
            output="screen",
        ),
        Node(
            package="boxbunny_core",
            executable="analytics_node",
            name="analytics_node",
            output="screen",
        ),
        Node(
            package="boxbunny_core",
            executable="llm_node",
            name="llm_node",
            output="screen",
            parameters=[{
                "model_path": str(ws_root / "models" / "llm" / "qwen2.5-3b-instruct-q4_k_m.gguf"),
                "n_gpu_layers": -1,
                "n_ctx": 2048,
                "fallback_tips_path": str(config_dir / "fallback_tips.json"),
            }],
        ),
        # --- Gesture Navigation ---
        # Uncomment to enable gesture control
        # Node(
        #     package="boxbunny_core",
        #     executable="gesture_node",
        #     name="gesture_node",
        #     output="screen",
        #     parameters=[{
        #         "enabled": True,
        #         "hold_duration_s": 0.7,
        #         "cooldown_s": 1.5,
        #         "min_confidence": 0.7,
        #         "swipe_threshold_px": 100.0,
        #         "process_interval": 3,
        #     }],
        # ),

        # --- GUI ---
        Node(
            package="boxbunny_gui",
            executable="gui_main",
            name="boxbunny_gui",
            output="screen",
        ),
    ])
