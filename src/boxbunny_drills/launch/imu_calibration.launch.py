"""Launch file for IMU Calibration GUI unit test.

This launches:
1. MPU6050 IMU driver node
2. IMU punch classifier node
3. IMU calibration/testing GUI

Usage: ros2 launch boxbunny_drills imu_calibration.launch.py
"""
from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    imu_config = PathJoinSubstitution([FindPackageShare("boxbunny_imu"), "config", "imu.yaml"])

    imu_driver = Node(
        package="boxbunny_imu",
        executable="mpu6050_node",
        parameters=[imu_config],
        output="screen",
    )

    imu_classifier = Node(
        package="boxbunny_imu",
        executable="imu_punch_classifier",
        parameters=[imu_config],
        output="screen",
    )

    imu_gui = Node(
        package="boxbunny_imu",
        executable="imu_punch_gui",
        output="screen",
    )

    return LaunchDescription([
        imu_driver,
        imu_classifier,
        imu_gui,
    ])
