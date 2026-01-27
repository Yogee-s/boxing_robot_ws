from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    enable_classifier = LaunchConfiguration("enable_classifier")

    imu_config = PathJoinSubstitution([FindPackageShare("boxbunny_imu"), "config", "imu.yaml"])

    imu = Node(
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
        condition=IfCondition(enable_classifier),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("enable_classifier", default_value="true"),
            imu,
            imu_classifier,
        ]
    )
