from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    realsense_config = PathJoinSubstitution(
        [FindPackageShare("boxbunny_bringup"), "config", "realsense.yaml"]
    )
    glove_config = PathJoinSubstitution(
        [FindPackageShare("boxbunny_vision"), "config", "glove_tracker.yaml"]
    )

    realsense = Node(
        package="realsense2_camera",
        executable="realsense2_camera_node",
        name="camera",
        parameters=[realsense_config],
        output="screen",
        prefix="env LD_PRELOAD=/usr/local/lib/librealsense2.so",
    )

    glove = Node(
        package="boxbunny_vision",
        executable="realsense_glove_tracker",
        parameters=[glove_config],
        output="screen",
    )

    return LaunchDescription([realsense, glove])
