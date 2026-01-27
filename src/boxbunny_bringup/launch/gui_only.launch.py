from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    gui_config = PathJoinSubstitution([FindPackageShare("boxbunny_gui"), "config", "gui.yaml"])
    gui = Node(
        package="boxbunny_gui",
        executable="gui_main",
        parameters=[gui_config],
        output="screen",
    )
    return LaunchDescription([gui])
