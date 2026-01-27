from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    llm_config = PathJoinSubstitution([FindPackageShare("boxbunny_llm"), "config", "llm.yaml"])
    llm = Node(
        package="boxbunny_llm",
        executable="trash_talk_node",
        parameters=[llm_config],
        output="screen",
    )
    return LaunchDescription([llm])
