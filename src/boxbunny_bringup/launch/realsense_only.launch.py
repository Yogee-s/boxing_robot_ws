from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    realsense_config = PathJoinSubstitution(
        [FindPackageShare("boxbunny_bringup"), "config", "realsense.yaml"]
    )
    glove_config = PathJoinSubstitution(
        [FindPackageShare("boxbunny_vision"), "config", "glove_tracker.yaml"]
    )

    # Declare arguments
    args = [
        DeclareLaunchArgument("rgb_width", default_value="640"),
        DeclareLaunchArgument("rgb_height", default_value="480"),
        DeclareLaunchArgument("rgb_fps", default_value="30"),
        DeclareLaunchArgument("depth_width", default_value="640"),
        DeclareLaunchArgument("depth_height", default_value="480"),
        DeclareLaunchArgument("depth_fps", default_value="30"),
    ]

    # Construct profiles
    rgb_profile = [
        LaunchConfiguration("rgb_width"), "x",
        LaunchConfiguration("rgb_height"), "x",
        LaunchConfiguration("rgb_fps")
    ]
    depth_profile = [
        LaunchConfiguration("depth_width"), "x",
        LaunchConfiguration("depth_height"), "x",
        LaunchConfiguration("depth_fps")
    ]

    realsense = Node(
        package="realsense2_camera",
        executable="realsense2_camera_node",
        name="camera",
        parameters=[
            realsense_config,
            {
                "rgb_camera.profile": rgb_profile,
                "depth_module.profile": depth_profile,
            }
        ],
        output="screen",
    )

    glove = Node(
        package="boxbunny_vision",
        executable="realsense_glove_tracker",
        parameters=[glove_config],
        output="screen",
    )

    return LaunchDescription(args + [realsense, glove])
