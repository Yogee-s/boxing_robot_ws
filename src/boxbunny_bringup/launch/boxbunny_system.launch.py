from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.conditions import IfCondition
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    enable_imu = LaunchConfiguration("enable_imu")
    enable_llm = LaunchConfiguration("enable_llm")
    enable_gui = LaunchConfiguration("enable_gui")

    realsense_config = PathJoinSubstitution(
        [FindPackageShare("boxbunny_bringup"), "config", "realsense.yaml"]
    )
    glove_config = PathJoinSubstitution(
        [FindPackageShare("boxbunny_vision"), "config", "glove_tracker.yaml"]
    )
    fusion_config = PathJoinSubstitution(
        [FindPackageShare("boxbunny_fusion"), "config", "fusion.yaml"]
    )
    drill_config = PathJoinSubstitution(
        [FindPackageShare("boxbunny_drills"), "config", "drill.yaml"]
    )
    imu_config = PathJoinSubstitution(
        [FindPackageShare("boxbunny_imu"), "config", "imu.yaml"]
    )
    analytics_config = PathJoinSubstitution(
        [FindPackageShare("boxbunny_analytics"), "config", "analytics.yaml"]
    )
    llm_config = PathJoinSubstitution(
        [FindPackageShare("boxbunny_llm"), "config", "llm.yaml"]
    )
    gui_config = PathJoinSubstitution(
        [FindPackageShare("boxbunny_gui"), "config", "gui.yaml"]
    )

    realsense = Node(
        package="realsense2_camera",
        executable="realsense2_camera_node",
        name="camera",
        parameters=[realsense_config],
        output="screen",
    )

    glove = Node(
        package="boxbunny_vision",
        executable="realsense_glove_tracker",
        parameters=[glove_config],
        output="screen",
    )

    fusion = Node(
        package="boxbunny_fusion",
        executable="punch_fusion_node",
        parameters=[fusion_config],
        output="screen",
    )

    drill = Node(
        package="boxbunny_drills",
        executable="reaction_drill_manager",
        parameters=[drill_config],
        output="screen",
    )

    imu = Node(
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

    analytics = Node(
        package="boxbunny_analytics",
        executable="punch_stats_node",
        parameters=[analytics_config],
        output="screen",
    )

    llm = Node(
        package="boxbunny_llm",
        executable="llm_talk_node",
        parameters=[llm_config],
        output="screen",
        condition=IfCondition(enable_llm),
    )

    gui = Node(
        package="boxbunny_gui",
        executable="gui_main",
        parameters=[gui_config],
        output="screen",
        condition=IfCondition(enable_gui),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("enable_imu", default_value="false"),
            DeclareLaunchArgument("enable_llm", default_value="true"),
            DeclareLaunchArgument("enable_gui", default_value="true"),
            realsense,
            glove,
            fusion,
            drill,
            imu,
            imu_classifier,
            analytics,
            llm,
            gui,
        ]
    )
