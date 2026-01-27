cmake_minimum_required(VERSION 3.8)
project(boxbunny_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/GloveDetection.msg"
  "msg/GloveDetections.msg"
  "msg/PunchEvent.msg"
  "msg/ImuPunch.msg"
  "msg/DrillEvent.msg"
  "msg/ImuDebug.msg"
  "msg/TrashTalk.msg"
  "srv/StartStopDrill.srv"
  "srv/CalibrateImuPunch.srv"
  "srv/GenerateLLM.srv"
  DEPENDENCIES builtin_interfaces std_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
