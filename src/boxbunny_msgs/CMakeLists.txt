cmake_minimum_required(VERSION 3.8)
project(boxbunny_msgs)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  # Messages
  "msg/PadImpact.msg"
  "msg/ArmStrike.msg"
  "msg/PunchEvent.msg"
  "msg/ArmStrikeEvent.msg"
  "msg/NavCommand.msg"
  "msg/PunchDetection.msg"
  "msg/ConfirmedPunch.msg"
  "msg/DefenseEvent.msg"
  "msg/PoseEstimate.msg"
  "msg/UserTracking.msg"
  "msg/SessionState.msg"
  "msg/SessionConfig.msg"
  "msg/RobotCommand.msg"
  "msg/HeightCommand.msg"
  "msg/RoundControl.msg"
  "msg/IMUStatus.msg"
  "msg/DrillDefinition.msg"
  "msg/DrillEvent.msg"
  "msg/DrillProgress.msg"
  "msg/CoachTip.msg"
  "msg/SessionPunchSummary.msg"
  # Services
  "srv/StartSession.srv"
  "srv/EndSession.srv"
  "srv/StartDrill.srv"
  "srv/SetImuMode.srv"
  "srv/CalibrateImuPunch.srv"
  "srv/GenerateLlm.srv"
  DEPENDENCIES std_msgs builtin_interfaces
)

ament_package()
