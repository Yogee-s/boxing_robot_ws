#!/usr/bin/env bash
set -euo pipefail

MICROROS_WS="${MICROROS_WS:-$HOME/microros_ws}"
ROBOT_WS="${ROBOT_WS:-$HOME/boxing_robot}"
ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"
TEENSY_BY_ID="/dev/serial/by-id/usb-Teensyduino_USB_Serial_15398210-if00"
TEENSY_FALLBACK="/dev/ttyACM0"
GUI_SCRIPT="${GUI_SCRIPT:-$ROBOT_WS/ros2_ws/arm_GUI_fair_3.py}"

if [[ ! -f "$ROS_SETUP" ]]; then
  echo "ROS setup not found: $ROS_SETUP" >&2
  exit 1
fi
if [[ ! -d "$MICROROS_WS" ]]; then
  echo "Micro-ROS workspace not found: $MICROROS_WS" >&2
  exit 1
fi
if [[ ! -f "$GUI_SCRIPT" ]]; then
  echo "GUI script not found: $GUI_SCRIPT" >&2
  exit 1
fi

DEVICE="$TEENSY_BY_ID"
if [[ ! -e "$DEVICE" ]]; then
  echo "Teensy by-id not found: $DEVICE. Falling back to $TEENSY_FALLBACK" >&2
  DEVICE="$TEENSY_FALLBACK"
fi

if ! command -v gnome-terminal >/dev/null 2>&1; then
  echo "gnome-terminal not found. Run these manually or install a terminal emulator." >&2
  echo "" >&2
  echo "Agent:" >&2
  echo "  cd \"$MICROROS_WS\"" >&2
  echo "  source \"$ROS_SETUP\"" >&2
  echo "  source install/local_setup.bash" >&2
  echo "  ros2 run micro_ros_agent micro_ros_agent serial --dev \"$DEVICE\"" >&2
  echo "" >&2
  echo "GUI:" >&2
  echo "  cd \"$ROBOT_WS\"" >&2
  echo "  source \"$ROS_SETUP\"" >&2
  echo "  python \"$GUI_SCRIPT\"" >&2
  exit 1
fi

gnome-terminal --title="Micro-ROS Agent" -- bash -lc "cd \"$MICROROS_WS\"; source \"$ROS_SETUP\"; source install/local_setup.bash; ros2 run micro_ros_agent micro_ros_agent serial --dev \"$DEVICE\"; exec bash"

gnome-terminal --title="Arm GUI (fair 3)" -- bash -lc "cd \"$ROBOT_WS\"; source \"$ROS_SETUP\"; python \"$GUI_SCRIPT\"; exec bash"
