"""Robot arm controller node for BoxBunny.

Bridges BoxBunny ROS commands to the V4 Arm Control GUI which handles
the actual motor control via micro-ROS.  The V4 GUI does calibration,
FSM strike execution (alignment, windup, apex, snap-back), and safety.

Architecture:
    BoxBunny GUI/sparring_engine
        → /boxbunny/robot/command (RobotCommand)
        → robot_node (this file)
        → /robot/strike_command (String JSON {slot, duration})
        → V4 Arm Control GUI (ROS Control tab)
        → motor_commands (Float64MultiArray)
        → Teensy firmware
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String

from boxbunny_msgs.msg import HeightCommand, RobotCommand, RoundControl

from boxbunny_core.constants import Topics

logger = logging.getLogger("boxbunny.robot_node")

# Punch code -> slot mapping (must match what's assigned in V4 GUI)
_CODE_TO_SLOT = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6}

# Speed string -> rad/s for V4 GUI speed override
_SPEED_MAP = {"slow": 5.0, "medium": 10.0, "fast": 20.0}


class RobotNode(Node):
    """ROS 2 node bridging BoxBunny commands to the V4 Arm Control GUI."""

    def __init__(self) -> None:
        super().__init__("robot_node")

        # Parameters
        self.declare_parameter("heartbeat_hz", 1.0)
        heartbeat_hz = self.get_parameter("heartbeat_hz").value

        # State
        self._round_active = False
        self._connected = False
        self._last_feedback_time = 0.0
        self._actual_positions = [0.0, 0.0, 0.0, 0.0]
        self._actual_currents = [0.0, 0.0, 0.0, 0.0]
        self._current_speed = "medium"

        # ── Publishers to V4 GUI ─────────────────────────────────────────
        self._pub_strike_cmd = self.create_publisher(
            String, Topics.ROBOT_STRIKE_COMMAND, 10
        )
        self._pub_system_enable = self.create_publisher(
            String, Topics.ROBOT_SYSTEM_ENABLE, 10
        )
        self._pub_height_cmd = self.create_publisher(
            String, Topics.ROBOT_HEIGHT_CMD, 10
        )
        self._pub_yaw_cmd = self.create_publisher(
            String, Topics.ROBOT_YAW_CMD, 10
        )
        self._pub_status = self.create_publisher(
            String, Topics.ROBOT_STATUS, 10
        )
        self._pub_strike_complete = self.create_publisher(
            String, Topics.ROBOT_STRIKE_COMPLETE, 10
        )

        # ── Subscribers from V4 GUI ──────────────────────────────────────
        # Motor feedback from Teensy (via V4 GUI / micro-ROS)
        self.create_subscription(
            Float64MultiArray, Topics.MOTOR_FEEDBACK,
            self._on_motor_feedback, 10
        )
        # Strike completion from V4 GUI
        self.create_subscription(
            String, Topics.ROBOT_STRIKE_FEEDBACK,
            self._on_strike_feedback, 10
        )
        # Person direction → yaw motor forwarding
        self.create_subscription(
            String, Topics.CV_PERSON_DIRECTION,
            self._on_person_direction, 10
        )

        # ── Subscribers from BoxBunny system ─────────────────────────────
        self.create_subscription(
            RobotCommand, Topics.ROBOT_COMMAND,
            self._on_robot_command, 10
        )
        self.create_subscription(
            HeightCommand, Topics.ROBOT_HEIGHT,
            self._on_height_command, 10
        )
        self.create_subscription(
            RoundControl, Topics.ROBOT_ROUND_CONTROL,
            self._on_round_control, 10
        )

        # Status timer
        self.create_timer(1.0 / max(0.1, heartbeat_hz), self._publish_status)

        logger.info("Robot node initialized (V4 GUI bridge mode)")

    # ── Motor feedback from Teensy ──────────────────────────────────────

    def _on_motor_feedback(self, msg: Float64MultiArray) -> None:
        """Track Teensy connection and motor state."""
        self._last_feedback_time = time.time()
        if not self._connected:
            self._connected = True
            logger.info("Teensy connected (receiving motor_feedback)")
        if len(msg.data) >= 8:
            self._actual_positions = list(msg.data[0:4])
            self._actual_currents = list(msg.data[4:8])

    # ── Strike feedback from V4 GUI ─────────────────────────────────────

    def _on_strike_feedback(self, msg: String) -> None:
        """Forward V4 GUI strike completion to BoxBunny system."""
        try:
            data = json.loads(msg.data)
            # Re-publish as /boxbunny/robot/strike_complete for the BoxBunny GUI
            slot = data.get("slot", 0)
            status = data.get("status", "unknown")
            duration = data.get("duration_actual", 0.0)
            strike = data.get("strike", "")
            # Map slot back to punch code
            code = str(slot) if 1 <= slot <= 6 else "0"
            complete = String()
            complete.data = json.dumps({
                "punch_code": code,
                "status": status,
                "duration_ms": int(duration * 1000),
                "strike": strike,
            })
            self._pub_strike_complete.publish(complete)
            logger.debug("Strike complete: slot=%d %s (%.1fs)", slot, status, duration)
        except (json.JSONDecodeError, TypeError) as e:
            logger.debug("Invalid strike feedback: %s", e)

    # ── Robot command handling ──────────────────────────────────────────

    def _on_robot_command(self, msg: RobotCommand) -> None:
        """Translate BoxBunny RobotCommand → V4 GUI /robot/strike_command."""
        if msg.command_type == "set_speed":
            self._current_speed = msg.speed
            logger.info("Robot speed set to %s", msg.speed)

        elif msg.command_type == "punch":
            slot = _CODE_TO_SLOT.get(msg.punch_code)
            if slot is None:
                logger.warning("Unknown punch code: %s", msg.punch_code)
                return
            speed = _SPEED_MAP.get(
                msg.speed or self._current_speed, 10.0
            )
            cmd = String()
            cmd.data = json.dumps({
                "slot": slot,
                "duration": 5.0,
                "speed": speed,
            })
            self._pub_strike_cmd.publish(cmd)
            logger.debug(
                "Strike command -> V4 GUI: slot=%d speed=%.1f",
                slot, speed,
            )

    # ── Height command handling ──────────────────────────────────────────

    def _on_height_command(self, msg: HeightCommand) -> None:
        """Convert BoxBunny HeightCommand to Teensy string commands."""
        height_msg = String()
        if msg.action == "stop":
            height_msg.data = "STOP"
        elif msg.action == "manual_up":
            height_msg.data = "UP:200"
        elif msg.action == "manual_down":
            height_msg.data = "DOWN:200"
        elif msg.action == "adjust":
            error = msg.current_height_px - msg.target_height_px
            pwm = min(255, int(abs(error) * 2))
            direction = "UP" if error > 0 else "DOWN"
            height_msg.data = f"{direction}:{pwm}"
        elif msg.action == "calibrate":
            logger.info("Height calibration requested")
            return
        else:
            return
        self._pub_height_cmd.publish(height_msg)

    def _on_person_direction(self, msg: String) -> None:
        """Forward person direction to Teensy yaw motor."""
        yaw_msg = String()
        yaw_msg.data = msg.data.upper()  # "LEFT", "RIGHT", "CENTRE"
        self._pub_yaw_cmd.publish(yaw_msg)
        logger.debug("Height command: %s", height_msg.data)

    # ── Round control ───────────────────────────────────────────────────

    def _on_round_control(self, msg: RoundControl) -> None:
        """Handle round start/stop — enable/disable V4 GUI."""
        if msg.action == "start":
            self._round_active = True
            enable_msg = String()
            enable_msg.data = "enable"
            self._pub_system_enable.publish(enable_msg)
            logger.info("Round started — V4 GUI enabled")
        elif msg.action == "stop":
            self._round_active = False
            disable_msg = String()
            disable_msg.data = "disable"
            self._pub_system_enable.publish(disable_msg)
            logger.info("Round stopped — V4 GUI disabled")

    # ── Status publishing ───────────────────────────────────────────────

    def _publish_status(self) -> None:
        """Publish robot arm status."""
        if self._connected and (time.time() - self._last_feedback_time) > 2.0:
            self._connected = False
            logger.warning("Teensy disconnected (no motor_feedback for >2s)")

        msg = String()
        msg.data = json.dumps({
            "status": "connected" if self._connected else "disconnected",
            "round_active": self._round_active,
            "speed": self._current_speed,
            "positions": [round(p, 2) for p in self._actual_positions],
        })
        self._pub_status.publish(msg)


def main(args=None) -> None:
    """Entry point for the robot node."""
    rclpy.init(args=args)
    node = RobotNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
