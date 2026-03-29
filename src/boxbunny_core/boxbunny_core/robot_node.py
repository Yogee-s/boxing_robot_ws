"""Robot arm controller node for BoxBunny.

Manages serial communication with the Teensy controlling the robot arm
and height adjustment motor. Loads punch waypoint sequences from JSON files.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from boxbunny_msgs.msg import HeightCommand, RobotCommand, RoundControl

logger = logging.getLogger("boxbunny.robot_node")


class RobotNode(Node):
    """ROS 2 node for robot arm and height motor control."""

    def __init__(self) -> None:
        super().__init__("robot_node")

        # Parameters
        self.declare_parameter("serial_port", "/dev/ttyACM0")
        self.declare_parameter("baud_rate", 115200)
        self.declare_parameter("heartbeat_hz", 10.0)
        self.declare_parameter("punch_sequences_dir", "")

        serial_port = self.get_parameter("serial_port").value
        baud_rate = self.get_parameter("baud_rate").value
        heartbeat_hz = self.get_parameter("heartbeat_hz").value
        seq_dir = self.get_parameter("punch_sequences_dir").value

        # Load punch sequences
        self._punch_sequences: Dict[str, List[Dict]] = {}
        self._load_punch_sequences(seq_dir)

        # Serial connection (lazy — connected on first use)
        self._serial = None
        self._serial_port = serial_port
        self._baud_rate = baud_rate
        self._connected = False

        # Motor state
        self._positions = [0.0, 0.0, 0.0, 0.0]  # M1-M4
        self._speeds = [50.0, 50.0, 50.0, 50.0]
        self._enabled = False
        self._round_active = False

        # Speed mapping
        self._speed_map = {"slow": 30.0, "medium": 50.0, "fast": 80.0}
        self._current_speed = "medium"

        # Subscribers
        self.create_subscription(RobotCommand, "/boxbunny/robot/command",
                                 self._on_robot_command, 10)
        self.create_subscription(HeightCommand, "/boxbunny/robot/height",
                                 self._on_height_command, 10)
        self.create_subscription(RoundControl, "/boxbunny/robot/round_control",
                                 self._on_round_control, 10)

        # Publisher
        self._pub_status = self.create_publisher(String, "/boxbunny/robot/status", 10)

        # Heartbeat timer
        if heartbeat_hz > 0:
            self.create_timer(1.0 / heartbeat_hz, self._heartbeat)

        # Status timer
        self.create_timer(2.0, self._publish_status)

        logger.info("Robot node initialized (port=%s, %d sequences loaded)",
                     serial_port, len(self._punch_sequences))

    def _load_punch_sequences(self, seq_dir: str) -> None:
        """Load punch waypoint sequences from JSON files."""
        if not seq_dir:
            ws_root = Path(__file__).resolve().parents[3]
            seq_dir = str(ws_root / "data" / "punch_sequences")

        seq_path = Path(seq_dir)
        if not seq_path.exists():
            logger.warning("Punch sequences directory not found: %s", seq_dir)
            return

        # Map filenames to punch codes
        code_map = {
            "1_Jab": "1", "2_Cross": "2", "3_Hook": "3",
            "4_R_Hook": "4", "5_L_UC": "5", "6_R_UC": "6",
            "Left_Hook": "3b", "Right_Hook": "4b",
        }

        for json_file in seq_path.glob("*.json"):
            try:
                with open(json_file, "r") as f:
                    sequence = json.load(f)
                stem = json_file.stem
                code = code_map.get(stem, stem)
                self._punch_sequences[code] = sequence
                logger.debug("Loaded punch sequence: %s -> code '%s' (%d waypoints)",
                             json_file.name, code, len(sequence))
            except (json.JSONDecodeError, OSError) as e:
                logger.error("Failed to load punch sequence %s: %s", json_file, e)

    def _connect_serial(self) -> bool:
        """Attempt to connect to the Teensy serial port."""
        if self._connected:
            return True
        try:
            import serial
            self._serial = serial.Serial(self._serial_port, self._baud_rate, timeout=0.1)
            self._connected = True
            logger.info("Connected to robot arm at %s", self._serial_port)
            return True
        except ImportError:
            logger.warning("pyserial not installed — robot arm in simulation mode")
            return False
        except Exception as e:
            logger.debug("Serial connection failed: %s", e)
            return False

    def _on_robot_command(self, msg: RobotCommand) -> None:
        """Handle robot punch or speed commands."""
        if msg.command_type == "set_speed":
            self._current_speed = msg.speed
            speed_val = self._speed_map.get(msg.speed, 50.0)
            self._speeds = [speed_val] * 4
            logger.info("Robot speed set to %s (%.0f)", msg.speed, speed_val)

        elif msg.command_type == "punch":
            code = msg.punch_code
            sequence = self._punch_sequences.get(code)
            if sequence is None:
                logger.warning("Unknown punch code: %s", code)
                return
            self._execute_punch(code, sequence)

    def _execute_punch(self, code: str, sequence: List[Dict]) -> None:
        """Execute a punch sequence by sending waypoints to Teensy."""
        logger.debug("Executing punch: code=%s (%d waypoints)", code, len(sequence))
        for waypoint in sequence:
            positions = waypoint.get("pos", [0, 0, 0, 0])
            speed_l = waypoint.get("spd_l", self._speeds[0])
            speed_r = waypoint.get("spd_r", self._speeds[2])
            self._send_motor_command(
                positions,
                [speed_l, speed_l, speed_r, speed_r],
                enabled=True,
            )
            time.sleep(0.05)  # Brief delay between waypoints

    def _on_height_command(self, msg: HeightCommand) -> None:
        """Handle height adjustment commands."""
        if msg.action == "stop":
            logger.info("Height motor stopped")
        elif msg.action in ("manual_up", "manual_down"):
            logger.info("Height manual: %s", msg.action)
        elif msg.action == "adjust":
            error = msg.current_height_px - msg.target_height_px
            logger.info("Height auto-adjust: error=%.1fpx", error)
            # In production: send proportional command to height motor via serial
        elif msg.action == "calibrate":
            logger.info("Height calibration requested")

    def _on_round_control(self, msg: RoundControl) -> None:
        """Handle round start/stop."""
        if msg.action == "start":
            self._round_active = True
            self._enabled = True
            logger.info("Round started — robot arm enabled")
        elif msg.action == "stop":
            self._round_active = False
            self._enabled = False
            self._send_motor_command([0, 0, 0, 0], [0, 0, 0, 0], enabled=False)
            logger.info("Round stopped — robot arm disabled")

    def _send_motor_command(
        self, positions: List[float], speeds: List[float], enabled: bool
    ) -> None:
        """Send motor command to Teensy via serial."""
        if not self._connected and not self._connect_serial():
            return
        # Protocol: [P1, P2, P3, P4, S1, S2, S3, S4, Mode]
        mode = 1.0 if enabled else 0.0
        command = positions[:4] + speeds[:4] + [mode]
        try:
            if self._serial:
                payload = ",".join(f"{v:.2f}" for v in command) + "\n"
                self._serial.write(payload.encode())
        except Exception as e:
            logger.error("Serial write failed: %s", e)
            self._connected = False

    def _heartbeat(self) -> None:
        """Send heartbeat to keep motor watchdog alive."""
        if not self._connected or not self._enabled:
            return
        self._send_motor_command(self._positions, self._speeds, self._enabled)

    def _publish_status(self) -> None:
        """Publish robot arm status."""
        msg = String()
        status = "connected" if self._connected else "disconnected"
        msg.data = json.dumps({
            "status": status,
            "port": self._serial_port,
            "round_active": self._round_active,
            "speed": self._current_speed,
            "sequences_loaded": len(self._punch_sequences),
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
