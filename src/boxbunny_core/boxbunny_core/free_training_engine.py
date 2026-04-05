"""Free training engine for BoxBunny.

Purely reactive counter-punch logic: when the user strikes a pad, the robot
throws back a random punch from the configured set for that pad.  Only one
strike at a time -- rapid pad hits during a robot strike are ignored (matches
the V4 GUI DynamicSparringTab behaviour).

This node is completely independent from the sparring engine.
"""
from __future__ import annotations

import json
import logging
import random
import time
from typing import Dict, List, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from boxbunny_core.config_loader import load_config
from boxbunny_core.constants import SessionState as SSConst, Topics
from boxbunny_msgs.msg import PunchEvent, RobotCommand, SessionState

logger = logging.getLogger("boxbunny.free_training_engine")


class FreeTrainingEngine(Node):
    """Reactive counter-punch engine for free training mode."""

    def __init__(self) -> None:
        super().__init__("free_training_engine")

        # Load config from boxbunny.yaml free_training section
        ft = load_config().free_training
        self._pad_strikes: Dict[str, List[str]] = ft.pad_counter_strikes
        self._cooldown_s: float = ft.counter_cooldown_ms / 1000.0
        self._speed: str = ft.speed

        # State
        self._active: bool = False
        self._robot_busy: bool = False
        self._last_counter: float = 0.0

        # Publishers
        self._pub_cmd = self.create_publisher(
            RobotCommand, Topics.ROBOT_COMMAND, 10,
        )

        # Subscribers
        self.create_subscription(
            SessionState, Topics.SESSION_STATE,
            self._on_session, 10,
        )
        self.create_subscription(
            PunchEvent, Topics.IMU_PUNCH_EVENT,
            self._on_punch_event, 10,
        )
        self.create_subscription(
            String, Topics.ROBOT_STRIKE_FEEDBACK,
            self._on_strike_feedback, 10,
        )
        # Session config for user-selected speed override
        self.create_subscription(
            String, "/boxbunny/session/config_json",
            self._on_session_config, 10,
        )

        logger.info(
            "Free training engine initialised "
            "(pads=%s, cooldown=%.1fs, speed=%s)",
            list(self._pad_strikes.keys()), self._cooldown_s, self._speed,
        )

    # ── Session lifecycle ────────────────────────────────────────────────

    def _on_session(self, msg: SessionState) -> None:
        """Activate only for free training mode when round is active."""
        was = self._active
        self._active = msg.state == SSConst.ACTIVE and msg.mode == "free"
        if self._active and not was:
            self._robot_busy = False
            self._last_counter = time.time()
            logger.info("Free training engine activated")
        elif was and not self._active:
            self._robot_busy = False
            logger.info("Free training engine deactivated")

    def _on_session_config(self, msg: String) -> None:
        """Read session config for user-selected speed override."""
        try:
            config = json.loads(msg.data)
            speed = config.get("speed")
            if speed and speed in ("slow", "medium", "fast"):
                self._speed = speed
                logger.info("Speed set to %s (from session config)", speed)
        except (json.JSONDecodeError, TypeError):
            pass

    # ── Strike feedback ──────────────────────────────────────────────────

    def _on_strike_feedback(self, msg: String) -> None:
        """Robot arm finished — clear busy flag and reset cooldown."""
        self._robot_busy = False
        self._last_counter = time.time()

    # ── Core reactive logic ──────────────────────────────────────────────

    def _on_punch_event(self, msg: PunchEvent) -> None:
        """React to user pad strike with a random counter-punch.

        Ignores the punch if:
        - Engine is not active (wrong mode or session not running)
        - Robot is busy executing a previous strike
        - Cooldown has not elapsed since last counter-punch
        """
        if not self._active:
            return
        if self._robot_busy:
            logger.debug("Ignoring pad=%s — robot busy", msg.pad)
            return

        now = time.time()
        if now - self._last_counter < self._cooldown_s:
            return

        strikes = self._pad_strikes.get(msg.pad)
        if not strikes:
            logger.debug("No counter-strikes configured for pad=%s", msg.pad)
            return

        punch_code = random.choice(strikes)

        cmd = RobotCommand()
        cmd.command_type = "punch"
        cmd.punch_code = punch_code
        cmd.speed = self._speed
        self._robot_busy = True
        self._pub_cmd.publish(cmd)

        logger.info(
            "Counter-punch: pad=%s -> code=%s speed=%s",
            msg.pad, punch_code, self._speed,
        )


def main(args: list[str] | None = None) -> None:
    """Entry point for the free training engine node."""
    rclpy.init(args=args)
    node = FreeTrainingEngine()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
