"""IMU processing node for BoxBunny.

Receives raw Teensy IMU data and republishes as processed events.
Handles dual-mode switching: NAVIGATION (pad taps = GUI nav) vs TRAINING
(pad impacts = punch data). Mode switches on SessionState changes.
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional

import rclpy
from rclpy.node import Node

from boxbunny_msgs.msg import (
    ArmStrike,
    ArmStrikeEvent,
    IMUStatus,
    NavCommand,
    PadImpact,
    PunchEvent,
    SessionState,
)

logger = logging.getLogger("boxbunny.imu_node")

FORCE_MAP = {"light": 0.33, "medium": 0.66, "hard": 1.0}

PAD_TO_NAV = {
    "left": "prev",
    "right": "next",
    "centre": "enter",
    "head": "back",
}


class ImuMode(Enum):
    NAVIGATION = "navigation"
    TRAINING = "training"
    TRANSITIONING = "transitioning"


@dataclass
class DebounceTracker:
    """Tracks debounce timers for navigation events."""

    per_pad: Dict[str, float] = field(default_factory=dict)
    global_last: float = 0.0
    pad_debounce_s: float = 0.3
    global_debounce_s: float = 0.2

    def can_fire(self, pad: str) -> bool:
        """Check if a navigation event can fire (debounce check)."""
        now = time.time()
        if now - self.global_last < self.global_debounce_s:
            return False
        if pad in self.per_pad and now - self.per_pad[pad] < self.pad_debounce_s:
            return False
        self.per_pad[pad] = now
        self.global_last = now
        return True


class ImuNode(Node):
    """ROS 2 node for IMU data processing and mode management."""

    def __init__(self) -> None:
        super().__init__("imu_node")

        # Parameters
        self.declare_parameter("nav_debounce_ms", 300)
        self.declare_parameter("nav_global_debounce_ms", 200)
        self.declare_parameter("mode_transition_ms", 200)
        self.declare_parameter("heartbeat_interval_s", 1.0)

        nav_debounce = self.get_parameter("nav_debounce_ms").value / 1000.0
        global_debounce = self.get_parameter("nav_global_debounce_ms").value / 1000.0
        self._transition_duration = self.get_parameter("mode_transition_ms").value / 1000.0
        heartbeat_interval = self.get_parameter("heartbeat_interval_s").value

        # State
        self._mode = ImuMode.NAVIGATION
        self._transition_start: Optional[float] = None
        self._target_mode: Optional[ImuMode] = None
        self._debounce = DebounceTracker(
            pad_debounce_s=nav_debounce,
            global_debounce_s=global_debounce,
        )

        # Subscribers
        self.create_subscription(PadImpact, "/boxbunny/imu/pad/impact", self._on_pad_impact, 10)
        self.create_subscription(ArmStrike, "/boxbunny/imu/arm/strike", self._on_arm_strike, 10)
        self.create_subscription(SessionState, "/boxbunny/session/state", self._on_session_state, 10)

        # Publishers
        self._pub_punch = self.create_publisher(PunchEvent, "/boxbunny/imu/punch_event", 10)
        self._pub_nav = self.create_publisher(NavCommand, "/boxbunny/imu/nav_event", 10)
        self._pub_arm = self.create_publisher(ArmStrikeEvent, "/boxbunny/imu/arm_event", 10)
        self._pub_status = self.create_publisher(IMUStatus, "/boxbunny/imu/status", 10)

        # Heartbeat timer
        self.create_timer(heartbeat_interval, self._publish_status)

        # Transition check timer (fast poll during transitions)
        self.create_timer(0.05, self._check_transition)

        logger.info("IMU node initialized in NAVIGATION mode")

    def _on_pad_impact(self, msg: PadImpact) -> None:
        """Handle raw pad impact from Teensy."""
        if self._mode == ImuMode.TRANSITIONING:
            return  # Drop events during mode transition

        if self._mode == ImuMode.NAVIGATION:
            self._handle_nav_tap(msg)
        elif self._mode == ImuMode.TRAINING:
            self._handle_punch_impact(msg)

    def _handle_nav_tap(self, msg: PadImpact) -> None:
        """Convert pad impact to navigation command."""
        command = PAD_TO_NAV.get(msg.pad)
        if command is None:
            return
        if not self._debounce.can_fire(msg.pad):
            return
        nav_msg = NavCommand()
        nav_msg.timestamp = msg.timestamp if msg.timestamp > 0 else time.time()
        nav_msg.command = command
        self._pub_nav.publish(nav_msg)
        logger.debug("Nav event: %s -> %s", msg.pad, command)

    def _handle_punch_impact(self, msg: PadImpact) -> None:
        """Convert pad impact to punch event."""
        punch_msg = PunchEvent()
        punch_msg.timestamp = msg.timestamp if msg.timestamp > 0 else time.time()
        punch_msg.pad = msg.pad
        punch_msg.level = msg.level
        punch_msg.force_normalized = FORCE_MAP.get(msg.level, 0.5)
        self._pub_punch.publish(punch_msg)
        logger.debug("Punch event: pad=%s level=%s force=%.2f",
                      msg.pad, msg.level, punch_msg.force_normalized)

    def _on_arm_strike(self, msg: ArmStrike) -> None:
        """Handle arm strike from Teensy — always forward regardless of mode."""
        arm_msg = ArmStrikeEvent()
        arm_msg.timestamp = msg.timestamp if msg.timestamp > 0 else time.time()
        arm_msg.arm = msg.arm
        arm_msg.contact = msg.contact
        self._pub_arm.publish(arm_msg)
        logger.debug("Arm event: arm=%s contact=%s", msg.arm, msg.contact)

    def _on_session_state(self, msg: SessionState) -> None:
        """Handle session state changes to switch IMU mode."""
        if msg.state in ("countdown", "active", "rest"):
            if self._mode != ImuMode.TRAINING:
                self._start_transition(ImuMode.TRAINING)
        elif msg.state in ("idle", "complete"):
            if self._mode != ImuMode.NAVIGATION:
                self._start_transition(ImuMode.NAVIGATION)

    def _start_transition(self, target: ImuMode) -> None:
        """Begin mode transition with grace period."""
        self._mode = ImuMode.TRANSITIONING
        self._target_mode = target
        self._transition_start = time.time()
        logger.info("IMU mode transitioning to %s (%.0fms grace)",
                     target.value, self._transition_duration * 1000)

    def _check_transition(self) -> None:
        """Check if transition grace period has elapsed."""
        if self._mode != ImuMode.TRANSITIONING:
            return
        if self._transition_start is None or self._target_mode is None:
            return
        if time.time() - self._transition_start >= self._transition_duration:
            self._mode = self._target_mode
            self._target_mode = None
            self._transition_start = None
            logger.info("IMU mode switched to %s", self._mode.value)

    def _publish_status(self) -> None:
        """Publish IMU connection status heartbeat."""
        status = IMUStatus()
        # In a real system, these would reflect actual Teensy connection state.
        # Default to all connected (will be overridden by actual hardware status).
        status.left_pad_connected = True
        status.centre_pad_connected = True
        status.right_pad_connected = True
        status.head_pad_connected = True
        status.left_arm_connected = True
        status.right_arm_connected = True
        status.is_simulator = False
        self._pub_status.publish(status)


def main(args=None) -> None:
    """Entry point for the IMU node."""
    rclpy.init(args=args)
    node = ImuNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
