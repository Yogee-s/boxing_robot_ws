"""Sparring engine for BoxBunny.

Generates unpredictable robot attack sequences using Markov-chain transition
matrices.  Five boxing styles, three difficulty levels, and reactive
behaviours make each round feel different.
"""
from __future__ import annotations

import logging
import random
import time
from typing import Dict, List, Optional

import rclpy
from rclpy.node import Node

from boxbunny_core.constants import (
    PunchType,
    SessionState as SSConst,
    Topics,
)
from boxbunny_msgs.msg import ConfirmedPunch, RobotCommand, SessionState

logger = logging.getLogger("boxbunny.sparring_engine")

# Row/column order: jab(0), cross(1), l_hook(2), r_hook(3), l_uc(4), r_uc(5)
PUNCH_CODES: List[str] = ["1", "2", "3", "4", "5", "6"]
PUNCH_NAMES: List[str] = [
    PunchType.JAB, PunchType.CROSS, PunchType.LEFT_HOOK,
    PunchType.RIGHT_HOOK, PunchType.LEFT_UPPERCUT, PunchType.RIGHT_UPPERCUT,
]

# -- Transition matrices (rows = current punch, cols = next punch proba) ------
# Each row sums to ~1.0.

STYLES: Dict[str, List[List[float]]] = {
    "boxer": [
        [0.15, 0.30, 0.20, 0.15, 0.10, 0.10],
        [0.25, 0.10, 0.25, 0.15, 0.10, 0.15],
        [0.20, 0.25, 0.10, 0.15, 0.15, 0.15],
        [0.20, 0.20, 0.15, 0.10, 0.15, 0.20],
        [0.15, 0.25, 0.20, 0.15, 0.10, 0.15],
        [0.20, 0.25, 0.15, 0.20, 0.10, 0.10],
    ],
    "brawler": [
        [0.10, 0.15, 0.25, 0.25, 0.10, 0.15],
        [0.10, 0.10, 0.25, 0.20, 0.15, 0.20],
        [0.10, 0.15, 0.15, 0.20, 0.15, 0.25],
        [0.10, 0.10, 0.25, 0.15, 0.20, 0.20],
        [0.05, 0.15, 0.20, 0.20, 0.15, 0.25],
        [0.05, 0.15, 0.25, 0.20, 0.20, 0.15],
    ],
    "counter_puncher": [
        [0.25, 0.30, 0.15, 0.10, 0.10, 0.10],
        [0.30, 0.15, 0.20, 0.10, 0.15, 0.10],
        [0.25, 0.25, 0.10, 0.15, 0.10, 0.15],
        [0.25, 0.20, 0.15, 0.10, 0.15, 0.15],
        [0.20, 0.30, 0.15, 0.15, 0.10, 0.10],
        [0.20, 0.30, 0.15, 0.10, 0.10, 0.15],
    ],
    "pressure": [
        [0.30, 0.25, 0.15, 0.10, 0.10, 0.10],
        [0.35, 0.15, 0.15, 0.15, 0.10, 0.10],
        [0.30, 0.20, 0.10, 0.15, 0.10, 0.15],
        [0.30, 0.20, 0.15, 0.10, 0.10, 0.15],
        [0.25, 0.25, 0.15, 0.15, 0.10, 0.10],
        [0.25, 0.25, 0.15, 0.10, 0.15, 0.10],
    ],
    "switch": [],  # filled dynamically
}

# Difficulty -> seconds between robot attacks
DIFFICULTY_INTERVAL: Dict[str, float] = {
    "easy": 2.0,
    "medium": 1.2,
    "hard": 0.7,
}

IDLE_THRESHOLD_S = 3.0  # Seconds of user inactivity before surprise attack
WEAKNESS_BIAS = 0.08    # Extra probability mass added for weak-defense punches


class SparringEngine(Node):
    """Generates robot attack sequences driven by Markov-chain style matrices."""

    def __init__(self) -> None:
        super().__init__("sparring_engine")

        # Parameters
        self.declare_parameter("style", "boxer")
        self.declare_parameter("difficulty", "medium")
        self.declare_parameter("switch_interval_s", 20.0)

        self._style_name: str = self.get_parameter("style").value
        self._difficulty: str = self.get_parameter("difficulty").value
        self._switch_interval: float = self.get_parameter("switch_interval_s").value

        # State
        self._session_active: bool = False
        self._last_attack_time: float = 0.0
        self._last_user_punch_time: float = 0.0
        self._current_punch_idx: int = 0  # index into PUNCH_CODES
        self._user_blocked_last: bool = False
        self._weakness_profile: Dict[int, float] = {}  # punch_idx -> miss_rate
        self._style_switched_at: float = 0.0
        self._active_style: str = self._style_name

        # Publishers
        self._pub_cmd = self.create_publisher(
            RobotCommand, Topics.ROBOT_COMMAND, 10
        )

        # Subscribers
        self.create_subscription(
            SessionState, Topics.SESSION_STATE, self._on_session_state, 10
        )
        self.create_subscription(
            ConfirmedPunch, Topics.PUNCH_CONFIRMED, self._on_user_punch, 50
        )

        # Attack timer -- ticks at a fast rate; actual pacing is gated internally
        self.create_timer(0.1, self._tick)

        logger.info(
            "Sparring engine initialised (style=%s, difficulty=%s)",
            self._style_name, self._difficulty,
        )

    # ------------------------------------------------------------------
    # Session tracking
    # ------------------------------------------------------------------

    def _on_session_state(self, msg: SessionState) -> None:
        """Activate/deactivate sparring based on session state."""
        was_active = self._session_active
        self._session_active = msg.state == SSConst.ACTIVE and msg.mode == "sparring"
        if self._session_active and not was_active:
            self._reset_round()
            logger.info("Sparring round started")
        elif not self._session_active and was_active:
            logger.info("Sparring round ended")

    def _reset_round(self) -> None:
        """Reset per-round state."""
        now = time.time()
        self._last_attack_time = now
        self._last_user_punch_time = now
        self._current_punch_idx = 0
        self._user_blocked_last = False
        self._style_switched_at = now
        self._active_style = self._style_name

    # ------------------------------------------------------------------
    # User activity tracking
    # ------------------------------------------------------------------

    def _on_user_punch(self, msg: ConfirmedPunch) -> None:
        """Track user punch timestamps for idle detection."""
        self._last_user_punch_time = (
            msg.timestamp if msg.timestamp > 0 else time.time()
        )

    def update_weakness_profile(self, profile: Dict[str, float]) -> None:
        """Accept a weakness profile mapping punch names to miss rates.

        This can be called externally (e.g. from analytics or GUI) to bias
        the engine toward punches the user defends poorly against.
        """
        for name, rate in profile.items():
            if name in PUNCH_NAMES:
                self._weakness_profile[PUNCH_NAMES.index(name)] = rate

    # ------------------------------------------------------------------
    # Core tick
    # ------------------------------------------------------------------

    def _tick(self) -> None:
        """Main loop -- decide whether to attack this tick."""
        if not self._session_active:
            return

        now = time.time()
        interval = DIFFICULTY_INTERVAL.get(self._difficulty, 1.2)

        # Handle style switching
        if self._style_name == "switch":
            if now - self._style_switched_at >= self._switch_interval:
                candidates = [s for s in STYLES if s != "switch" and s != self._active_style]
                self._active_style = random.choice(candidates)
                self._style_switched_at = now
                logger.info("Style switched to '%s'", self._active_style)

        # Surprise attack if user idle
        user_idle = (now - self._last_user_punch_time) > IDLE_THRESHOLD_S
        if user_idle and (now - self._last_attack_time) > interval * 0.6:
            self._attack(now)
            return

        # Regular pacing
        if (now - self._last_attack_time) >= interval:
            self._attack(now)

    # ------------------------------------------------------------------
    # Attack generation
    # ------------------------------------------------------------------

    def _attack(self, now: float) -> None:
        """Select and publish the next robot punch."""
        next_idx = self._select_next_punch()

        # Reactive: if user blocked the last punch, pick a different angle
        if self._user_blocked_last and next_idx == self._current_punch_idx:
            alternatives = [i for i in range(len(PUNCH_CODES)) if i != next_idx]
            next_idx = random.choice(alternatives)
            self._user_blocked_last = False

        self._current_punch_idx = next_idx
        self._last_attack_time = now

        msg = RobotCommand()
        msg.command_type = "punch"
        msg.punch_code = PUNCH_CODES[next_idx]
        msg.speed = self._difficulty_to_speed()
        self._pub_cmd.publish(msg)

        logger.debug(
            "Robot attack: %s (style=%s)", PUNCH_NAMES[next_idx], self._active_style
        )

    def _select_next_punch(self) -> int:
        """Use the Markov transition matrix + weakness bias to pick next punch."""
        matrix = STYLES.get(self._active_style, STYLES["boxer"])
        if not matrix:
            matrix = STYLES["boxer"]

        row = matrix[self._current_punch_idx]
        weights = list(row)

        # Apply weakness bias
        if self._weakness_profile:
            for idx, miss_rate in self._weakness_profile.items():
                if 0 <= idx < len(weights):
                    weights[idx] += WEAKNESS_BIAS * miss_rate

            # Re-normalise
            total = sum(weights)
            if total > 0:
                weights = [w / total for w in weights]

        # Weighted random selection
        r = random.random()
        cumulative = 0.0
        for idx, w in enumerate(weights):
            cumulative += w
            if r <= cumulative:
                return idx
        return len(weights) - 1

    def _difficulty_to_speed(self) -> str:
        """Map difficulty level to robot speed string."""
        return {"easy": "slow", "medium": "medium", "hard": "fast"}.get(
            self._difficulty, "medium"
        )

    def set_user_blocked(self) -> None:
        """Signal that the user successfully blocked the last robot punch.

        Called externally (e.g., from the punch processor's defense pipeline).
        """
        self._user_blocked_last = True


def main(args: list[str] | None = None) -> None:
    """Entry point for the sparring engine node."""
    rclpy.init(args=args)
    node = SparringEngine()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
