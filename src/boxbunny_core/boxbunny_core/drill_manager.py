"""Drill session manager for BoxBunny.

Loads combo drill definitions from YAML, validates detected punch sequences
against expected combos, tracks accuracy/timing/streak, and publishes
drill progress events.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
import rclpy
from rclpy.node import Node

from boxbunny_core.constants import (
    Difficulty,
    PunchType,
    Services,
    SessionState as SSConst,
    Topics,
)
from boxbunny_msgs.msg import (
    ConfirmedPunch,
    DrillDefinition,
    DrillEvent,
    DrillProgress,
    SessionState,
)
from boxbunny_msgs.srv import StartDrill

logger = logging.getLogger("boxbunny.drill_manager")

# Punch code -> canonical name
_CODE_TO_NAME: Dict[str, str] = {
    "1": PunchType.JAB,
    "2": PunchType.CROSS,
    "3": PunchType.LEFT_HOOK,
    "4": PunchType.RIGHT_HOOK,
    "5": PunchType.LEFT_UPPERCUT,
    "6": PunchType.RIGHT_UPPERCUT,
}

DEFAULT_TOLERANCE_MS = 500.0


class _ActiveDrill:
    """Mutable state for a drill in progress."""

    def __init__(self, definition: Dict[str, Any], tolerance_ms: float) -> None:
        self.name: str = definition["name"]
        self.combo: List[str] = [
            _CODE_TO_NAME.get(str(c), str(c)) for c in definition["combo"]
        ]
        self.timing_ms: float = float(definition.get("timing_ms", 300))
        self.total_reps: int = int(definition.get("reps", 10))
        self.tolerance_s: float = tolerance_ms / 1000.0

        # Per-attempt state
        self.detected: List[str] = []
        self.punch_times: List[float] = []

        # Session-level counters
        self.reps_completed: int = 0
        self.reps_missed: int = 0
        self.accuracy_sum: float = 0.0
        self.timing_sum: float = 0.0
        self.current_streak: int = 0
        self.best_streak: int = 0


class DrillManager(Node):
    """Manages combo drill sessions and validates punch sequences."""

    def __init__(self) -> None:
        super().__init__("drill_manager")

        # Parameters
        self.declare_parameter("drills_yaml", "")
        self.declare_parameter("timing_tolerance_ms", DEFAULT_TOLERANCE_MS)
        yaml_path = self.get_parameter("drills_yaml").value
        self._default_tolerance = float(
            self.get_parameter("timing_tolerance_ms").value
        )

        # Load drill catalogue
        self._catalogue: Dict[str, Dict[str, Any]] = {}
        self._difficulty_params: Dict[str, Dict[str, Any]] = {}
        self._load_drills(yaml_path)

        # Active drill state
        self._drill: Optional[_ActiveDrill] = None
        self._session_active: bool = False
        self._combo_start_time: float = 0.0

        # Publishers
        self._pub_definition = self.create_publisher(
            DrillDefinition, Topics.DRILL_DEFINITION, 10
        )
        self._pub_event = self.create_publisher(DrillEvent, Topics.DRILL_EVENT, 10)
        self._pub_progress = self.create_publisher(
            DrillProgress, Topics.DRILL_PROGRESS, 10
        )

        # Subscribers
        self.create_subscription(
            ConfirmedPunch, Topics.PUNCH_CONFIRMED, self._on_punch, 50
        )
        self.create_subscription(
            SessionState, Topics.SESSION_STATE, self._on_session_state, 10
        )

        # Service
        self.create_service(StartDrill, Services.START_DRILL, self._handle_start)

        # Timeout timer -- checks for stalled combos every 500ms
        self.create_timer(0.5, self._check_timeout)

        logger.info(
            "Drill manager initialised (%d drills loaded)", len(self._catalogue)
        )

    # ------------------------------------------------------------------
    # Config loading
    # ------------------------------------------------------------------

    def _load_drills(self, yaml_path: str) -> None:
        """Load drill definitions from YAML config."""
        if not yaml_path:
            ws_root = Path(__file__).resolve().parents[3]
            yaml_path = str(ws_root / "config" / "drills.yaml")

        path = Path(yaml_path)
        if not path.exists():
            logger.warning("Drills YAML not found: %s", yaml_path)
            return

        try:
            with open(path, "r") as fh:
                raw = yaml.safe_load(fh) or {}
        except Exception as exc:
            logger.error("Failed to parse drills YAML: %s", exc)
            return

        self._difficulty_params = raw.get("difficulty", {})
        for level in (Difficulty.BEGINNER, Difficulty.INTERMEDIATE, Difficulty.ADVANCED):
            for drill in raw.get(level, []):
                key = f"{level}/{drill['name']}"
                drill["difficulty"] = level
                self._catalogue[key] = drill

        logger.info(
            "Loaded %d drills across %d difficulty levels",
            len(self._catalogue),
            len(self._difficulty_params),
        )

    # ------------------------------------------------------------------
    # Service handler
    # ------------------------------------------------------------------

    def _handle_start(
        self, request: StartDrill.Request, response: StartDrill.Response
    ) -> StartDrill.Response:
        """Handle StartDrill service -- begin a new drill session."""
        difficulty = request.difficulty or Difficulty.BEGINNER
        key = f"{difficulty}/{request.drill_name}"
        definition = self._catalogue.get(key)

        if definition is None:
            response.success = False
            response.message = f"Drill not found: {key}"
            logger.warning("StartDrill failed: unknown drill '%s'", key)
            return response

        # Resolve tolerance from difficulty params or default
        diff_params = self._difficulty_params.get(difficulty, {})
        tolerance = float(diff_params.get("timing_tolerance_ms", self._default_tolerance))

        self._drill = _ActiveDrill(definition, tolerance)
        self._combo_start_time = 0.0

        # Publish the drill definition
        defn_msg = DrillDefinition()
        defn_msg.drill_name = self._drill.name
        defn_msg.difficulty = difficulty
        defn_msg.combo_sequence = list(self._drill.combo)
        defn_msg.total_combos = self._drill.total_reps
        defn_msg.target_speed = self._drill.timing_ms
        self._pub_definition.publish(defn_msg)

        response.success = True
        response.drill_id = key
        response.message = f"Drill '{self._drill.name}' started ({self._drill.total_reps} reps)"
        logger.info("Drill started: %s (tolerance=%.0fms)", key, tolerance)
        return response

    # ------------------------------------------------------------------
    # Subscriber callbacks
    # ------------------------------------------------------------------

    def _on_session_state(self, msg: SessionState) -> None:
        """Track session lifecycle."""
        self._session_active = msg.state == SSConst.ACTIVE
        if msg.state == SSConst.COMPLETE and self._drill is not None:
            logger.info("Session complete -- drill stopped")
            self._drill = None

    def _on_punch(self, msg: ConfirmedPunch) -> None:
        """Process each confirmed punch against the active drill combo."""
        if self._drill is None or not self._session_active:
            return

        now = msg.timestamp if msg.timestamp > 0 else time.time()

        # First punch of a new combo attempt
        if not self._drill.detected:
            self._combo_start_time = now
            self._emit_event("combo_started", now)

        self._drill.detected.append(msg.punch_type)
        self._drill.punch_times.append(now)

        expected_len = len(self._drill.combo)

        # Check for wrong punch at this position
        idx = len(self._drill.detected) - 1
        if idx < expected_len and self._drill.detected[idx] != self._drill.combo[idx]:
            self._finish_attempt(now, partial=True)
            return

        # Check if full combo is complete
        if len(self._drill.detected) >= expected_len:
            self._finish_attempt(now, partial=False)

    # ------------------------------------------------------------------
    # Combo evaluation
    # ------------------------------------------------------------------

    def _finish_attempt(self, timestamp: float, *, partial: bool) -> None:
        """Score a completed or partially completed combo attempt."""
        d = self._drill
        if d is None:
            return

        expected = d.combo
        detected = d.detected
        accuracy = self._compute_accuracy(expected, detected)
        timing_score = self._compute_timing_score(d)

        if partial or accuracy < 0.5:
            event_type = "combo_partial" if detected else "combo_missed"
            d.reps_missed += 1
            d.current_streak = 0
        else:
            event_type = "combo_completed"
            d.reps_completed += 1
            d.accuracy_sum += accuracy
            d.timing_sum += timing_score
            d.current_streak += 1
            d.best_streak = max(d.best_streak, d.current_streak)

        self._emit_event(event_type, timestamp, accuracy, timing_score, detected, expected)
        self._emit_progress(timestamp)

        # Reset for next attempt
        d.detected = []
        d.punch_times = []
        self._combo_start_time = 0.0

        # Check if drill is finished
        total_attempts = d.reps_completed + d.reps_missed
        if total_attempts >= d.total_reps:
            logger.info(
                "Drill '%s' finished: %d/%d completed, best streak=%d",
                d.name, d.reps_completed, total_attempts, d.best_streak,
            )
            self._drill = None

    @staticmethod
    def _compute_accuracy(expected: List[str], detected: List[str]) -> float:
        """Fraction of expected punches matched in order."""
        if not expected:
            return 1.0
        matches = sum(
            1 for e, d in zip(expected, detected) if e == d
        )
        return matches / len(expected)

    @staticmethod
    def _compute_timing_score(drill: _ActiveDrill) -> float:
        """Score 0-1 based on inter-punch timing vs target."""
        if len(drill.punch_times) < 2 or drill.timing_ms <= 0:
            return 1.0
        target_s = drill.timing_ms / 1000.0
        gaps = [
            drill.punch_times[i + 1] - drill.punch_times[i]
            for i in range(len(drill.punch_times) - 1)
        ]
        errors = [abs(g - target_s) for g in gaps]
        avg_error = sum(errors) / len(errors)
        # Score: 1.0 when perfect, decays toward 0 as error grows
        return max(0.0, 1.0 - avg_error / max(target_s, 0.1))

    # ------------------------------------------------------------------
    # Timeout
    # ------------------------------------------------------------------

    def _check_timeout(self) -> None:
        """Mark a combo as missed if no punch arrives within tolerance."""
        if self._drill is None or self._combo_start_time == 0.0:
            return
        elapsed = time.time() - self._combo_start_time
        expected_duration_s = (
            len(self._drill.combo) * (self._drill.timing_ms / 1000.0)
            + self._drill.tolerance_s * 2
        )
        if elapsed > expected_duration_s:
            self._finish_attempt(time.time(), partial=True)

    # ------------------------------------------------------------------
    # Message helpers
    # ------------------------------------------------------------------

    def _emit_event(
        self,
        event_type: str,
        timestamp: float,
        accuracy: float = 0.0,
        timing_score: float = 0.0,
        detected: Optional[List[str]] = None,
        expected: Optional[List[str]] = None,
    ) -> None:
        """Publish a DrillEvent."""
        msg = DrillEvent()
        msg.timestamp = timestamp
        msg.event_type = event_type
        msg.combo_index = (
            (self._drill.reps_completed + self._drill.reps_missed)
            if self._drill
            else 0
        )
        msg.accuracy = accuracy
        msg.timing_score = timing_score
        msg.detected_punches = detected or []
        msg.expected_punches = expected or []
        self._pub_event.publish(msg)

    def _emit_progress(self, timestamp: float) -> None:
        """Publish a DrillProgress."""
        d = self._drill
        if d is None:
            return
        total_attempts = d.reps_completed + d.reps_missed
        msg = DrillProgress()
        msg.timestamp = timestamp
        msg.combos_completed = d.reps_completed
        msg.combos_remaining = max(0, d.total_reps - total_attempts)
        msg.overall_accuracy = (
            d.accuracy_sum / d.reps_completed if d.reps_completed > 0 else 0.0
        )
        msg.current_streak = float(d.current_streak)
        msg.best_streak = d.best_streak
        self._pub_progress.publish(msg)


def main(args: list[str] | None = None) -> None:
    """Entry point for the drill manager node."""
    rclpy.init(args=args)
    node = DrillManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
