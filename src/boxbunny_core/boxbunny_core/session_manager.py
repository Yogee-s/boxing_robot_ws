"""Session lifecycle manager for BoxBunny.

Manages training session start/pause/stop, countdown, rounds, rest periods.
Publishes SessionState (the signal that triggers IMU mode switch).
Accumulates punch and defense data. Auto-saves periodically.
"""

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import rclpy
from rclpy.node import Node

from boxbunny_msgs.msg import (
    ConfirmedPunch,
    DefenseEvent,
    HeightCommand,
    SessionConfig,
    SessionPunchSummary,
    SessionState,
    UserTracking,
)
from boxbunny_msgs.srv import EndSession, StartSession

logger = logging.getLogger("boxbunny.session_manager")


@dataclass
class RoundData:
    """Data collected during a single round."""

    punches: List[Dict] = field(default_factory=list)
    defense_events: List[Dict] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0


@dataclass
class SessionData:
    """All data for an active session."""

    session_id: str = ""
    mode: str = "training"
    difficulty: str = "beginner"
    username: str = "guest"
    config: Dict = field(default_factory=dict)
    rounds: List[RoundData] = field(default_factory=list)
    current_round: int = 0
    total_rounds: int = 3
    work_time_s: int = 180
    rest_time_s: int = 60
    started_at: float = 0.0
    total_punches: int = 0
    punch_distribution: Dict[str, int] = field(default_factory=dict)
    force_distribution: Dict[str, float] = field(default_factory=dict)
    force_counts: Dict[str, int] = field(default_factory=dict)
    pad_distribution: Dict[str, int] = field(default_factory=dict)
    robot_punches_thrown: int = 0
    robot_punches_landed: int = 0
    defense_breakdown: Dict[str, int] = field(default_factory=dict)
    depth_samples: List[float] = field(default_factory=list)
    lateral_samples: List[float] = field(default_factory=list)


class SessionManager(Node):
    """Manages the lifecycle of training sessions."""

    def __init__(self) -> None:
        super().__init__("session_manager")

        # Parameters
        self.declare_parameter("countdown_seconds", 3)
        self.declare_parameter("autosave_interval_s", 10.0)
        self._countdown_seconds = self.get_parameter("countdown_seconds").value
        autosave_interval = self.get_parameter("autosave_interval_s").value

        # State
        self._current_state = "idle"
        self._session: Optional[SessionData] = None
        self._countdown_remaining = 0
        self._round_timer_start = 0.0
        self._rest_timer_start = 0.0
        self._last_autosave = 0.0
        self._height_adjusted = False

        # Publishers
        self._pub_state = self.create_publisher(SessionState, "/boxbunny/session/state", 10)
        self._pub_summary = self.create_publisher(
            SessionPunchSummary, "/boxbunny/punch/session_summary", 10
        )
        self._pub_height = self.create_publisher(HeightCommand, "/boxbunny/robot/height", 10)

        # Subscribers
        self.create_subscription(
            ConfirmedPunch, "/boxbunny/punch/confirmed", self._on_confirmed_punch, 50
        )
        self.create_subscription(
            DefenseEvent, "/boxbunny/punch/defense", self._on_defense_event, 50
        )
        self.create_subscription(
            UserTracking, "/boxbunny/cv/user_tracking", self._on_user_tracking, 10
        )
        self.create_subscription(
            SessionConfig, "/boxbunny/session/config", self._on_session_config, 10
        )

        # Services
        self.create_service(StartSession, "/boxbunny/session/start", self._handle_start)
        self.create_service(EndSession, "/boxbunny/session/end", self._handle_end)

        # Timers
        self.create_timer(1.0, self._tick)
        self.create_timer(autosave_interval, self._autosave)

        self._publish_state()
        logger.info("Session manager initialized")

    def _publish_state(self) -> None:
        """Publish current session state."""
        msg = SessionState()
        msg.state = self._current_state
        msg.mode = self._session.mode if self._session else ""
        msg.username = self._session.username if self._session else "guest"
        self._pub_state.publish(msg)

    def _set_state(self, new_state: str) -> None:
        """Update session state and publish."""
        if new_state != self._current_state:
            old_state = self._current_state
            self._current_state = new_state
            self._publish_state()
            logger.info("Session state: %s -> %s", old_state, new_state)

    def _handle_start(
        self, request: StartSession.Request, response: StartSession.Response
    ) -> StartSession.Response:
        """Handle StartSession service request."""
        if self._current_state != "idle":
            response.success = False
            response.message = f"Cannot start: session already in state '{self._current_state}'"
            return response

        session_id = str(uuid.uuid4())[:12]
        config = json.loads(request.config_json) if request.config_json else {}

        self._session = SessionData(
            session_id=session_id,
            mode=request.mode,
            difficulty=request.difficulty,
            username=request.username or "guest",
            config=config,
            total_rounds=config.get("rounds", 3),
            work_time_s=config.get("work_time_sec", 180),
            rest_time_s=config.get("rest_time_sec", 60),
            started_at=time.time(),
        )

        self._height_adjusted = False
        self._countdown_remaining = self._countdown_seconds
        self._set_state("countdown")

        response.success = True
        response.session_id = session_id
        response.message = f"Session {session_id} starting ({request.mode})"
        logger.info("Session started: %s mode=%s user=%s",
                     session_id, request.mode, request.username)
        return response

    def _handle_end(
        self, request: EndSession.Request, response: EndSession.Response
    ) -> EndSession.Response:
        """Handle EndSession service request."""
        if self._session is None:
            response.success = False
            response.message = "No active session"
            return response

        summary = self._build_summary()
        self._publish_session_summary(summary)
        self._set_state("complete")

        response.success = True
        response.summary_json = json.dumps(summary)
        response.message = "Session ended"

        # Reset after a brief delay to allow subscribers to process
        self.create_timer(2.0, self._reset_to_idle)
        return response

    def _tick(self) -> None:
        """Called every second to manage session timers."""
        if self._session is None:
            return

        if self._current_state == "countdown":
            self._countdown_remaining -= 1
            if self._countdown_remaining <= 0:
                self._start_round()

        elif self._current_state == "active":
            elapsed = time.time() - self._round_timer_start
            if elapsed >= self._session.work_time_s:
                self._end_round()

        elif self._current_state == "rest":
            elapsed = time.time() - self._rest_timer_start
            if elapsed >= self._session.rest_time_s:
                self._countdown_remaining = self._countdown_seconds
                self._set_state("countdown")

    def _start_round(self) -> None:
        """Start a new round."""
        if self._session is None:
            return
        self._session.current_round += 1
        self._session.rounds.append(RoundData(start_time=time.time()))
        self._round_timer_start = time.time()
        self._set_state("active")
        logger.info("Round %d/%d started", self._session.current_round,
                     self._session.total_rounds)

    def _end_round(self) -> None:
        """End the current round."""
        if self._session is None or not self._session.rounds:
            return
        self._session.rounds[-1].end_time = time.time()
        logger.info("Round %d/%d ended", self._session.current_round,
                     self._session.total_rounds)

        if self._session.current_round >= self._session.total_rounds:
            summary = self._build_summary()
            self._publish_session_summary(summary)
            self._set_state("complete")
            self.create_timer(3.0, self._reset_to_idle)
        else:
            self._rest_timer_start = time.time()
            self._set_state("rest")

    def _on_confirmed_punch(self, msg: ConfirmedPunch) -> None:
        """Accumulate confirmed punch data."""
        if self._session is None or self._current_state != "active":
            return
        self._session.total_punches += 1
        pt = msg.punch_type or "unclassified"
        self._session.punch_distribution[pt] = self._session.punch_distribution.get(pt, 0) + 1
        pad = msg.pad or "unknown"
        self._session.pad_distribution[pad] = self._session.pad_distribution.get(pad, 0) + 1
        if msg.force_normalized > 0:
            self._session.force_distribution[pt] = (
                self._session.force_distribution.get(pt, 0.0) + msg.force_normalized
            )
            self._session.force_counts[pt] = self._session.force_counts.get(pt, 0) + 1
        if self._session.rounds:
            self._session.rounds[-1].punches.append({
                "type": pt, "pad": pad, "force": msg.force_normalized,
                "cv_conf": msg.cv_confidence, "ts": msg.timestamp,
            })

    def _on_defense_event(self, msg: DefenseEvent) -> None:
        """Accumulate defense event data."""
        if self._session is None or self._current_state != "active":
            return
        self._session.robot_punches_thrown += 1
        if msg.struck:
            self._session.robot_punches_landed += 1
        dt = msg.defense_type or "unknown"
        self._session.defense_breakdown[dt] = self._session.defense_breakdown.get(dt, 0) + 1
        if self._session.rounds:
            self._session.rounds[-1].defense_events.append({
                "arm": msg.arm, "struck": msg.struck, "type": dt, "ts": msg.timestamp,
            })

    def _on_user_tracking(self, msg: UserTracking) -> None:
        """Collect depth and movement data."""
        if self._session is None or not msg.user_detected:
            return
        self._session.depth_samples.append(msg.depth)
        self._session.lateral_samples.append(abs(msg.lateral_displacement))

        # Height auto-adjustment during countdown
        if self._current_state == "countdown" and not self._height_adjusted:
            height_msg = HeightCommand()
            height_msg.current_height_px = msg.bbox_top_y
            height_msg.target_height_px = 0.15 * 540  # 15% of 540p frame
            height_msg.action = "adjust"
            self._pub_height.publish(height_msg)
            self._height_adjusted = True

    def _on_session_config(self, msg: SessionConfig) -> None:
        """Handle session config updates (e.g., from GUI)."""
        pass  # Config is primarily set via StartSession service

    def _build_summary(self) -> Dict:
        """Build session summary statistics."""
        s = self._session
        if s is None:
            return {}
        avg_force = {}
        for pt, total in s.force_distribution.items():
            count = s.force_counts.get(pt, 1)
            avg_force[pt] = round(total / max(count, 1), 3)
        defense_rate = 0.0
        if s.robot_punches_thrown > 0:
            defended = s.robot_punches_thrown - s.robot_punches_landed
            defense_rate = defended / s.robot_punches_thrown
        avg_depth = sum(s.depth_samples) / max(len(s.depth_samples), 1)
        depth_range = (max(s.depth_samples) - min(s.depth_samples)) if s.depth_samples else 0.0
        lateral_total = sum(s.lateral_samples)

        return {
            "session_id": s.session_id,
            "mode": s.mode,
            "difficulty": s.difficulty,
            "total_punches": s.total_punches,
            "punch_distribution": s.punch_distribution,
            "force_distribution": avg_force,
            "pad_distribution": s.pad_distribution,
            "robot_punches_thrown": s.robot_punches_thrown,
            "robot_punches_landed": s.robot_punches_landed,
            "defense_rate": round(defense_rate, 3),
            "defense_breakdown": s.defense_breakdown,
            "avg_depth": round(avg_depth, 3),
            "depth_range": round(depth_range, 3),
            "lateral_movement": round(lateral_total, 1),
            "rounds_completed": s.current_round,
            "duration_sec": round(time.time() - s.started_at, 1),
        }

    def _publish_session_summary(self, summary: Dict) -> None:
        """Publish the session summary message."""
        msg = SessionPunchSummary()
        msg.total_punches = summary.get("total_punches", 0)
        msg.punch_distribution_json = json.dumps(summary.get("punch_distribution", {}))
        msg.force_distribution_json = json.dumps(summary.get("force_distribution", {}))
        msg.pad_distribution_json = json.dumps(summary.get("pad_distribution", {}))
        msg.robot_punches_thrown = summary.get("robot_punches_thrown", 0)
        msg.robot_punches_landed = summary.get("robot_punches_landed", 0)
        msg.defense_rate = summary.get("defense_rate", 0.0)
        msg.defense_type_breakdown_json = json.dumps(summary.get("defense_breakdown", {}))
        msg.avg_depth = summary.get("avg_depth", 0.0)
        msg.depth_range = summary.get("depth_range", 0.0)
        msg.lateral_movement = summary.get("lateral_movement", 0.0)
        msg.session_duration_sec = summary.get("duration_sec", 0.0)
        msg.rounds_completed = summary.get("rounds_completed", 0)
        self._pub_summary.publish(msg)

    def _autosave(self) -> None:
        """Periodically save session data (crash recovery)."""
        if self._session is None or self._current_state == "idle":
            return
        logger.debug("Autosaving session %s (%d punches)",
                      self._session.session_id, self._session.total_punches)
        # In production, this would write to the database via the db manager

    def _reset_to_idle(self) -> None:
        """Reset session state to idle."""
        self._session = None
        self._set_state("idle")


def main(args=None) -> None:
    """Entry point for the session manager node."""
    rclpy.init(args=args)
    node = SessionManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
