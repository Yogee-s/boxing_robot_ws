"""Full-screen sparring session — clean dark layout with live stats."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from boxbunny_gui.theme import Color, Icon, Size, font, badge_style
from boxbunny_gui.widgets import PunchCounter, TimerDisplay

if TYPE_CHECKING:
    from boxbunny_gui.gui_bridge import GuiBridge
    from boxbunny_gui.nav.router import PageRouter

logger = logging.getLogger(__name__)


def _stat_box(title: str, value: str, accent: str) -> tuple:
    """Compact stat display. Returns (widget, value_label)."""
    box = QWidget()
    box.setFixedHeight(70)
    box.setStyleSheet(f"""
        QWidget {{
            background-color: #131920;
            border: 1px solid #1E2832;
            border-left: 3px solid {accent};
            border-radius: {Size.RADIUS}px;
        }}
    """)
    lay = QVBoxLayout(box)
    lay.setContentsMargins(14, 6, 14, 6)
    lay.setSpacing(0)

    hdr = QLabel(title)
    hdr.setAlignment(Qt.AlignCenter)
    hdr.setStyleSheet(
        f"font-size: 10px; font-weight: 700; color: {Color.TEXT_DISABLED};"
        " letter-spacing: 0.8px; background: transparent; border: none;"
    )
    lay.addWidget(hdr)

    val = QLabel(value)
    val.setAlignment(Qt.AlignCenter)
    val.setStyleSheet(
        f"font-size: 24px; font-weight: 700; color: {Color.TEXT};"
        " background: transparent; border: none;"
    )
    lay.addWidget(val)
    return box, val


class SparringSessionPage(QWidget):
    """Live sparring session with offense/defense tracking."""

    def __init__(
        self,
        router: PageRouter,
        bridge: Optional[GuiBridge] = None,
        sound: Optional[Any] = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._router = router
        self._bridge = bridge
        self._sound = sound
        self._config: Dict[str, Any] = {}
        self._total_attacks: int = 0
        self._blocks_detected: int = 0
        self._block_frames: int = 0  # consecutive block frames >70%
        self._block_counted: bool = False  # True after this block pose is counted
        self._punch_dist: Dict[str, int] = {}
        self._username: str = ""
        self._session_active: bool = False
        self._session_id: str = ""
        self._build_ui()
        self._connect_bridge()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(28, 10, 28, 22)
        root.setSpacing(0)

        # ── Top: round + mode ────────────────────────────────────────────
        top = QHBoxLayout()
        self._round_lbl = QLabel("Round 1/3")
        self._round_lbl.setStyleSheet(
            f"font-size: 15px; font-weight: 700; color: {Color.TEXT};"
            " background-color: #1A1214;"
            " border: 1px solid #3D1A22;"
            f" border-left: 3px solid {Color.DANGER};"
            f" border-radius: {Size.RADIUS_SM}px;"
            " padding: 6px 16px;"
        )
        top.addWidget(self._round_lbl)
        top.addStretch()
        mode_lbl = QLabel("SPARRING")
        mode_lbl.setStyleSheet(badge_style(Color.DANGER))
        top.addWidget(mode_lbl)
        root.addLayout(top)

        # ── Timer (main visual) ──────────────────────────────────────────
        self._timer = TimerDisplay(
            font_size=Size.TEXT_TIMER_SM, show_ring=True
        )
        self._timer.finished.connect(self._on_timer_done)
        root.addWidget(self._timer, stretch=1)

        root.addSpacing(6)

        # ── Incoming attack indicator ────────────────────────────────────
        attack_row = QHBoxLayout()
        attack_tag = QLabel("INCOMING")
        attack_tag.setStyleSheet(
            f"font-size: 10px; font-weight: 700; color: {Color.TEXT_DISABLED};"
            " letter-spacing: 1px;"
        )
        attack_row.addWidget(attack_tag)
        self._attack_lbl = QLabel("Waiting...")
        self._attack_lbl.setStyleSheet(
            f"font-size: 16px; font-weight: 700; color: {Color.WARNING};"
        )
        self._attack_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        attack_row.addWidget(self._attack_lbl, stretch=1)
        root.addLayout(attack_row)

        root.addSpacing(10)

        # ── Stats row ────────────────────────────────────────────────────
        stats_row = QHBoxLayout()
        stats_row.setSpacing(10)

        self._punch_counter = PunchCounter(label="YOUR PUNCHES")
        stats_row.addWidget(self._punch_counter)

        counters_box, self._counters_lbl = _stat_box("COUNTERS", "0", Color.SUCCESS)
        self._counter_count: int = 0
        stats_row.addWidget(counters_box)

        # CV prediction + FPS (fixed width so layout doesn't jump)
        cv_box, self._cv_pred_lbl = _stat_box("CV MODEL", "--", Color.INFO)
        cv_box.setFixedWidth(200)
        stats_row.addWidget(cv_box)

        fps_box, self._cv_fps_lbl = _stat_box("FPS", "--", Color.WARNING)
        fps_box.setFixedWidth(70)
        stats_row.addWidget(fps_box)

        root.addLayout(stats_row)

        root.addSpacing(10)

        # ── Stop button ──────────────────────────────────────────────────
        bottom = QHBoxLayout()
        bottom.addStretch()
        self._btn_stop = QPushButton(f"{Icon.STOP}  STOP")
        self._btn_stop.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_stop.setFixedSize(180, 48)
        self._btn_stop.setStyleSheet(f"""
            QPushButton {{
                background-color: {Color.DANGER}; color: white;
                font-size: 15px; font-weight: 700;
                border: none; border-radius: {Size.RADIUS}px;
            }}
            QPushButton:hover {{ background-color: {Color.DANGER_DARK}; }}
        """)
        self._btn_stop.clicked.connect(self._on_stop)
        bottom.addWidget(self._btn_stop)
        bottom.addStretch()
        root.addLayout(bottom)

    def _connect_bridge(self) -> None:
        if self._bridge is None:
            return
        self._bridge.punch_confirmed.connect(self._on_punch)
        self._bridge.defense_event.connect(self._on_defense)
        self._bridge.debug_info.connect(self._on_debug_info)
        self._bridge.strike_complete.connect(self._on_strike_complete)
        # Counter-punches: listen to RobotCommand directly so the counter
        # ticks the instant sparring_engine fires, bypassing the
        # strike_complete pipeline (V4 GUI publishes a duplicate payload
        # without the source tag, which used to mask counters).
        self._bridge.robot_command_issued.connect(self._on_robot_command_issued)

    _PT_DISPLAY = {
        "jab": "Jab", "cross": "Cross",
        "left_hook": "L Hook", "right_hook": "R Hook",
        "left_uppercut": "L Upper", "right_uppercut": "R Upper",
    }

    def _on_punch(self, data: Dict[str, Any]) -> None:
        if not self._session_active:
            return
        self._punch_counter.increment()
        import time as _t
        self._last_user_punch_t = _t.time()
        # Track punch distribution
        pt = data.get("punch_type", "?")
        display = self._PT_DISPLAY.get(pt, pt)
        self._punch_dist[display] = self._punch_dist.get(display, 0) + 1
        # CV prediction label is handled by _on_debug_info (raw CV only).
        # Fused results are only shown on the results page.

    def _on_defense(self, data: Dict[str, Any]) -> None:
        if not self._session_active:
            return

    def _on_strike_complete(self, data: Dict[str, Any]) -> None:
        """Robot arm finished — count as a robot attack."""
        if not self._session_active:
            return
        self._total_attacks += 1
        _PUNCH_NAMES = {
            "1": "JAB", "2": "CROSS", "3": "L HOOK", "4": "R HOOK",
            "5": "L UPPER", "6": "R UPPER",
        }
        code = data.get("punch_code", "")
        name = _PUNCH_NAMES.get(code, code)
        self._attack_lbl.setText(name)

    def _on_robot_command_issued(self, data: Dict[str, Any]) -> None:
        """Tick the COUNTERS counter the moment sparring_engine fires one."""
        if not self._session_active:
            return
        if data.get("command_type") != "punch":
            return
        if data.get("source") != "counter":
            return
        self._counter_count += 1
        self._counters_lbl.setText(str(self._counter_count))

    def _on_debug_info(self, data: Dict[str, Any]) -> None:
        """Show raw CV prediction + FPS."""
        if not self._session_active:
            return
        action = data.get("action", "idle")
        confidence = data.get("confidence", 0.0)
        fps = data.get("fps", 0.0)
        _COLORS = {
            "jab": Color.JAB, "cross": Color.CROSS,
            "left_hook": Color.L_HOOK, "right_hook": Color.R_HOOK,
            "left_uppercut": Color.L_UPPERCUT, "right_uppercut": Color.R_UPPERCUT,
            "block": Color.BLOCK,
        }
        color = _COLORS.get(action, Color.TEXT_DISABLED)
        fps_color = Color.SUCCESS if fps >= 25 else (Color.WARNING if fps >= 15 else Color.DANGER)
        self._cv_fps_lbl.setText(f"{fps:.0f}")
        self._cv_fps_lbl.setStyleSheet(
            f"font-size: 24px; font-weight: 700; color: {fps_color};"
            " background: transparent; border: none;")
        # Count blocks: must hold block pose for 10+ consecutive frames
        # (>70% confidence) before counting. Resets when pose changes.
        if action == "block" and confidence >= 0.7:
            self._block_frames += 1
            if self._block_frames >= 10 and not self._block_counted:
                self._block_counted = True
                self._blocks_detected += 1
        else:
            self._block_frames = 0
            self._block_counted = False

        if action in ("idle", ""):
            self._cv_pred_lbl.setText("IDLE")
            self._cv_pred_lbl.setStyleSheet(
                f"font-size: 16px; font-weight: 700; color: {Color.TEXT_DISABLED};"
                " background: transparent; border: none;")
        else:
            name = action.upper().replace("_", " ")
            self._cv_pred_lbl.setText(f"{name} {confidence:.0%}")
            self._cv_pred_lbl.setStyleSheet(
                f"font-size: 14px; font-weight: 700; color: {color};"
                " background: transparent; border: none;")

    def _end_session(self) -> None:
        """End the ROS session so engines deactivate."""
        if not self._bridge:
            return
        if self._session_id:
            self._bridge.call_end_session(
                session_id=self._session_id,
                callback=lambda ok, summary, msg: logger.info(
                    "Sparring session ended: ok=%s", ok),
            )
            self._session_id = ""

    def _on_timer_done(self) -> None:
        if not self._session_active:
            return
        if self._sound:
            self._sound.play("bell_end")
        self._session_active = False
        self._end_session()
        self._go_results()

    def _on_stop(self) -> None:
        self._session_active = False
        self._timer.pause()
        self._end_session()
        logger.info("Sparring stopped by user")
        self._go_results()

    def _go_results(self) -> None:
        defense_rate = round(
            self._blocks_detected / max(self._total_attacks, 1), 3
        )
        self._router.replace(
            "sparring_results", config=self._config,
            username=self._username,
            total_punches=self._punch_counter._count,
            robot_attacks=self._total_attacks,
            blocks_detected=self._blocks_detected,
            punch_dist=dict(self._punch_dist),
            defense_rate=defense_rate,
        )

    def _countdown_tick(self) -> None:
        if not self._session_active:
            return
        if self._countdown_remaining > 0:
            self._timer.set_overlay(str(self._countdown_remaining))
            self._countdown_remaining -= 1
            QTimer.singleShot(1000, self._countdown_tick)
        else:
            self._timer.set_overlay("GO!")
            if self._sound:
                self._sound.play("bell_start")
            QTimer.singleShot(500, self._start_round)

    def _start_round(self) -> None:
        self._timer.clear_overlay()
        self._attack_lbl.setText("Waiting...")
        self._timer.start(self._work_time)

    def _parse_seconds(self, val: str) -> int:
        return int(val.rstrip("s")) if val.rstrip("s").isdigit() else 90

    def on_enter(self, **kwargs: Any) -> None:
        self._config = kwargs.get("config", {})
        self._username = kwargs.get("username", "")
        rounds = self._config.get("Rounds", "3")
        work_time = self._parse_seconds(
            self._config.get("Duration", self._config.get("Work", "90s"))
        )
        self._session_active = True
        self._round_lbl.setText(f"Round 1/{rounds}")
        self._total_attacks = 0
        self._counter_count = 0
        self._last_user_punch_t = 0.0
        self._blocks_detected = 0
        self._block_frames = 0
        self._block_counted = False
        self._punch_dist = {}
        self._counters_lbl.setText("0")
        self._attack_lbl.setText("Waiting...")
        self._cv_pred_lbl.setText("--")
        self._cv_fps_lbl.setText("--")
        # cv_hdr_lbl is part of the _stat_box header — no separate handle needed
        self._punch_counter.set_count(0)
        self._work_time = work_time
        # Start ROS session so sparring_engine activates
        self._start_ros_session()
        # 3-second countdown
        self._countdown_remaining = 3
        self._timer.set_time(work_time)
        self._timer.set_overlay("Get Ready")
        QTimer.singleShot(1000, self._countdown_tick)
        logger.debug("SparringSessionPage entered")

    def _start_ros_session(self) -> None:
        """Start the ROS session for sparring mode."""
        if self._bridge is None:
            return
        import json
        config_json = json.dumps(self._config, default=str)
        self._bridge.call_start_session(
            mode="sparring",
            difficulty=self._config.get("difficulty", "medium"),
            config_json=config_json,
            username=self._username,
            callback=self._on_session_started,
        )

    def _on_session_started(
        self, success: bool, session_id: str, message: str,
    ) -> None:
        if success:
            self._session_id = session_id
            logger.info("Sparring session started: %s", session_id)
        else:
            # Previous session might still be cleaning up — retry after 2s
            logger.warning("Session start failed: %s — retrying...", message)
            QTimer.singleShot(2000, self._start_ros_session)

    def on_leave(self) -> None:
        self._session_active = False
        self._timer.pause()
        self._end_session()
