"""Active training session — cycles through combo punches at configured speed.

Timer with progress bar, live combo sequence display with current-punch
highlight, round counter, punch counter, and stop button.
"""
from __future__ import annotations

import logging
import random
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from boxbunny_gui.theme import Color, Icon, Size, font, badge_style, back_link_style
from boxbunny_gui.widgets import PunchCounter, TimerDisplay

if TYPE_CHECKING:
    from boxbunny_gui.curriculum import ComboCurriculum
    from boxbunny_gui.gui_bridge import GuiBridge
    from boxbunny_gui.nav.router import PageRouter

logger = logging.getLogger(__name__)

_PUNCH_NAMES: Dict[str, str] = {
    "1": "JAB", "2": "CROSS", "3": "L HOOK", "4": "R HOOK",
    "5": "L UPPER", "6": "R UPPER",
    "1b": "BODY JAB", "2b": "BODY CROSS",
    "3b": "BODY HOOK", "4b": "BODY R HOOK",
    "slip": "SLIP-L", "slipr": "SLIP-R",
    "block": "BLOCK-L", "blockr": "BLOCK-R",
}

_PUNCH_COLORS: Dict[str, str] = {
    "1": Color.JAB, "2": Color.CROSS, "3": Color.L_HOOK,
    "4": Color.R_HOOK, "5": Color.L_UPPERCUT, "6": Color.R_UPPERCUT,
    "1b": Color.JAB, "2b": Color.CROSS, "3b": Color.L_HOOK,
    "4b": Color.R_HOOK,
    "slip": Color.BLOCK, "slipr": Color.BLOCK,
    "block": Color.BLOCK, "blockr": Color.BLOCK,
}

# Defense token → robot punch code mapping
# Slip-L: robot throws jab (1) for user to slip
# Slip-R: robot throws cross (2) for user to slip
# Block-L: robot throws a random left punch (1, 3, or 5)
# Block-R: robot throws a random right punch (2, 4, or 6)
_DEFENSE_PUNCH_MAP: Dict[str, list] = {
    "slip": ["1"],           # Jab
    "slipr": ["2"],          # Cross
    "block": ["1", "3", "5"],  # Any left punch
    "blockr": ["2", "4", "6"],  # Any right punch
}
_DEFENSIVE_TOKENS = set(_DEFENSE_PUNCH_MAP.keys())

# Robot command substitution: some punches can't be safely thrown by the
# robot (e.g. uppercuts go upward). Substitute with same-side straight punch.
_ROBOT_SUBSTITUTE: Dict[str, str] = {
    "5": "1",  # Left uppercut → robot throws jab (same side)
    "6": "2",  # Right uppercut → robot throws cross (same side)
}


def _parse_speed_ms(speed_str: str) -> int:
    """Parse speed string like 'Medium (2s)' to milliseconds."""
    m = re.search(r"([\d.]+)\s*s", speed_str)
    if m:
        return int(float(m.group(1)) * 1000)
    return 2000  # default Medium


class TrainingSessionPage(QWidget):
    """Active training session that cycles combo punches at the configured speed."""

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
        self._current_round: int = 1
        self._total_rounds: int = 3
        self._curriculum: Optional[ComboCurriculum] = None
        self._combo_id: Optional[str] = None
        self._difficulty: Optional[str] = None
        self._username: str = ""
        self._session_active: bool = False
        self._counting_active: bool = False
        self._paused: bool = False
        self._session_id: str = ""

        # Tracking state — must exist before _connect_bridge() because
        # ble_status_changed can fire immediately after signal connection
        # and _on_ble_status reads these attributes.
        self._tracking_active: bool = False

        # Drill cycling state
        self._combo_tokens: List[str] = []
        self._seq_boundaries: List[int] = []  # cumulative end indices per sequence
        self._drill_idx: int = 0
        self._combos_completed: int = 0
        self._reps_completed: int = 0
        self._robot_speed: str = "medium"
        self._waiting_for_strike: bool = False
        self._drill_generation: int = 0  # invalidates stale timeouts
        self._last_strike_complete_t: float = 0.0  # debounce duplicate completions
        # CV prediction accumulator (group consecutive high-conf detections)
        self._cv_current_type: str = ""
        self._cv_current_frames: int = 0
        self._cv_current_conf_sum: float = 0.0
        self._cv_current_peak: float = 0.0
        self._cv_events: List[Dict[str, Any]] = []  # completed events
        self._drill_timer = QTimer(self)
        self._drill_timer.timeout.connect(self._drill_tick)

        self._build_ui()
        self._connect_bridge()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(28, 10, 28, 22)
        root.setSpacing(0)

        # ── Top: back button + round + combo name + mode badge ───────────
        top = QHBoxLayout()
        self._btn_back = QPushButton(f"{Icon.BACK}  Back")
        self._btn_back.setStyleSheet(back_link_style())
        self._btn_back.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_back.clicked.connect(self._on_back)
        self._btn_back.setVisible(False)
        top.addWidget(self._btn_back)
        top.addSpacing(12)

        self._round_lbl = QLabel("Round 1/3")
        self._round_lbl.setStyleSheet(
            f"font-size: 15px; font-weight: 700; color: {Color.TEXT};"
            " background-color: #1A1510;"
            " border: 1px solid #3D2E1A;"
            f" border-left: 3px solid {Color.PRIMARY};"
            f" border-radius: {Size.RADIUS_SM}px;"
            " padding: 6px 16px;"
        )
        top.addWidget(self._round_lbl)
        top.addSpacing(16)

        self._combo_name_lbl = QLabel("")
        self._combo_name_lbl.setStyleSheet(
            f"font-size: 18px; font-weight: 700; color: {Color.TEXT};"
        )
        top.addWidget(self._combo_name_lbl)
        top.addStretch()

        mode_lbl = QLabel("TRAINING")
        mode_lbl.setStyleSheet(badge_style(Color.PRIMARY))
        top.addWidget(mode_lbl)
        root.addLayout(top)

        # ── Timer ────────────────────────────────────────────────────────
        self._timer = TimerDisplay(font_size=Size.TEXT_TIMER_SM, show_ring=True)
        self._timer.finished.connect(self._on_timer_done)
        root.addWidget(self._timer, stretch=1)

        # ── Current punch cue (VERY BIG) ────────────────────────────────
        self._cue_lbl = QLabel("")
        self._cue_lbl.setAlignment(Qt.AlignCenter)
        self._cue_lbl.setFixedHeight(70)
        self._cue_lbl.setStyleSheet(
            f"font-size: 48px; font-weight: 800; color: {Color.PRIMARY};"
            " letter-spacing: 3px;"
        )
        root.addWidget(self._cue_lbl)

        # ── Next punch preview ───────────────────────────────────────────
        self._next_lbl = QLabel("")
        self._next_lbl.setAlignment(Qt.AlignCenter)
        self._next_lbl.setStyleSheet(
            f"font-size: 14px; font-weight: 600; color: {Color.TEXT_DISABLED};"
        )
        root.addWidget(self._next_lbl)

        root.addSpacing(6)

        # ── Combo sequence with highlight (scrollable) ───────────────────
        from PySide6.QtWidgets import QScrollArea, QScroller
        self._combo_scroll = QScrollArea()
        self._combo_scroll.setWidgetResizable(True)
        self._combo_scroll.setFixedHeight(50)
        self._combo_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._combo_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._combo_scroll.setStyleSheet(
            " background-color: #131920;"
            " border: 1px solid #1E2832;"
            f" border-radius: {Size.RADIUS}px;"
        )
        self._combo_seq_lbl = QLabel("")
        self._combo_seq_lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self._combo_seq_lbl.setTextFormat(Qt.TextFormat.RichText)
        self._combo_seq_lbl.setStyleSheet(
            f"font-size: 16px; font-weight: 700; color: {Color.TEXT_SECONDARY};"
            " background: transparent; border: none;"
            " padding: 6px 16px;"
        )
        self._combo_scroll.setWidget(self._combo_seq_lbl)
        QScroller.grabGesture(
            self._combo_scroll.viewport(),
            QScroller.ScrollerGestureType.LeftMouseButtonGesture,
        )
        root.addWidget(self._combo_scroll)

        root.addSpacing(10)

        # ── Stats row ────────────────────────────────────────────────────
        stats_row = QHBoxLayout()
        stats_row.setSpacing(10)

        self._punch_counter = PunchCounter(label="PUNCHES")
        stats_row.addWidget(self._punch_counter)

        stats_row.addStretch()

        # Speed selector (free training only) — dropdown + custom slider
        from boxbunny_gui.pages.training.training_config_page import _SpeedTile
        self._speed_tile = _SpeedTile(self)
        self._speed_tile.setVisible(False)
        stats_row.addWidget(self._speed_tile)

        # Combos completed counter (shown for combo drills)
        self._combos_box = QWidget()
        self._combos_box.setFixedHeight(70)
        self._combos_box.setStyleSheet(f"""
            QWidget {{
                background-color: #131920;
                border: 1px solid #1E2832;
                border-left: 3px solid {Color.SUCCESS};
                border-radius: {Size.RADIUS}px;
            }}
        """)
        combos_lay = QVBoxLayout(self._combos_box)
        combos_lay.setContentsMargins(14, 6, 14, 6)
        combos_lay.setSpacing(0)
        combos_hdr = QLabel("COMBOS")
        combos_hdr.setAlignment(Qt.AlignCenter)
        combos_hdr.setStyleSheet(
            f"font-size: 10px; font-weight: 700; color: {Color.TEXT_DISABLED};"
            " letter-spacing: 0.8px; background: transparent; border: none;"
        )
        combos_lay.addWidget(combos_hdr)
        self._combos_lbl = QLabel("0")
        self._combos_lbl.setAlignment(Qt.AlignCenter)
        self._combos_lbl.setStyleSheet(
            f"font-size: 24px; font-weight: 700; color: {Color.TEXT};"
            " background: transparent; border: none;"
        )
        combos_lay.addWidget(self._combos_lbl)
        stats_row.addWidget(self._combos_box)

        # Reps counter (shown only for multi-sequence custom drills)
        self._reps_box = QWidget()
        self._reps_box.setFixedHeight(70)
        self._reps_box.setStyleSheet(f"""
            QWidget {{
                background-color: #131920;
                border: 1px solid #1E2832;
                border-left: 3px solid {Color.PRIMARY};
                border-radius: {Size.RADIUS}px;
            }}
        """)
        reps_lay = QVBoxLayout(self._reps_box)
        reps_lay.setContentsMargins(14, 6, 14, 6)
        reps_lay.setSpacing(0)
        reps_hdr = QLabel("REPS")
        reps_hdr.setAlignment(Qt.AlignCenter)
        reps_hdr.setStyleSheet(
            f"font-size: 10px; font-weight: 700; color: {Color.TEXT_DISABLED};"
            " letter-spacing: 0.8px; background: transparent; border: none;"
        )
        reps_lay.addWidget(reps_hdr)
        self._reps_lbl = QLabel("0")
        self._reps_lbl.setAlignment(Qt.AlignCenter)
        self._reps_lbl.setStyleSheet(
            f"font-size: 24px; font-weight: 700; color: {Color.TEXT};"
            " background: transparent; border: none;"
        )
        reps_lay.addWidget(self._reps_lbl)
        self._reps_box.setVisible(False)
        stats_row.addWidget(self._reps_box)

        # CV prediction block (free training only)
        self._cv_box = QWidget()
        self._cv_box.setFixedSize(200, 70)
        self._cv_box.setStyleSheet(f"""
            QWidget {{
                background-color: #131920;
                border: 1px solid #1E2832;
                border-left: 3px solid {Color.INFO};
                border-radius: {Size.RADIUS}px;
            }}
        """)
        cv_lay = QVBoxLayout(self._cv_box)
        cv_lay.setContentsMargins(14, 6, 14, 6)
        cv_lay.setSpacing(0)
        self._cv_hdr_lbl = QLabel("CV MODEL")
        self._cv_hdr_lbl.setAlignment(Qt.AlignCenter)
        self._cv_hdr_lbl.setStyleSheet(
            f"font-size: 10px; font-weight: 700; color: {Color.TEXT_DISABLED};"
            " letter-spacing: 0.8px; background: transparent; border: none;"
        )
        cv_lay.addWidget(self._cv_hdr_lbl)
        self._cv_pred_lbl = QLabel("--")
        self._cv_pred_lbl.setAlignment(Qt.AlignCenter)
        self._cv_pred_lbl.setStyleSheet(
            f"font-size: 16px; font-weight: 700; color: {Color.TEXT_DISABLED};"
            " background: transparent; border: none;"
        )
        cv_lay.addWidget(self._cv_pred_lbl)
        self._cv_box.setVisible(False)
        stats_row.addWidget(self._cv_box)

        # FPS block (free training only)
        self._fps_box = QWidget()
        self._fps_box.setFixedHeight(70)
        self._fps_box.setStyleSheet(f"""
            QWidget {{
                background-color: #131920;
                border: 1px solid #1E2832;
                border-left: 3px solid {Color.WARNING};
                border-radius: {Size.RADIUS}px;
            }}
        """)
        fps_lay = QVBoxLayout(self._fps_box)
        fps_lay.setContentsMargins(14, 6, 14, 6)
        fps_lay.setSpacing(0)
        fps_hdr = QLabel("FPS")
        fps_hdr.setAlignment(Qt.AlignCenter)
        fps_hdr.setStyleSheet(
            f"font-size: 10px; font-weight: 700; color: {Color.TEXT_DISABLED};"
            " letter-spacing: 0.8px; background: transparent; border: none;"
        )
        fps_lay.addWidget(fps_hdr)
        self._cv_fps_lbl = QLabel("--")
        self._cv_fps_lbl.setAlignment(Qt.AlignCenter)
        self._cv_fps_lbl.setStyleSheet(
            f"font-size: 24px; font-weight: 700; color: {Color.TEXT};"
            " background: transparent; border: none;"
        )
        fps_lay.addWidget(self._cv_fps_lbl)
        self._fps_box.setVisible(False)
        stats_row.addWidget(self._fps_box)

        root.addLayout(stats_row)

        root.addSpacing(10)

        # ── Free-training tracking row (visible only in free training) ──
        # User can connect the base motor to live person-tracking from here.
        # Auto-stops on session end via _safety_stop_tracking().
        self._track_row = QWidget()
        track_lay = QHBoxLayout(self._track_row)
        track_lay.setContentsMargins(0, 0, 0, 0)
        track_lay.setSpacing(12)
        track_title = QLabel("Base Tracking")
        track_title.setStyleSheet(
            f"font-size: 14px; font-weight: 700; color: {Color.TEXT};"
            " background: transparent; border: none;"
        )
        track_lay.addWidget(track_title)
        self._track_status_lbl = QLabel("BLE: …")
        self._track_status_lbl.setStyleSheet(
            f"font-size: 12px; color: {Color.TEXT_SECONDARY};"
            " background: transparent; border: none;"
        )
        track_lay.addWidget(self._track_status_lbl)
        track_lay.addStretch()
        self._btn_track = QPushButton("Start Tracking")
        self._btn_track.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_track.setFixedHeight(40)
        self._btn_track.setMinimumWidth(160)
        self._btn_track.setEnabled(False)  # locked until BLE connects
        self._btn_track.clicked.connect(self._on_track_toggle)
        track_lay.addWidget(self._btn_track)
        self._track_row.setVisible(False)
        self._ble_connected: bool = False
        root.addWidget(self._track_row)

        root.addSpacing(10)

        # Pause + Stop buttons — centered
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        self._btn_pause = QPushButton("PAUSE")
        self._btn_pause.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_pause.setFixedSize(140, 48)
        self._btn_pause.setStyleSheet(f"""
            QPushButton {{
                background-color: {Color.SURFACE};
                color: {Color.TEXT};
                font-size: 15px; font-weight: 700;
                border: 1px solid {Color.BORDER_LIGHT};
                border-radius: {Size.RADIUS}px;
            }}
            QPushButton:hover {{
                border-color: {Color.WARNING};
                background-color: {Color.SURFACE_HOVER};
            }}
        """)
        self._btn_pause.clicked.connect(self._on_pause_resume)
        btn_row.addWidget(self._btn_pause)

        btn_row.addSpacing(12)

        self._btn_stop = QPushButton(f"{Icon.STOP}  STOP")
        self._btn_stop.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_stop.setFixedSize(140, 48)
        self._btn_stop.setStyleSheet(f"""
            QPushButton {{
                background-color: {Color.DANGER}; color: white;
                font-size: 15px; font-weight: 700;
                border: none; border-radius: {Size.RADIUS}px;
            }}
            QPushButton:hover {{ background-color: {Color.DANGER_DARK}; }}
        """)
        self._btn_stop.clicked.connect(self._on_stop)
        btn_row.addWidget(self._btn_stop)

        btn_row.addStretch()
        root.addLayout(btn_row)

    # ── Bridge signals ───────────────────────────────────────────────────

    def _connect_bridge(self) -> None:
        if self._bridge is None:
            return
        self._bridge.punch_confirmed.connect(self._on_punch)
        self._bridge.drill_progress.connect(self._on_drill_progress)
        self._bridge.coach_tip.connect(self._on_coach_tip)
        self._bridge.session_state_changed.connect(self._on_session_state)
        self._bridge.strike_complete.connect(self._on_strike_complete)
        self._bridge.debug_info.connect(self._on_debug_info)
        self._bridge.ble_status_changed.connect(self._on_ble_status)

    def _on_ble_status(self, status: Dict[str, Any]) -> None:
        self._ble_connected = bool(status.get("connected"))
        self._track_status_lbl.setText(
            "BLE: connected" if self._ble_connected else "BLE: disconnected"
        )
        # Safety: never let the user request tracking when BLE is dead.
        # If we lose the link mid-session, force the local state back to
        # idle so the next status flip starts in a sane spot.
        if not self._ble_connected and self._tracking_active:
            self._tracking_active = False
        self._apply_track_button_style(active=self._tracking_active)

    def _apply_track_button_style(self, *, active: bool) -> None:
        # Disable while BLE is offline — user can't start tracking the
        # base motor if there's nothing to send commands to.
        self._btn_track.setEnabled(self._ble_connected)
        if active:
            self._btn_track.setText("Stop Tracking")
            self._btn_track.setStyleSheet(f"""
                QPushButton {{
                    background-color: {Color.WARNING}; color: #FFFFFF;
                    font-size: 13px; font-weight: 700;
                    border: none; border-radius: {Size.RADIUS}px;
                    padding: 0 16px;
                }}
                QPushButton:hover {{ background-color: {Color.WARNING_DARK}; }}
                QPushButton:disabled {{
                    background-color: {Color.SURFACE};
                    color: {Color.TEXT_DISABLED};
                }}
            """)
        else:
            self._btn_track.setText("Start Tracking")
            self._btn_track.setStyleSheet(f"""
                QPushButton {{
                    background-color: {Color.PRIMARY}; color: #FFFFFF;
                    font-size: 13px; font-weight: 700;
                    border: none; border-radius: {Size.RADIUS}px;
                    padding: 0 16px;
                }}
                QPushButton:hover {{ background-color: {Color.PRIMARY_DARK}; }}
                QPushButton:disabled {{
                    background-color: {Color.SURFACE};
                    color: {Color.TEXT_DISABLED};
                }}
            """)

    def _on_track_toggle(self) -> None:
        if self._bridge is None:
            logger.warning("Tracking toggled but GUI bridge is offline")
            return
        if self._tracking_active:
            self._bridge.call_tracking_stop(self._on_track_stop_result)
        else:
            self._bridge.call_tracking_start(self._on_track_start_result)

    def _on_track_start_result(self, success: bool, message: str) -> None:
        if success:
            self._tracking_active = True
            self._apply_track_button_style(active=True)
            logger.info("Tracking started: %s", message)
        else:
            logger.warning("Tracking start failed: %s", message)

    def _on_track_stop_result(self, success: bool, message: str) -> None:
        self._tracking_active = False
        self._apply_track_button_style(active=False)
        if success:
            logger.info("Tracking stopped: %s", message)
        else:
            logger.warning("Tracking stop reported failure: %s", message)

    def _on_punch(self, data: Dict[str, Any]) -> None:
        if not self._counting_active:
            return
        self._punch_counter.increment()
        # CV prediction label is handled by _on_debug_info (raw CV only).
        # Fused results are only shown on the results page.

    def _on_debug_info(self, data: Dict[str, Any]) -> None:
        """Show raw CV prediction and FPS, accumulate for results."""
        if not self._session_active:
            return
        action = data.get("action", "idle")
        confidence = data.get("confidence", 0.0)
        fps = data.get("fps", 0.0)
        # Accumulate CV predictions (always, not just when CV box visible)
        if self._counting_active:
            self._accumulate_cv(action, confidence)
        if not self._cv_box.isVisible():
            return
        color = {
            "jab": Color.JAB, "cross": Color.CROSS,
            "left_hook": Color.L_HOOK, "right_hook": Color.R_HOOK,
            "left_uppercut": Color.L_UPPERCUT, "right_uppercut": Color.R_UPPERCUT,
            "block": Color.BLOCK,
        }.get(action, Color.TEXT_DISABLED)
        # FPS block
        fps_color = Color.SUCCESS if fps >= 25 else (Color.WARNING if fps >= 15 else Color.DANGER)
        self._cv_fps_lbl.setText(f"{fps:.0f}")
        self._cv_fps_lbl.setStyleSheet(
            f"font-size: 24px; font-weight: 700; color: {fps_color};"
            " background: transparent; border: none;"
        )
        # CV prediction block — raw CV predictions only
        if action in ("idle", ""):
            self._cv_pred_lbl.setText("IDLE")
            self._cv_pred_lbl.setStyleSheet(
                f"font-size: 16px; font-weight: 700; color: {Color.TEXT_DISABLED};"
                " background: transparent; border: none;"
            )
        else:
            name = action.upper().replace("_", " ")
            self._cv_pred_lbl.setText(f"{name} {confidence:.0%}")
            self._cv_pred_lbl.setStyleSheet(
                f"font-size: 14px; font-weight: 700; color: {color};"
                " background: transparent; border: none;"
            )

    def _accumulate_cv(self, action: str, confidence: float) -> None:
        """Group consecutive CV predictions into events."""
        is_punch = action not in ("idle", "block", "") and confidence >= 0.50
        if is_punch and action == self._cv_current_type:
            self._cv_current_frames += 1
            self._cv_current_conf_sum += confidence
            self._cv_current_peak = max(self._cv_current_peak, confidence)
        else:
            self._close_cv_event()
            if is_punch:
                self._cv_current_type = action
                self._cv_current_frames = 1
                self._cv_current_conf_sum = confidence
                self._cv_current_peak = confidence

    def _close_cv_event(self) -> None:
        """Flush current CV event if it had enough frames."""
        if self._cv_current_frames >= 3 and self._cv_current_peak >= 0.60:
            self._cv_events.append({
                "type": self._cv_current_type,
                "frames": self._cv_current_frames,
                "peak": round(self._cv_current_peak, 3),
                "avg": round(self._cv_current_conf_sum / self._cv_current_frames, 3),
            })
        self._cv_current_type = ""
        self._cv_current_frames = 0
        self._cv_current_conf_sum = 0.0
        self._cv_current_peak = 0.0

    def _build_cv_summary(self) -> Dict[str, Dict]:
        """Build per-punch-type summary from accumulated CV events."""
        self._close_cv_event()
        summary: Dict[str, Dict] = {}
        for evt in self._cv_events:
            t = evt["type"]
            if t not in summary:
                summary[t] = {"count": 0, "total_frames": 0, "peak": 0.0}
            summary[t]["count"] += 1
            summary[t]["total_frames"] += evt["frames"]
            summary[t]["peak"] = max(summary[t]["peak"], evt["peak"])
        return summary

    def _on_drill_progress(self, data: Dict[str, Any]) -> None:
        if not self._session_active:
            return

    def _on_coach_tip(self, text: str, tip_type: str) -> None:
        if not self._session_active:
            return

    def _on_session_state(self, state: str, mode: str) -> None:
        if not self._session_active:
            return
        if state == "rest":
            self._drill_timer.stop()
            self._score_round()
            self._router.replace(
                "training_rest", config=self._config,
                round_num=self._current_round,
                total_rounds=self._total_rounds,
                curriculum=self._curriculum,
                combo_id=self._combo_id,
                difficulty=self._difficulty,
                username=self._username,
            )

    # ── Drill cycling ────────────────────────────────────────────────────

    def _on_strike_complete(self, data: Dict[str, Any]) -> None:
        """Handle robot arm strike completion — advance to next drill punch.

        Debounces within 500ms to prevent double-advancement when both
        the simulator and V4 GUI publish feedback for the same command.
        """
        if not self._session_active:
            return
        import time as _t
        now = _t.time()
        if now - self._last_strike_complete_t < 0.5:
            return  # duplicate from second feedback path
        self._last_strike_complete_t = now
        logger.debug("Strike complete: %s", data.get("status", "?"))
        # Advance to next punch now that arm is done
        if self._combo_tokens and self._waiting_for_strike:
            self._waiting_for_strike = False
            self._drill_tick()

    def _strike_timeout(self, generation: int) -> None:
        """Safety fallback: advance drill if strike_complete was lost."""
        if generation != self._drill_generation:
            return  # stale timeout from a previous punch — ignore
        if self._session_active and self._waiting_for_strike and self._combo_tokens:
            logger.warning("Strike timeout — advancing drill without feedback")
            self._waiting_for_strike = False
            self._drill_tick()

    def _drill_tick(self) -> None:
        """Advance to the next punch in the combo sequence.

        Waits for strike_complete feedback before sending the next punch.
        This ensures the arm finishes each move before the next is sent.
        """
        if not self._session_active or not self._combo_tokens:
            return
        if self._waiting_for_strike:
            return  # still waiting for arm to finish current punch

        self._drill_idx += 1
        # Check if we crossed a sequence boundary → count as a combo
        if self._seq_boundaries:
            for boundary in self._seq_boundaries:
                if self._drill_idx == boundary:
                    self._combos_completed += 1
                    self._combos_lbl.setText(str(self._combos_completed))
                    break
        if self._drill_idx >= len(self._combo_tokens):
            # Completed all sequences → count as a rep, restart
            self._drill_idx = 0
            if not self._seq_boundaries:
                # Single sequence (no boundaries) — count as combo
                self._combos_completed += 1
                self._combos_lbl.setText(str(self._combos_completed))
            if len(self._seq_boundaries) > 1:
                self._reps_completed += 1
                self._reps_lbl.setText(str(self._reps_completed))

        self._update_cue()

        # Publish punch command and wait for strike_complete before next
        token = self._combo_tokens[self._drill_idx]
        self._waiting_for_strike = True
        self._drill_generation += 1
        gen = self._drill_generation
        if self._bridge is not None:
            if token in _DEFENSE_PUNCH_MAP:
                # Defense: robot throws a punch for user to defend against
                choices = _DEFENSE_PUNCH_MAP[token]
                punch_code = random.choice(choices)
                self._bridge.publish_punch_command(punch_code, self._robot_speed, "drill")
            else:
                # Substitute unsafe punches (e.g. uppercuts → straight punch)
                robot_code = _ROBOT_SUBSTITUTE.get(token, token)
                self._bridge.publish_punch_command(robot_code, self._robot_speed, "drill")
        # Safety timeout: if strike_complete never arrives, advance anyway
        # Generation check prevents stale timeouts from advancing the drill
        QTimer.singleShot(10000, lambda g=gen: self._strike_timeout(g))

    def _update_cue(self) -> None:
        """Update the current-punch cue, next preview, and sequence bar."""
        if not self._combo_tokens:
            self._cue_lbl.setText("")
            self._next_lbl.setText("")
            self._combo_scroll.setVisible(False)
            return

        idx = self._drill_idx
        total = len(self._combo_tokens)
        token = self._combo_tokens[idx]
        name = _PUNCH_NAMES.get(token, token.upper())
        color = _PUNCH_COLORS.get(token.rstrip("b"), Color.PRIMARY)

        # Big current punch cue
        self._cue_lbl.setText(name)
        self._cue_lbl.setStyleSheet(
            f"font-size: 48px; font-weight: 800; color: {color};"
            " letter-spacing: 3px;"
        )

        # Next punch preview — show current sequence number if multi-sequence
        next_idx = (idx + 1) % total
        next_token = self._combo_tokens[next_idx]
        next_name = _PUNCH_NAMES.get(next_token, next_token.upper())
        seq_info = ""
        if self._seq_boundaries:
            seq_num = 1
            for b in self._seq_boundaries:
                if idx < b:
                    break
                seq_num += 1
            seq_info = f"Seq {seq_num}/{len(self._seq_boundaries)}  \u2022  "
        step_text = f"{seq_info}Step {idx + 1}/{total}"
        at_boundary = (idx + 1) in self._seq_boundaries
        if idx + 1 >= total:
            self._next_lbl.setText(f"{step_text}  —  all sequences done, restarting")
        elif at_boundary:
            self._next_lbl.setText(f"{step_text}  —  next sequence")
        else:
            self._next_lbl.setText(f"{step_text}  —  next: {next_name}")
        self._next_lbl.setStyleSheet(
            f"font-size: 14px; font-weight: 600; color: {Color.TEXT_DISABLED};"
        )

        # Sequence bar with current punch highlighted + sequence separators
        parts = []
        for i, t in enumerate(self._combo_tokens):
            # Insert separator between sequences
            if i > 0 and i in self._seq_boundaries[:-1]:
                parts.append(
                    f'<span style="color:{Color.PRIMARY}; font-size:16px;'
                    f' font-weight:800;">\u2502</span>'
                )
            pname = _PUNCH_NAMES.get(t, t.upper())
            pcolor = _PUNCH_COLORS.get(t.rstrip("b"), Color.TEXT_SECONDARY)
            if i == idx:
                parts.append(
                    f'<span style="color:{pcolor}; font-size:18px;'
                    f' font-weight:800;">\u25B6 {pname}</span>'
                )
            else:
                parts.append(
                    f'<span style="color:{Color.TEXT_DISABLED};'
                    f' font-size:14px;">{pname}</span>'
                )
        self._combo_seq_lbl.setText("&nbsp; &rarr; &nbsp;".join(parts))
        self._combo_seq_lbl.setVisible(True)
        self._combo_scroll.setVisible(True)
        # Auto-scroll to keep current punch centered
        QTimer.singleShot(10, lambda: self._scroll_to_current(idx, total))

    def _scroll_to_current(self, idx: int, total: int) -> None:
        """Scroll the combo sequence bar so the current punch is centered."""
        sb = self._combo_scroll.horizontalScrollBar()
        if sb.maximum() == 0:
            return  # fits in view, no scrolling needed
        # Estimate position as fraction of total width
        frac = idx / max(total - 1, 1)
        # Center the current punch by offsetting half the viewport
        viewport_w = self._combo_scroll.viewport().width()
        label_w = self._combo_seq_lbl.sizeHint().width()
        target = int(frac * label_w - viewport_w / 2)
        sb.setValue(max(0, min(target, sb.maximum())))

    # ── Timer / round lifecycle ──────────────────────────────────────────

    def _on_timer_done(self) -> None:
        if not self._session_active:
            return
        if self._sound:
            self._sound.play("bell_end")
        self._session_active = False
        self._counting_active = False
        self._drill_timer.stop()
        self._score_round()
        if self._current_round < self._total_rounds:
            self._router.replace(
                "training_rest", config=self._config,
                round_num=self._current_round,
                total_rounds=self._total_rounds,
                curriculum=self._curriculum,
                combo_id=self._combo_id,
                difficulty=self._difficulty,
                username=self._username,
            )
        else:
            self._end_ros_session()
            self._router.replace(
                "training_results", config=self._config,
                curriculum=self._curriculum,
                combo_id=self._combo_id,
                difficulty=self._difficulty,
                username=self._username,
                total_punches=self._punch_counter._count,
                combos_completed=self._combos_completed,
                cv_summary=self._build_cv_summary(),
            )

    def _on_back(self) -> None:
        """Back button — return to home (only available before START)."""
        self._session_active = False
        self._router.back()

    def _sync_speed_from_tile(self) -> None:
        """Read the speed tile value and update config + robot speed."""
        self._config["Speed"] = self._speed_tile.value
        self._robot_speed = self._speed_tile.speed_for_ros
        logger.info("Free training speed: %s (%s)",
                     self._speed_tile.value, self._robot_speed)

    def imu_start(self) -> None:
        """Called by centre pad IMU — triggers START if waiting."""
        if getattr(self, '_waiting_for_start', False):
            self._on_pause_resume()

    def _on_pause_resume(self) -> None:
        if not self._session_active:
            return
        # Handle START press for free training
        if getattr(self, '_waiting_for_start', False):
            self._waiting_for_start = False
            self._sync_speed_from_tile()
            self._speed_tile.setVisible(False)
            self._btn_back.setVisible(False)
            self._btn_stop.setVisible(True)
            self._btn_pause.setText("PAUSE")
            self._btn_pause.setStyleSheet(f"""
                QPushButton {{
                    background-color: {Color.SURFACE};
                    color: {Color.TEXT};
                    font-size: 15px; font-weight: 700;
                    border: 1px solid {Color.BORDER_LIGHT};
                    border-radius: {Size.RADIUS}px;
                }}
                QPushButton:hover {{
                    border-color: {Color.WARNING};
                    background-color: {Color.SURFACE_HOVER};
                }}
            """)
            self._start_ros_session()
            self._timer.set_overlay("Get Ready")
            QTimer.singleShot(1000, self._countdown_tick)
            return
        if self._paused:
            # Resume
            self._paused = False
            self._counting_active = True
            self._timer.resume()
            self._btn_pause.setText("PAUSE")
            self._btn_pause.setStyleSheet(f"""
                QPushButton {{
                    background-color: {Color.SURFACE};
                    color: {Color.TEXT};
                    font-size: 15px; font-weight: 700;
                    border: 1px solid {Color.BORDER_LIGHT};
                    border-radius: {Size.RADIUS}px;
                }}
                QPushButton:hover {{
                    border-color: {Color.WARNING};
                    background-color: {Color.SURFACE_HOVER};
                }}
            """)
            logger.info("Training resumed")
        else:
            # Pause
            self._paused = True
            self._counting_active = False
            self._timer.pause()
            self._drill_timer.stop()
            self._btn_pause.setText("RESUME")
            self._btn_pause.setStyleSheet(f"""
                QPushButton {{
                    background-color: {Color.WARNING};
                    color: #000000;
                    font-size: 15px; font-weight: 700;
                    border: none;
                    border-radius: {Size.RADIUS}px;
                }}
                QPushButton:hover {{
                    background-color: {Color.WARNING_DARK};
                }}
            """)
            logger.info("Training paused")

    def _on_stop(self) -> None:
        self._session_active = False
        self._counting_active = False
        self._drill_timer.stop()
        self._timer.pause()
        self._score_round()
        self._end_ros_session()
        self._safety_stop_tracking()
        logger.info("Training session stopped by user")
        self._router.replace(
            "training_results", config=self._config,
            curriculum=self._curriculum,
            combo_id=self._combo_id,
            difficulty=self._difficulty,
            username=self._username,
            total_punches=self._punch_counter._count,
            combos_completed=self._combos_completed,
            cv_summary=self._build_cv_summary(),
        )

    def _start_ros_session(self) -> None:
        """Tell the ROS session manager to start the drill."""
        if self._bridge is None:
            return
        import json
        # Free training = no combo tokens, sparring engine needs mode="free"
        mode = "free" if not self._combo_tokens else "training"
        config_json = json.dumps(self._config, default=str)
        self._bridge.call_start_session(
            mode=mode,
            difficulty=self._difficulty or "beginner",
            config_json=config_json,
            username=self._username,
            callback=self._on_session_started,
        )

    def _on_session_started(
        self, success: bool, session_id: str, message: str,
    ) -> None:
        if success:
            self._session_id = session_id
            logger.info("ROS session started: %s", session_id)
        else:
            logger.warning("ROS session start failed: %s — retrying...", message)
            QTimer.singleShot(2000, self._start_ros_session)

    def _end_ros_session(self) -> None:
        """Tell the ROS session manager to end the drill."""
        if self._bridge is None:
            return
        if self._session_id:
            self._bridge.call_end_session(
                session_id=self._session_id,
                callback=lambda ok, summary, msg: logger.info(
                    "ROS session ended: ok=%s", ok
                ),
            )
            self._session_id = ""

    def _score_round(self) -> None:
        if not self._curriculum or not self._combo_id:
            return
        # Score based on combos completed (1-5 scale)
        # 0 combos = 1.0, 3+ combos = 4.0, 5+ = 5.0
        combos = self._combos_completed
        if combos >= 5:
            score = 5.0
        elif combos >= 3:
            score = 4.0
        elif combos >= 1:
            score = 3.0
        else:
            score = 1.5
        self._curriculum.update_score(self._combo_id, score)
        logger.info(
            "Round %d scored: combo=%s combos_done=%d score=%.1f",
            self._current_round, self._combo_id, combos, score,
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
        """Begin the round — start the timer and the drill cycling."""
        self._counting_active = True
        self._timer.clear_overlay()
        self._timer.start(self._work_time)
        # Restore labels and update description for free training
        self._cue_lbl.setVisible(True)
        self._next_lbl.setVisible(True)
        if not self._combo_tokens:
            self._cue_lbl.setText("")
            self._next_lbl.setText("Throw any punch — robot will counter-strike")

        # Map GUI speed to robot speed
        speed_str = self._config.get("Speed", "Medium (2s)")
        if "Slow" in speed_str:
            self._robot_speed = "slow"
        elif "Fast" in speed_str:
            self._robot_speed = "fast"
        else:
            self._robot_speed = "medium"

        # Start drill cycling if we have a combo
        if self._combo_tokens:
            self._drill_idx = 0
            self._waiting_for_strike = False
            self._update_cue()

            # Publish the first punch — subsequent punches sent on strike_complete
            first = self._combo_tokens[0]
            self._waiting_for_strike = True
            self._drill_generation += 1
            gen = self._drill_generation
            if self._bridge is not None:
                if first in _DEFENSE_PUNCH_MAP:
                    self._bridge.publish_punch_command(
                        random.choice(_DEFENSE_PUNCH_MAP[first]),
                        self._robot_speed,
                    )
                else:
                    robot_code = _ROBOT_SUBSTITUTE.get(first, first)
                    self._bridge.publish_punch_command(robot_code, self._robot_speed, "drill")
            # Safety timeout for first punch
            QTimer.singleShot(10000, lambda g=gen: self._strike_timeout(g))

            logger.info(
                "Drill started: %s (wait for strike_complete between punches)",
                self._combo_tokens,
            )

    def _parse_seconds(self, val: str) -> int:
        return int(val.rstrip("s")) if val.rstrip("s").isdigit() else 90

    # ── Lifecycle ────────────────────────────────────────────────────────

    def on_enter(self, **kwargs: Any) -> None:
        self._config = kwargs.get("config", {})
        self._total_rounds = int(self._config.get("Rounds", "3"))
        self._current_round = kwargs.get("round_num", 1)
        self._curriculum = kwargs.get("curriculum")
        self._combo_id = (
            kwargs.get("combo_id")
            or self._config.get("combo", {}).get("id")
        )
        self._difficulty = kwargs.get("difficulty")
        self._username = kwargs.get("username", "")
        # Safety: remember whether tracking was enabled so on_leave / _on_stop
        # can explicitly halt it. Motor must not run once the session ends.
        # Always start each session with tracking off — the user must
        # explicitly press Start Tracking.
        self._tracking_active = False
        work_time = self._parse_seconds(self._config.get("Work Time", "90s"))

        # Parse combo sequence
        combo = self._config.get("combo", {})
        seq = combo.get("seq", "")
        self._combo_tokens = seq.split("-") if seq else []
        self._drill_idx = 0
        self._combos_completed = 0
        self._reps_completed = 0
        self._cv_events = []
        self._cv_current_type = ""
        self._cv_current_frames = 0
        # Build cumulative sequence boundaries for multi-sequence drills
        seq_lengths = combo.get("seq_lengths", [])
        self._seq_boundaries = []
        if len(seq_lengths) > 1:
            cumulative = 0
            for length in seq_lengths:
                cumulative += length
                self._seq_boundaries.append(cumulative)

        self._session_active = True
        self._counting_active = False
        self._paused = False
        self._session_id = ""
        self._btn_pause.setText("PAUSE")
        self._round_lbl.setText(
            f"Round {self._current_round}/{self._total_rounds}"
        )
        self._punch_counter.set_count(0)
        self._combos_lbl.setText("0")
        self._reps_lbl.setText("0")
        self._reps_box.setVisible(len(self._seq_boundaries) > 1)

        # Show combo name
        name = combo.get("name", "")
        self._combo_name_lbl.setText(name if name else "Free Training")

        # Show initial sequence (all dimmed until round starts)
        if self._combo_tokens:
            parts = []
            for t in self._combo_tokens:
                pname = _PUNCH_NAMES.get(t, t.upper())
                parts.append(
                    f'<span style="color:{Color.TEXT_DISABLED};'
                    f' font-size:14px;">{pname}</span>'
                )
            self._combo_seq_lbl.setText("&nbsp; &rarr; &nbsp;".join(parts))
            self._combo_scroll.setVisible(True)
            self._cue_lbl.setText("GET READY")
            self._cue_lbl.setStyleSheet(
                f"font-size: 32px; font-weight: 800; color: {Color.TEXT_DISABLED};"
                " letter-spacing: 3px;"
            )
            self._next_lbl.setText(
                f"Combo: {name}  —  "
                f"{len(self._combo_tokens)} punches per cycle"
            )
        else:
            self._combo_scroll.setVisible(False)
            self._cue_lbl.setText("")
            self._next_lbl.setText("Free Training — no combo sequence")

        self._work_time = work_time
        self._countdown_remaining = 3
        self._timer.set_time(work_time)

        if not self._combo_tokens:
            # Free training — show START button + speed selector
            self._round_lbl.setVisible(False)
            self._timer.set_overlay("Tap START")
            self._cue_lbl.setText("")
            self._cue_lbl.setVisible(False)
            self._next_lbl.setVisible(False)
            self._btn_pause.setText("START")
            self._btn_pause.setStyleSheet(
                f"background: {Color.PRIMARY}; color: #fff;"
                " font-size: 18px; font-weight: bold; border-radius: 12px;"
                " padding: 12px 32px;"
            )
            self._speed_tile.setVisible(True)
            self._robot_speed = "medium"
            self._combos_box.setVisible(False)
            self._reps_box.setVisible(False)
            self._cv_box.setVisible(True)
            self._fps_box.setVisible(True)
            self._cv_pred_lbl.setText("--")
            self._cv_fps_lbl.setText("--")
            self._btn_back.setVisible(True)
            self._btn_stop.setVisible(False)
            self._waiting_for_start = True
            # Tracking row is only meaningful in free training
            self._track_row.setVisible(True)
            self._apply_track_button_style(active=False)
        else:
            # Combo drill — auto-countdown as before
            self._round_lbl.setVisible(True)
            self._waiting_for_start = False
            self._speed_tile.setVisible(False)
            self._combos_box.setVisible(True)
            self._cv_box.setVisible(False)
            self._fps_box.setVisible(False)
            self._btn_back.setVisible(False)
            self._btn_stop.setVisible(True)
            self._track_row.setVisible(False)
            self._timer.set_overlay("Get Ready")
            # Start the ROS session on the first round
            if self._current_round == 1:
                self._start_ros_session()
            QTimer.singleShot(1000, self._countdown_tick)
        logger.info(
            "Training round %d/%d (combo=%s, seq=%s)",
            self._current_round, self._total_rounds,
            self._combo_id, self._combo_tokens,
        )

    def on_leave(self) -> None:
        self._session_active = False
        self._counting_active = False
        self._drill_timer.stop()
        self._timer.pause()
        self._end_ros_session()
        self._safety_stop_tracking()

    def _safety_stop_tracking(self) -> None:
        """Always halt base-tracking when leaving/stopping the session.

        Safety: motor must not keep running once training ends. Called
        unconditionally so even spurious paths that missed setting
        ``_tracking_active`` still halt the motor.
        """
        if self._bridge is None:
            return
        try:
            self._bridge.call_tracking_stop()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to stop tracking: %s", exc)
        self._tracking_active = False
