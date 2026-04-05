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

from boxbunny_gui.theme import Color, Icon, Size, font, badge_style
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
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._router = router
        self._bridge = bridge
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

        # Drill cycling state
        self._combo_tokens: List[str] = []
        self._drill_idx: int = 0
        self._combos_completed: int = 0
        self._robot_speed: str = "medium"
        self._drill_timer = QTimer(self)
        self._drill_timer.timeout.connect(self._drill_tick)

        self._build_ui()
        self._connect_bridge()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(28, 10, 28, 22)
        root.setSpacing(0)

        # ── Top: round + combo name + mode badge ─────────────────────────
        top = QHBoxLayout()
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
        top.addStretch()

        self._combo_name_lbl = QLabel("")
        self._combo_name_lbl.setAlignment(Qt.AlignCenter)
        self._combo_name_lbl.setStyleSheet(
            f"font-size: 14px; font-weight: 600; color: {Color.TEXT};"
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

        # ── Combo sequence with highlight ────────────────────────────────
        self._combo_seq_lbl = QLabel("")
        self._combo_seq_lbl.setAlignment(Qt.AlignCenter)
        self._combo_seq_lbl.setTextFormat(Qt.TextFormat.RichText)
        self._combo_seq_lbl.setStyleSheet(
            f"font-size: 16px; font-weight: 700; color: {Color.TEXT_SECONDARY};"
            " background-color: #131920;"
            " border: 1px solid #1E2832;"
            f" border-radius: {Size.RADIUS}px;"
            " padding: 10px 20px;"
        )
        root.addWidget(self._combo_seq_lbl)

        root.addSpacing(10)

        # ── Stats row ────────────────────────────────────────────────────
        stats_row = QHBoxLayout()
        stats_row.setSpacing(10)

        self._punch_counter = PunchCounter(label="PUNCHES")
        stats_row.addWidget(self._punch_counter)

        stats_row.addStretch()

        # Speed selector (free training only)
        _SPEED_ACCENT = "#C88D2E"
        self._speed_options = ["Slow", "Medium", "Fast"]
        self._speed_index = 1
        self._speed_tile = QPushButton("Speed\nMedium")
        self._speed_tile.setCursor(Qt.CursorShape.PointingHandCursor)
        self._speed_tile.setFixedHeight(70)
        self._speed_tile.setStyleSheet(f"""
            QPushButton {{
                background-color: {Color.SURFACE}; color: {Color.TEXT};
                border: 1px solid {Color.BORDER};
                border-left: 3px solid {_SPEED_ACCENT};
                border-radius: {Size.RADIUS}px;
                font-size: 15px; font-weight: 600; padding: 10px 14px;
            }}
            QPushButton:hover {{
                background-color: {Color.SURFACE_HOVER};
                border-color: {_SPEED_ACCENT};
                border-left: 3px solid {_SPEED_ACCENT};
            }}
            QPushButton:pressed {{
                background-color: {_SPEED_ACCENT}; color: #FFFFFF;
                border-color: {_SPEED_ACCENT};
                border-left: 3px solid {_SPEED_ACCENT};
            }}
        """)
        self._speed_tile.clicked.connect(self._cycle_speed)
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

        # CV prediction block (free training only)
        self._cv_box = QWidget()
        self._cv_box.setFixedHeight(70)
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

    def _on_punch(self, data: Dict[str, Any]) -> None:
        if not self._counting_active:
            return
        self._punch_counter.increment()
        # Show fusion-filtered result in the CV box during free training
        if not self._combo_tokens and self._cv_box.isVisible():
            pt = data.get("punch_type", "?")
            color = {
                "jab": Color.JAB, "cross": Color.CROSS,
                "left_hook": Color.L_HOOK, "right_hook": Color.R_HOOK,
                "left_uppercut": Color.L_UPPERCUT, "right_uppercut": Color.R_UPPERCUT,
            }.get(pt, Color.TEXT)
            name = pt.upper().replace("_", " ")
            imu_tag = "IMU" if data.get("imu_confirmed") else "CV"
            conf = data.get("cv_confidence", 0.0)
            self._cv_hdr_lbl.setText("FUSION")
            self._cv_pred_lbl.setText(f"{name} {conf:.0%}")
            self._cv_pred_lbl.setStyleSheet(
                f"font-size: 16px; font-weight: 700; color: {color};"
                " background: transparent; border: none;"
            )

    def _on_debug_info(self, data: Dict[str, Any]) -> None:
        """Show raw CV prediction and FPS."""
        if not self._session_active or not self._cv_box.isVisible():
            return
        action = data.get("action", "idle")
        confidence = data.get("confidence", 0.0)
        fps = data.get("fps", 0.0)
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
        # CV prediction block — always show raw CV predictions
        # (header changes to FUSION momentarily when a confirmed punch arrives)
        if action in ("idle", ""):
            self._cv_pred_lbl.setText("IDLE")
            self._cv_pred_lbl.setStyleSheet(
                f"font-size: 16px; font-weight: 700; color: {Color.TEXT_DISABLED};"
                " background: transparent; border: none;"
            )
            if not self._counting_active:
                self._cv_hdr_lbl.setText("CV MODEL")
        else:
            name = action.upper().replace("_", " ")
            self._cv_pred_lbl.setText(f"{name} {confidence:.0%}")
            self._cv_pred_lbl.setStyleSheet(
                f"font-size: 14px; font-weight: 700; color: {color};"
                " background: transparent; border: none;"
            )
            if not self._counting_active:
                self._cv_hdr_lbl.setText("CV MODEL")

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
        """Handle robot arm strike completion feedback (informational only)."""
        if not self._session_active:
            return
        logger.debug("Strike complete: %s", data.get("status", "?"))

    def _drill_tick(self) -> None:
        """Advance to the next punch in the combo sequence.

        Sends commands on each timer tick without waiting for arm completion.
        The V4 GUI handles queueing and execution timing internally.
        """
        if not self._session_active or not self._combo_tokens:
            return

        self._drill_idx += 1
        if self._drill_idx >= len(self._combo_tokens):
            # Completed one full combo cycle
            self._drill_idx = 0
            self._combos_completed += 1
            self._combos_lbl.setText(str(self._combos_completed))

        self._update_cue()

        # Publish punch command to robot — fire and forget
        token = self._combo_tokens[self._drill_idx]
        if self._bridge is not None:
            if token in _DEFENSE_PUNCH_MAP:
                # Defense: robot throws a punch for user to defend against
                choices = _DEFENSE_PUNCH_MAP[token]
                punch_code = random.choice(choices)
                self._bridge.publish_punch_command(punch_code, self._robot_speed)
            else:
                self._bridge.publish_punch_command(token, self._robot_speed)

    def _update_cue(self) -> None:
        """Update the current-punch cue, next preview, and sequence bar."""
        if not self._combo_tokens:
            self._cue_lbl.setText("")
            self._next_lbl.setText("")
            self._combo_seq_lbl.setVisible(False)
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

        # Next punch preview
        next_idx = (idx + 1) % total
        next_token = self._combo_tokens[next_idx]
        next_name = _PUNCH_NAMES.get(next_token, next_token.upper())
        step_text = f"Step {idx + 1}/{total}"
        if idx + 1 >= total:
            self._next_lbl.setText(f"{step_text}  —  combo complete, restarting")
        else:
            self._next_lbl.setText(f"{step_text}  —  next: {next_name}")
        self._next_lbl.setStyleSheet(
            f"font-size: 14px; font-weight: 600; color: {Color.TEXT_DISABLED};"
        )

        # Sequence bar with current punch highlighted
        parts = []
        for i, t in enumerate(self._combo_tokens):
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

    # ── Timer / round lifecycle ──────────────────────────────────────────

    def _on_timer_done(self) -> None:
        if not self._session_active:
            return
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
            )

    def _cycle_speed(self) -> None:
        """Cycle through speed options (tap to change, like training config)."""
        self._speed_index = (self._speed_index + 1) % len(self._speed_options)
        label = self._speed_options[self._speed_index]
        self._speed_tile.setText(f"Speed\n{label}")
        speed_val = {"Slow": "slow", "Medium": "medium", "Fast": "fast"}[label]
        self._robot_speed = speed_val
        self._config["Speed"] = label
        logger.info("Free training speed set to %s", speed_val)

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
            self._speed_tile.setVisible(False)
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
            self._drill_timer.start()
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
        logger.info("Training session stopped by user")
        self._router.replace(
            "training_results", config=self._config,
            curriculum=self._curriculum,
            combo_id=self._combo_id,
            difficulty=self._difficulty,
            username=self._username,
            total_punches=self._punch_counter._count,
            combos_completed=self._combos_completed,
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
            logger.warning("ROS session start failed: %s", message)

    def _end_ros_session(self) -> None:
        """Tell the ROS session manager to end the drill."""
        if self._bridge is None or not self._session_id:
            return
        self._bridge.call_end_session(
            session_id=self._session_id,
            callback=lambda ok, summary, msg: logger.info(
                "ROS session ended: ok=%s", ok
            ),
        )

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
            QTimer.singleShot(500, self._start_round)

    def _start_round(self) -> None:
        """Begin the round — start the timer and the drill cycling."""
        self._counting_active = True
        self._timer.clear_overlay()
        self._timer.start(self._work_time)

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
            self._update_cue()
            speed_ms = _parse_speed_ms(speed_str)
            self._drill_timer.setInterval(speed_ms)
            self._drill_timer.start()

            # Publish the first punch immediately
            first = self._combo_tokens[0]
            if self._bridge is not None:
                if first in _DEFENSE_PUNCH_MAP:
                    self._bridge.publish_punch_command(
                        random.choice(_DEFENSE_PUNCH_MAP[first]),
                        self._robot_speed,
                    )
                else:
                    self._bridge.publish_punch_command(first, self._robot_speed)

            logger.info(
                "Drill cycling started: %s at %dms intervals (speed=%s)",
                self._combo_tokens, speed_ms, self._robot_speed,
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
        work_time = self._parse_seconds(self._config.get("Work Time", "90s"))

        # Parse combo sequence
        seq = self._config.get("combo", {}).get("seq", "")
        self._combo_tokens = seq.split("-") if seq else []
        self._drill_idx = 0
        self._combos_completed = 0

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

        # Show combo name
        combo = self._config.get("combo", {})
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
            self._combo_seq_lbl.setVisible(True)
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
            self._combo_seq_lbl.setVisible(False)
            self._cue_lbl.setText("")
            self._next_lbl.setText("Free Training — no combo sequence")

        self._work_time = work_time
        self._countdown_remaining = 3
        self._timer.set_time(work_time)

        if not self._combo_tokens:
            # Free training — show START button + speed selector
            self._timer.set_overlay("Tap START")
            self._cue_lbl.setText("FREE TRAINING")
            self._cue_lbl.setStyleSheet(
                f"font-size: 32px; font-weight: 800; color: {Color.PRIMARY};"
                " letter-spacing: 3px;"
            )
            self._btn_pause.setText("START")
            self._btn_pause.setStyleSheet(
                f"background: {Color.PRIMARY}; color: #fff;"
                " font-size: 18px; font-weight: bold; border-radius: 12px;"
                " padding: 12px 32px;"
            )
            self._speed_tile.setVisible(True)
            self._speed_index = 1
            self._speed_tile.setText("Speed\nMedium")
            self._robot_speed = "medium"
            self._combos_box.setVisible(False)
            self._cv_box.setVisible(True)
            self._fps_box.setVisible(True)
            self._cv_pred_lbl.setText("--")
            self._cv_fps_lbl.setText("--")
            self._waiting_for_start = True
        else:
            # Combo drill — auto-countdown as before
            self._waiting_for_start = False
            self._speed_tile.setVisible(False)
            self._combos_box.setVisible(True)
            self._cv_box.setVisible(False)
            self._fps_box.setVisible(False)
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
