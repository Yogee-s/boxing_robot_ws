"""Full-screen active training session page — premium treatment.

Timer with thick ring, combo display with highlighting, round counter,
live punch counter, coach tip bar, and stop button. Integrates with
ComboCurriculum for Anki-style scoring at round end.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from boxbunny_gui.theme import Color, Icon, Size, font, DANGER_BTN, badge_style
from boxbunny_gui.widgets import (
    BigButton, CoachTipBar, ComboDisplay, PunchCounter, TimerDisplay,
)

if TYPE_CHECKING:
    from boxbunny_gui.curriculum import ComboCurriculum
    from boxbunny_gui.gui_bridge import GuiBridge
    from boxbunny_gui.nav.router import PageRouter

logger = logging.getLogger(__name__)


class TrainingSessionPage(QWidget):
    """Active training session with live ROS data display."""

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
        self._build_ui()
        self._connect_bridge()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 10, 24, 14)
        root.setSpacing(6)

        # Coach tip bar (collapsible, top)
        self._coach_bar = CoachTipBar(parent=self)
        root.addWidget(self._coach_bar)

        # Top row: round label (left) + combo name badge (center) + mode badge (right)
        top_row = QHBoxLayout()
        self._round_lbl = QLabel("Round 1/3")
        self._round_lbl.setStyleSheet(
            f"color: {Color.TEXT}; font-size: 18px; font-weight: 700;"
            f" background-color: {Color.SURFACE};"
            f" border-left: {Size.ACCENT_BAR_W}px solid {Color.PRIMARY};"
            f" border-radius: {Size.RADIUS_SM}px;"
            " padding: 6px 16px;"
        )
        top_row.addWidget(self._round_lbl)
        top_row.addStretch()

        self._combo_name_lbl = QLabel("")
        self._combo_name_lbl.setStyleSheet(
            f"color: {Color.PRIMARY}; font-size: 15px; font-weight: 600;"
            f" background-color: {Color.SURFACE};"
            f" border: 1px solid {Color.PRIMARY}30;"
            f" border-radius: {Size.RADIUS_SM}px;"
            " padding: 4px 14px;"
        )
        self._combo_name_lbl.setAlignment(Qt.AlignCenter)
        top_row.addWidget(self._combo_name_lbl)
        top_row.addStretch()

        mode_lbl = QLabel("TRAINING")
        mode_lbl.setStyleSheet(badge_style(Color.PRIMARY))
        top_row.addWidget(mode_lbl)
        root.addLayout(top_row)

        # Center: timer -- dominant element
        self._timer = TimerDisplay(font_size=Size.TEXT_TIMER, show_ring=True)
        self._timer.finished.connect(self._on_timer_done)
        root.addWidget(self._timer, stretch=1)

        # Combo sequence display
        self._combo_seq_lbl = QLabel("")
        self._combo_seq_lbl.setAlignment(Qt.AlignCenter)
        self._combo_seq_lbl.setStyleSheet(
            f"color: {Color.INFO}; font-size: 20px; font-weight: 700;"
        )
        root.addWidget(self._combo_seq_lbl)

        # Combo display (punch badges)
        self._combo_display = ComboDisplay(parent=self)
        root.addWidget(self._combo_display)

        # Bottom row: punch counter (left) + stop button (right)
        bottom = QHBoxLayout()
        bottom.setContentsMargins(0, 6, 0, 0)
        self._punch_counter = PunchCounter(label="PUNCHES")
        bottom.addWidget(self._punch_counter)
        bottom.addStretch()

        self._btn_stop = QPushButton(f"{Icon.STOP}  STOP")
        self._btn_stop.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_stop.setFixedSize(120, 52)
        self._btn_stop.setStyleSheet(f"""
            QPushButton {{
                background-color: {Color.DANGER}; color: white;
                font-size: 16px; font-weight: 700;
                border: none; border-radius: {Size.RADIUS_LG}px;
            }}
            QPushButton:hover {{ background-color: {Color.DANGER_DARK}; }}
            QPushButton:pressed {{ background-color: #C33C3C; }}
        """)
        self._btn_stop.clicked.connect(self._on_stop)
        bottom.addWidget(
            self._btn_stop, alignment=Qt.AlignmentFlag.AlignVCenter
        )
        root.addLayout(bottom)

    def _connect_bridge(self) -> None:
        if self._bridge is None:
            return
        self._bridge.punch_confirmed.connect(self._on_punch)
        self._bridge.drill_progress.connect(self._on_drill_progress)
        self._bridge.coach_tip.connect(self._on_coach_tip)
        self._bridge.session_state_changed.connect(self._on_session_state)

    def _on_punch(self, data: Dict[str, Any]) -> None:
        self._punch_counter.increment()

    def _on_drill_progress(self, data: Dict[str, Any]) -> None:
        pass

    def _on_coach_tip(self, text: str, tip_type: str) -> None:
        self._coach_bar.show_tip(text, tip_type)

    def _on_session_state(self, state: str, mode: str) -> None:
        if state == "rest":
            self._score_round()
            self._router.replace(
                "training_rest", config=self._config,
                round_num=self._current_round,
                total_rounds=self._total_rounds,
                curriculum=self._curriculum,
                combo_id=self._combo_id,
                difficulty=self._difficulty,
            )

    def _on_timer_done(self) -> None:
        self._score_round()
        if self._current_round < self._total_rounds:
            self._router.replace(
                "training_rest", config=self._config,
                round_num=self._current_round,
                total_rounds=self._total_rounds,
                curriculum=self._curriculum,
                combo_id=self._combo_id,
                difficulty=self._difficulty,
            )
        else:
            self._router.replace(
                "training_results", config=self._config,
                curriculum=self._curriculum,
                combo_id=self._combo_id,
                difficulty=self._difficulty,
            )

    def _on_stop(self) -> None:
        self._timer.pause()
        self._score_round()
        logger.info("Training session stopped by user")
        self._router.replace(
            "training_results", config=self._config,
            curriculum=self._curriculum,
            combo_id=self._combo_id,
            difficulty=self._difficulty,
        )

    def _score_round(self) -> None:
        """Score the current round via the curriculum (placeholder score)."""
        if self._curriculum and self._combo_id:
            score = 3.0
            self._curriculum.update_score(self._combo_id, score)
            logger.info(
                "Round %d scored: combo=%s score=%.1f",
                self._current_round, self._combo_id, score,
            )

    def _parse_seconds(self, val: str) -> int:
        return int(val.rstrip("s")) if val.rstrip("s").isdigit() else 90

    def _update_combo_display(self) -> None:
        """Show the current combo name and sequence."""
        combo = self._config.get("combo", {})
        name = combo.get("name", "")
        seq = combo.get("seq", "")
        self._combo_name_lbl.setText(name)
        if seq:
            tokens = seq.split("-") if isinstance(seq, str) else []
            display_parts = []
            for t in tokens:
                base = t.rstrip("b")
                pname = _PUNCH_NAMES.get(base, t.upper())
                if t.endswith("b"):
                    pname = f"Body {pname}"
                display_parts.append(pname)
            self._combo_seq_lbl.setText("  →  ".join(display_parts))
        else:
            self._combo_seq_lbl.setText("")

    # ── Lifecycle ──────────────────────────────────────────────────────
    def on_enter(self, **kwargs: Any) -> None:
        self._config = kwargs.get("config", {})
        self._total_rounds = int(self._config.get("Rounds", "3"))
        self._current_round = kwargs.get("round_num", 1)
        self._curriculum = kwargs.get("curriculum")
        self._combo_id = kwargs.get("combo_id") or self._config.get("combo", {}).get("id")
        self._difficulty = kwargs.get("difficulty")
        work_time = self._parse_seconds(self._config.get("Work Time", "90s"))

        self._round_lbl.setText(
            f"Round {self._current_round}/{self._total_rounds}"
        )
        self._punch_counter.set_count(0)
        self._update_combo_display()
        self._timer.start(work_time)
        logger.info(
            "Training session round %d/%d (combo=%s)",
            self._current_round, self._total_rounds, self._combo_id,
        )

    def on_leave(self) -> None:
        self._timer.pause()


# Punch name mapping for display
_PUNCH_NAMES = {
    "1": "Jab", "2": "Cross", "3": "Hook", "4": "R.Hook",
    "5": "L.Upper", "6": "R.Upper",
    "slip": "Slip", "block": "Block",
}
