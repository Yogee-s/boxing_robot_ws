"""Coach station mode -- full-screen during coaching sessions.

State machine: Ready -> Active -> Completed -> Ready.
Ready: Start button + config name + participant number.
Active: timer, round counter, punch count.
Completed: quick stats, Next Person / End Session buttons.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from boxbunny_gui.theme import Color, Size, font, DANGER_BTN, GHOST_BTN, PRIMARY_BTN, SURFACE_BTN
from boxbunny_gui.widgets import BigButton, PunchCounter, StatCard, TimerDisplay

if TYPE_CHECKING:
    from boxbunny_gui.gui_bridge import GuiBridge
    from boxbunny_gui.nav.router import PageRouter

logger = logging.getLogger(__name__)

_STATE_READY = "ready"
_STATE_ACTIVE = "active"
_STATE_COMPLETED = "completed"


class StationPage(QWidget):
    """Full-screen coach station with ready/active/completed states."""

    def __init__(
        self,
        router: PageRouter,
        bridge: Optional[GuiBridge] = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._router = router
        self._bridge = bridge
        self._state: str = _STATE_READY
        self._config_name: str = "Jab-Cross Drill"
        self._participant: int = 1
        self._punch_count: int = 0
        self._build_ui()
        if self._bridge:
            self._bridge.punch_confirmed.connect(self._on_punch)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(Size.SPACING, Size.SPACING, Size.SPACING, Size.SPACING)
        root.setSpacing(Size.SPACING)

        # Top bar (minimal)
        top = QHBoxLayout()
        self._config_lbl = QLabel(self._config_name)
        self._config_lbl.setFont(font(20, bold=True))
        self._config_lbl.setWordWrap(True)
        top.addWidget(self._config_lbl)
        top.addStretch()
        self._participant_lbl = QLabel(f"Participant #{self._participant}")
        self._participant_lbl.setStyleSheet(f"color: {Color.PRIMARY}; font-size: 18px;")
        top.addWidget(self._participant_lbl)
        root.addLayout(top)

        # Ready state
        self._ready_widget = QWidget()
        ready_lay = QVBoxLayout(self._ready_widget)
        ready_lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._btn_start = BigButton("START", stylesheet=PRIMARY_BTN)
        self._btn_start.setFixedSize(300, 120)
        self._btn_start.clicked.connect(self._go_active)
        ready_lay.addWidget(self._btn_start, alignment=Qt.AlignmentFlag.AlignCenter)
        root.addWidget(self._ready_widget)

        # Active state
        self._active_widget = QWidget()
        active_lay = QVBoxLayout(self._active_widget)
        self._timer = TimerDisplay(font_size=Size.TEXT_TIMER, show_ring=True)
        self._timer.finished.connect(self._go_completed)
        active_lay.addWidget(self._timer, stretch=1)

        act_row = QHBoxLayout()
        self._round_lbl = QLabel("Round 1/3")
        self._round_lbl.setFont(font(22, bold=True))
        act_row.addWidget(self._round_lbl)
        act_row.addStretch()
        self._punch_counter = PunchCounter(label="PUNCHES")
        act_row.addWidget(self._punch_counter)
        active_lay.addLayout(act_row)

        self._active_widget.setVisible(False)
        root.addWidget(self._active_widget)

        # Completed state
        self._completed_widget = QWidget()
        comp_lay = QVBoxLayout(self._completed_widget)
        stats_row = QHBoxLayout()
        self._stat_punches = StatCard("Punches", "--")
        self._stat_score = StatCard("Score", "--%")
        self._stat_duration = StatCard("Duration", "--")
        stats_row.addWidget(self._stat_punches)
        stats_row.addWidget(self._stat_score)
        stats_row.addWidget(self._stat_duration)
        comp_lay.addLayout(stats_row)
        comp_lay.addStretch()

        btn_row = QHBoxLayout()
        self._btn_next = BigButton("Next Person", stylesheet=PRIMARY_BTN)
        self._btn_next.setFixedHeight(80)
        self._btn_next.clicked.connect(self._next_person)
        btn_row.addWidget(self._btn_next, stretch=2)
        self._btn_end = BigButton("End Session", stylesheet=DANGER_BTN)
        self._btn_end.setFixedWidth(180)
        self._btn_end.clicked.connect(self._end_session)
        btn_row.addWidget(self._btn_end)
        comp_lay.addLayout(btn_row)

        self._completed_widget.setVisible(False)
        root.addWidget(self._completed_widget)

    def _set_state(self, state: str) -> None:
        self._state = state
        self._ready_widget.setVisible(state == _STATE_READY)
        self._active_widget.setVisible(state == _STATE_ACTIVE)
        self._completed_widget.setVisible(state == _STATE_COMPLETED)

    def _go_active(self) -> None:
        self._punch_count = 0
        self._punch_counter.set_count(0)
        self._timer.start(90)  # TODO: read from config
        self._set_state(_STATE_ACTIVE)
        logger.info("Station active for participant #%d", self._participant)

    def _go_completed(self) -> None:
        self._stat_punches.set_value(str(self._punch_count))
        self._stat_score.set_value("--%")  # TODO: real score
        self._stat_duration.set_value("1:30")  # TODO: real duration
        self._set_state(_STATE_COMPLETED)
        logger.info("Station completed for participant #%d", self._participant)

    def _next_person(self) -> None:
        self._participant += 1
        self._participant_lbl.setText(f"Participant #{self._participant}")
        self._set_state(_STATE_READY)

    def _end_session(self) -> None:
        logger.info("Coach station session ended after %d participants", self._participant)
        self._router.navigate("home_coach")

    def _on_punch(self, data: Dict[str, Any]) -> None:
        if self._state == _STATE_ACTIVE:
            self._punch_count += 1
            self._punch_counter.set_count(self._punch_count)

    # ── Lifecycle ──────────────────────────────────────────────────────
    def on_enter(self, **kwargs: Any) -> None:
        self._config_name = kwargs.get("config_name", self._config_name)
        self._participant = 1
        self._config_lbl.setText(self._config_name)
        self._participant_lbl.setText(f"Participant #{self._participant}")
        self._set_state(_STATE_READY)
        logger.debug("StationPage entered")

    def on_leave(self) -> None:
        self._timer.reset()
