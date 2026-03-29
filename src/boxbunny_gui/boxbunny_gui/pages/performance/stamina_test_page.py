"""Stamina test page: throw as many punches as possible in a timed period.

Large timer countdown, live punch count, punches-per-minute display,
and target pad indicators.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from boxbunny_gui.theme import Color, Size, font, DANGER_BTN, GHOST_BTN, PRIMARY_BTN
from boxbunny_gui.widgets import BigButton, PunchCounter, StatCard, TimerDisplay

if TYPE_CHECKING:
    from boxbunny_gui.gui_bridge import GuiBridge
    from boxbunny_gui.nav.router import PageRouter

logger = logging.getLogger(__name__)

_DEFAULT_DURATION = 120  # seconds
_STATE_READY = "ready"
_STATE_ACTIVE = "active"
_STATE_RESULTS = "results"


class StaminaTestPage(QWidget):
    """Timed stamina test with live punch rate tracking."""

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
        self._punch_count: int = 0
        self._elapsed: int = 0
        self._peak_rate: float = 0.0
        self._build_ui()
        if self._bridge:
            self._bridge.punch_confirmed.connect(self._on_punch)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(Size.SPACING, Size.SPACING_SM, Size.SPACING, Size.SPACING_SM)
        root.setSpacing(Size.SPACING)

        # Top bar
        top = QHBoxLayout()
        btn_back = BigButton("Back", stylesheet=GHOST_BTN)
        btn_back.setFixedWidth(100)
        btn_back.clicked.connect(lambda: self._on_back())
        top.addWidget(btn_back)
        title = QLabel("Stamina Test")
        title.setFont(font(Size.TEXT_SUBHEADER, bold=True))
        top.addWidget(title)
        top.addStretch()
        root.addLayout(top)

        # Timer
        self._timer = TimerDisplay(font_size=Size.TEXT_TIMER, show_ring=True)
        self._timer.finished.connect(self._on_done)
        self._timer.tick.connect(self._on_tick)
        root.addWidget(self._timer, stretch=1)

        # Live stats row
        stats = QHBoxLayout()
        self._punch_counter = PunchCounter(label="PUNCHES")
        stats.addWidget(self._punch_counter)

        rate_col = QVBoxLayout()
        rate_col.setAlignment(Qt.AlignmentFlag.AlignCenter)
        rate_label = QLabel("PUNCHES/MIN")
        rate_label.setStyleSheet(
            f"color: {Color.TEXT_SECONDARY}; font-size: 14px; font-weight: bold;"
        )
        rate_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._rate_lbl = QLabel("0")
        self._rate_lbl.setFont(font(36, bold=True))
        self._rate_lbl.setStyleSheet(f"color: {Color.PRIMARY};")
        self._rate_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        rate_col.addWidget(rate_label)
        rate_col.addWidget(self._rate_lbl)
        stats.addLayout(rate_col)

        # Target pad indicator placeholder
        pad_col = QVBoxLayout()
        pad_col.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pad_label = QLabel("TARGET")
        pad_label.setStyleSheet(
            f"color: {Color.TEXT_SECONDARY}; font-size: 14px; font-weight: bold;"
        )
        pad_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._pad_lbl = QLabel("--")
        self._pad_lbl.setFont(font(28, bold=True))
        self._pad_lbl.setStyleSheet(f"color: {Color.WARNING};")
        self._pad_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pad_col.addWidget(pad_label)
        pad_col.addWidget(self._pad_lbl)
        stats.addLayout(pad_col)
        root.addLayout(stats)

        # Start / Stop button
        self._btn_action = BigButton("Start", stylesheet=PRIMARY_BTN)
        self._btn_action.setFixedHeight(70)
        self._btn_action.clicked.connect(self._toggle)
        root.addWidget(self._btn_action)

        # Results overlay
        self._results_widget = QWidget()
        res_lay = QVBoxLayout(self._results_widget)
        res_row = QHBoxLayout()
        self._stat_total = StatCard("Total Punches", "--")
        self._stat_peak = StatCard("Peak Rate", "--/min")
        self._stat_fatigue = StatCard("Fatigue", "--")
        res_row.addWidget(self._stat_total)
        res_row.addWidget(self._stat_peak)
        res_row.addWidget(self._stat_fatigue)
        res_lay.addLayout(res_row)
        btn_done = BigButton("Done", stylesheet=PRIMARY_BTN)
        btn_done.clicked.connect(lambda: self._router.navigate("performance_menu"))
        res_lay.addWidget(btn_done)
        self._results_widget.setVisible(False)
        root.addWidget(self._results_widget)

    def _toggle(self) -> None:
        if self._state == _STATE_READY:
            self._start_test()
        else:
            self._on_done()

    def _start_test(self) -> None:
        self._state = _STATE_ACTIVE
        self._punch_count = 0
        self._elapsed = 0
        self._peak_rate = 0.0
        self._punch_counter.set_count(0)
        self._timer.start(_DEFAULT_DURATION)
        self._btn_action.setText("Stop")
        self._btn_action.setStyleSheet(DANGER_BTN)
        self._results_widget.setVisible(False)

    def _on_tick(self, remaining: int) -> None:
        self._elapsed = _DEFAULT_DURATION - remaining
        if self._elapsed > 0:
            rate = self._punch_count / (self._elapsed / 60.0)
            self._rate_lbl.setText(str(int(rate)))
            self._peak_rate = max(self._peak_rate, rate)

    def _on_punch(self, data: Dict[str, Any]) -> None:
        if self._state != _STATE_ACTIVE:
            return
        self._punch_count += 1
        self._punch_counter.set_count(self._punch_count)

    def _on_done(self) -> None:
        self._timer.pause()
        self._state = _STATE_RESULTS
        self._btn_action.setVisible(False)
        self._stat_total.set_value(str(self._punch_count))
        self._stat_peak.set_value(f"{self._peak_rate:.0f}/min")
        # TODO: compute fatigue curve
        self._stat_fatigue.set_value("--")
        self._results_widget.setVisible(True)

    def _on_back(self) -> None:
        self._timer.reset()
        self._router.back()

    # ── Lifecycle ──────────────────────────────────────────────────────
    def on_enter(self, **kwargs: Any) -> None:
        self._state = _STATE_READY
        self._timer.set_time(_DEFAULT_DURATION)
        self._btn_action.setText("Start")
        self._btn_action.setStyleSheet(PRIMARY_BTN)
        self._btn_action.setVisible(True)
        self._results_widget.setVisible(False)
        self._punch_counter.set_count(0)
        self._rate_lbl.setText("0")
        logger.debug("StaminaTestPage entered")

    def on_leave(self) -> None:
        self._timer.reset()
