"""Stamina test page: throw as many punches as possible in a timed period.

Large timer countdown, live punch count, punches-per-minute display,
and target pad indicators.
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

from PySide6.QtWidgets import QPushButton as _QPushButton

from boxbunny_gui.theme import (
    Color, Icon, Size, font, badge_style, back_link_style,
    DANGER_BTN, GHOST_BTN, PRIMARY_BTN,
)
from boxbunny_gui.widgets import BigButton, PunchCounter, StatCard, TimerDisplay

if TYPE_CHECKING:
    from boxbunny_gui.gui_bridge import GuiBridge
    from boxbunny_gui.nav.router import PageRouter

logger = logging.getLogger(__name__)

_DEFAULT_DURATION = 120  # seconds
_STATE_READY = "ready"
_STATE_ACTIVE = "active"
_STATE_RESULTS = "results"


def _stat_col(label: str, value: str, color: str) -> tuple:
    """Clean stat tile — no child border artifacts."""
    frame = QWidget()
    frame.setObjectName("stile")
    frame.setFixedHeight(70)
    frame.setStyleSheet(f"""
        QWidget#stile {{
            background-color: #131920;
            border: 1px solid #1E2832;
            border-left: 3px solid {color};
            border-radius: {Size.RADIUS}px;
        }}
        QWidget#stile QLabel {{
            background: transparent; border: none;
        }}
    """)
    col = QVBoxLayout(frame)
    col.setAlignment(Qt.AlignmentFlag.AlignCenter)
    col.setContentsMargins(14, 6, 14, 6)
    col.setSpacing(0)
    h = QLabel(label)
    h.setStyleSheet(
        f"font-size: 10px; font-weight: 700; color: {Color.TEXT_DISABLED};"
        " letter-spacing: 0.8px;"
    )
    h.setAlignment(Qt.AlignmentFlag.AlignCenter)
    v = QLabel(value)
    v.setStyleSheet(f"font-size: 24px; font-weight: 700; color: {Color.TEXT};")
    v.setAlignment(Qt.AlignmentFlag.AlignCenter)
    col.addWidget(h)
    col.addWidget(v)
    return frame, v


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
        root.setContentsMargins(30, Size.SPACING_SM, 30, 22)
        root.setSpacing(Size.SPACING_SM)

        # Top bar
        top = QHBoxLayout()
        btn_back = _QPushButton(f"{Icon.BACK}  Back")
        btn_back.setStyleSheet(back_link_style())
        btn_back.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_back.clicked.connect(self._on_back)
        top.addWidget(btn_back)
        title = QLabel("Stamina Test")
        title.setStyleSheet(f"font-size: 20px; font-weight: 700; color: {Color.TEXT};")
        top.addWidget(title)
        top.addStretch()

        self._duration_badge = QLabel("2:00")
        self._duration_badge.setStyleSheet(badge_style(Color.PRIMARY))
        top.addWidget(self._duration_badge)
        root.addLayout(top)

        # Timer
        self._timer = TimerDisplay(font_size=Size.TEXT_TIMER, show_ring=True)
        self._timer.finished.connect(self._on_done)
        self._timer.tick.connect(self._on_tick)
        root.addWidget(self._timer, stretch=1)

        # Live stats row
        stats = QHBoxLayout()
        stats.setSpacing(12)
        self._punch_counter = PunchCounter(label="PUNCHES")
        stats.addWidget(self._punch_counter)

        rate_widget, self._rate_lbl = _stat_col(
            "PUNCHES/MIN", "0", Color.PRIMARY
        )
        stats.addWidget(rate_widget)

        pad_widget, self._pad_lbl = _stat_col(
            "TARGET", "--", Color.WARNING
        )
        stats.addWidget(pad_widget)
        root.addLayout(stats)

        # Start / Stop button
        self._btn_action = BigButton("Start", stylesheet=PRIMARY_BTN)
        self._btn_action.setFixedHeight(70)
        self._btn_action.clicked.connect(self._toggle)
        root.addWidget(self._btn_action)

        # Results overlay
        self._results_widget = QWidget()
        res_lay = QVBoxLayout(self._results_widget)
        res_lay.setSpacing(16)
        res_lay.setContentsMargins(0, 8, 0, 0)

        res_title = QLabel("Results")
        res_title.setFont(font(20, bold=True))
        res_title.setStyleSheet(f"color: {Color.PRIMARY};")
        res_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        res_lay.addWidget(res_title)

        res_row = QHBoxLayout()
        res_row.setSpacing(12)
        self._stat_total = StatCard(
            "Total Punches", "--", accent=Color.PRIMARY,
        )
        self._stat_peak = StatCard(
            "Peak Rate", "--/min", accent=Color.WARNING,
        )
        self._stat_fatigue = StatCard(
            "Fatigue", "--", accent=Color.DANGER,
        )
        res_row.addWidget(self._stat_total)
        res_row.addWidget(self._stat_peak)
        res_row.addWidget(self._stat_fatigue)
        res_lay.addLayout(res_row)

        btn_done = BigButton("Done", stylesheet=PRIMARY_BTN)
        btn_done.setFixedHeight(70)
        btn_done.clicked.connect(
            lambda: self._router.navigate("performance")
        )
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
        self._duration_badge.setStyleSheet(badge_style(Color.DANGER))

    def _on_tick(self, remaining: int) -> None:
        self._elapsed = _DEFAULT_DURATION - remaining
        mins, secs = divmod(remaining, 60)
        self._duration_badge.setText(f"{mins}:{secs:02d}")
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
        self._duration_badge.setText("2:00")
        self._duration_badge.setStyleSheet(badge_style(Color.PRIMARY))
        logger.debug("StaminaTestPage entered")

    def on_leave(self) -> None:
        self._timer.reset()
