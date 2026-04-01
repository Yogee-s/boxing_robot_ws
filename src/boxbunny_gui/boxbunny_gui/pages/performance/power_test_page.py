"""Power test page with IMU force measurement.

State machine: instructions -> countdown -> active (10 punches) -> results.
Shows force bar per punch and final stats.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

from PySide6.QtWidgets import QPushButton as _QPushButton

from boxbunny_gui.theme import Color, Icon, Size, font, GHOST_BTN, PRIMARY_BTN, badge_style, back_link_style
from boxbunny_gui.widgets import BigButton, StatCard, TimerDisplay

if TYPE_CHECKING:
    from boxbunny_gui.gui_bridge import GuiBridge
    from boxbunny_gui.nav.router import PageRouter

logger = logging.getLogger(__name__)

_TARGET_PUNCHES = 10
_STATE_INSTRUCTIONS = "instructions"
_STATE_COUNTDOWN = "countdown"
_STATE_ACTIVE = "active"
_STATE_RESULTS = "results"


class PowerTestPage(QWidget):
    """IMU-based power test: 10 max-effort punches."""

    def __init__(
        self,
        router: PageRouter,
        bridge: Optional[GuiBridge] = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._router = router
        self._bridge = bridge
        self._state: str = _STATE_INSTRUCTIONS
        self._forces: List[float] = []
        self._build_ui()
        if self._bridge:
            self._bridge.punch_confirmed.connect(self._on_punch)

    def _build_ui(self) -> None:
        self._root = QVBoxLayout(self)
        self._root.setContentsMargins(30, Size.SPACING_SM, 30, Size.SPACING_SM)
        self._root.setSpacing(Size.SPACING_SM)

        # Top bar
        top = QHBoxLayout()
        btn_back = _QPushButton(f"{Icon.BACK}  Back")
        btn_back.setStyleSheet(back_link_style())
        btn_back.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_back.clicked.connect(self._on_back)
        top.addWidget(btn_back)
        self._title = QLabel("Power Test")
        self._title.setStyleSheet(f"font-size: 20px; font-weight: 700; color: {Color.TEXT};")
        top.addWidget(self._title)
        top.addStretch()
        self._root.addLayout(top)

        # Instructions panel
        self._instr_widget = QWidget()
        self._instr_widget.setObjectName("instrPanel")
        self._instr_widget.setStyleSheet(f"""
            QWidget#instrPanel {{
                background-color: #131920;
                border: 1px solid #1E2832;
                border-radius: {Size.RADIUS_LG}px;
            }}
            QWidget#instrPanel QLabel {{
                background: transparent; border: none;
            }}
        """)
        instr_lay = QVBoxLayout(self._instr_widget)
        instr_lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        instr_lay.setContentsMargins(30, 30, 30, 30)
        instr_lay.setSpacing(16)

        self._imu_warn = QLabel("IMU Required \u2014 connect pads to proceed")
        self._imu_warn.setStyleSheet(
            f"color: {Color.WARNING}; font-size: 14px; font-weight: 600;"
            " background-color: rgba(255, 171, 64, 0.1);"
            " border-radius: 8px; padding: 10px 16px;"
        )
        self._imu_warn.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._imu_warn.setWordWrap(True)
        self._imu_warn.setVisible(False)
        instr_lay.addWidget(self._imu_warn)

        instr_text = QLabel(
            "Throw 10 punches as hard as you can.\n"
            "We will measure your peak force."
        )
        instr_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        instr_text.setWordWrap(True)
        instr_text.setStyleSheet(
            f"font-size: 16px; color: {Color.TEXT};"
        )
        instr_lay.addWidget(instr_text)

        self._btn_begin = BigButton("Begin Test", stylesheet=PRIMARY_BTN)
        self._btn_begin.setFixedHeight(60)
        self._btn_begin.setFixedWidth(280)
        self._btn_begin.clicked.connect(self._start_countdown)
        btn_wrap = QHBoxLayout()
        btn_wrap.setAlignment(Qt.AlignmentFlag.AlignCenter)
        btn_wrap.addWidget(self._btn_begin)
        instr_lay.addLayout(btn_wrap)
        self._root.addWidget(self._instr_widget, stretch=1)

        # Countdown timer
        self._countdown = TimerDisplay(
            font_size=Size.TEXT_TIMER, show_ring=False
        )
        self._countdown.finished.connect(self._start_active)
        self._countdown.setVisible(False)
        self._root.addWidget(self._countdown, stretch=1)

        # Active phase: counter + force bars
        self._active_widget = QWidget()
        active_lay = QVBoxLayout(self._active_widget)
        active_lay.setSpacing(12)
        active_lay.setContentsMargins(0, 0, 0, 0)

        self._count_lbl = QLabel("0 / 10")
        self._count_lbl.setFont(font(28, bold=True))
        self._count_lbl.setStyleSheet(
            "background: transparent; " + badge_style(Color.PRIMARY)
        )
        self._count_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._count_lbl.setFixedHeight(44)
        count_wrap = QHBoxLayout()
        count_wrap.setAlignment(Qt.AlignmentFlag.AlignCenter)
        count_wrap.addWidget(self._count_lbl)
        active_lay.addLayout(count_wrap)

        self._bars_layout = QHBoxLayout()
        self._bars_layout.setSpacing(8)
        self._bars: list[QProgressBar] = []
        for _ in range(_TARGET_PUNCHES):
            bar = QProgressBar()
            bar.setOrientation(Qt.Orientation.Vertical)
            bar.setFixedWidth(48)
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setTextVisible(False)
            bar.setStyleSheet(
                f"QProgressBar {{ background-color: {Color.SURFACE};"
                f" border: 1px solid {Color.BORDER};"
                f" border-radius: 8px; }}"
                f" QProgressBar::chunk {{ background-color: {Color.DANGER};"
                f" border-radius: 7px; }}"
            )
            self._bars_layout.addWidget(bar)
            self._bars.append(bar)
        active_lay.addLayout(self._bars_layout, stretch=1)
        self._active_widget.setVisible(False)
        self._root.addWidget(self._active_widget, stretch=1)

        # Results panel
        self._results_widget = QWidget()
        res_lay = QVBoxLayout(self._results_widget)
        res_lay.setSpacing(16)
        res_lay.setContentsMargins(0, 8, 0, 0)

        res_title = QLabel("Results")
        res_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        res_title.setStyleSheet(
            f"font-size: 22px; font-weight: 700; color: {Color.TEXT};"
            " background-color: #1A1214;"
            " border: 1px solid #3D1A22;"
            f" border-radius: {Size.RADIUS}px;"
            " padding: 10px 20px;"
        )
        res_lay.addWidget(res_title)

        self._stat_peak = StatCard("Peak Force", "--", accent=Color.DANGER)
        self._stat_avg = StatCard("Average Force", "--", accent=Color.PRIMARY)
        res_row = QHBoxLayout()
        res_row.setSpacing(Size.SPACING)
        res_row.addWidget(self._stat_peak)
        res_row.addWidget(self._stat_avg)
        res_lay.addLayout(res_row)

        btn_home = BigButton("Done", stylesheet=PRIMARY_BTN)
        btn_home.setFixedHeight(60)
        btn_home.clicked.connect(
            lambda: self._router.navigate("performance")
        )
        res_lay.addWidget(btn_home)
        self._results_widget.setVisible(False)
        self._root.addWidget(self._results_widget)

    def _set_state(self, state: str) -> None:
        self._state = state
        self._instr_widget.setVisible(state == _STATE_INSTRUCTIONS)
        self._countdown.setVisible(state == _STATE_COUNTDOWN)
        self._active_widget.setVisible(state == _STATE_ACTIVE)
        self._results_widget.setVisible(state == _STATE_RESULTS)

    def _start_countdown(self) -> None:
        self._set_state(_STATE_COUNTDOWN)
        self._countdown.start(3)

    def _start_active(self) -> None:
        self._forces.clear()
        for bar in self._bars:
            bar.setValue(0)
        self._count_lbl.setText(f"0 / {_TARGET_PUNCHES}")
        self._set_state(_STATE_ACTIVE)

    def _on_punch(self, data: Dict[str, Any]) -> None:
        if self._state != _STATE_ACTIVE:
            return
        force = data.get("force", 0.0)
        self._forces.append(force)
        idx = len(self._forces) - 1
        if idx < _TARGET_PUNCHES:
            self._bars[idx].setValue(int(force * 100))
        self._count_lbl.setText(
            f"{len(self._forces)} / {_TARGET_PUNCHES}"
        )
        if len(self._forces) >= _TARGET_PUNCHES:
            self._show_results()

    def _show_results(self) -> None:
        peak = max(self._forces) if self._forces else 0
        avg = (
            sum(self._forces) / len(self._forces)
            if self._forces else 0
        )
        self._stat_peak.set_value(f"{peak:.0%}")
        self._stat_avg.set_value(f"{avg:.0%}")
        self._set_state(_STATE_RESULTS)

    def _on_back(self) -> None:
        self._countdown.reset()
        self._router.back()

    # ── Lifecycle ──────────────────────────────────────────────────────
    def on_enter(self, **kwargs: Any) -> None:
        self._forces.clear()
        imu_available = (
            self._bridge is not None and self._bridge.online
        )
        self._imu_warn.setVisible(not imu_available)
        self._btn_begin.setEnabled(True)
        self._set_state(_STATE_INSTRUCTIONS)
        logger.debug("PowerTestPage entered")

    def on_leave(self) -> None:
        self._countdown.reset()
