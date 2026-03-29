"""Power test page with IMU force measurement.

State machine: instructions -> countdown -> active (10 punches) -> results.
Shows force bar per punch and final stats.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

from boxbunny_gui.theme import Color, Size, font, DANGER_BTN, GHOST_BTN, PRIMARY_BTN
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
        self._root.setContentsMargins(Size.SPACING, Size.SPACING, Size.SPACING, Size.SPACING)
        self._root.setSpacing(Size.SPACING)

        # Back button (always visible)
        top = QHBoxLayout()
        btn_back = BigButton("Back", stylesheet=GHOST_BTN)
        btn_back.setFixedWidth(100)
        btn_back.clicked.connect(self._on_back)
        top.addWidget(btn_back)
        self._title = QLabel("Power Test")
        self._title.setFont(font(Size.TEXT_SUBHEADER, bold=True))
        top.addWidget(self._title)
        top.addStretch()
        self._root.addLayout(top)

        # Instructions panel
        self._instr_widget = QWidget()
        instr_lay = QVBoxLayout(self._instr_widget)
        instr_lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._imu_warn = QLabel("IMU Required -- connect pads to proceed")
        self._imu_warn.setFont(font(18, bold=True))
        self._imu_warn.setStyleSheet(f"color: {Color.WARNING};")
        self._imu_warn.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._imu_warn.setWordWrap(True)
        self._imu_warn.setVisible(False)
        instr_lay.addWidget(self._imu_warn)
        instr_text = QLabel("Throw 10 punches as hard as you can.\nWe will measure your peak force.")
        instr_text.setFont(font(18))
        instr_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        instr_text.setWordWrap(True)
        instr_lay.addWidget(instr_text)
        self._btn_begin = BigButton("Begin Test", stylesheet=PRIMARY_BTN)
        self._btn_begin.setFixedHeight(70)
        self._btn_begin.clicked.connect(self._start_countdown)
        instr_lay.addWidget(self._btn_begin)
        self._root.addWidget(self._instr_widget)

        # Countdown timer (reused for countdown phase)
        self._countdown = TimerDisplay(font_size=Size.TEXT_TIMER, show_ring=False)
        self._countdown.finished.connect(self._start_active)
        self._countdown.setVisible(False)
        self._root.addWidget(self._countdown, stretch=1)

        # Active phase: counter + force bars
        self._active_widget = QWidget()
        active_lay = QVBoxLayout(self._active_widget)
        self._count_lbl = QLabel("0 / 10")
        self._count_lbl.setFont(font(36, bold=True))
        self._count_lbl.setStyleSheet(f"color: {Color.PRIMARY};")
        self._count_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        active_lay.addWidget(self._count_lbl)
        self._bars_layout = QHBoxLayout()
        self._bars_layout.setSpacing(4)
        self._bars: list[QProgressBar] = []
        for _ in range(_TARGET_PUNCHES):
            bar = QProgressBar()
            bar.setOrientation(Qt.Orientation.Vertical)
            bar.setFixedWidth(40)
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setTextVisible(False)
            bar.setStyleSheet(
                f"QProgressBar {{ background-color: {Color.SURFACE_LIGHT}; border-radius: 4px; }}"
                f" QProgressBar::chunk {{ background-color: {Color.PRIMARY}; border-radius: 4px; }}"
            )
            self._bars_layout.addWidget(bar)
            self._bars.append(bar)
        active_lay.addLayout(self._bars_layout)
        self._active_widget.setVisible(False)
        self._root.addWidget(self._active_widget, stretch=1)

        # Results panel
        self._results_widget = QWidget()
        res_lay = QVBoxLayout(self._results_widget)
        self._stat_peak = StatCard("Peak Force", "--")
        self._stat_avg = StatCard("Average Force", "--")
        res_row = QHBoxLayout()
        res_row.addWidget(self._stat_peak)
        res_row.addWidget(self._stat_avg)
        res_lay.addLayout(res_row)
        btn_home = BigButton("Done", stylesheet=PRIMARY_BTN)
        btn_home.clicked.connect(lambda: self._router.navigate("performance_menu"))
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
        self._count_lbl.setText(f"{len(self._forces)} / {_TARGET_PUNCHES}")
        if len(self._forces) >= _TARGET_PUNCHES:
            self._show_results()

    def _show_results(self) -> None:
        peak = max(self._forces) if self._forces else 0
        avg = sum(self._forces) / len(self._forces) if self._forces else 0
        self._stat_peak.set_value(f"{peak:.0%}")
        self._stat_avg.set_value(f"{avg:.0%}")
        self._set_state(_STATE_RESULTS)

    def _on_back(self) -> None:
        self._countdown.reset()
        self._router.back()

    # ── Lifecycle ──────────────────────────────────────────────────────
    def on_enter(self, **kwargs: Any) -> None:
        self._forces.clear()
        imu_available = self._bridge is not None and self._bridge.online
        self._imu_warn.setVisible(not imu_available)
        self._btn_begin.setEnabled(imu_available or True)  # allow offline testing
        self._set_state(_STATE_INSTRUCTIONS)
        logger.debug("PowerTestPage entered")

    def on_leave(self) -> None:
        self._countdown.reset()
