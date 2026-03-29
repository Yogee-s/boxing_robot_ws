"""Reaction time test page.

Random-delay stimulus (green screen flash), 10 trials, results with
tier classification.
"""
from __future__ import annotations

import logging
import random
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from boxbunny_gui.theme import Color, Size, font, GHOST_BTN, PRIMARY_BTN
from boxbunny_gui.widgets import BigButton, StatCard

if TYPE_CHECKING:
    from boxbunny_gui.gui_bridge import GuiBridge
    from boxbunny_gui.nav.router import PageRouter

logger = logging.getLogger(__name__)

_TOTAL_TRIALS = 10
_MIN_DELAY_MS = 1000
_MAX_DELAY_MS = 4000

_TIERS = [
    (150, "Lightning"),
    (200, "Fast"),
    (280, "Average"),
    (380, "Developing"),
    (9999, "Slow"),
]


class ReactionTestPage(QWidget):
    """10-trial reaction time test with random stimulus delay."""

    def __init__(
        self,
        router: PageRouter,
        bridge: Optional[GuiBridge] = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._router = router
        self._bridge = bridge
        self._trial: int = 0
        self._times: List[float] = []
        self._stimulus_on: bool = False
        self._stimulus_time: float = 0.0
        self._delay_timer = QTimer(self)
        self._delay_timer.setSingleShot(True)
        self._delay_timer.timeout.connect(self._show_stimulus)
        self._build_ui()
        if self._bridge:
            self._bridge.punch_confirmed.connect(self._on_punch)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(Size.SPACING, Size.SPACING, Size.SPACING, Size.SPACING)
        root.setSpacing(Size.SPACING)

        # Top bar
        top = QHBoxLayout()
        btn_back = BigButton("Back", stylesheet=GHOST_BTN)
        btn_back.setFixedWidth(100)
        btn_back.clicked.connect(lambda: self._abort())
        top.addWidget(btn_back)
        self._title = QLabel("Reaction Time Test")
        self._title.setFont(font(Size.TEXT_SUBHEADER, bold=True))
        top.addWidget(self._title)
        top.addStretch()
        self._trial_lbl = QLabel(f"Trial 0/{_TOTAL_TRIALS}")
        self._trial_lbl.setFont(font(18))
        self._trial_lbl.setStyleSheet(f"color: {Color.TEXT_SECONDARY};")
        top.addWidget(self._trial_lbl)
        root.addLayout(top)

        # Stimulus area (large central region)
        self._stimulus = QWidget()
        self._stimulus.setMinimumHeight(250)
        self._stimulus_lbl = QLabel("Tap Start to begin")
        self._stimulus_lbl.setFont(font(32, bold=True))
        self._stimulus_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        stim_lay = QVBoxLayout(self._stimulus)
        stim_lay.addWidget(self._stimulus_lbl)
        self._set_stimulus_bg(Color.SURFACE)
        root.addWidget(self._stimulus, stretch=1)

        # Start button
        self._btn_start = BigButton("Start", stylesheet=PRIMARY_BTN)
        self._btn_start.setFixedHeight(70)
        self._btn_start.clicked.connect(self._begin_test)
        root.addWidget(self._btn_start)

        # Results panel
        self._results_widget = QWidget()
        res_lay = QVBoxLayout(self._results_widget)
        stats = QHBoxLayout()
        self._stat_avg = StatCard("Average", "-- ms")
        self._stat_best = StatCard("Best", "-- ms")
        self._stat_worst = StatCard("Worst", "-- ms")
        self._stat_tier = StatCard("Tier", "--")
        stats.addWidget(self._stat_avg)
        stats.addWidget(self._stat_best)
        stats.addWidget(self._stat_worst)
        stats.addWidget(self._stat_tier)
        res_lay.addLayout(stats)
        btn_done = BigButton("Done", stylesheet=PRIMARY_BTN)
        btn_done.clicked.connect(lambda: self._router.navigate("performance_menu"))
        res_lay.addWidget(btn_done)
        self._results_widget.setVisible(False)
        root.addWidget(self._results_widget)

    def _set_stimulus_bg(self, color: str) -> None:
        self._stimulus.setStyleSheet(
            f"background-color: {color}; border-radius: {Size.RADIUS_LG}px;"
        )

    def _begin_test(self) -> None:
        self._trial = 0
        self._times.clear()
        self._btn_start.setVisible(False)
        self._results_widget.setVisible(False)
        self._next_trial()

    def _next_trial(self) -> None:
        self._trial += 1
        self._trial_lbl.setText(f"Trial {self._trial}/{_TOTAL_TRIALS}")
        self._stimulus_on = False
        self._set_stimulus_bg(Color.SURFACE)
        self._stimulus_lbl.setText("Wait for green...")
        self._stimulus_lbl.setStyleSheet(f"color: {Color.TEXT_SECONDARY};")
        delay = random.randint(_MIN_DELAY_MS, _MAX_DELAY_MS)
        self._delay_timer.start(delay)

    def _show_stimulus(self) -> None:
        self._stimulus_on = True
        self._stimulus_time = time.monotonic()
        self._set_stimulus_bg(Color.PRIMARY)
        self._stimulus_lbl.setText("PUNCH NOW!")
        self._stimulus_lbl.setStyleSheet(f"color: {Color.BG}; font-weight: bold;")

    def _on_punch(self, data: Dict[str, Any]) -> None:
        if not self._stimulus_on:
            return
        reaction_ms = (time.monotonic() - self._stimulus_time) * 1000
        self._times.append(reaction_ms)
        self._stimulus_on = False
        self._set_stimulus_bg(Color.SURFACE)
        self._stimulus_lbl.setText(f"{reaction_ms:.0f} ms")
        self._stimulus_lbl.setStyleSheet(f"color: {Color.PRIMARY};")

        if self._trial >= _TOTAL_TRIALS:
            QTimer.singleShot(800, self._show_results)
        else:
            QTimer.singleShot(800, self._next_trial)

    def _show_results(self) -> None:
        avg = sum(self._times) / len(self._times) if self._times else 0
        best = min(self._times) if self._times else 0
        worst = max(self._times) if self._times else 0
        tier = next((t for ms, t in _TIERS if avg <= ms), "Slow")

        self._stat_avg.set_value(f"{avg:.0f} ms")
        self._stat_best.set_value(f"{best:.0f} ms")
        self._stat_worst.set_value(f"{worst:.0f} ms")
        self._stat_tier.set_value(tier)
        self._results_widget.setVisible(True)
        self._stimulus_lbl.setText("Test Complete")
        self._stimulus_lbl.setStyleSheet(f"color: {Color.TEXT};")

    def _abort(self) -> None:
        self._delay_timer.stop()
        self._router.back()

    # ── Lifecycle ──────────────────────────────────────────────────────
    def on_enter(self, **kwargs: Any) -> None:
        self._times.clear()
        self._trial = 0
        self._stimulus_on = False
        self._btn_start.setVisible(True)
        self._results_widget.setVisible(False)
        self._trial_lbl.setText(f"Trial 0/{_TOTAL_TRIALS}")
        self._set_stimulus_bg(Color.SURFACE)
        self._stimulus_lbl.setText("Tap Start to begin")
        self._stimulus_lbl.setStyleSheet(f"color: {Color.TEXT_SECONDARY};")
        logger.debug("ReactionTestPage entered")

    def on_leave(self) -> None:
        self._delay_timer.stop()
