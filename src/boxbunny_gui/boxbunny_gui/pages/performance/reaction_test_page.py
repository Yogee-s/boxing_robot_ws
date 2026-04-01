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

from PySide6.QtWidgets import QPushButton as _QPushButton

from boxbunny_gui.theme import Color, Icon, Size, font, badge_style, back_link_style, GHOST_BTN, PRIMARY_BTN
from boxbunny_gui.widgets import BigButton, StatCard

if TYPE_CHECKING:
    from boxbunny_gui.gui_bridge import GuiBridge
    from boxbunny_gui.nav.router import PageRouter

logger = logging.getLogger(__name__)

_TOTAL_TRIALS = 3
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
        root.setContentsMargins(30, Size.SPACING_SM, 30, Size.SPACING_SM)
        root.setSpacing(12)

        # Top bar
        top = QHBoxLayout()
        btn_back = _QPushButton(f"{Icon.BACK}  Back")
        btn_back.setStyleSheet(back_link_style())
        btn_back.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_back.clicked.connect(self._abort)
        top.addWidget(btn_back)
        self._title = QLabel("Reaction Time")
        self._title.setStyleSheet(f"font-size: 20px; font-weight: 700; color: {Color.TEXT};")
        top.addWidget(self._title)
        top.addStretch()
        self._trial_lbl = QLabel(f"Trial 0/{_TOTAL_TRIALS}")
        self._trial_lbl.setStyleSheet(badge_style(Color.WARNING))
        top.addWidget(self._trial_lbl)
        root.addLayout(top)

        # Stimulus area (large central region)
        self._stimulus = QWidget()
        self._stimulus.setMinimumHeight(200)
        self._stimulus_lbl = QLabel("Tap Start to begin")
        self._stimulus_lbl.setFont(font(28, bold=True))
        self._stimulus_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._stimulus_lbl.setWordWrap(True)
        stim_lay = QVBoxLayout(self._stimulus)
        stim_lay.setContentsMargins(20, 20, 20, 20)
        stim_lay.addWidget(self._stimulus_lbl)
        self._set_stimulus_bg(Color.SURFACE)
        root.addWidget(self._stimulus, stretch=1)

        root.addSpacing(10)

        # Trial indicators row
        trials_row = QHBoxLayout()
        trials_row.setSpacing(12)
        trials_row.addStretch()
        self._trial_cards: list[QLabel] = []
        for i in range(_TOTAL_TRIALS):
            card = QLabel(f"#{i + 1}")
            card.setFixedSize(90, 40)
            card.setAlignment(Qt.AlignCenter)
            card.setStyleSheet(f"""
                font-size: 13px; font-weight: 600; color: {Color.TEXT_DISABLED};
                background-color: #131920;
                border: 1px solid #1E2832;
                border-radius: {Size.RADIUS_SM}px;
            """)
            trials_row.addWidget(card)
            self._trial_cards.append(card)
        trials_row.addStretch()
        root.addLayout(trials_row)

        root.addSpacing(10)

        # Start button
        self._btn_start = BigButton("Start", stylesheet=PRIMARY_BTN)
        self._btn_start.setFixedHeight(60)
        self._btn_start.clicked.connect(self._begin_test)
        root.addWidget(self._btn_start)

        # Results panel
        self._results_widget = QWidget()
        res_lay = QVBoxLayout(self._results_widget)
        res_lay.setSpacing(12)
        res_lay.setContentsMargins(0, 4, 0, 0)

        res_title = QLabel("Results")
        res_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        res_title.setStyleSheet(
            f"font-size: 22px; font-weight: 700; color: {Color.TEXT};"
            " background-color: #1A1810;"
            " border: 1px solid #3D351A;"
            f" border-radius: {Size.RADIUS}px;"
            " padding: 10px 20px;"
        )
        res_lay.addWidget(res_title)

        stats = QHBoxLayout()
        stats.setSpacing(8)
        self._stat_avg = StatCard(
            "Average", "-- ms", accent=Color.PRIMARY,
        )
        self._stat_best = StatCard(
            "Best", "-- ms", accent=Color.PRIMARY_LIGHT,
        )
        self._stat_worst = StatCard(
            "Worst", "-- ms", accent=Color.DANGER,
        )
        self._stat_tier = StatCard(
            "Tier", "--", accent=Color.WARNING,
        )
        stats.addWidget(self._stat_avg)
        stats.addWidget(self._stat_best)
        stats.addWidget(self._stat_worst)
        stats.addWidget(self._stat_tier)
        res_lay.addLayout(stats)

        btn_done = BigButton("Done", stylesheet=PRIMARY_BTN)
        btn_done.setFixedHeight(60)
        btn_done.clicked.connect(
            lambda: self._router.navigate("performance")
        )
        res_lay.addWidget(btn_done)
        self._results_widget.setVisible(False)
        root.addWidget(self._results_widget)

    def _set_stimulus_bg(self, color: str) -> None:
        border = Color.BORDER if color == Color.SURFACE else color
        self._stimulus.setStyleSheet(
            f"background-color: {color};"
            f" border-radius: 14px;"
            f" border: 1px solid {border};"
        )

    def _reset_trial_cards(self) -> None:
        for i, card in enumerate(self._trial_cards):
            card.setText(f"#{i + 1}")
            card.setStyleSheet(f"""
                font-size: 13px; font-weight: 600; color: {Color.TEXT_DISABLED};
                background-color: #131920;
                border: 1px solid #1E2832;
                border-radius: {Size.RADIUS_SM}px;
            """)

    def _begin_test(self) -> None:
        self._trial = 0
        self._times.clear()
        self._reset_trial_cards()
        self._btn_start.setVisible(False)
        self._results_widget.setVisible(False)
        self._next_trial()

    def _next_trial(self) -> None:
        self._trial += 1
        self._trial_lbl.setText(f"Trial {self._trial}/{_TOTAL_TRIALS}")
        self._stimulus_on = False
        self._set_stimulus_bg(Color.SURFACE)
        self._stimulus_lbl.setText("Wait for green...")
        self._stimulus_lbl.setStyleSheet(
            f"background: transparent; color: {Color.TEXT_DISABLED}; font-size: 32px; font-weight: 700;"
            " letter-spacing: 1px;"
        )
        delay = random.randint(_MIN_DELAY_MS, _MAX_DELAY_MS)
        self._delay_timer.start(delay)

    def _show_stimulus(self) -> None:
        self._stimulus_on = True
        self._stimulus_time = time.monotonic()
        self._set_stimulus_bg(Color.SUCCESS)
        self._stimulus_lbl.setText("PUNCH NOW!")
        self._stimulus_lbl.setStyleSheet(
            "background: transparent; color: #FFFFFF; font-size: 44px; font-weight: 800;"
            " letter-spacing: 2px;"
        )

    def _on_punch(self, data: Dict[str, Any]) -> None:
        if not self._stimulus_on:
            return
        reaction_ms = (time.monotonic() - self._stimulus_time) * 1000
        self._times.append(reaction_ms)
        self._stimulus_on = False
        self._set_stimulus_bg(Color.SURFACE)
        self._stimulus_lbl.setText(f"{reaction_ms:.0f} ms")
        self._stimulus_lbl.setStyleSheet(
            f"background: transparent; color: {Color.PRIMARY}; font-size: 44px; font-weight: 800;"
        )

        # Update trial card
        idx = len(self._times) - 1
        if idx < len(self._trial_cards):
            ms = reaction_ms
            if ms < 200:
                color = Color.SUCCESS
            elif ms < 350:
                color = Color.PRIMARY
            else:
                color = Color.DANGER
            self._trial_cards[idx].setText(f"{ms:.0f}ms")
            self._trial_cards[idx].setStyleSheet(f"""
                font-size: 13px; font-weight: 700; color: {color};
                background-color: #131920;
                border: 1px solid {color};
                border-radius: {Size.RADIUS_SM}px;
            """)

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
        self._stimulus_lbl.setStyleSheet(
            f"background: transparent; color: {Color.PRIMARY}; font-size: 28px; font-weight: 700;"
        )

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
        self._trial_lbl.setStyleSheet(badge_style(Color.WARNING))
        self._set_stimulus_bg(Color.SURFACE)
        self._stimulus_lbl.setText("Tap Start to begin")
        self._stimulus_lbl.setStyleSheet(
            f"background: transparent; color: {Color.TEXT_SECONDARY}; font-size: 28px; font-weight: 700;"
        )
        self._reset_trial_cards()
        logger.debug("ReactionTestPage entered")

    def on_leave(self) -> None:
        self._delay_timer.stop()
