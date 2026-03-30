"""Rest period between training rounds — calming dark treatment.

Large countdown timer, brief stats from last round, and skip button.
Subtly different background for visual distinction from active training.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from boxbunny_gui.theme import Color, Icon, Size, font, SURFACE_BTN
from boxbunny_gui.widgets import BigButton, StatCard, TimerDisplay

if TYPE_CHECKING:
    from boxbunny_gui.nav.router import PageRouter

logger = logging.getLogger(__name__)


class TrainingRestPage(QWidget):
    """Rest countdown between training rounds."""

    def __init__(self, router: PageRouter, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._router = router
        self._config: Dict[str, Any] = {}
        self._round_num: int = 1
        self._total_rounds: int = 3
        self._build_ui()

    def _build_ui(self) -> None:
        # Slightly blue-tinted surface for calming distinction
        self.setStyleSheet(f"background-color: {Color.SURFACE};")

        root = QVBoxLayout(self)
        root.setContentsMargins(40, 24, 40, 20)
        root.setSpacing(10)

        # Large REST title in calming blue tint
        title = QLabel("REST")
        title.setFont(font(42, bold=True))
        title.setStyleSheet(
            f"background: transparent; color: {Color.INFO};"
            " letter-spacing: 8px;"
        )
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(title)

        # Next round subtitle
        self._next_round_lbl = QLabel("Next round coming up...")
        self._next_round_lbl.setStyleSheet(
            f"background: transparent; color: {Color.TEXT_DISABLED};"
            " font-size: 14px;"
        )
        self._next_round_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(self._next_round_lbl)

        root.addSpacing(4)

        # Timer — centered and calm (uses blue color via widget)
        self._timer = TimerDisplay(font_size=Size.TEXT_TIMER_SM, show_ring=True)
        self._timer.finished.connect(self._on_rest_done)
        root.addWidget(self._timer, stretch=1)

        # Quick stats from last round
        stats_row = QHBoxLayout()
        stats_row.setSpacing(14)
        stats_row.addStretch()
        self._stat_punches = StatCard("Punches", "--", accent=Color.PRIMARY)
        self._stat_punches.setFixedWidth(180)
        self._stat_accuracy = StatCard("Accuracy", "--%", accent=Color.INFO)
        self._stat_accuracy.setFixedWidth(180)
        stats_row.addWidget(self._stat_punches)
        stats_row.addWidget(self._stat_accuracy)
        stats_row.addStretch()
        root.addLayout(stats_row)

        root.addSpacing(8)

        # Skip button — subtle, centered
        self._btn_skip = BigButton(
            f"Skip Rest  {Icon.NEXT}", stylesheet=SURFACE_BTN
        )
        self._btn_skip.setFixedSize(200, 50)
        self._btn_skip.clicked.connect(self._skip)
        root.addWidget(
            self._btn_skip, alignment=Qt.AlignmentFlag.AlignCenter
        )

    def _on_rest_done(self) -> None:
        self._advance_to_next_round()

    def _skip(self) -> None:
        self._timer.pause()
        self._advance_to_next_round()

    def _advance_to_next_round(self) -> None:
        next_round = self._round_num + 1
        logger.info("Advancing to round %d/%d", next_round, self._total_rounds)
        self._router.replace(
            "training_session",
            config=self._config,
            round_num=next_round,
            curriculum=self._curriculum,
            combo_id=self._combo_id,
            difficulty=self._difficulty,
        )

    def _parse_seconds(self, val: str) -> int:
        return int(val.rstrip("s")) if val.rstrip("s").isdigit() else 30

    # ── Lifecycle ──────────────────────────────────────────────────────
    def on_enter(self, **kwargs: Any) -> None:
        self._config = kwargs.get("config", {})
        self._round_num = kwargs.get("round_num", 1)
        self._total_rounds = kwargs.get("total_rounds", 3)
        self._curriculum = kwargs.get("curriculum")
        self._combo_id = kwargs.get("combo_id")
        self._difficulty = kwargs.get("difficulty")
        rest_time = self._parse_seconds(self._config.get("Rest Time", "30s"))
        self._timer.start(rest_time)

        self._next_round_lbl.setText(
            f"Round {self._round_num + 1} of {self._total_rounds} coming up..."
        )
        self._stat_punches.set_value("--")
        self._stat_accuracy.set_value("--%")
        logger.debug(
            "TrainingRestPage entered (after round %d)", self._round_num
        )

    def on_leave(self) -> None:
        self._timer.pause()
