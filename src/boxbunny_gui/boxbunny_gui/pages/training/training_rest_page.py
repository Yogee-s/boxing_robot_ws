"""Rest period between training rounds.

Large countdown timer, brief stats from last round, and skip button.
Visually distinct background shade.
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

from boxbunny_gui.theme import Color, Size, font, SURFACE_BTN
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
        self.setStyleSheet(f"background-color: {Color.SURFACE};")

        root = QVBoxLayout(self)
        root.setContentsMargins(Size.SPACING_LG, Size.SPACING_LG,
                                Size.SPACING_LG, Size.SPACING_LG)
        root.setSpacing(Size.SPACING)

        # Title
        title = QLabel("REST")
        title.setFont(font(28, bold=True))
        title.setStyleSheet(f"color: {Color.TEXT_SECONDARY};")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(title)

        # Timer
        self._timer = TimerDisplay(font_size=Size.TEXT_TIMER, show_ring=True)
        self._timer.finished.connect(self._on_rest_done)
        root.addWidget(self._timer, stretch=1)

        # Quick stats from last round
        stats_row = QHBoxLayout()
        self._stat_punches = StatCard("Punches", "--")
        self._stat_accuracy = StatCard("Accuracy", "--%")
        stats_row.addWidget(self._stat_punches)
        stats_row.addWidget(self._stat_accuracy)
        root.addLayout(stats_row)

        # Skip button
        self._btn_skip = BigButton("Skip Rest", stylesheet=SURFACE_BTN)
        self._btn_skip.setFixedHeight(Size.BUTTON_H)
        self._btn_skip.clicked.connect(self._skip)
        root.addWidget(self._btn_skip, alignment=Qt.AlignmentFlag.AlignCenter)

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
        )

    def _parse_seconds(self, val: str) -> int:
        return int(val.rstrip("s")) if val.rstrip("s").isdigit() else 30

    # ── Lifecycle ──────────────────────────────────────────────────────
    def on_enter(self, **kwargs: Any) -> None:
        self._config = kwargs.get("config", {})
        self._round_num = kwargs.get("round_num", 1)
        self._total_rounds = kwargs.get("total_rounds", 3)
        rest_time = self._parse_seconds(self._config.get("Rest Time", "30s"))
        self._timer.start(rest_time)

        # TODO: receive actual round stats from bridge
        self._stat_punches.set_value("--")
        self._stat_accuracy.set_value("--%")
        logger.debug("TrainingRestPage entered (after round %d)", self._round_num)

    def on_leave(self) -> None:
        self._timer.pause()
