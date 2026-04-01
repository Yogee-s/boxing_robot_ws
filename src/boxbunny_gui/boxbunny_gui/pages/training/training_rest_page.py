"""Rest period between training rounds — clean calming design."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from boxbunny_gui.theme import Color, Icon, Size, font
from boxbunny_gui.widgets import TimerDisplay

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
        root = QVBoxLayout(self)
        root.setContentsMargins(32, 16, 32, 22)
        root.setSpacing(0)

        root.addStretch(1)

        # REST title
        title = QLabel("REST")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(
            f"font-size: 36px; font-weight: 700; color: {Color.INFO};"
            " letter-spacing: 8px;"
        )
        root.addWidget(title)
        root.addSpacing(4)

        self._next_round_lbl = QLabel("Next round coming up...")
        self._next_round_lbl.setAlignment(Qt.AlignCenter)
        self._next_round_lbl.setStyleSheet(
            f"font-size: 14px; color: {Color.TEXT_SECONDARY};"
        )
        root.addWidget(self._next_round_lbl)

        # Timer
        self._timer = TimerDisplay(font_size=Size.TEXT_TIMER_SM, show_ring=True)
        self._timer.finished.connect(self._on_rest_done)
        root.addWidget(self._timer, stretch=1)

        root.addStretch(1)

        # Skip button — centered
        bottom = QHBoxLayout()
        bottom.addStretch()
        self._btn_skip = QPushButton(f"Skip Rest  {Icon.NEXT}")
        self._btn_skip.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_skip.setFixedSize(180, 44)
        self._btn_skip.setStyleSheet(f"""
            QPushButton {{
                font-size: 14px; font-weight: 600;
                background-color: {Color.SURFACE};
                color: {Color.TEXT};
                border: 1px solid {Color.BORDER_LIGHT};
                border-radius: {Size.RADIUS}px;
            }}
            QPushButton:hover {{
                border-color: {Color.INFO};
                background-color: {Color.SURFACE_HOVER};
            }}
        """)
        self._btn_skip.clicked.connect(self._skip)
        bottom.addWidget(self._btn_skip)
        bottom.addStretch()
        root.addLayout(bottom)

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
        logger.debug("TrainingRestPage entered (after round %d)", self._round_num)

    def on_leave(self) -> None:
        self._timer.pause()
