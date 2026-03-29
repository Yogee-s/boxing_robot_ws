"""Post-training session results page.

Stat cards grid, punch distribution bar chart placeholder, AI coach
summary, and action buttons.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from boxbunny_gui.theme import Color, Size, font, GHOST_BTN, PRIMARY_BTN, SURFACE_BTN, section_title_style
from boxbunny_gui.widgets import BigButton, StatCard

if TYPE_CHECKING:
    from boxbunny_gui.gui_bridge import GuiBridge
    from boxbunny_gui.nav.router import PageRouter

logger = logging.getLogger(__name__)


class TrainingResultsPage(QWidget):
    """Post-session results display with AI coach feedback."""

    def __init__(
        self,
        router: PageRouter,
        bridge: Optional[GuiBridge] = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._router = router
        self._bridge = bridge
        self._config: Dict[str, Any] = {}
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 12, 24, 12)
        root.setSpacing(10)

        # Title row with checkmark accent
        title = QLabel("\u2713  Session Complete")
        title.setFont(font(Size.TEXT_SUBHEADER, bold=True))
        title.setStyleSheet(f"color: {Color.PRIMARY};")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(title)

        # Stat cards (2x2) with proper spacing
        stats = QGridLayout()
        stats.setSpacing(12)
        stats.setContentsMargins(8, 0, 8, 0)
        self._stat_punches = StatCard("Total Punches", "0")
        self._stat_accuracy = StatCard("Accuracy", "0%")
        self._stat_best = StatCard("Best Round", "--")
        self._stat_fatigue = StatCard("Fatigue Index", "--")
        stats.addWidget(self._stat_punches, 0, 0)
        stats.addWidget(self._stat_accuracy, 0, 1)
        stats.addWidget(self._stat_best, 1, 0)
        stats.addWidget(self._stat_fatigue, 1, 1)
        root.addLayout(stats)

        # Punch distribution placeholder -- polished card
        self._chart_frame = QFrame()
        self._chart_frame.setFixedHeight(56)
        self._chart_frame.setStyleSheet(
            f"background-color: {Color.SURFACE};"
            f" border: 1px solid {Color.BORDER};"
            f" border-radius: {Size.RADIUS}px;"
        )
        chart_lay = QVBoxLayout(self._chart_frame)
        chart_lay.setContentsMargins(16, 8, 16, 8)
        chart_title = QLabel("PUNCH DISTRIBUTION")
        chart_title.setStyleSheet(
            "background: transparent;"
            f" color: {Color.TEXT_DISABLED}; font-size: 10px;"
            " font-weight: 700; letter-spacing: 0.8px;"
        )
        chart_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        chart_lay.addWidget(chart_title)
        chart_bar = QLabel("\u2588 \u2588 \u2588 \u2588 \u2588 \u2588")
        chart_bar.setStyleSheet(
            "background: transparent;"
            f" color: {Color.TEXT_DISABLED}; font-size: 16px; letter-spacing: 4px;"
        )
        chart_bar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        chart_lay.addWidget(chart_bar)
        root.addWidget(self._chart_frame)

        # AI Coach summary -- with section header and card styling
        coach_section = QVBoxLayout()
        coach_section.setSpacing(6)
        coach_header = QLabel("\u2728  AI COACH")
        coach_header.setStyleSheet(section_title_style())
        coach_section.addWidget(coach_header)

        self._coach_lbl = QLabel("AI Coach analysis loading...")
        self._coach_lbl.setStyleSheet(
            f"color: {Color.TEXT_SECONDARY}; font-size: 14px;"
            " line-height: 1.4;"
            f" background-color: {Color.SURFACE};"
            f" border: 1px solid {Color.BORDER};"
            f" border-left: 3px solid {Color.PRIMARY};"
            f" border-radius: {Size.RADIUS}px;"
            " padding: 14px 16px;"
        )
        self._coach_lbl.setWordWrap(True)
        self._coach_lbl.setMinimumHeight(50)
        self._coach_lbl.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        coach_section.addWidget(self._coach_lbl)
        root.addLayout(coach_section)

        root.addStretch()

        # Action buttons row — no fixed widths, let them share space
        bottom = QHBoxLayout()
        bottom.setSpacing(8)

        self._btn_home = BigButton("Home", stylesheet=GHOST_BTN)
        self._btn_home.setFixedHeight(42)
        self._btn_home.clicked.connect(
            lambda: self._router.navigate("home")
        )
        bottom.addWidget(self._btn_home, stretch=1)

        self._btn_save = BigButton("Save", stylesheet=SURFACE_BTN)
        self._btn_save.setFixedHeight(42)
        self._btn_save.clicked.connect(self._on_save_preset)
        bottom.addWidget(self._btn_save, stretch=1)

        self._btn_again = BigButton("Train Again", stylesheet=PRIMARY_BTN)
        self._btn_again.setFixedHeight(42)
        self._btn_again.clicked.connect(self._on_train_again)
        bottom.addWidget(self._btn_again, stretch=1)
        root.addLayout(bottom)

    def _on_save_preset(self) -> None:
        logger.info("Save preset requested")

    def _on_train_again(self) -> None:
        self._router.navigate(
            "training_config", combo=self._config.get("combo", {})
        )

    def _request_llm_summary(self) -> None:
        if self._bridge is None:
            self._coach_lbl.setText("AI Coach unavailable in offline mode.")
            return
        self._bridge.call_generate_llm(
            prompt="Summarize this boxing training session in 1-2 sentences.",
            context_json="{}",
            system_prompt_key="coach_summary",
            callback=self._on_llm_response,
        )

    def _on_llm_response(
        self, success: bool, response: str, gen_time: float,
    ) -> None:
        if success:
            self._coach_lbl.setText(response)
        else:
            self._coach_lbl.setText("AI Coach analysis unavailable.")

    # ── Lifecycle ──────────────────────────────────────────────────────
    def on_enter(self, **kwargs: Any) -> None:
        self._config = kwargs.get("config", {})
        self._stat_punches.set_value("--")
        self._stat_accuracy.set_value("--%")
        self._stat_best.set_value("--")
        self._stat_fatigue.set_value("--")
        self._request_llm_summary()
        logger.debug("TrainingResultsPage entered")

    def on_leave(self) -> None:
        pass
