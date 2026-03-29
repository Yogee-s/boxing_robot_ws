"""Post-training session results page.

Stat cards grid, punch distribution bar chart placeholder, AI coach
summary, QR code for detailed breakdown, and action buttons.
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

from boxbunny_gui.theme import Color, Size, font, GHOST_BTN, PRIMARY_BTN, SURFACE_BTN
from boxbunny_gui.widgets import BigButton, QRWidget, StatCard

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
        root.setContentsMargins(Size.SPACING, Size.SPACING_SM, Size.SPACING, Size.SPACING_SM)
        root.setSpacing(Size.SPACING)

        # Title
        title = QLabel("Session Complete")
        title.setFont(font(Size.TEXT_SUBHEADER, bold=True))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(title)

        # Stat cards (2x2)
        stats = QGridLayout()
        stats.setSpacing(Size.SPACING_SM)
        self._stat_punches = StatCard("Total Punches", "0")
        self._stat_accuracy = StatCard("Accuracy", "0%")
        self._stat_best = StatCard("Best Round", "--")
        self._stat_fatigue = StatCard("Fatigue Index", "--")
        stats.addWidget(self._stat_punches, 0, 0)
        stats.addWidget(self._stat_accuracy, 0, 1)
        stats.addWidget(self._stat_best, 1, 0)
        stats.addWidget(self._stat_fatigue, 1, 1)
        root.addLayout(stats)

        # Punch distribution placeholder
        self._chart_frame = QFrame()
        self._chart_frame.setFixedHeight(60)
        self._chart_frame.setStyleSheet(
            f"background-color: {Color.SURFACE}; border-radius: {Size.RADIUS_SM}px;"
        )
        chart_lay = QHBoxLayout(self._chart_frame)
        # TODO: replace with QPainter bar chart or coloured QFrame bars
        chart_placeholder = QLabel("Punch Distribution Chart")
        chart_placeholder.setStyleSheet(f"color: {Color.TEXT_DISABLED}; font-size: 14px;")
        chart_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        chart_lay.addWidget(chart_placeholder)
        root.addWidget(self._chart_frame)

        # AI Coach summary
        self._coach_lbl = QLabel("AI Coach analysis loading...")
        self._coach_lbl.setStyleSheet(
            f"color: {Color.TEXT_SECONDARY}; font-size: 15px;"
            f" background-color: {Color.SURFACE}; border-radius: {Size.RADIUS_SM}px;"
            f" padding: {Size.SPACING_SM}px;"
        )
        self._coach_lbl.setWordWrap(True)
        self._coach_lbl.setAlignment(Qt.AlignmentFlag.AlignLeft)
        root.addWidget(self._coach_lbl)

        # QR + buttons
        bottom = QHBoxLayout()
        self._qr = QRWidget(size=64)
        self._qr.set_text("https://boxbunny.local/session/latest")
        bottom.addWidget(self._qr)
        bottom.addStretch()

        self._btn_save = BigButton("Save as Preset", stylesheet=SURFACE_BTN)
        self._btn_save.setFixedWidth(170)
        self._btn_save.clicked.connect(self._on_save_preset)
        bottom.addWidget(self._btn_save)

        self._btn_again = BigButton("Train Again", stylesheet=PRIMARY_BTN)
        self._btn_again.setFixedWidth(150)
        self._btn_again.clicked.connect(self._on_train_again)
        bottom.addWidget(self._btn_again)

        self._btn_home = BigButton("Home", stylesheet=GHOST_BTN)
        self._btn_home.setFixedWidth(100)
        self._btn_home.clicked.connect(lambda: self._router.navigate("home_individual"))
        bottom.addWidget(self._btn_home)
        root.addLayout(bottom)

    def _on_save_preset(self) -> None:
        # TODO: save current config as user preset
        logger.info("Save preset requested")

    def _on_train_again(self) -> None:
        self._router.navigate("training_config", combo=self._config.get("combo", {}))

    def _request_llm_summary(self) -> None:
        if self._bridge is None:
            self._coach_lbl.setText("AI Coach unavailable in offline mode.")
            return
        # TODO: build real context from session data
        self._bridge.call_generate_llm(
            prompt="Summarize this boxing training session in 1-2 sentences.",
            context_json="{}",
            system_prompt_key="coach_summary",
            callback=self._on_llm_response,
        )

    def _on_llm_response(self, success: bool, response: str, gen_time: float) -> None:
        if success:
            self._coach_lbl.setText(response)
        else:
            self._coach_lbl.setText("AI Coach analysis unavailable.")

    # ── Lifecycle ──────────────────────────────────────────────────────
    def on_enter(self, **kwargs: Any) -> None:
        self._config = kwargs.get("config", {})
        # TODO: populate stat cards from actual session data
        self._stat_punches.set_value("--")
        self._stat_accuracy.set_value("--%")
        self._stat_best.set_value("--")
        self._stat_fatigue.set_value("--")
        self._request_llm_summary()
        logger.debug("TrainingResultsPage entered")

    def on_leave(self) -> None:
        pass
