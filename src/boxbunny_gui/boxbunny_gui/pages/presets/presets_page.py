"""Preset management page.

List of user's presets as PresetCards with Create New button.
Tap starts session; long-press placeholder for edit/delete/favourite.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from boxbunny_gui.theme import Color, Size, font, GHOST_BTN, PRIMARY_BTN, SURFACE_BTN
from boxbunny_gui.widgets import BigButton, PresetCard

if TYPE_CHECKING:
    from boxbunny_gui.nav.router import PageRouter

logger = logging.getLogger(__name__)

# Placeholder presets (keys match PresetCard.set_preset expected format)
_DEMO_PRESETS: List[Dict[str, Any]] = [
    {"id": 1, "name": "Morning Warmup", "mode": "training",
     "summary": "3 rounds, Jab-Cross", "favorite": True},
    {"id": 2, "name": "Heavy Bag", "mode": "training",
     "summary": "5 rounds, Full Combo", "favorite": False},
    {"id": 3, "name": "Quick Session", "mode": "training",
     "summary": "2 rounds, Jab-Cross-Hook", "favorite": True},
]


class PresetsPage(QWidget):
    """Browse, create, and manage training presets."""

    def __init__(self, router: PageRouter, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._router = router
        self._cards: list[PresetCard] = []
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(Size.SPACING, Size.SPACING_SM, Size.SPACING, Size.SPACING_SM)
        root.setSpacing(Size.SPACING)

        # Top bar
        top = QHBoxLayout()
        btn_back = BigButton("Back", stylesheet=GHOST_BTN)
        btn_back.setFixedWidth(100)
        btn_back.clicked.connect(lambda: self._router.back())
        top.addWidget(btn_back)
        title = QLabel("Presets")
        title.setFont(font(Size.TEXT_SUBHEADER, bold=True))
        top.addWidget(title)
        top.addStretch()
        self._btn_create = BigButton("Create New", stylesheet=PRIMARY_BTN)
        self._btn_create.setFixedWidth(160)
        self._btn_create.clicked.connect(self._on_create)
        top.addWidget(self._btn_create)
        root.addLayout(top)

        # Scrollable preset list
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self._list_widget = QWidget()
        self._list_layout = QVBoxLayout(self._list_widget)
        self._list_layout.setSpacing(Size.SPACING_SM)
        self._list_layout.setContentsMargins(0, 0, 0, 0)
        scroll.setWidget(self._list_widget)
        root.addWidget(scroll, stretch=1)

    def _populate(self) -> None:
        for card in self._cards:
            self._list_layout.removeWidget(card)
            card.deleteLater()
        self._cards.clear()

        for preset in _DEMO_PRESETS:
            card = PresetCard(parent=self)
            card.set_preset(preset)
            card.clicked.connect(lambda pid, p=preset: self._on_tap(p))
            self._list_layout.addWidget(card)
            self._cards.append(card)
        self._list_layout.addStretch()

    def _on_tap(self, preset: Dict[str, Any]) -> None:
        logger.info("Preset tapped: %s", preset["name"])
        # TODO: load full preset config and navigate to training_config
        self._router.navigate("training_config", combo={"name": preset["name"]})

    def _on_create(self) -> None:
        logger.info("Create new preset")
        self._router.navigate("training_config")

    # ── Lifecycle ──────────────────────────────────────────────────────
    def on_enter(self, **kwargs: Any) -> None:
        self._populate()
        logger.debug("PresetsPage entered")

    def on_leave(self) -> None:
        pass
