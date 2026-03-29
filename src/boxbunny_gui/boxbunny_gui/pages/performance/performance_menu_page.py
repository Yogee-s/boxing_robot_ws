"""Performance test selection menu.

Three large cards: Power Test, Stamina Test, Reaction Time Test.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from boxbunny_gui.theme import Color, Size, font, GHOST_BTN, mode_card_style
from boxbunny_gui.widgets import BigButton

if TYPE_CHECKING:
    from boxbunny_gui.nav.router import PageRouter

logger = logging.getLogger(__name__)

_TESTS = [
    {
        "name": "Power Test",
        "desc": "Measure your punch force with 10 max-effort hits",
        "route": "power_test",
        "accent": Color.DANGER,
    },
    {
        "name": "Stamina Test",
        "desc": "Throw as many punches as you can in 2 minutes",
        "route": "stamina_test",
        "accent": Color.PRIMARY,
    },
    {
        "name": "Reaction Time",
        "desc": "Punch when the screen flashes \u2014 10 trials",
        "route": "reaction_test",
        "accent": Color.WARNING,
    },
]


class PerformanceMenuPage(QWidget):
    """Selection screen for the three performance tests."""

    def __init__(self, router: PageRouter, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._router = router
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(40, Size.SPACING, 40, Size.SPACING)
        root.setSpacing(Size.SPACING_SM)

        # Back + title
        top = QHBoxLayout()
        btn_back = BigButton("Back", stylesheet=GHOST_BTN)
        btn_back.setFixedWidth(100)
        btn_back.clicked.connect(lambda: self._router.back())
        top.addWidget(btn_back)
        title = QLabel("Performance Tests")
        title.setFont(font(Size.TEXT_SUBHEADER, bold=True))
        top.addWidget(title)
        top.addStretch()
        root.addLayout(top)

        subtitle = QLabel("Select a test to measure your boxing performance")
        subtitle.setStyleSheet(
            f"color: {Color.TEXT_SECONDARY}; font-size: 14px;"
            " padding-left: 4px;"
        )
        root.addWidget(subtitle)
        root.addSpacing(8)

        # Test cards as QPushButtons using mode_card_style
        for test in _TESTS:
            accent = test["accent"]
            card = QPushButton()
            card.setCursor(Qt.CursorShape.PointingHandCursor)
            card.setFixedHeight(120)
            card.setStyleSheet(mode_card_style(accent))

            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(20, 14, 20, 14)
            card_layout.setSpacing(6)

            name_lbl = QLabel(test["name"])
            name_lbl.setStyleSheet(
                "background: transparent;"
                f" color: {Color.TEXT}; font-size: 18px; font-weight: 700;"
            )
            card_layout.addWidget(name_lbl)

            desc_lbl = QLabel(test["desc"])
            desc_lbl.setStyleSheet(
                "background: transparent;"
                f" color: {Color.TEXT_SECONDARY}; font-size: 14px;"
            )
            desc_lbl.setWordWrap(True)
            card_layout.addWidget(desc_lbl)

            card.clicked.connect(
                lambda _c=False, r=test["route"]: self._router.navigate(r)
            )
            root.addWidget(card)

        root.addStretch()

    def on_enter(self, **kwargs: Any) -> None:
        logger.debug("PerformanceMenuPage entered")

    def on_leave(self) -> None:
        pass
