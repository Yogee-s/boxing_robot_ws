"""Performance test selection menu.

Three large premium cards: Power Test, Stamina Test, Reaction Time.
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

from boxbunny_gui.theme import (
    Color, Icon, Size, font, GHOST_BTN,
    mode_card_style_v2, back_link_style,
)
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
        "desc": "Punch when the screen flashes — 10 trials",
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
        root.setContentsMargins(32, Size.SPACING, 32, Size.SPACING)
        root.setSpacing(Size.SPACING_SM)

        # Back + title
        top = QHBoxLayout()
        btn_back = QPushButton(f"{Icon.BACK}  Back")
        btn_back.setStyleSheet(back_link_style())
        btn_back.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_back.clicked.connect(lambda: self._router.back())
        top.addWidget(btn_back)
        top.addStretch()
        title = QLabel("Performance Tests")
        title.setFont(font(Size.TEXT_SUBHEADER, bold=True))
        top.addWidget(title)
        top.addStretch()
        # Balance spacer
        spacer = QLabel()
        spacer.setFixedWidth(80)
        top.addWidget(spacer)
        root.addLayout(top)

        subtitle = QLabel("Select a test to measure your boxing performance")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet(
            f"color: {Color.TEXT_SECONDARY}; font-size: 14px;"
        )
        root.addWidget(subtitle)
        root.addSpacing(8)

        # Test cards as QPushButtons with icon layout
        for test in _TESTS:
            accent = test["accent"]
            card = QPushButton()
            card.setCursor(Qt.CursorShape.PointingHandCursor)
            card.setFixedHeight(120)
            card.setStyleSheet(mode_card_style_v2(accent))

            card_layout = QHBoxLayout(card)
            card_layout.setContentsMargins(20, 14, 20, 14)
            card_layout.setSpacing(16)

            # Text column
            text_col = QVBoxLayout()
            text_col.setSpacing(4)
            text_col.setContentsMargins(0, 0, 0, 0)

            name_lbl = QLabel(test["name"])
            name_lbl.setStyleSheet(
                "background: transparent;"
                f" color: {Color.TEXT}; font-size: 18px; font-weight: 700;"
                " border: none;"
            )
            name_lbl.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            text_col.addWidget(name_lbl)

            desc_lbl = QLabel(test["desc"])
            desc_lbl.setStyleSheet(
                "background: transparent;"
                f" color: {Color.TEXT_SECONDARY}; font-size: 13px;"
                " border: none;"
            )
            desc_lbl.setWordWrap(True)
            desc_lbl.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            text_col.addWidget(desc_lbl)

            card_layout.addLayout(text_col, stretch=1)

            # Arrow
            arrow = QLabel("→")
            arrow.setStyleSheet(
                f"color: {Color.TEXT_DISABLED}; font-size: 20px;"
                " background: transparent; border: none;"
            )
            arrow.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            card_layout.addWidget(arrow)

            card.clicked.connect(
                lambda _c=False, r=test["route"]: self._router.navigate(r)
            )
            root.addWidget(card)

        root.addStretch()

    def on_enter(self, **kwargs: Any) -> None:
        logger.debug("PerformanceMenuPage entered")

    def on_leave(self) -> None:
        pass
