"""Performance test selection menu — warm-tinted cards with keywords."""
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

from boxbunny_gui.theme import Color, Icon, Size, font, back_link_style

if TYPE_CHECKING:
    from boxbunny_gui.nav.router import PageRouter

logger = logging.getLogger(__name__)

_KW = f"color:{Color.PRIMARY_LIGHT}; font-weight:600"
_TESTS = [
    {
        "name": "Power Test",
        "desc": f'Measure your <span style="{_KW}">punch force</span> '
                f'with <span style="{_KW}">10 max-effort</span> hits',
        "route": "power_test",
        "accent": Color.DANGER,
        "bg": "#1A1214", "border": "#3D1A22",
    },
    {
        "name": "Stamina Test",
        "desc": f'Throw as many <span style="{_KW}">punches</span> '
                f'as you can in <span style="{_KW}">2 minutes</span>',
        "route": "stamina_test",
        "accent": Color.PRIMARY,
        "bg": "#1A1510", "border": "#3D2E1A",
    },
    {
        "name": "Reaction Time",
        "desc": f'Punch when the screen <span style="{_KW}">flashes</span> '
                f'\u2014 <span style="{_KW}">3 trials</span>',
        "route": "reaction_test",
        "accent": Color.WARNING,
        "bg": "#1A1810", "border": "#3D351A",
    },
]


class PerformanceMenuPage(QWidget):
    def __init__(self, router: PageRouter, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._router = router
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(32, 14, 32, 22)
        root.setSpacing(0)

        # Top bar
        top = QHBoxLayout()
        btn_back = QPushButton(f"{Icon.BACK}  Back")
        btn_back.setStyleSheet(back_link_style())
        btn_back.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_back.clicked.connect(lambda: self._router.back())
        top.addWidget(btn_back)
        top.addStretch()
        root.addLayout(top)

        root.addStretch(1)

        # Title
        title = QLabel("Performance Tests")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(
            f"font-size: 36px; font-weight: 700; color: {Color.TEXT};"
        )
        root.addWidget(title)

        sub = QLabel("Select a test to measure your boxing performance")
        sub.setAlignment(Qt.AlignCenter)
        sub.setStyleSheet(f"font-size: 16px; color: {Color.TEXT_SECONDARY};")
        root.addWidget(sub)

        root.addSpacing(20)

        # 3 cards in a horizontal row — centered, compact
        cards_row = QHBoxLayout()
        cards_row.setSpacing(14)

        for test in _TESTS:
            accent = test["accent"]
            bg = test.get("bg", Color.SURFACE)
            border = test.get("border", Color.BORDER)

            card = QPushButton()
            card.setCursor(Qt.CursorShape.PointingHandCursor)
            card.setFixedHeight(120)
            card.setStyleSheet(f"""
                QPushButton {{
                    background-color: {bg};
                    border: 1px solid {border};
                    border-bottom: 3px solid {accent};
                    border-radius: {Size.RADIUS}px;
                }}
                QPushButton:hover {{
                    background-color: {Color.SURFACE_HOVER};
                    border: 1px solid {accent};
                    border-bottom: 3px solid {accent};
                }}
            """)

            lay = QVBoxLayout(card)
            lay.setContentsMargins(14, 16, 14, 16)
            lay.setSpacing(0)
            lay.setAlignment(Qt.AlignCenter)

            name_lbl = QLabel(test["name"])
            name_lbl.setAlignment(Qt.AlignCenter)
            name_lbl.setStyleSheet(
                "background: transparent; border: none;"
                f" font-size: 24px; font-weight: 700; color: {Color.TEXT};"
            )
            name_lbl.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            lay.addWidget(name_lbl)

            card.clicked.connect(
                lambda _c=False, r=test["route"]: self._router.navigate(r)
            )
            cards_row.addWidget(card)

        root.addLayout(cards_row)

        root.addStretch(2)

    def on_enter(self, **kwargs: Any) -> None:
        logger.debug("PerformanceMenuPage entered")

    def on_leave(self) -> None:
        pass
