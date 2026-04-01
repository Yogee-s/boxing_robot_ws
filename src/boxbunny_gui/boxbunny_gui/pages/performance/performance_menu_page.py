"""Performance test selection menu — big centred cards with hold-for-description."""
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
from boxbunny_gui.widgets import HoldTooltipCard

if TYPE_CHECKING:
    from boxbunny_gui.nav.router import PageRouter

logger = logging.getLogger(__name__)

_KW = f"color:{Color.PRIMARY_LIGHT}; font-weight:600"
_TESTS = [
    {
        "name": "Power",
        "tag": "Max Force",
        "desc": f'Measure your <span style="{_KW}">punch force</span> '
                f'with <span style="{_KW}">10 max-effort</span> hits',
        "route": "power_test",
        "accent": Color.DANGER,
        "bg": "#1A1214",
    },
    {
        "name": "Stamina",
        "tag": "Endurance",
        "desc": f'Throw as many <span style="{_KW}">punches</span> '
                f'as you can in <span style="{_KW}">2 minutes</span>',
        "route": "stamina_test",
        "accent": Color.PRIMARY,
        "bg": "#1A1510",
    },
    {
        "name": "Reaction",
        "tag": "Speed",
        "desc": f'Punch when the screen <span style="{_KW}">flashes</span> '
                f'\u2014 <span style="{_KW}">3 trials</span>',
        "route": "reaction_test",
        "accent": Color.WARNING,
        "bg": "#1A1810",
    },
]


class PerformanceMenuPage(QWidget):
    def __init__(self, router: PageRouter, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._router = router
        self._username: str = ""
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
        btn_back.clicked.connect(self._go_back)
        top.addWidget(btn_back)
        bar_title = QLabel("Performance Tests")
        bar_title.setStyleSheet(
            f"font-size: 20px; font-weight: 700; color: {Color.TEXT};"
        )
        top.addWidget(bar_title)
        top.addStretch()
        root.addLayout(top)

        # Centre everything vertically
        root.addStretch(1)

        # Subtitle
        sub = QLabel("Select a test to measure your boxing performance")
        sub.setAlignment(Qt.AlignCenter)
        sub.setStyleSheet(f"font-size: 15px; color: {Color.TEXT_SECONDARY};")
        root.addWidget(sub)

        root.addSpacing(24)

        # 3 big cards in a horizontal row
        cards_row = QHBoxLayout()
        cards_row.setContentsMargins(40, 0, 40, 0)
        cards_row.setSpacing(18)

        for test in _TESTS:
            accent = test["accent"]
            bg = test["bg"]

            card = HoldTooltipCard(desc_html=test["desc"])
            card.setCursor(Qt.CursorShape.PointingHandCursor)
            card.setFixedHeight(180)
            card.setStyleSheet(f"""
                QPushButton {{
                    background-color: {bg};
                    border: 1px solid {Color.BORDER};
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
            lay.setContentsMargins(14, 14, 14, 14)
            lay.setSpacing(6)
            lay.setAlignment(Qt.AlignCenter)

            name_lbl = QLabel(test["name"])
            name_lbl.setAlignment(Qt.AlignCenter)
            name_lbl.setStyleSheet(
                "background: transparent; border: none;"
                f" font-size: 28px; font-weight: 700; color: {Color.TEXT};"
            )
            name_lbl.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            lay.addWidget(name_lbl)

            tag_lbl = QLabel(test["tag"])
            tag_lbl.setAlignment(Qt.AlignCenter)
            tag_lbl.setStyleSheet(
                "background: transparent; border: none;"
                f" font-size: 14px; font-weight: 600; color: {Color.TEXT_SECONDARY};"
                " letter-spacing: 0.5px;"
            )
            tag_lbl.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            lay.addWidget(tag_lbl)

            card.clicked.connect(
                lambda _c=False, r=test["route"]: self._router.navigate(r)
            )
            cards_row.addWidget(card)

        root.addLayout(cards_row)

        root.addSpacing(12)

        # Hint
        hint = QLabel("Hold card for details")
        hint.setAlignment(Qt.AlignCenter)
        hint.setStyleSheet(
            f"font-size: 16px; font-weight: 600; color: {Color.PRIMARY_LIGHT};"
        )
        root.addWidget(hint)

        root.addStretch(1)

    def _go_back(self) -> None:
        if self._username:
            self._router.navigate("home", username=self._username)
        else:
            self._router.navigate("home_guest")

    def on_enter(self, **kwargs: Any) -> None:
        self._username = kwargs.get("username", "")
        logger.debug("PerformanceMenuPage entered")

    def on_leave(self) -> None:
        pass
