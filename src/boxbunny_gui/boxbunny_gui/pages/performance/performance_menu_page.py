"""Performance test selection menu.

Three large cards: Power Test, Stamina Test, Reaction Time Test.
Each shows test name, description, and last score if available.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from boxbunny_gui.theme import Color, Size, font, GHOST_BTN
from boxbunny_gui.widgets import BigButton

if TYPE_CHECKING:
    from boxbunny_gui.nav.router import PageRouter

logger = logging.getLogger(__name__)

_TESTS = [
    {
        "name": "Power Test",
        "desc": "Measure your punch force with 10 max-effort hits",
        "route": "power_test",
        "last_score": None,
    },
    {
        "name": "Stamina Test",
        "desc": "Throw as many punches as you can in 2 minutes",
        "route": "stamina_test",
        "last_score": None,
    },
    {
        "name": "Reaction Time Test",
        "desc": "Punch when the screen flashes -- 10 trials",
        "route": "reaction_test",
        "last_score": None,
    },
]


class _TestCard(QFrame):
    """Large card for a performance test."""

    def __init__(self, test: dict, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.test = test
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(110)
        self.setStyleSheet(
            f"QFrame {{ background-color: {Color.SURFACE};"
            f" border-radius: {Size.RADIUS_LG}px; }}"
            f" QFrame:hover {{ background-color: {Color.SURFACE_HOVER}; }}"
        )
        lay = QVBoxLayout(self)
        lay.setContentsMargins(Size.SPACING, Size.SPACING, Size.SPACING, Size.SPACING)

        top = QHBoxLayout()
        name = QLabel(test["name"])
        name.setFont(font(22, bold=True))
        top.addWidget(name)
        top.addStretch()
        if test["last_score"] is not None:
            score = QLabel(f"Last: {test['last_score']}")
            score.setStyleSheet(f"color: {Color.PRIMARY}; font-size: 16px;")
            top.addWidget(score)
        lay.addLayout(top)

        desc = QLabel(test["desc"])
        desc.setStyleSheet(f"color: {Color.TEXT_SECONDARY}; font-size: 15px;")
        desc.setWordWrap(True)
        lay.addWidget(desc)


class PerformanceMenuPage(QWidget):
    """Selection screen for the three performance tests."""

    def __init__(self, router: PageRouter, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._router = router
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(Size.SPACING, Size.SPACING_SM, Size.SPACING, Size.SPACING_SM)
        root.setSpacing(Size.SPACING)

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

        # Test cards
        for test in _TESTS:
            card = _TestCard(test, self)
            card.mousePressEvent = lambda _e, r=test["route"]: self._router.navigate(r)
            root.addWidget(card)

        root.addStretch()

    # ── Lifecycle ──────────────────────────────────────────────────────
    def on_enter(self, **kwargs: Any) -> None:
        # TODO: load last scores from database
        logger.debug("PerformanceMenuPage entered")

    def on_leave(self) -> None:
        pass
