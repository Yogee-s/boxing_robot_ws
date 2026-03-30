"""Guest home page — shown after skill assessment.

Clean mode selection with accent-colored cards.
"""
from __future__ import annotations

import logging
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGridLayout, QHBoxLayout, QLabel, QPushButton,
    QVBoxLayout, QWidget,
)

from boxbunny_gui.theme import (
    Color, Icon, Size, close_btn_style, subtle_btn_style,
)

logger = logging.getLogger(__name__)

_MODES = [
    {
        "name": "Training",
        "desc": "Practice combos with guided drills",
        "accent": Color.PRIMARY,
        "route": "training_select",
    },
    {
        "name": "Sparring",
        "desc": "Fight against the robot AI",
        "accent": Color.DANGER,
        "route": "sparring_select",
    },
    {
        "name": "Free Training",
        "desc": "Open session, no structure",
        "accent": Color.INFO,
        "route": "training_session",
    },
    {
        "name": "Performance",
        "desc": "Test your power, stamina and speed",
        "accent": Color.PURPLE,
        "route": "performance",
    },
]


def _mode_card(mode: dict) -> QPushButton:
    """Mode card with left accent bar, accent-colored title, and arrow."""
    accent = mode["accent"]
    btn = QPushButton()
    btn.setCursor(Qt.CursorShape.PointingHandCursor)
    btn.setFixedHeight(120)
    btn.setStyleSheet(f"""
        QPushButton {{
            background-color: {Color.SURFACE};
            border: 1px solid {Color.BORDER};
            border-left: 4px solid {accent};
            border-radius: {Size.RADIUS}px;
            text-align: left;
        }}
        QPushButton:hover {{
            background-color: {Color.SURFACE_HOVER};
            border-color: {accent}40;
            border-left: 4px solid {accent};
        }}
        QPushButton:pressed {{
            background-color: {Color.SURFACE_LIGHT};
            border-left: 4px solid {accent};
        }}
    """)

    lay = QHBoxLayout(btn)
    lay.setContentsMargins(20, 14, 18, 14)
    lay.setSpacing(14)

    # Text column
    text_col = QVBoxLayout()
    text_col.setSpacing(4)
    text_col.setContentsMargins(0, 0, 0, 0)

    title = QLabel(mode["name"])
    title.setStyleSheet(
        "background: transparent;"
        f" font-size: 18px; font-weight: 700; color: {accent};"
        " border: none;"
    )
    title.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
    text_col.addWidget(title)

    desc = QLabel(mode["desc"])
    desc.setStyleSheet(
        f"background: transparent; font-size: 13px;"
        f" color: {Color.TEXT_SECONDARY}; border: none;"
    )
    desc.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
    text_col.addWidget(desc)

    lay.addLayout(text_col, stretch=1)

    # Arrow in accent color
    arrow = QLabel(Icon.NEXT)
    arrow.setStyleSheet(
        f"color: {accent}60; font-size: 20px;"
        " background: transparent; border: none;"
    )
    arrow.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
    lay.addWidget(arrow)

    return btn


class HomeGuestPage(QWidget):
    """Menu for guest users — 2x2 card grid with welcome header."""

    def __init__(self, router=None, **kwargs):
        super().__init__()
        self._router = router

        root = QVBoxLayout(self)
        root.setContentsMargins(40, 12, 40, 12)
        root.setSpacing(0)

        # ── Top bar (pinned to top) ─────────────────────────────────────
        top = QHBoxLayout()
        top.setSpacing(10)

        title = QLabel("Welcome")
        title.setStyleSheet(
            f"font-size: 24px; font-weight: 700; color: {Color.TEXT};"
        )
        top.addWidget(title)
        top.addStretch()

        close_btn = QPushButton(f"{Icon.CLOSE}")
        close_btn.setStyleSheet(close_btn_style())
        close_btn.setFixedSize(44, 32)
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn.clicked.connect(lambda: self.window().close())
        top.addWidget(close_btn)
        root.addLayout(top)

        sub = QLabel("Choose a mode to get started")
        sub.setStyleSheet(
            f"font-size: 13px; color: {Color.TEXT_SECONDARY};"
        )
        root.addWidget(sub)

        # Center the grid vertically
        root.addStretch(1)

        # ── 2x2 Mode grid ───────────────────────────────────────────────
        grid = QGridLayout()
        grid.setSpacing(12)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)

        for i, mode in enumerate(_MODES):
            btn = _mode_card(mode)
            btn.clicked.connect(
                lambda _c=False, r=mode["route"]: self._nav(r)
            )
            grid.addWidget(btn, i // 2, i % 2)

        root.addLayout(grid)

        root.addStretch(1)

        # ── Bottom ───────────────────────────────────────────────────────
        bottom = QHBoxLayout()
        bottom.addStretch()

        back = QPushButton(f"{Icon.BACK}  Back")
        back.setCursor(Qt.CursorShape.PointingHandCursor)
        back.setFixedSize(100, 30)
        back.setStyleSheet(subtle_btn_style())
        back.clicked.connect(lambda: self._nav("auth"))
        bottom.addWidget(back)
        bottom.addStretch()
        root.addLayout(bottom)

    def _nav(self, page: str):
        if self._router:
            self._router.navigate(page)

    def on_enter(self, **kwargs: Any) -> None:
        pass

    def on_leave(self) -> None:
        pass
