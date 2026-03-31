"""Guest home page — shown after skill assessment.

Matches the individual home page styling with warm-tinted cards.
"""
from __future__ import annotations

import logging
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGridLayout, QHBoxLayout, QLabel, QPushButton,
    QVBoxLayout, QWidget,
)

from boxbunny_gui.theme import Color, Icon, Size, subtle_btn_style

logger = logging.getLogger(__name__)

_H = f"color:{Color.PRIMARY_LIGHT}; font-weight:600"
_MODES = [
    {
        "name": "Techniques",
        "desc": f'Practice <span style="{_H}">punch combinations</span> with '
                f'<span style="{_H}">guided drills</span>',
        "accent": Color.PRIMARY,
        "bg": "#1A1510", "border": "#3D2E1A",
        "route": "training_select",
    },
    {
        "name": "Sparring",
        "desc": f'<span style="{_H}">Fight</span> against the '
                f'<span style="{_H}">robot AI</span>',
        "accent": Color.DANGER,
        "bg": "#1A1214", "border": "#3D1A22",
        "route": "sparring_select",
    },
    {
        "name": "Free Training",
        "desc": f'<span style="{_H}">Open session</span>, no structure',
        "accent": Color.INFO,
        "bg": "#101820", "border": "#1A2E40",
        "route": "training_session",
    },
    {
        "name": "Performance",
        "desc": f'Test your <span style="{_H}">power</span>, '
                f'<span style="{_H}">stamina</span> and '
                f'<span style="{_H}">speed</span>',
        "accent": Color.PURPLE,
        "bg": "#16101E", "border": "#2A1A3D",
        "route": "performance",
    },
]


def _mode_card(mode: dict) -> QPushButton:
    accent = mode["accent"]
    bg = mode.get("bg", Color.SURFACE)
    border = mode.get("border", Color.BORDER)
    btn = QPushButton()
    btn.setCursor(Qt.CursorShape.PointingHandCursor)
    btn.setFixedHeight(110)
    btn.setStyleSheet(f"""
        QPushButton {{
            background-color: {bg};
            border: 1px solid {border};
            border-left: 3px solid {accent};
            border-radius: {Size.RADIUS}px;
            text-align: left;
        }}
        QPushButton:hover {{
            background-color: {Color.SURFACE_HOVER};
            border: 1px solid {accent};
            border-left: 3px solid {accent};
        }}
        QPushButton:pressed {{
            background-color: {accent};
            border-color: {accent};
            border-left: 3px solid {accent};
        }}
    """)

    lay = QHBoxLayout(btn)
    lay.setContentsMargins(18, 14, 16, 14)
    lay.setSpacing(0)

    text_col = QVBoxLayout()
    text_col.setSpacing(4)
    text_col.setContentsMargins(0, 0, 0, 0)

    title = QLabel(mode["name"])
    title.setStyleSheet(
        "background: transparent; border: none;"
        f" font-size: 17px; font-weight: 700; color: {Color.TEXT};"
    )
    title.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
    text_col.addWidget(title)

    desc = QLabel(mode["desc"])
    desc.setTextFormat(Qt.TextFormat.RichText)
    desc.setStyleSheet(
        "background: transparent; border: none;"
        f" font-size: 12px; color: {Color.TEXT_SECONDARY};"
    )
    desc.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
    text_col.addWidget(desc)

    lay.addLayout(text_col, stretch=1)

    arrow = QLabel(Icon.NEXT)
    arrow.setStyleSheet(
        f"color: {accent}; font-size: 16px;"
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
        root.setContentsMargins(32, 14, 32, 10)
        root.setSpacing(0)

        # ── Top bar ──────────────────────────────────────────────────────
        top = QHBoxLayout()
        top.setSpacing(10)

        title = QLabel("Welcome")
        title.setStyleSheet(
            f"font-size: 22px; font-weight: 700; color: {Color.TEXT};"
        )
        top.addWidget(title)
        top.addStretch()

        close_btn = QPushButton("Close")
        close_btn.setStyleSheet(f"""
            QPushButton {{
                font-size: 13px; font-weight: 600; padding: 6px 14px;
                background-color: {Color.SURFACE}; color: {Color.TEXT_SECONDARY};
                border: 1px solid {Color.BORDER_LIGHT}; border-radius: 8px;
            }}
            QPushButton:hover {{
                background-color: {Color.DANGER}; color: white;
                border-color: {Color.DANGER};
            }}
        """)
        close_btn.setFixedHeight(32)
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn.clicked.connect(lambda: self.window().close())
        top.addWidget(close_btn)
        root.addLayout(top)

        sub = QLabel("Choose a mode to get started")
        sub.setStyleSheet(f"font-size: 13px; color: {Color.TEXT_SECONDARY};")
        root.addWidget(sub)

        root.addStretch(1)

        # ── 2x2 Mode grid ───────────────────────────────────────────────
        grid = QGridLayout()
        grid.setSpacing(10)
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
