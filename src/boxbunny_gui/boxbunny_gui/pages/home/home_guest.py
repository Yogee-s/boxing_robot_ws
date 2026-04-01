"""Guest home page — shown after skill assessment."""
from __future__ import annotations

import logging
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QPushButton,
    QVBoxLayout, QWidget,
)

from boxbunny_gui.theme import Color, Icon, Size, subtle_btn_style

logger = logging.getLogger(__name__)

_KW = f"color:{Color.PRIMARY_LIGHT}; font-weight:600"
_MODES = [
    {
        "name": "Techniques",
        "desc": f'Practice <span style="{_KW}">punch combinations</span> with '
                f'<span style="{_KW}">guided drills</span>',
        "accent": Color.PRIMARY,
        "tint": ("#1C1610", "#2E221A"),
        "route": "training_select",
    },
    {
        "name": "Sparring",
        "desc": f'<span style="{_KW}">Fight</span> against the '
                f'<span style="{_KW}">robot AI</span>',
        "accent": Color.DANGER,
        "tint": ("#1C1214", "#2E1A1E"),
        "route": "sparring_select",
    },
    {
        "name": "Free Training",
        "desc": f'<span style="{_KW}">Open session</span>, no structure',
        "accent": Color.INFO,
        "tint": ("#111820", "#1A2530"),
        "route": "training_session",
    },
    {
        "name": "Performance",
        "desc": f'Test your <span style="{_KW}">power</span>, '
                f'<span style="{_KW}">stamina</span> and '
                f'<span style="{_KW}">speed</span>',
        "accent": Color.PURPLE,
        "tint": ("#181420", "#221C30"),
        "route": "performance",
    },
]


def _mode_card(mode: dict) -> QPushButton:
    accent = mode["accent"]
    bg, border = mode.get("tint", (Color.SURFACE, Color.BORDER))
    btn = QPushButton()
    btn.setCursor(Qt.CursorShape.PointingHandCursor)
    btn.setFixedHeight(120)
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
    """)

    lay = QVBoxLayout(btn)
    lay.setContentsMargins(18, 16, 18, 16)
    lay.setSpacing(8)

    title = QLabel(mode["name"])
    title.setStyleSheet(
        "background: transparent; border: none;"
        f" font-size: 24px; font-weight: 700; color: {Color.TEXT};"
    )
    title.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
    lay.addWidget(title)

    lay.addStretch()

    desc = QLabel(mode["desc"])
    desc.setTextFormat(Qt.TextFormat.RichText)
    desc.setStyleSheet(
        "background: transparent; border: none;"
        f" font-size: 14px; color: {Color.TEXT_SECONDARY};"
    )
    desc.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
    lay.addWidget(desc)

    return btn


class HomeGuestPage(QWidget):
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
            f"font-size: 28px; font-weight: 700; color: {Color.PRIMARY};"
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

        # ── 2x2 card grid — tall cards, narrower ────────────────────────
        grid = QHBoxLayout()
        grid.setSpacing(12)

        col1 = QVBoxLayout()
        col1.setSpacing(12)
        col2 = QVBoxLayout()
        col2.setSpacing(12)

        for i, mode in enumerate(_MODES):
            btn = _mode_card(mode)
            btn.clicked.connect(
                lambda _c=False, r=mode["route"]: self._nav(r)
            )
            if i % 2 == 0:
                col1.addWidget(btn)
            else:
                col2.addWidget(btn)

        grid.addLayout(col1)
        grid.addLayout(col2)
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
