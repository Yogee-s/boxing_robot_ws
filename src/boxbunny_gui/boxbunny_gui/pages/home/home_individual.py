"""Home page for logged-in users.

Colorful grid layout with mode cards. Premium dark theme.
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
    Color, close_btn_style, top_bar_btn_style,
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
    {
        "name": "History",
        "desc": "Past sessions and progress",
        "accent": Color.WARNING,
        "route": "history",
    },
]


def _mode_card(mode: dict, height: int = 100) -> QPushButton:
    """Fixed-height mode card with centered content."""
    accent = mode["accent"]
    btn = QPushButton()
    btn.setCursor(Qt.CursorShape.PointingHandCursor)
    btn.setFixedHeight(height)
    btn.setStyleSheet(f"""
        QPushButton {{
            background-color: {Color.SURFACE};
            color: {Color.TEXT};
            border: 1px solid {Color.BORDER};
            border-left: 4px solid {accent};
            border-radius: 14px;
            padding: 0px;
        }}
        QPushButton:hover {{
            background-color: {Color.SURFACE_HOVER};
            border-color: {accent}50;
            border-left: 4px solid {accent};
        }}
        QPushButton:pressed {{
            background-color: {Color.SURFACE_LIGHT};
        }}
    """)

    lay = QVBoxLayout(btn)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(4)
    lay.setAlignment(Qt.AlignmentFlag.AlignCenter)

    title = QLabel(mode["name"])
    title.setAlignment(Qt.AlignmentFlag.AlignCenter)
    title.setStyleSheet(
        "background: transparent;"
        f" font-size: 20px; font-weight: 700; color: {Color.TEXT};"
    )
    title.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
    lay.addWidget(title)

    desc = QLabel(mode["desc"])
    desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
    desc.setStyleSheet(
        f"background: transparent; font-size: 13px;"
        f" color: {Color.TEXT_SECONDARY};"
    )
    desc.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
    lay.addWidget(desc)

    return btn


class HomeIndividualPage(QWidget):
    """Main menu for authenticated users — colorful card grid."""

    def __init__(self, router=None, **kwargs):
        super().__init__()
        self._router = router

        root = QVBoxLayout(self)
        root.setContentsMargins(40, 20, 40, 20)
        root.setSpacing(10)

        # ── Top bar ──────────────────────────────────────────────────────
        top = QHBoxLayout()
        top.setSpacing(12)
        self._name_label = QLabel("Welcome back!")
        self._name_label.setStyleSheet(
            f"font-size: 22px; font-weight: 700; color: {Color.TEXT};"
        )
        top.addWidget(self._name_label)
        top.addStretch()

        settings_btn = QPushButton("Settings")
        settings_btn.setStyleSheet(top_bar_btn_style())
        settings_btn.clicked.connect(lambda: self._nav("settings"))
        top.addWidget(settings_btn)

        top.addSpacing(8)

        close_btn = QPushButton("\u2715")
        close_btn.setFixedSize(36, 36)
        close_btn.setStyleSheet(close_btn_style())
        close_btn.clicked.connect(lambda: self.window().close())
        top.addWidget(close_btn)
        root.addLayout(top)

        root.addStretch()

        # ── Mode grid: 2x2 + 1 bottom ───────────────────────────────────
        grid = QGridLayout()
        grid.setSpacing(10)

        for i, mode in enumerate(_MODES):
            is_last = i == len(_MODES) - 1
            btn = _mode_card(mode, height=55 if is_last else 100)
            btn.clicked.connect(
                lambda _c=False, r=mode["route"]: self._nav(r)
            )
            if i < 4:
                grid.addWidget(btn, i // 2, i % 2)
            else:
                grid.addWidget(btn, 2, 0, 1, 2)

        root.addLayout(grid)

        root.addStretch()

        # ── Bottom ───────────────────────────────────────────────────────
        bottom = QHBoxLayout()
        bottom.addStretch()

        logout = QPushButton("Log Out")
        logout.setCursor(Qt.CursorShape.PointingHandCursor)
        logout.setFixedSize(110, 32)
        logout.setStyleSheet(f"""
            QPushButton {{
                font-size: 13px; font-weight: 600;
                background-color: transparent; color: {Color.TEXT_SECONDARY};
                border: 1px solid {Color.BORDER_LIGHT}; border-radius: 8px;
            }}
            QPushButton:hover {{
                color: {Color.DANGER}; border-color: {Color.DANGER};
            }}
            QPushButton:pressed {{
                background-color: {Color.DANGER}; color: white;
                border-color: {Color.DANGER};
            }}
        """)
        logout.clicked.connect(lambda: self._nav("auth"))
        bottom.addWidget(logout)
        bottom.addStretch()
        root.addLayout(bottom)

    def _nav(self, page: str):
        if self._router:
            self._router.navigate(page)

    def on_enter(self, username: str = "Guest", **kwargs: Any):
        self._username = username
        self._name_label.setText(f"Welcome, {username}!")

    def on_leave(self) -> None:
        pass
