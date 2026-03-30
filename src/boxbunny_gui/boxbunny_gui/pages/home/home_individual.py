"""Home page for logged-in users.

Premium dark dashboard with icon-enriched mode cards and welcome section.
"""
from __future__ import annotations

import logging
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton,
    QVBoxLayout, QWidget,
)

from boxbunny_gui.theme import (
    Color, Icon, Size, close_btn_style, top_bar_btn_style,
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


def _mode_card(mode: dict, height: int = 90) -> QPushButton:
    """Mode card with left accent bar, accent-colored title, and arrow."""
    accent = mode["accent"]
    btn = QPushButton()
    btn.setCursor(Qt.CursorShape.PointingHandCursor)
    btn.setFixedHeight(height)
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
    lay.setContentsMargins(20, 12, 18, 12)
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
        f" color: {Color.TEXT_SECONDARY};"
        " border: none;"
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


class HomeIndividualPage(QWidget):
    """Main menu for authenticated users — premium icon card grid."""

    def __init__(self, router=None, **kwargs):
        super().__init__()
        self._router = router

        root = QVBoxLayout(self)
        root.setContentsMargins(32, 16, 32, 12)
        root.setSpacing(8)

        # ── Top bar ──────────────────────────────────────────────────────
        top = QHBoxLayout()
        top.setSpacing(12)

        # User avatar + welcome
        self._avatar_lbl = QLabel("G")
        self._avatar_lbl.setFixedSize(40, 40)
        self._avatar_lbl.setAlignment(Qt.AlignCenter)
        self._avatar_lbl.setStyleSheet(f"""
            background-color: {Color.PRIMARY_MUTED};
            color: {Color.PRIMARY};
            font-size: 18px; font-weight: 700;
            border: 2px solid {Color.PRIMARY};
            border-radius: 20px;
        """)
        top.addWidget(self._avatar_lbl)

        self._name_label = QLabel("Welcome back!")
        self._name_label.setStyleSheet(
            f"font-size: 20px; font-weight: 700; color: {Color.TEXT};"
        )
        top.addWidget(self._name_label)
        top.addStretch()

        settings_btn = QPushButton("Settings")
        settings_btn.setStyleSheet(top_bar_btn_style())
        settings_btn.setFixedSize(80, 32)
        settings_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        settings_btn.clicked.connect(lambda: self._nav("settings"))
        top.addWidget(settings_btn)

        close_btn = QPushButton(f"{Icon.CLOSE}")
        close_btn.setStyleSheet(close_btn_style())
        close_btn.setFixedSize(44, 32)
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn.clicked.connect(lambda: self.window().close())
        top.addWidget(close_btn)
        root.addLayout(top)

        root.addSpacing(4)

        # ── Mode grid: 2+2+1 layout ─────────────────────────────────────
        grid = QGridLayout()
        grid.setSpacing(10)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)

        for i, mode in enumerate(_MODES):
            is_last = i == len(_MODES) - 1
            btn = _mode_card(mode, height=65 if is_last else 110)
            btn.clicked.connect(
                lambda _c=False, r=mode["route"]: self._nav(r)
            )
            if i < 4:
                grid.addWidget(btn, i // 2, i % 2)
            else:
                grid.addWidget(btn, 2, 0, 1, 2)

        root.addLayout(grid, stretch=1)

        root.addSpacing(4)

        # ── Bottom ───────────────────────────────────────────────────────
        bottom = QHBoxLayout()
        bottom.addStretch()

        logout = QPushButton("Log Out")
        logout.setCursor(Qt.CursorShape.PointingHandCursor)
        logout.setFixedSize(110, 30)
        logout.setStyleSheet(f"""
            QPushButton {{
                font-size: 12px; font-weight: 600;
                background-color: transparent; color: {Color.TEXT_DISABLED};
                border: 1px solid {Color.BORDER}; border-radius: 8px;
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
        # Update avatar initial
        initial = username[0].upper() if username else "G"
        self._avatar_lbl.setText(initial)

    def on_leave(self) -> None:
        pass
