"""Guest home page — shown after skill assessment.

Colorful mode cards with centered text. Premium, aesthetic design.
"""
from __future__ import annotations

import logging
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGridLayout, QHBoxLayout, QLabel, QPushButton,
    QVBoxLayout, QWidget,
)

from boxbunny_gui.theme import Color, badge_style, close_btn_style

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
    """Fixed-height mode card with centered content."""
    accent = mode["accent"]
    btn = QPushButton()
    btn.setCursor(Qt.CursorShape.PointingHandCursor)
    btn.setFixedHeight(120)
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
        f" font-size: 22px; font-weight: 700; color: {Color.TEXT};"
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


class HomeGuestPage(QWidget):
    """Menu for guest (unauthenticated) users — 2x2 colorful grid."""

    def __init__(self, router=None, **kwargs):
        super().__init__()
        self._router = router

        root = QVBoxLayout(self)
        root.setContentsMargins(40, 20, 40, 20)
        root.setSpacing(12)

        # ── Top bar ──────────────────────────────────────────────────────
        top = QHBoxLayout()
        top.setSpacing(10)
        title = QLabel("Guest Mode")
        title.setStyleSheet(
            f"font-size: 22px; font-weight: 700; color: {Color.TEXT};"
        )
        top.addWidget(title)

        badge = QLabel("No account")
        badge.setStyleSheet(badge_style())
        top.addWidget(badge)
        top.addStretch()

        close_btn = QPushButton("\u2715")
        close_btn.setFixedSize(36, 36)
        close_btn.setStyleSheet(close_btn_style())
        close_btn.clicked.connect(lambda: self.window().close())
        top.addWidget(close_btn)
        root.addLayout(top)

        root.addStretch()

        # ── 2x2 Mode grid — centered with fixed card heights ────────────
        grid = QGridLayout()
        grid.setSpacing(12)
        for i, mode in enumerate(_MODES):
            btn = _mode_card(mode)
            btn.clicked.connect(
                lambda _c=False, r=mode["route"]: self._nav(r)
            )
            grid.addWidget(btn, i // 2, i % 2)

        root.addLayout(grid)

        root.addStretch()

        # ── Bottom ───────────────────────────────────────────────────────
        bottom = QHBoxLayout()
        bottom.addStretch()

        back = QPushButton("\u2190  Back to Start")
        back.setCursor(Qt.CursorShape.PointingHandCursor)
        back.setFixedSize(160, 34)
        back.setStyleSheet(f"""
            QPushButton {{
                font-size: 13px; font-weight: 600;
                background-color: transparent; color: {Color.TEXT_SECONDARY};
                border: 1px solid {Color.BORDER_LIGHT}; border-radius: 8px;
            }}
            QPushButton:hover {{
                color: {Color.TEXT}; border-color: {Color.PRIMARY};
            }}
            QPushButton:pressed {{
                background-color: {Color.SURFACE};
            }}
        """)
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
