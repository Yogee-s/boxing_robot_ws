"""Home page for logged-in users.

Premium dark dashboard with clean mode cards and welcome section.
"""
from __future__ import annotations

import logging
from typing import Any

from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QColor, QLinearGradient, QPainter, QPainterPath
from PySide6.QtWidgets import (
    QGridLayout, QHBoxLayout, QLabel, QPushButton,
    QVBoxLayout, QWidget,
)

from boxbunny_gui.theme import Color, Icon, Size

logger = logging.getLogger(__name__)

_H = f"color:{Color.PRIMARY_LIGHT}; font-weight:600"
_MODES = [
    {
        "name": "Techniques",
        "desc": f'Practice <span style="{_H}">punch combinations</span> with '
                f'<span style="{_H}">guided drills</span>',
        "accent": Color.PRIMARY,
        "route": "training_select",
    },
    {
        "name": "Sparring",
        "desc": f'<span style="{_H}">Fight</span> against the '
                f'<span style="{_H}">robot AI</span>',
        "accent": Color.DANGER,
        "route": "sparring_select",
    },
    {
        "name": "Free Training",
        "desc": f'<span style="{_H}">Open session</span>, no structure',
        "accent": Color.INFO,
        "route": "training_session",
    },
    {
        "name": "Performance",
        "desc": f'Test your <span style="{_H}">power</span>, '
                f'<span style="{_H}">stamina</span> and '
                f'<span style="{_H}">speed</span>',
        "accent": Color.PURPLE,
        "route": "performance",
    },
    {
        "name": "History",
        "desc": f'Past <span style="{_H}">sessions</span> and '
                f'<span style="{_H}">progress</span>',
        "accent": Color.WARNING,
        "route": "history",
    },
]


# ── Small avatar for the top bar ─────────────────────────────────────────

class _MiniAvatar(QWidget):
    """Small circular gradient avatar with person silhouette."""

    def __init__(self, size: int = 38, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(size, size)
        self._sz = size

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        s = self._sz

        clip = QPainterPath()
        clip.addEllipse(QRectF(0, 0, s, s))
        p.setClipPath(clip)

        grad = QLinearGradient(0, 0, 0, s)
        grad.setColorAt(0.0, QColor(Color.PRIMARY))
        grad.setColorAt(1.0, QColor(Color.PRIMARY_DARK))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(grad)
        p.drawEllipse(QRectF(0, 0, s, s))

        person = QColor(255, 255, 255, 200)
        p.setBrush(person)
        head_r = s * 0.16
        p.drawEllipse(QRectF(s / 2 - head_r, s * 0.24, head_r * 2, head_r * 2))
        p.drawEllipse(QRectF(s * 0.22, s * 0.56, s * 0.56, s * 0.44))
        p.end()


# ── Mode card ────────────────────────────────────────────────────────────

def _mode_card(mode: dict, height: int = 100) -> QPushButton:
    """Mode card with subtle accent tint."""
    accent = mode["accent"]
    btn = QPushButton()
    btn.setCursor(Qt.CursorShape.PointingHandCursor)
    btn.setFixedHeight(height)
    # Muted tint per accent
    _TINTS = {
        Color.PRIMARY: ("#1C1610", "#2E221A"),
        Color.DANGER:  ("#1C1214", "#2E1A1E"),
        Color.INFO:    ("#111820", "#1A2530"),
        Color.PURPLE:  ("#181420", "#221C30"),
        Color.WARNING: ("#1C1810", "#2E261A"),
    }
    bg_t, border_t = _TINTS.get(accent, ("#141920", "#1E2530"))
    btn.setStyleSheet(f"""
        QPushButton {{
            background-color: {bg_t};
            border: 1px solid {border_t};
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

    return btn


# ═══════════════════════════════════════════════════════════════════════════

class HomeIndividualPage(QWidget):
    """Main menu for authenticated users."""

    def __init__(self, router=None, **kwargs):
        super().__init__()
        self._router = router
        self._username = "Guest"

        root = QVBoxLayout(self)
        root.setContentsMargins(32, 14, 32, 10)
        root.setSpacing(0)

        # ── Top bar ──────────────────────────────────────────────────────
        top = QHBoxLayout()
        top.setSpacing(10)

        self._avatar = _MiniAvatar(size=38)
        top.addWidget(self._avatar)

        self._name_label = QLabel("Welcome back!")
        self._name_label.setStyleSheet(
            f"font-size: 20px; font-weight: 700; color: {Color.TEXT};"
        )
        top.addWidget(self._name_label)
        top.addStretch()

        settings_btn = QPushButton("Settings")
        settings_btn.setStyleSheet(f"""
            QPushButton {{
                font-size: 13px; font-weight: 600; padding: 6px 16px;
                background-color: {Color.SURFACE}; color: {Color.TEXT_SECONDARY};
                border: 1px solid {Color.BORDER_LIGHT}; border-radius: 8px;
            }}
            QPushButton:hover {{
                color: {Color.TEXT}; border-color: {Color.PRIMARY};
                background-color: {Color.SURFACE_LIGHT};
            }}
        """)
        settings_btn.setFixedHeight(44)
        settings_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        settings_btn.clicked.connect(lambda: self._nav("settings"))
        top.addWidget(settings_btn)

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
        close_btn.setFixedHeight(44)
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn.clicked.connect(lambda: self.window().close())
        top.addWidget(close_btn)
        root.addLayout(top)

        root.addSpacing(12)

        # ── Mode grid ────────────────────────────────────────────────────
        grid = QGridLayout()
        grid.setSpacing(10)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)

        for i, mode in enumerate(_MODES):
            is_last = i == len(_MODES) - 1
            btn = _mode_card(mode, height=60 if is_last else 100)
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
        logout.setFixedSize(100, 28)
        logout.setStyleSheet(f"""
            QPushButton {{
                font-size: 12px; font-weight: 600;
                background-color: transparent; color: {Color.TEXT_DISABLED};
                border: 1px solid {Color.BORDER}; border-radius: 8px;
            }}
            QPushButton:hover {{
                color: {Color.DANGER}; border-color: {Color.DANGER};
            }}
        """)
        logout.clicked.connect(lambda: self._nav("auth"))
        bottom.addWidget(logout)
        bottom.addStretch()
        root.addLayout(bottom)

    def _nav(self, page: str):
        if self._router:
            self._router.navigate(page, username=self._username)

    def on_enter(self, username: str = "Guest", **kwargs: Any):
        self._username = username
        self._name_label.setText(f"Welcome, {username}!")

    def on_leave(self) -> None:
        pass
