"""Guest home page — shown after skill assessment."""
from __future__ import annotations

import logging
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGridLayout, QHBoxLayout, QLabel, QPushButton,
    QVBoxLayout, QWidget,
)

from boxbunny_gui.theme import (
    Color, Icon, Size,
    top_bar_btn_style, close_btn_style,
)
from boxbunny_gui.widgets import HoldTooltipCard

logger = logging.getLogger(__name__)

_KW = f"color:{Color.PRIMARY_LIGHT}; font-weight:600"
_MODES = [
    {
        "name": "Techniques",
        "tag": "Combo Drills",
        "desc": f'Practice <span style="{_KW}">punch combinations</span> with '
                f'<span style="{_KW}">guided drills</span>',
        "accent": Color.PRIMARY,
        "route": "training_select",
    },
    {
        "name": "Sparring",
        "tag": "vs Robot AI",
        "desc": f'<span style="{_KW}">Fight</span> against the '
                f'<span style="{_KW}">robot AI</span>',
        "accent": Color.DANGER,
        "route": "sparring_select",
    },
    {
        "name": "Free Training",
        "tag": "Open Session",
        "desc": f'<span style="{_KW}">Open session</span>, no structure',
        "accent": Color.INFO,
        "route": "training_session",
    },
    {
        "name": "Performance",
        "tag": "Power / Speed",
        "desc": f'Test your <span style="{_KW}">power</span>, '
                f'<span style="{_KW}">stamina</span> and '
                f'<span style="{_KW}">speed</span>',
        "accent": Color.PURPLE,
        "route": "performance",
    },
]


# Richer tinted backgrounds per mode — visible against dark BG
_CARD_TINTS = {
    Color.PRIMARY: ("#231810", "#2E2014"),   # warm orange
    Color.DANGER:  ("#231418", "#2E1A20"),   # red
    Color.INFO:    ("#101E2E", "#162838"),   # blue
    Color.PURPLE:  ("#1C1430", "#241C3A"),   # purple
}


def _mode_card(mode: dict) -> HoldTooltipCard:
    """Tinted mode card — each card has a unique colour."""
    accent = mode["accent"]
    bg, bg_hover = _CARD_TINTS.get(accent, (Color.SURFACE, Color.SURFACE_LIGHT))

    btn = HoldTooltipCard(desc_html=mode["desc"])
    btn.setCursor(Qt.CursorShape.PointingHandCursor)
    btn.setFixedHeight(120)
    btn.setStyleSheet(f"""
        QPushButton {{
            background-color: {bg};
            border: 1px solid {Color.BORDER};
            border-bottom: 3px solid {accent};
            border-radius: {Size.RADIUS}px;
        }}
        QPushButton:hover {{
            background-color: {bg_hover};
            border: 1px solid {accent};
            border-bottom: 3px solid {accent};
        }}
    """)

    lay = QVBoxLayout(btn)
    lay.setContentsMargins(12, 8, 12, 8)
    lay.setSpacing(4)
    lay.setAlignment(Qt.AlignCenter)

    title = QLabel(mode["name"])
    title.setAlignment(Qt.AlignCenter)
    title.setStyleSheet(
        "background: transparent; border: none;"
        f" font-size: 26px; font-weight: 700; color: {Color.TEXT};"
    )
    title.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
    lay.addWidget(title)

    tag = QLabel(mode["tag"])
    tag.setAlignment(Qt.AlignCenter)
    tag.setStyleSheet(
        "background: transparent; border: none;"
        f" font-size: 14px; font-weight: 600; color: {Color.TEXT_SECONDARY};"
        " letter-spacing: 0.5px;"
    )
    tag.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
    lay.addWidget(tag)

    return btn


class HomeGuestPage(QWidget):
    def __init__(self, router=None, preset_overlay=None, **kwargs):
        super().__init__()
        self._router = router
        self._preset_overlay = preset_overlay
        self._level = "Beginner"
        self._guest_history: list = []

        root = QVBoxLayout(self)
        root.setContentsMargins(32, 18, 32, 18)
        root.setSpacing(0)

        # ── Top bar ──────────────────────────────────────────────────────
        top = QHBoxLayout()
        top.setSpacing(10)

        self._title = QLabel("Welcome, Guest")
        self._title.setStyleSheet(
            f"font-size: 28px; font-weight: 700; color: {Color.PRIMARY};"
        )
        top.addWidget(self._title)
        top.addStretch()

        quick_btn = QPushButton("Quick Start")
        quick_btn.setStyleSheet(f"""
            QPushButton {{
                font-size: 13px; font-weight: 700;
                background-color: {Color.PRIMARY}; color: #FFFFFF;
                border: none; border-radius: 8px;
                padding: 0 16px;
            }}
            QPushButton:hover {{ background-color: {Color.PRIMARY_DARK}; }}
        """)
        quick_btn.setFixedHeight(44)
        quick_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        quick_btn.clicked.connect(self._open_presets)
        top.addWidget(quick_btn)

        settings_btn = QPushButton("Settings")
        settings_btn.setStyleSheet(top_bar_btn_style())
        settings_btn.setFixedHeight(44)
        settings_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        settings_btn.clicked.connect(lambda: self._nav("settings"))
        top.addWidget(settings_btn)

        close_btn = QPushButton("Close")
        close_btn.setStyleSheet(close_btn_style())
        close_btn.setFixedHeight(44)
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn.clicked.connect(lambda: self.window().close())
        top.addWidget(close_btn)
        root.addLayout(top)

        # ── Section label ────────────────────────────────────────────────
        root.addSpacing(4)
        section_row = QHBoxLayout()
        section_lbl = QLabel("SELECT MODE")
        section_lbl.setStyleSheet(
            f"font-size: 11px; font-weight: 700; color: {Color.TEXT_DISABLED};"
            " letter-spacing: 2px;"
        )
        section_row.addWidget(section_lbl)

        divider = QWidget()
        divider.setFixedHeight(1)
        divider.setStyleSheet(f"background-color: {Color.BORDER};")
        section_row.addWidget(divider, stretch=1)
        root.addLayout(section_row)

        # Push cards to vertical centre
        root.addStretch(1)

        # ── Card grid (2x2) — with side margins to keep cards from edges
        grid = QGridLayout()
        grid.setContentsMargins(80, 0, 80, 0)
        grid.setSpacing(14)
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
        bottom.setSpacing(12)

        back = QPushButton(f"{Icon.BACK}  Back")
        back.setCursor(Qt.CursorShape.PointingHandCursor)
        back.setFixedSize(120, 44)
        back.setStyleSheet(f"""
            QPushButton {{
                font-size: 13px; font-weight: 600;
                background-color: {Color.SURFACE}; color: {Color.TEXT_SECONDARY};
                border: 1px solid {Color.BORDER_LIGHT}; border-radius: 8px;
            }}
            QPushButton:hover {{
                color: {Color.TEXT}; border-color: {Color.PRIMARY};
                background-color: {Color.SURFACE_LIGHT};
            }}
        """)
        back.clicked.connect(lambda: self._nav("auth"))
        bottom.addWidget(back)

        bottom.addStretch()

        hint = QLabel("Hold card for details")
        hint.setStyleSheet(
            f"font-size: 16px; font-weight: 600; color: {Color.PRIMARY_LIGHT};"
        )
        bottom.addWidget(hint)

        bottom.addStretch()

        history_btn = QPushButton("History")
        history_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        history_btn.setFixedSize(160, 48)
        history_btn.setStyleSheet(f"""
            QPushButton {{
                font-size: 15px; font-weight: 700;
                background-color: transparent; color: {Color.PRIMARY_LIGHT};
                border: 2px solid {Color.PRIMARY_LIGHT}; border-radius: 10px;
            }}
            QPushButton:hover {{
                color: #FFFFFF; border-color: {Color.PRIMARY};
                background-color: {Color.PRIMARY};
            }}
        """)
        history_btn.clicked.connect(lambda: self._nav("history"))
        bottom.addWidget(history_btn)

        root.addLayout(bottom)

    def _open_presets(self):
        if self._preset_overlay:
            self._preset_overlay.toggle()

    def _nav(self, page: str):
        if self._router:
            self._router.navigate(
                page,
                user_level=self._level,
                guest_history=self._guest_history,
            )

    def on_enter(self, **kwargs: Any) -> None:
        level = kwargs.get("level", "")
        if level:
            level_title = level.title() if level.lower() in (
                "beginner", "intermediate", "advanced"
            ) else level
            self._level = level_title
        logger.info("HomeGuestPage entered (level=%s)", self._level)

    def on_leave(self) -> None:
        pass
