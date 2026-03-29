"""Combo browser with difficulty tabs and mastery progress.

Scrollable list of combo cards with coloured punch-sequence indicators.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from boxbunny_gui.theme import Color, Size, font, GHOST_BTN, tab_btn_style, back_link_style
from boxbunny_gui.widgets import BigButton

if TYPE_CHECKING:
    from boxbunny_gui.nav.router import PageRouter

logger = logging.getLogger(__name__)

# Punch colour mapping using theme colors
_PUNCH_COLORS = {1: Color.JAB, 2: Color.CROSS, 3: Color.L_HOOK, 4: Color.L_UPPERCUT}
_PUNCH_NAMES = {1: "Jab", 2: "Cross", 3: "Hook", 4: "Upper"}

_COMBOS: List[Dict[str, Any]] = [
    {"id": "jc", "name": "Jab-Cross", "seq": [1, 2], "diff": "beginner", "mastery": 65},
    {"id": "jch", "name": "Jab-Cross-Hook", "seq": [1, 2, 3], "diff": "beginner", "mastery": 30},
    {"id": "jchc", "name": "Jab-Cross-Hook-Cross", "seq": [1, 2, 3, 2], "diff": "intermediate", "mastery": 10},
    {"id": "djc", "name": "Double Jab-Cross", "seq": [1, 1, 2], "diff": "intermediate", "mastery": 0},
    {"id": "full", "name": "Full Combo", "seq": [1, 2, 3, 4, 2], "diff": "advanced", "mastery": 0},
]

_TABS = ["All", "Beginner", "Intermediate", "Advanced"]


class _ComboCard(QPushButton):
    """Single combo card with color-coded punch dots and mastery bar."""

    def __init__(self, combo: Dict[str, Any], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.combo = combo
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(80)
        self.setText("")  # clear default text, we use child widgets

        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {Color.SURFACE};
                color: {Color.TEXT};
                border: 1px solid {Color.BORDER};
                border-radius: 14px;
                padding: 0px;
            }}
            QPushButton:hover {{
                background-color: {Color.SURFACE_HOVER};
                border-color: {Color.PRIMARY}40;
            }}
            QPushButton:pressed {{
                background-color: {Color.SURFACE_LIGHT};
            }}
        """)

        # Card layout with child labels
        lay = QHBoxLayout(self)
        lay.setContentsMargins(18, 12, 18, 12)
        lay.setSpacing(14)

        # Left: name + difficulty badge
        left_col = QVBoxLayout()
        left_col.setSpacing(4)
        name_lbl = QLabel(combo["name"])
        name_lbl.setStyleSheet(
            f"background: transparent; color: {Color.TEXT};"
            " font-size: 16px; font-weight: 600;"
        )
        left_col.addWidget(name_lbl)

        diff_lbl = QLabel(combo["diff"].title())
        diff_color = {
            "beginner": Color.PRIMARY,
            "intermediate": Color.WARNING,
            "advanced": Color.DANGER,
        }.get(combo["diff"], Color.TEXT_SECONDARY)
        diff_lbl.setStyleSheet(
            f"background: transparent; color: {diff_color};"
            " font-size: 12px; font-weight: 600;"
        )
        left_col.addWidget(diff_lbl)
        lay.addLayout(left_col)

        lay.addStretch()

        # Center: color-coded punch dots
        dots_row = QHBoxLayout()
        dots_row.setSpacing(6)
        for punch_id in combo["seq"]:
            dot = QLabel("\u25CF")
            dot.setStyleSheet(
                f"background: transparent;"
                f" color: {_PUNCH_COLORS.get(punch_id, Color.TEXT_DISABLED)};"
                " font-size: 18px;"
            )
            dot.setAlignment(Qt.AlignmentFlag.AlignCenter)
            dots_row.addWidget(dot)
        lay.addLayout(dots_row)

        lay.addStretch()

        # Right: mastery percentage
        pct = combo["mastery"]
        pct_color = Color.PRIMARY if pct >= 50 else (
            Color.WARNING if pct > 0 else Color.TEXT_DISABLED
        )
        pct_lbl = QLabel(f"{pct}%")
        pct_lbl.setStyleSheet(
            f"background: transparent; color: {pct_color};"
            " font-size: 20px; font-weight: 700;"
        )
        pct_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        pct_lbl.setFixedWidth(56)
        lay.addWidget(pct_lbl)


class ComboSelectPage(QWidget):
    """Browse and select a combo to train."""

    def __init__(self, router: PageRouter, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._router = router
        self._active_tab: str = "All"
        self._cards: list[_ComboCard] = []
        self._tab_btns: list[QPushButton] = []
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 16, 24, 16)
        root.setSpacing(14)

        # Back link + title row
        top = QHBoxLayout()
        btn_back = QPushButton("\u2190  Back")
        btn_back.setStyleSheet(back_link_style())
        btn_back.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_back.clicked.connect(lambda: self._router.back())
        top.addWidget(btn_back)
        top.addStretch()
        title = QLabel("Select Combo")
        title.setFont(font(Size.TEXT_SUBHEADER, bold=True))
        top.addWidget(title)
        top.addStretch()
        # Spacer to balance the back button
        spacer = QLabel()
        spacer.setFixedWidth(80)
        top.addWidget(spacer)
        root.addLayout(top)

        # Difficulty filter tabs
        tabs = QHBoxLayout()
        tabs.setSpacing(8)
        for tab_name in _TABS:
            btn = QPushButton(tab_name)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setStyleSheet(tab_btn_style(tab_name == self._active_tab))
            btn.clicked.connect(
                lambda _c=False, t=tab_name: self._set_tab(t)
            )
            tabs.addWidget(btn)
            self._tab_btns.append(btn)
        tabs.addStretch()
        root.addLayout(tabs)

        # Scrollable combo list
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self._list_widget = QWidget()
        self._list_layout = QVBoxLayout(self._list_widget)
        self._list_layout.setSpacing(8)
        self._list_layout.setContentsMargins(0, 0, 4, 0)
        scroll.setWidget(self._list_widget)
        root.addWidget(scroll, stretch=1)

    def _set_tab(self, tab: str) -> None:
        self._active_tab = tab
        # Update tab button styles
        for i, tab_name in enumerate(_TABS):
            self._tab_btns[i].setStyleSheet(
                tab_btn_style(tab_name == self._active_tab)
            )
        self._populate()

    def _populate(self) -> None:
        for card in self._cards:
            self._list_layout.removeWidget(card)
            card.deleteLater()
        self._cards.clear()

        for combo in _COMBOS:
            if (self._active_tab != "All"
                    and combo["diff"] != self._active_tab.lower()):
                continue
            card = _ComboCard(combo, self)
            card.clicked.connect(
                lambda _c=False, c=combo: self._select(c)
            )
            self._list_layout.addWidget(card)
            self._cards.append(card)
        self._list_layout.addStretch()

    def _select(self, combo: Dict[str, Any]) -> None:
        logger.info("Selected combo: %s", combo["name"])
        self._router.navigate(
            "training_config", combo_id=combo["id"], combo=combo
        )

    # ── Lifecycle ──────────────────────────────────────────────────────
    def on_enter(self, **kwargs: Any) -> None:
        self._active_tab = "All"
        for i, tab_name in enumerate(_TABS):
            self._tab_btns[i].setStyleSheet(tab_btn_style(tab_name == "All"))
        self._populate()
        logger.debug("ComboSelectPage entered")

    def on_leave(self) -> None:
        pass
