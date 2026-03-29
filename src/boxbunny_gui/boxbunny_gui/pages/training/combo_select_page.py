"""Combo browser with difficulty tabs and mastery progress.

Scrollable list of combo cards with coloured punch-sequence indicators.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from boxbunny_gui.theme import Color, Size, font, GHOST_BTN, PRIMARY_BTN, SURFACE_BTN
from boxbunny_gui.widgets import BigButton

if TYPE_CHECKING:
    from boxbunny_gui.nav.router import PageRouter

logger = logging.getLogger(__name__)

# Punch colour mapping: 1=jab(blue), 2=cross(red), 3=hook(orange), 4=upper(purple)
_PUNCH_COLORS = {1: "#2196F3", 2: "#F44336", 3: "#FF9800", 4: "#9C27B0"}

_COMBOS: List[Dict[str, Any]] = [
    {"id": "jc", "name": "Jab-Cross", "seq": [1, 2], "diff": "beginner", "mastery": 65},
    {"id": "jch", "name": "Jab-Cross-Hook", "seq": [1, 2, 3], "diff": "beginner", "mastery": 30},
    {"id": "jchc", "name": "Jab-Cross-Hook-Cross", "seq": [1, 2, 3, 2], "diff": "intermediate", "mastery": 10},
    {"id": "djc", "name": "Double Jab-Cross", "seq": [1, 1, 2], "diff": "intermediate", "mastery": 0},
    {"id": "full", "name": "Full Combo", "seq": [1, 2, 3, 4, 2], "diff": "advanced", "mastery": 0},
]

_TABS = ["Beginner", "Intermediate", "Advanced", "All"]


class _ComboCard(QFrame):
    """Single combo card with name, sequence dots, and mastery bar."""

    def __init__(self, combo: Dict[str, Any], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.combo = combo
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(80)
        self.setStyleSheet(
            f"QFrame {{ background-color: {Color.SURFACE};"
            f" border-radius: {Size.RADIUS}px; }}"
            f" QFrame:hover {{ background-color: {Color.SURFACE_HOVER}; }}"
        )
        lay = QHBoxLayout(self)
        lay.setContentsMargins(Size.SPACING, Size.SPACING_SM, Size.SPACING, Size.SPACING_SM)

        # Name + difficulty
        info = QVBoxLayout()
        name_lbl = QLabel(combo["name"])
        name_lbl.setFont(font(18, bold=True))
        diff_lbl = QLabel(combo["diff"].title())
        diff_lbl.setStyleSheet(f"color: {Color.TEXT_SECONDARY}; font-size: 13px;")
        info.addWidget(name_lbl)
        info.addWidget(diff_lbl)
        lay.addLayout(info)

        # Sequence dots
        dots = QHBoxLayout()
        dots.setSpacing(6)
        for p in combo["seq"]:
            dot = QLabel()
            dot.setFixedSize(20, 20)
            dot.setStyleSheet(
                f"background-color: {_PUNCH_COLORS.get(p, Color.TEXT_SECONDARY)};"
                " border-radius: 10px;"
            )
            dots.addWidget(dot)
        dots.addStretch()
        lay.addLayout(dots)

        # Mastery bar
        bar = QProgressBar()
        bar.setFixedWidth(80)
        bar.setFixedHeight(12)
        bar.setTextVisible(False)
        bar.setValue(combo["mastery"])
        bar.setStyleSheet(
            f"QProgressBar {{ background-color: {Color.SURFACE_LIGHT};"
            f" border-radius: 6px; }}"
            f" QProgressBar::chunk {{ background-color: {Color.PRIMARY};"
            f" border-radius: 6px; }}"
        )
        lay.addWidget(bar)


class ComboSelectPage(QWidget):
    """Browse and select a combo to train."""

    def __init__(self, router: PageRouter, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._router = router
        self._active_tab: str = "All"
        self._cards: list[_ComboCard] = []
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
        title = QLabel("Select Combo")
        title.setFont(font(Size.TEXT_SUBHEADER, bold=True))
        top.addWidget(title)
        top.addStretch()
        root.addLayout(top)

        # Difficulty tabs — use plain QPushButton to avoid BigButton min-height
        tabs = QHBoxLayout()
        tabs.setSpacing(8)
        self._tab_btns: list[QPushButton] = []
        for tab in _TABS:
            btn = QPushButton(tab)
            btn.setStyleSheet(f"""
                QPushButton {{
                    font-size: 14px; font-weight: 600; padding: 8px 18px;
                    background-color: {Color.SURFACE_LIGHT}; color: {Color.TEXT_SECONDARY};
                    border: 1px solid {Color.BORDER}; border-radius: 8px;
                    min-height: 0; min-width: 0;
                }}
                QPushButton:hover {{ color: {Color.TEXT}; border-color: {Color.PRIMARY}; }}
            """)
            btn.clicked.connect(lambda _c=False, t=tab: self._set_tab(t))
            tabs.addWidget(btn)
            self._tab_btns.append(btn)
        tabs.addStretch()
        root.addLayout(tabs)

        # Scrollable combo list
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self._list_widget = QWidget()
        self._list_layout = QVBoxLayout(self._list_widget)
        self._list_layout.setSpacing(Size.SPACING_SM)
        self._list_layout.setContentsMargins(0, 0, 0, 0)
        scroll.setWidget(self._list_widget)
        root.addWidget(scroll, stretch=1)

    def _set_tab(self, tab: str) -> None:
        self._active_tab = tab
        self._populate()

    def _populate(self) -> None:
        for card in self._cards:
            self._list_layout.removeWidget(card)
            card.deleteLater()
        self._cards.clear()

        for combo in _COMBOS:
            if self._active_tab != "All" and combo["diff"] != self._active_tab.lower():
                continue
            card = _ComboCard(combo, self)
            card.mousePressEvent = lambda _e, c=combo: self._select(c)
            self._list_layout.addWidget(card)
            self._cards.append(card)
        self._list_layout.addStretch()

    def _select(self, combo: Dict[str, Any]) -> None:
        logger.info("Selected combo: %s", combo["name"])
        self._router.navigate("training_config", combo_id=combo["id"], combo=combo)

    # ── Lifecycle ──────────────────────────────────────────────────────
    def on_enter(self, **kwargs: Any) -> None:
        self._active_tab = "All"
        self._populate()
        logger.debug("ComboSelectPage entered")

    def on_leave(self) -> None:
        pass
