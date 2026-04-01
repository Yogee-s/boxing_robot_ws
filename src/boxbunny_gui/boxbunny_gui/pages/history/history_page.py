"""Training history timeline with filter buttons and scrollable session list.

Each session card shows date, mode icon, duration, punch count, and score.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from boxbunny_gui.theme import Color, Size, font, GHOST_BTN, tab_btn_style
from boxbunny_gui.widgets import BigButton

if TYPE_CHECKING:
    from boxbunny_gui.nav.router import PageRouter

logger = logging.getLogger(__name__)

_FILTERS = ["All", "Training", "Sparring", "Performance"]

# Mode accent colors for the left border
_MODE_COLORS = {
    "Training": Color.PRIMARY,
    "Sparring": Color.DANGER,
    "Performance": Color.WARNING,
}

# Mode icons (emoji stand-ins)
_MODE_ICONS = {
    "Training": "\U0001F94A",
    "Sparring": "\u2694\uFE0F",
    "Performance": "\u26A1",
}

# Placeholder history data
_DEMO_HISTORY: List[Dict[str, str]] = [
    {"date": "2026-03-29", "mode": "Training", "duration": "12m", "punches": "142", "score": "78%"},
    {"date": "2026-03-28", "mode": "Sparring", "duration": "9m", "punches": "98", "score": "65%"},
    {"date": "2026-03-27", "mode": "Performance", "duration": "3m", "punches": "64", "score": "220ms"},
    {"date": "2026-03-26", "mode": "Training", "duration": "15m", "punches": "187", "score": "82%"},
]


def _stat_pill(label: str, value: str) -> QLabel:
    """Tiny stat pill with muted label + bright value."""
    pill = QLabel(f"{label} {value}")
    pill.setStyleSheet(
        "background: transparent;"
        f" font-size: 12px; font-weight: 600; color: {Color.TEXT_SECONDARY};"
        f" background-color: {Color.SURFACE_LIGHT}; border-radius: 8px;"
        " padding: 3px 10px;"
    )
    return pill


class _SessionCard(QFrame):
    """Single history entry card with structured layout."""

    def __init__(
        self, session: Dict[str, str], parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.session = session
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        accent = _MODE_COLORS.get(session["mode"], Color.TEXT_SECONDARY)
        self.setObjectName("histCard")
        self.setFixedHeight(80)
        self.setStyleSheet(f"""
            QFrame#histCard {{
                background-color: {Color.SURFACE};
                border: 1px solid {Color.BORDER};
                border-top: 3px solid {accent};
                border-radius: 12px;
            }}
            QFrame#histCard:hover {{
                background-color: {Color.SURFACE_HOVER};
                border-color: {accent}50;
                border-top: 3px solid {accent};
            }}
        """)

        row = QHBoxLayout(self)
        row.setContentsMargins(16, 10, 16, 10)
        row.setSpacing(14)

        # Left: mode + date
        left = QVBoxLayout()
        left.setSpacing(2)
        icon = _MODE_ICONS.get(session["mode"], "")
        mode_lbl = QLabel(f"{icon}  {session['mode']}")
        mode_lbl.setStyleSheet(
            "background: transparent;"
            f" font-size: 15px; font-weight: 700; color: {Color.TEXT};"
        )
        left.addWidget(mode_lbl)
        date_lbl = QLabel(session["date"])
        date_lbl.setStyleSheet(
            "background: transparent;"
            f" font-size: 12px; color: {Color.TEXT_DISABLED};"
        )
        left.addWidget(date_lbl)
        row.addLayout(left)

        row.addStretch()

        # Right: stat pills
        pills = QHBoxLayout()
        pills.setSpacing(6)
        pills.addWidget(_stat_pill("\u23F1", session["duration"]))
        pills.addWidget(_stat_pill("\U0001F44A", session["punches"]))

        score_lbl = QLabel(session["score"])
        score_lbl.setStyleSheet(
            "background: transparent;"
            f" font-size: 13px; font-weight: 700; color: {accent};"
            f" background-color: {accent}18; border-radius: 8px;"
            " padding: 3px 12px;"
        )
        pills.addWidget(score_lbl)
        row.addLayout(pills)


class HistoryPage(QWidget):
    """Scrollable training history with filter buttons."""

    def __init__(self, router: PageRouter, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._router = router
        self._active_filter: str = "All"
        self._cards: list[_SessionCard] = []
        self._filter_btns: list[QPushButton] = []
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(
            Size.SPACING_LG, Size.SPACING, Size.SPACING_LG, Size.SPACING_SM
        )
        root.setSpacing(Size.SPACING)

        # Top bar
        top = QHBoxLayout()
        btn_back = BigButton("Back", stylesheet=GHOST_BTN)
        btn_back.setFixedWidth(100)
        btn_back.clicked.connect(lambda: self._router.back())
        top.addWidget(btn_back)
        title = QLabel("History")
        title.setStyleSheet(f"font-size: 20px; font-weight: 700; color: {Color.TEXT};")
        top.addWidget(title)
        top.addStretch()

        # Session count badge
        self._count_lbl = QLabel()
        self._count_lbl.setStyleSheet(
            f"font-size: 12px; font-weight: 600; color: {Color.TEXT_DISABLED};"
            f" background-color: {Color.SURFACE}; border-radius: 8px;"
            " padding: 4px 12px;"
        )
        top.addWidget(self._count_lbl)
        root.addLayout(top)

        # Filter tabs with active state
        filters = QHBoxLayout()
        filters.setSpacing(8)
        for f in _FILTERS:
            btn = QPushButton(f)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setStyleSheet(tab_btn_style(f == self._active_filter))
            btn.clicked.connect(
                lambda _c=False, flt=f: self._set_filter(flt)
            )
            filters.addWidget(btn)
            self._filter_btns.append(btn)
        filters.addStretch()
        root.addLayout(filters)

        # Scrollable list
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self._list_widget = QWidget()
        self._list_layout = QVBoxLayout(self._list_widget)
        self._list_layout.setSpacing(10)
        self._list_layout.setContentsMargins(2, 4, 2, 4)
        scroll.setWidget(self._list_widget)
        root.addWidget(scroll, stretch=1)

    def _set_filter(self, flt: str) -> None:
        self._active_filter = flt
        for i, f_name in enumerate(_FILTERS):
            self._filter_btns[i].setStyleSheet(
                tab_btn_style(f_name == self._active_filter)
            )
        self._populate()

    def _populate(self) -> None:
        for card in self._cards:
            self._list_layout.removeWidget(card)
            card.deleteLater()
        self._cards.clear()

        # Remove old stretch / empty label
        while self._list_layout.count():
            item = self._list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        filtered = [
            s for s in _DEMO_HISTORY
            if self._active_filter == "All" or s["mode"] == self._active_filter
        ]
        self._count_lbl.setText(f"{len(filtered)} sessions")

        if not filtered:
            empty = QLabel("No sessions yet")
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            empty.setStyleSheet(
                f"color: {Color.TEXT_DISABLED}; font-size: 15px;"
                " padding: 40px;"
            )
            self._list_layout.addWidget(empty)

        for session in filtered:
            card = _SessionCard(session, self)
            self._list_layout.addWidget(card)
            self._cards.append(card)
        self._list_layout.addStretch()

    # ── Lifecycle ──────────────────────────────────────────────────────
    def on_enter(self, **kwargs: Any) -> None:
        self._active_filter = "All"
        for i, f_name in enumerate(_FILTERS):
            self._filter_btns[i].setStyleSheet(
                tab_btn_style(f_name == "All")
            )
        self._populate()
        logger.debug("HistoryPage entered")

    def on_leave(self) -> None:
        pass
