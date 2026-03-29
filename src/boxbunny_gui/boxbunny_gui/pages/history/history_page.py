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
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from boxbunny_gui.theme import Color, Size, font, GHOST_BTN, SURFACE_BTN
from boxbunny_gui.widgets import BigButton

if TYPE_CHECKING:
    from boxbunny_gui.nav.router import PageRouter

logger = logging.getLogger(__name__)

_FILTERS = ["All", "Training", "Sparring", "Performance"]

# Placeholder history data
_DEMO_HISTORY: List[Dict[str, str]] = [
    {"date": "2026-03-29", "mode": "Training", "duration": "12m", "punches": "142", "score": "78%"},
    {"date": "2026-03-28", "mode": "Sparring", "duration": "9m", "punches": "98", "score": "65%"},
    {"date": "2026-03-27", "mode": "Performance", "duration": "3m", "punches": "64", "score": "220ms"},
    {"date": "2026-03-26", "mode": "Training", "duration": "15m", "punches": "187", "score": "82%"},
]


class _SessionCard(QFrame):
    """Single history entry card."""

    def __init__(self, session: Dict[str, str], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.session = session
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(72)
        self.setStyleSheet(
            f"QFrame {{ background-color: {Color.SURFACE};"
            f" border-radius: {Size.RADIUS}px; }}"
            f" QFrame:hover {{ background-color: {Color.SURFACE_HOVER}; }}"
        )
        lay = QHBoxLayout(self)
        lay.setContentsMargins(Size.SPACING, Size.SPACING_SM, Size.SPACING, Size.SPACING_SM)

        # Mode icon placeholder + date
        left = QVBoxLayout()
        mode_lbl = QLabel(session["mode"])
        mode_lbl.setFont(font(16, bold=True))
        date_lbl = QLabel(session["date"])
        date_lbl.setStyleSheet(f"color: {Color.TEXT_SECONDARY}; font-size: 13px;")
        left.addWidget(mode_lbl)
        left.addWidget(date_lbl)
        lay.addLayout(left)

        lay.addStretch()

        # Stats
        dur = QLabel(session["duration"])
        dur.setStyleSheet(f"color: {Color.TEXT_SECONDARY}; font-size: 14px;")
        lay.addWidget(dur)

        punches = QLabel(f"{session['punches']} punches")
        punches.setStyleSheet(f"color: {Color.TEXT}; font-size: 14px;")
        lay.addWidget(punches)

        score = QLabel(session["score"])
        score.setFont(font(16, bold=True))
        score.setStyleSheet(f"color: {Color.PRIMARY};")
        lay.addWidget(score)


class HistoryPage(QWidget):
    """Scrollable training history with filter buttons."""

    def __init__(self, router: PageRouter, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._router = router
        self._active_filter: str = "All"
        self._cards: list[_SessionCard] = []
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(Size.SPACING, Size.SPACING_SM, Size.SPACING, Size.SPACING_SM)
        root.setSpacing(Size.SPACING)

        # Top bar
        top = QHBoxLayout()
        btn_back = BigButton("Back", stylesheet=GHOST_BTN)
        btn_back.setFixedWidth(100)
        btn_back.clicked.connect(lambda: self._router.back())
        top.addWidget(btn_back)
        title = QLabel("History")
        title.setFont(font(Size.TEXT_SUBHEADER, bold=True))
        top.addWidget(title)
        top.addStretch()
        root.addLayout(top)

        # Filters
        filters = QHBoxLayout()
        filters.setSpacing(Size.SPACING_SM)
        self._filter_btns: list[BigButton] = []
        for f in _FILTERS:
            btn = BigButton(f, stylesheet=SURFACE_BTN)
            btn.setFixedHeight(40)
            btn.setFixedWidth(120)
            btn.clicked.connect(lambda _c=False, flt=f: self._set_filter(flt))
            filters.addWidget(btn)
            self._filter_btns.append(btn)
        filters.addStretch()
        root.addLayout(filters)

        # Scrollable list
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self._list_widget = QWidget()
        self._list_layout = QVBoxLayout(self._list_widget)
        self._list_layout.setSpacing(Size.SPACING_SM)
        self._list_layout.setContentsMargins(0, 0, 0, 0)
        scroll.setWidget(self._list_widget)
        root.addWidget(scroll, stretch=1)

    def _set_filter(self, flt: str) -> None:
        self._active_filter = flt
        self._populate()

    def _populate(self) -> None:
        for card in self._cards:
            self._list_layout.removeWidget(card)
            card.deleteLater()
        self._cards.clear()

        for session in _DEMO_HISTORY:
            if self._active_filter != "All" and session["mode"] != self._active_filter:
                continue
            card = _SessionCard(session, self)
            # TODO: tap to view detailed results
            self._list_layout.addWidget(card)
            self._cards.append(card)
        self._list_layout.addStretch()

    # ── Lifecycle ──────────────────────────────────────────────────────
    def on_enter(self, **kwargs: Any) -> None:
        self._active_filter = "All"
        self._populate()
        logger.debug("HistoryPage entered")

    def on_leave(self) -> None:
        pass
