"""User-account selection grid with search-as-you-type filtering.

Each account is shown as a dark-surface card with display name and level
badge.  The grid is scrollable and adapts to 2-3 columns.
"""

from __future__ import annotations

import logging

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QLabel,
    QLineEdit,
    QScrollArea,
    QScroller,
    QVBoxLayout,
    QWidget,
)

from boxbunny_gui.theme import Color, Size

log = logging.getLogger(__name__)

_LEVEL_COLORS: dict[str, str] = {
    "beginner": Color.PRIMARY,
    "intermediate": "#42A5F5",
    "advanced": Color.DANGER,
}
_CARD_W = 120
_CARD_H = 100


class _AccountCard(QFrame):
    """A single clickable user card."""

    clicked = Signal(int)

    def __init__(self, user_id: int, display_name: str, level: str, parent=None) -> None:
        super().__init__(parent)
        self.user_id = user_id
        self.display_name = display_name
        self.level = level.lower()

        self.setFixedSize(_CARD_W, _CARD_H)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._apply_style(hovered=False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 12, 8, 8)
        layout.setSpacing(6)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        name_lbl = QLabel(display_name)
        name_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        name_lbl.setWordWrap(True)
        name_lbl.setStyleSheet(
            f"color: {Color.TEXT}; font-size: 18px; font-weight: bold;"
            " background: transparent;"
        )

        level_color = _LEVEL_COLORS.get(self.level, Color.TEXT_SECONDARY)
        level_lbl = QLabel(level.capitalize())
        level_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        level_lbl.setStyleSheet(
            f"color: {level_color}; font-size: 12px; font-weight: bold;"
            f" background-color: {Color.SURFACE_LIGHT};"
            f" border-radius: 4px; padding: 2px 8px;"
        )

        layout.addWidget(name_lbl)
        layout.addWidget(level_lbl)

    def _apply_style(self, hovered: bool) -> None:
        bg = Color.SURFACE_LIGHT if hovered else Color.SURFACE
        self.setStyleSheet(
            f"QFrame {{ background-color: {bg};"
            f" border-radius: {Size.RADIUS}px; }}"
        )

    def enterEvent(self, event) -> None:  # noqa: N802
        self._apply_style(hovered=True)
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802
        self._apply_style(hovered=False)
        super().leaveEvent(event)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        self.clicked.emit(self.user_id)


class AccountPicker(QWidget):
    """Scrollable grid of user accounts with real-time search filter.

    Signals
    -------
    account_selected(int)
        Emitted with the ``user_id`` of the tapped card.
    """

    account_selected = Signal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._cards: list[_AccountCard] = []
        self._users: list[dict] = []

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(Size.SPACING_SM)

        # search field
        self._search = QLineEdit()
        self._search.setPlaceholderText("Search users...")
        self._search.setMinimumHeight(Size.MIN_TOUCH)
        self._search.textChanged.connect(self._on_filter)
        outer.addWidget(self._search)

        # scrollable grid
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        self._grid_widget = QWidget()
        self._grid_widget.setStyleSheet("background: transparent;")
        self._grid = QGridLayout(self._grid_widget)
        self._grid.setSpacing(Size.SPACING_SM)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._scroll.setWidget(self._grid_widget)
        QScroller.grabGesture(
            self._scroll.viewport(),
            QScroller.ScrollerGestureType.LeftMouseButtonGesture,
        )
        outer.addWidget(self._scroll, 1)

    # -- public API -----------------------------------------------------------
    def set_users(self, users: list[dict]) -> None:
        """Load user list.  Each dict must have ``id``, ``display_name``, ``level``."""
        self._users = users
        self._rebuild()

    def filter_text(self, text: str) -> None:
        """Programmatically set the search text."""
        self._search.setText(text)

    # -- internals ------------------------------------------------------------
    def _on_filter(self, text: str) -> None:
        needle = text.strip().lower()
        for card in self._cards:
            card.setVisible(needle in card.display_name.lower())

    def _rebuild(self) -> None:
        # clear old cards
        for card in self._cards:
            self._grid.removeWidget(card)
            card.deleteLater()
        self._cards.clear()

        cols = self._column_count()
        for idx, user in enumerate(self._users):
            card = _AccountCard(
                user_id=user.get("id", idx),
                display_name=user.get("display_name", "?"),
                level=user.get("level", "beginner"),
            )
            card.clicked.connect(self.account_selected.emit)
            self._grid.addWidget(card, idx // cols, idx % cols)
            self._cards.append(card)

    def _column_count(self) -> int:
        w = self.width()
        if w > 400:
            return 3
        return 2

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self._users:
            self._rebuild()
