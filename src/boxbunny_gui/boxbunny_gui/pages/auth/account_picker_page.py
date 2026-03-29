"""Grid of user account cards with search and QR sign-up link.

Receives optional ``coach_only`` kwarg to filter user type.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from boxbunny_gui.theme import Color, Size, font, GHOST_BTN
from boxbunny_gui.widgets import BigButton, QRWidget

if TYPE_CHECKING:
    from boxbunny_gui.nav.router import PageRouter

logger = logging.getLogger(__name__)

# Placeholder user data -- replaced at runtime by database query
_DEMO_USERS: List[Dict[str, str]] = [
    {"id": "u1", "name": "Alex", "level": "Intermediate", "type": "user"},
    {"id": "u2", "name": "Jordan", "level": "Beginner", "type": "user"},
    {"id": "c1", "name": "Coach Mike", "level": "Coach", "type": "coach"},
]


class _UserCard(QFrame):
    """Clickable card showing display_name + level badge."""

    def __init__(self, user: Dict[str, str], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.user = user
        self.setFixedSize(200, 100)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(
            f"QFrame {{ background-color: {Color.SURFACE};"
            f" border-radius: {Size.RADIUS}px; }}"
            f" QFrame:hover {{ background-color: {Color.SURFACE_HOVER}; }}"
        )
        lay = QVBoxLayout(self)
        lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        name_lbl = QLabel(user["name"])
        name_lbl.setFont(font(20, bold=True))
        name_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge = QLabel(user["level"])
        badge.setStyleSheet(f"color: {Color.PRIMARY}; font-size: 14px;")
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(name_lbl)
        lay.addWidget(badge)

    def mousePressEvent(self, event: Any) -> None:  # noqa: N802
        super().mousePressEvent(event)


class AccountPickerPage(QWidget):
    """Grid of selectable user account cards."""

    def __init__(self, router: PageRouter, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._router = router
        self._coach_only: bool = False
        self._cards: list[_UserCard] = []
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(Size.SPACING_LG, Size.SPACING, Size.SPACING_LG, Size.SPACING)
        root.setSpacing(Size.SPACING)

        # Top bar: back + search
        top = QHBoxLayout()
        self._btn_back = BigButton("Back", stylesheet=GHOST_BTN)
        self._btn_back.setFixedWidth(100)
        self._btn_back.clicked.connect(lambda: self._router.back())
        top.addWidget(self._btn_back)
        self._search = QLineEdit()
        self._search.setPlaceholderText("Search users...")
        self._search.setStyleSheet(
            f"QLineEdit {{ background-color: {Color.SURFACE}; color: {Color.TEXT};"
            f" border: 1px solid {Color.BORDER_LIGHT}; border-radius: {Size.RADIUS_SM}px;"
            f" padding: 8px 12px; font-size: 16px; }}"
            f" QLineEdit:focus {{ border-color: {Color.PRIMARY}; }}"
        )
        self._search.setFixedHeight(40)
        self._search.textChanged.connect(self._filter)
        top.addWidget(self._search)
        root.addLayout(top)

        # Scrollable grid
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self._grid_widget = QWidget()
        self._grid = QGridLayout(self._grid_widget)
        self._grid.setSpacing(Size.SPACING)
        scroll.setWidget(self._grid_widget)
        root.addWidget(scroll, stretch=1)

        # Bottom: QR sign-up
        bottom = QHBoxLayout()
        bottom.addStretch()
        signup_lbl = QLabel("New? Scan QR to sign up")
        signup_lbl.setStyleSheet(f"color: {Color.TEXT_SECONDARY}; font-size: 14px;")
        bottom.addWidget(signup_lbl)
        self._qr = QRWidget(size=60)
        self._qr.set_text("https://boxbunny.local/signup")
        bottom.addWidget(self._qr)
        bottom.addStretch()
        root.addLayout(bottom)

    def _populate(self) -> None:
        # Clear
        for card in self._cards:
            self._grid.removeWidget(card)
            card.deleteLater()
        self._cards.clear()

        users = _DEMO_USERS
        if self._coach_only:
            users = [u for u in users if u["type"] == "coach"]

        for i, user in enumerate(users):
            card = _UserCard(user, self)
            card.mousePressEvent = lambda _e, u=user: self._select_user(u)
            self._grid.addWidget(card, i // 4, i % 4)
            self._cards.append(card)

    def _select_user(self, user: Dict[str, str]) -> None:
        logger.info("Selected user: %s", user["name"])
        self._router.navigate("pattern_lock", user_id=user["id"], user_name=user["name"])

    def _filter(self, text: str) -> None:
        text_lower = text.lower()
        for card in self._cards:
            card.setVisible(text_lower in card.user["name"].lower())

    # ── Lifecycle ──────────────────────────────────────────────────────
    def on_enter(self, **kwargs: Any) -> None:
        self._coach_only = kwargs.get("coach_only", False)
        self._search.clear()
        self._populate()
        logger.debug("AccountPickerPage entered (coach_only=%s)", self._coach_only)

    def on_leave(self) -> None:
        pass
