"""Grid of user account cards with search and sign-up button.

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
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from boxbunny_gui.theme import Color, Size, font, back_link_style, badge_style

if TYPE_CHECKING:
    from boxbunny_gui.nav.router import PageRouter

logger = logging.getLogger(__name__)

_DEMO_USERS: List[Dict[str, str]] = [
    {"id": "u1", "name": "Alex", "level": "Intermediate", "type": "user"},
    {"id": "u2", "name": "Jordan", "level": "Beginner", "type": "user"},
    {"id": "c1", "name": "Coach Mike", "level": "Coach", "type": "coach"},
]


class _UserCard(QFrame):
    """Clickable card showing avatar initial, display_name, and level badge.

    Uses QFrame with a click signal for proper internal layout.
    """

    clicked = None  # will be set per-instance via a signal-like pattern

    def __init__(self, user: Dict[str, str], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.user = user
        self.setFixedSize(210, 130)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._callback = None

        self.setStyleSheet(f"""
            QFrame {{
                background-color: {Color.SURFACE};
                border: 1px solid {Color.BORDER};
                border-radius: 14px;
            }}
            QFrame:hover {{
                background-color: {Color.SURFACE_HOVER};
                border-color: {Color.PRIMARY};
            }}
        """)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 14, 12, 12)
        lay.setSpacing(6)
        lay.setAlignment(Qt.AlignCenter)

        # Avatar circle
        initial = user["name"][0].upper()
        avatar = QLabel(initial)
        avatar.setFixedSize(44, 44)
        avatar.setAlignment(Qt.AlignCenter)
        avatar.setStyleSheet(f"""
            background: transparent;
            background-color: {Color.PRIMARY_MUTED};
            color: {Color.PRIMARY};
            font-size: 20px; font-weight: 700;
            border: 2px solid {Color.PRIMARY};
            border-radius: 22px;
        """)
        lay.addWidget(avatar, alignment=Qt.AlignCenter)

        # Name
        name_lbl = QLabel(user["name"])
        name_lbl.setAlignment(Qt.AlignCenter)
        name_lbl.setStyleSheet(
            f"font-size: 16px; font-weight: 600; color: {Color.TEXT};"
            " background: transparent; border: none;"
        )
        lay.addWidget(name_lbl, alignment=Qt.AlignCenter)

        # Level badge
        level_lbl = QLabel(user["level"])
        level_lbl.setAlignment(Qt.AlignCenter)
        level_lbl.setStyleSheet(
            "background: transparent; " + badge_style(Color.TEXT_SECONDARY)
        )
        lay.addWidget(level_lbl, alignment=Qt.AlignCenter)

    def mousePressEvent(self, event) -> None:
        if self._callback:
            self._callback()
        super().mousePressEvent(event)

    def connect_clicked(self, callback) -> None:
        """Register a click callback."""
        self._callback = callback


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
        root.setContentsMargins(
            Size.SPACING_LG * 2, Size.SPACING, Size.SPACING_LG * 2, Size.SPACING
        )
        root.setSpacing(Size.SPACING)

        # Top bar: back + title + search
        top = QHBoxLayout()
        self._btn_back = QPushButton("\u2190  Back")
        self._btn_back.setStyleSheet(back_link_style())
        self._btn_back.clicked.connect(lambda: self._router.back())
        top.addWidget(self._btn_back)

        title = QLabel("Select Account")
        title.setFont(font(Size.TEXT_SUBHEADER, bold=True))
        top.addWidget(title)
        top.addStretch()

        self._search = QLineEdit()
        self._search.setPlaceholderText("Search users...")
        self._search.setFixedSize(250, 42)
        self._search.textChanged.connect(self._filter)
        top.addWidget(self._search)
        root.addLayout(top)

        # Scrollable grid
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self._grid_widget = QWidget()
        self._grid = QGridLayout(self._grid_widget)
        self._grid.setSpacing(Size.SPACING_LG)
        scroll.setWidget(self._grid_widget)
        root.addWidget(scroll, stretch=1)

        # Bottom: sign-up hint
        bottom = QHBoxLayout()
        bottom.addStretch()
        signup_hint = QLabel("Don't have an account?")
        signup_hint.setStyleSheet(
            f"color: {Color.TEXT_SECONDARY}; font-size: 14px;"
        )
        bottom.addWidget(signup_hint)

        signup_btn = QPushButton("Sign Up")
        signup_btn.setFixedSize(110, 38)
        signup_btn.setStyleSheet(f"""
            QPushButton {{
                font-size: 14px; font-weight: 600;
                background-color: transparent; color: {Color.PRIMARY};
                border: 1px solid {Color.PRIMARY}; border-radius: {Size.RADIUS_SM}px;
            }}
            QPushButton:hover {{
                background-color: {Color.PRIMARY}; color: {Color.BG};
            }}
            QPushButton:pressed {{
                background-color: {Color.PRIMARY_PRESSED}; color: {Color.BG};
            }}
        """)
        signup_btn.clicked.connect(lambda: self._router.navigate("signup"))
        bottom.addWidget(signup_btn)
        bottom.addStretch()
        root.addLayout(bottom)

    def _populate(self) -> None:
        for card in self._cards:
            self._grid.removeWidget(card)
            card.deleteLater()
        self._cards.clear()

        users = _DEMO_USERS
        if self._coach_only:
            users = [u for u in users if u["type"] == "coach"]

        for i, user in enumerate(users):
            card = _UserCard(user, self)
            card.connect_clicked(lambda u=user: self._select_user(u))
            self._grid.addWidget(card, i // 4, i % 4)
            self._cards.append(card)

    def _select_user(self, user: Dict[str, str]) -> None:
        logger.info("Selected user: %s", user["name"])
        self._router.navigate(
            "pattern_lock", user_id=user["id"], user_name=user["name"]
        )

    def _filter(self, text: str) -> None:
        text_lower = text.lower()
        for card in self._cards:
            card.setVisible(text_lower in card.user["name"].lower())

    def on_enter(self, **kwargs: Any) -> None:
        self._coach_only = kwargs.get("coach_only", False)
        self._search.clear()
        self._populate()
        logger.debug("AccountPickerPage entered (coach_only=%s)", self._coach_only)

    def on_leave(self) -> None:
        pass
