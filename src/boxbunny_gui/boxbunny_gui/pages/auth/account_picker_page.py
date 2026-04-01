"""Grid of user account cards with search and sign-up button — premium styling."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List

from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import (
    QColor, QLinearGradient, QPainter, QPainterPath, QPen,
)
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

from boxbunny_gui.theme import (
    Color, Icon, Size, font, back_link_style,
    outline_btn_style,
)

if TYPE_CHECKING:
    from boxbunny_gui.nav.router import PageRouter

logger = logging.getLogger(__name__)

_DEMO_USERS: List[Dict[str, str]] = [
    {"id": "u1", "name": "Alex", "level": "Intermediate", "type": "user",
     "has_pattern": True},
    {"id": "u2", "name": "Jordan", "level": "Beginner", "type": "user",
     "has_pattern": False},
    {"id": "c1", "name": "Coach Mike", "level": "Coach", "type": "coach",
     "has_pattern": True},
]

# Gradient pairs — (top-color, bottom-color) for each avatar
_AVATAR_GRADIENTS = [
    ("#FF6B35", "#E8522A"),   # warm orange
    ("#58A6FF", "#3984DB"),   # sky blue
    ("#BC8CFF", "#9B6AE0"),   # lavender
    ("#56D364", "#3CB950"),   # fresh green
    ("#FFAB40", "#E89530"),   # amber
    ("#FF5C5C", "#D94343"),   # coral
]

# Hover accent derived from first color in gradient
_AVATAR_ACCENTS = [g[0] for g in _AVATAR_GRADIENTS]


# ── Avatar widget ────────────────────────────────────────────────────────

class _AvatarWidget(QWidget):
    """Circular avatar with gradient background and person silhouette."""

    def __init__(self, size: int = 56, color_idx: int = 0,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(size, size)
        self._sz = size
        self._c1, self._c2 = _AVATAR_GRADIENTS[
            color_idx % len(_AVATAR_GRADIENTS)
        ]

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        s = self._sz

        # ── Clip to circle ───────────────────────────────────────────────
        clip = QPainterPath()
        clip.addEllipse(QRectF(0, 0, s, s))
        p.setClipPath(clip)

        # ── Gradient background ──────────────────────────────────────────
        grad = QLinearGradient(0, 0, 0, s)
        grad.setColorAt(0.0, QColor(self._c1))
        grad.setColorAt(1.0, QColor(self._c2))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(grad)
        p.drawEllipse(QRectF(0, 0, s, s))

        # ── Person silhouette (white, semi-transparent) ──────────────────
        person = QColor(255, 255, 255, 210)
        p.setBrush(person)
        p.setPen(Qt.PenStyle.NoPen)

        # Head
        head_r = s * 0.17
        head_cx = s * 0.5
        head_cy = s * 0.34
        p.drawEllipse(QRectF(
            head_cx - head_r, head_cy - head_r, head_r * 2, head_r * 2
        ))

        # Shoulders / body — wide ellipse clipped by the circle
        body_w = s * 0.62
        body_h = s * 0.50
        body_top = s * 0.58
        p.drawEllipse(QRectF(
            (s - body_w) / 2, body_top, body_w, body_h
        ))

        p.end()


# ── Card styling ─────────────────────────────────────────────────────────

def _card_style(accent: str) -> str:
    return f"""
        _UserCard {{
            background-color: #131920;
            border: 1px solid #1E2832;
            border-bottom: 2px solid {accent}30;
            border-radius: {Size.RADIUS_LG}px;
        }}
        _UserCard:hover {{
            background-color: {Color.SURFACE_HOVER};
            border-color: {accent};
            border-bottom: 2px solid {accent};
        }}
        _UserCard QLabel {{
            background: transparent;
            border: none;
        }}
    """


class _UserCard(QFrame):
    """Clickable card showing avatar, display_name, and level badge."""

    clicked = None

    def __init__(self, user: Dict[str, str], color_idx: int = 0,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.user = user
        self.setFixedSize(210, 155)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._callback = None
        accent = _AVATAR_ACCENTS[color_idx % len(_AVATAR_ACCENTS)]

        self.setStyleSheet(_card_style(accent))

        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 16, 12, 14)
        lay.setSpacing(8)
        lay.setAlignment(Qt.AlignCenter)

        # Avatar
        avatar = _AvatarWidget(size=56, color_idx=color_idx, parent=self)
        lay.addWidget(avatar, alignment=Qt.AlignCenter)

        # Name
        name_lbl = QLabel(user["name"])
        name_lbl.setAlignment(Qt.AlignCenter)
        name_lbl.setStyleSheet(
            f"font-size: 16px; font-weight: 600; color: {Color.TEXT};"
        )
        lay.addWidget(name_lbl, alignment=Qt.AlignCenter)

        # Level badge — muted pill
        level_text = user["level"].upper()
        level_lbl = QLabel(level_text)
        level_lbl.setAlignment(Qt.AlignCenter)
        level_lbl.setStyleSheet(
            f"font-size: 10px; font-weight: 700; color: {Color.TEXT_SECONDARY};"
            f" background-color: {Color.SURFACE_LIGHT};"
            f" border: 1px solid {Color.BORDER};"
            f" border-radius: 6px; padding: 3px 10px;"
            " letter-spacing: 0.8px;"
        )
        lay.addWidget(level_lbl, alignment=Qt.AlignCenter)

    def mousePressEvent(self, event) -> None:
        if self._callback:
            self._callback()
        super().mousePressEvent(event)

    def connect_clicked(self, callback) -> None:
        self._callback = callback


def _load_users_from_db() -> List[Dict[str, str]]:
    """Try to load real users from the database, fall back to demo list."""
    try:
        from boxbunny_gui.db_helper import list_users, get_user
        rows = list_users()
        if not rows:
            return _DEMO_USERS
        users = []
        for row in rows:
            full_user = get_user(row["id"])
            has_pattern = bool(full_user and full_user.get("pattern_hash"))
            users.append({
                "id": str(row["id"]),
                "name": row.get("display_name") or row.get("username", "?"),
                "username": row.get("username", ""),
                "level": row.get("level", "beginner").capitalize(),
                "type": row.get("user_type", "individual"),
                "has_pattern": has_pattern,
            })
        return users
    except Exception as exc:
        logger.warning("Could not load users from DB: %s", exc)
        return _DEMO_USERS


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
        self._btn_back = QPushButton(f"{Icon.BACK}  Back")
        self._btn_back.setStyleSheet(back_link_style())
        self._btn_back.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_back.clicked.connect(lambda: self._router.back())
        top.addWidget(self._btn_back)

        title = QLabel("Select Account")
        title.setStyleSheet(f"font-size: 20px; font-weight: 700; color: {Color.TEXT};")
        top.addWidget(title)
        top.addStretch()

        self._search = QLineEdit()
        self._search.setPlaceholderText("Search users...")
        self._search.setFixedSize(250, 42)
        self._search.setStyleSheet(f"""
            QLineEdit {{
                font-size: 14px;
                background-color: {Color.SURFACE};
                color: {Color.TEXT};
                border: 1px solid {Color.BORDER};
                border-radius: {Size.RADIUS}px;
                padding: 0 14px;
            }}
            QLineEdit:focus {{
                border-color: {Color.PRIMARY};
            }}
        """)
        self._search.textChanged.connect(self._filter)
        top.addWidget(self._search)
        root.addLayout(top)

        # Scrollable grid
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(
            "QScrollArea { background: transparent; border: none; }"
            " QWidget { background: transparent; }"
        )
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
        signup_btn.setFixedSize(140, 48)
        signup_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        signup_btn.setStyleSheet(outline_btn_style())
        signup_btn.clicked.connect(lambda: self._router.navigate("signup"))
        bottom.addWidget(signup_btn)
        bottom.addStretch()
        root.addLayout(bottom)

    def _populate(self) -> None:
        for card in self._cards:
            self._grid.removeWidget(card)
            card.deleteLater()
        self._cards.clear()

        users = _load_users_from_db()
        if self._coach_only:
            users = [u for u in users if u["type"] == "coach"]

        for i, user in enumerate(users):
            card = _UserCard(user, color_idx=i, parent=self)
            card.connect_clicked(lambda u=user: self._select_user(u))
            self._grid.addWidget(card, i // 4, i % 4)
            self._cards.append(card)

    def _select_user(self, user: Dict[str, str]) -> None:
        logger.info("Selected user: %s", user["name"])
        self._router.navigate(
            "pattern_lock",
            user_id=user["id"],
            user_name=user["name"],
            username=user.get("username", ""),
            has_pattern=user.get("has_pattern", True),
        )

    def _filter(self, text: str) -> None:
        text_lower = text.lower().strip()
        if not text_lower:
            for card in self._cards:
                card.setVisible(True)
            return
        for card in self._cards:
            name = card.user["name"].lower()
            username = card.user.get("username", "").lower()
            card.setVisible(
                name.startswith(text_lower)
                or text_lower in name
                or username.startswith(text_lower)
            )

    def on_enter(self, **kwargs: Any) -> None:
        self._coach_only = kwargs.get("coach_only", False)
        self._search.clear()
        self._populate()
        logger.debug("AccountPickerPage entered (coach_only=%s)", self._coach_only)

    def on_leave(self) -> None:
        pass
