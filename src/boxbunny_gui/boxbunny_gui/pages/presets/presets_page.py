"""Preset management page.

List of user's presets as PresetCards with Create New button.
Tap starts session; long-press placeholder for edit/delete/favourite.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QScroller,
    QVBoxLayout,
    QWidget,
)

from boxbunny_gui.theme import Color, Size, font, GHOST_BTN, PRIMARY_BTN, SURFACE_BTN
from boxbunny_gui.widgets import BigButton, PresetCard

if TYPE_CHECKING:
    from boxbunny_gui.nav.router import PageRouter

logger = logging.getLogger(__name__)

_DB_PATH = Path(__file__).resolve().parents[5] / "data" / "boxbunny_main.db"


class PresetsPage(QWidget):
    """Browse, create, and manage training presets."""

    def __init__(self, router: PageRouter, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._router = router
        self._username: str = ""
        self._cards: list[PresetCard] = []
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
        title = QLabel("Presets")
        title.setStyleSheet(f"font-size: 20px; font-weight: 700; color: {Color.TEXT};")
        top.addWidget(title)
        top.addStretch()
        self._btn_create = BigButton("Create New", stylesheet=PRIMARY_BTN)
        self._btn_create.setFixedWidth(160)
        self._btn_create.clicked.connect(self._on_create)
        top.addWidget(self._btn_create)
        root.addLayout(top)

        # Empty state label
        self._empty_lbl = QLabel("No presets yet — create one after a training session!")
        self._empty_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_lbl.setStyleSheet(f"color: {Color.TEXT_DISABLED}; font-size: 14px;")
        self._empty_lbl.hide()
        root.addWidget(self._empty_lbl)

        # Scrollable preset list
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self._list_widget = QWidget()
        self._list_layout = QVBoxLayout(self._list_widget)
        self._list_layout.setSpacing(Size.SPACING_SM)
        self._list_layout.setContentsMargins(0, 0, 0, 0)
        scroll.setWidget(self._list_widget)
        QScroller.grabGesture(
            scroll.viewport(),
            QScroller.ScrollerGestureType.LeftMouseButtonGesture,
        )
        root.addWidget(scroll, stretch=1)

    def _load_presets(self) -> List[Dict[str, Any]]:
        """Load presets from database for the current user."""
        if not self._username or not _DB_PATH.exists():
            return []
        try:
            conn = sqlite3.connect(str(_DB_PATH))
            conn.row_factory = sqlite3.Row
            user_row = conn.execute(
                "SELECT id FROM users WHERE username = ?", (self._username,)
            ).fetchone()
            if not user_row:
                conn.close()
                return []
            rows = conn.execute(
                "SELECT * FROM presets WHERE user_id = ? "
                "ORDER BY is_favorite DESC, use_count DESC",
                (user_row["id"],),
            ).fetchall()
            conn.close()
            return [dict(r) for r in rows]
        except Exception as exc:
            logger.warning("Failed to load presets: %s", exc)
            return []

    def _populate(self) -> None:
        for card in self._cards:
            self._list_layout.removeWidget(card)
            card.deleteLater()
        self._cards.clear()

        presets = self._load_presets()
        if not presets:
            self._empty_lbl.show()
        else:
            self._empty_lbl.hide()

        for preset in presets:
            card = PresetCard(parent=self)
            card.set_preset({
                "id": preset["id"],
                "name": preset["name"],
                "mode": preset.get("preset_type", "training"),
                "summary": preset.get("description", ""),
                "favorite": bool(preset.get("is_favorite", False)),
            })
            card.clicked.connect(lambda pid, p=preset: self._on_tap(p))
            self._list_layout.addWidget(card)
            self._cards.append(card)
        self._list_layout.addStretch()

    def _on_tap(self, preset: Dict[str, Any]) -> None:
        logger.info("Preset tapped: %s", preset["name"])
        try:
            config = json.loads(preset.get("config_json", "{}"))
        except (json.JSONDecodeError, TypeError):
            config = {}
        preset_type = preset.get("preset_type", "training")

        if preset_type == "sparring":
            self._router.navigate(
                "sparring_select",
                username=self._username,
            )
        else:
            combo = {
                "name": config.get("combo_name", preset["name"]),
                "seq": config.get("combo_seq", ""),
                "id": config.get("combo_id"),
            }
            self._router.navigate(
                "training_config",
                combo=combo,
                difficulty=config.get("difficulty", "beginner"),
                username=self._username,
            )

    def _on_create(self) -> None:
        logger.info("Create new preset")
        self._router.navigate("training_config", username=self._username)

    # ── Lifecycle ──────────────────────────────────────────────────────
    def on_enter(self, **kwargs: Any) -> None:
        self._username = kwargs.get("username", "")
        self._populate()
        logger.debug("PresetsPage entered")

    def on_leave(self) -> None:
        pass
