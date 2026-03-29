"""Quick-start preset card for the home screen.

A clickable dark-surface card showing preset name, mode badge, and a
brief config summary.  Optional favourite star in the top-right corner.
"""

from __future__ import annotations

import logging

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
)

from boxbunny_gui.theme import Color, Size

log = logging.getLogger(__name__)

_MODE_COLORS: dict[str, str] = {
    "reaction": Color.PRIMARY,
    "shadow": "#42A5F5",
    "defence": Color.WARNING,
    "training": "#2196F3",
    "sparring": "#FF5722",
    "performance": Color.WARNING,
    "free": "#9C27B0",
    "circuit": "#00BCD4",
}


class PresetCard(QFrame):
    """Clickable card representing a saved drill preset.

    Dimensions are 250x100 px with a dark surface background and
    rounded corners.  Hover lightens the background.

    Signals
    -------
    clicked(int)
        Emitted with the ``preset_id`` when the card is tapped.
    """

    clicked = Signal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._preset_id: int = 0
        self._favorite: bool = False

        self.setFixedSize(200, 80)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._apply_bg(hovered=False)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(Size.SPACING, Size.SPACING_SM, Size.SPACING, Size.SPACING_SM)
        layout.setSpacing(Size.SPACING_SM)

        # left: text column
        text_col = QVBoxLayout()
        text_col.setSpacing(4)

        self._name_lbl = QLabel()
        self._name_lbl.setStyleSheet(
            f"color: {Color.TEXT}; font-size: 16px; font-weight: bold;"
            " background: transparent;"
        )

        self._mode_lbl = QLabel()
        self._mode_lbl.setStyleSheet(
            "font-size: 12px; font-weight: bold;"
            f" background-color: {Color.SURFACE_LIGHT};"
            f" border-radius: 4px; padding: 2px 8px;"
        )

        self._summary_lbl = QLabel()
        self._summary_lbl.setStyleSheet(
            f"color: {Color.TEXT_SECONDARY}; font-size: 14px; background: transparent;"
        )
        self._summary_lbl.setWordWrap(True)

        text_col.addWidget(self._name_lbl)
        text_col.addWidget(self._mode_lbl, 0, Qt.AlignmentFlag.AlignLeft)
        text_col.addWidget(self._summary_lbl)
        layout.addLayout(text_col, 1)

        # right: favourite star
        self._star_lbl = QLabel()
        self._star_lbl.setFixedSize(28, 28)
        self._star_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._star_lbl.setStyleSheet("font-size: 20px; background: transparent;")
        layout.addWidget(self._star_lbl, 0, Qt.AlignmentFlag.AlignTop)

    # -- public API -----------------------------------------------------------
    def set_preset(self, data: dict) -> None:
        """Load preset data.

        Expected keys: ``id``, ``name``, ``mode``, ``summary``,
        ``favorite`` (bool, optional).
        """
        self._preset_id = data.get("id", 0)
        self._name_lbl.setText(data.get("name", ""))

        mode = data.get("mode", data.get("preset_type", "")).lower()
        self._mode_lbl.setText(mode.capitalize())
        mode_color = _MODE_COLORS.get(mode, Color.TEXT_SECONDARY)
        self._mode_lbl.setStyleSheet(
            f"color: {mode_color}; font-size: 12px; font-weight: bold;"
            f" background-color: {Color.SURFACE_LIGHT};"
            f" border-radius: 4px; padding: 2px 8px;"
        )

        self._summary_lbl.setText(data.get("summary", ""))

        self._favorite = data.get("favorite", data.get("is_favorite", False))
        self._star_lbl.setText("\u2605" if self._favorite else "\u2606")
        self._star_lbl.setStyleSheet(
            f"font-size: 20px; background: transparent;"
            f" color: {'#FFD600' if self._favorite else Color.TEXT_SECONDARY};"
        )

    # -- hover / click --------------------------------------------------------
    def _apply_bg(self, hovered: bool) -> None:
        bg = Color.SURFACE_LIGHT if hovered else Color.SURFACE
        self.setStyleSheet(
            f"QFrame {{ background-color: {bg}; border-radius: {Size.RADIUS}px; }}"
        )

    def enterEvent(self, event) -> None:  # noqa: N802
        self._apply_bg(hovered=True)
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802
        self._apply_bg(hovered=False)
        super().leaveEvent(event)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        self.clicked.emit(self._preset_id)
