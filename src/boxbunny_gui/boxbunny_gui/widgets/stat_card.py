"""Compact stat-display card with optional trend indicator.

Dark-surface rounded frame suitable for dashboard grids.
"""

from __future__ import annotations

import logging

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout

from boxbunny_gui.theme import Color, Size

log = logging.getLogger(__name__)

_TREND_ARROWS: dict[str, tuple[str, str]] = {
    "up": ("\u25B2", Color.PRIMARY),
    "down": ("\u25BC", Color.DANGER),
    "neutral": ("\u2014", Color.TEXT_SECONDARY),
}


class StatCard(QFrame):
    """A small stat card showing *title*, *value*, and an optional trend arrow.

    Parameters
    ----------
    title : str
        Short label (e.g. "Avg. Speed").
    value : str
        Primary metric (e.g. "42 ms").
    subtitle : str
        Extra context line below the value.
    trend : str
        One of ``"up"``, ``"down"``, ``"neutral"``.
    """

    def __init__(
        self,
        title: str,
        value: str,
        subtitle: str = "",
        trend: str = "neutral",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setFixedHeight(100)
        self.setStyleSheet(
            f"QFrame {{ background-color: {Color.SURFACE};"
            f" border-radius: {Size.RADIUS}px; }}"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(Size.SPACING, Size.SPACING_SM, Size.SPACING, Size.SPACING_SM)
        layout.setSpacing(2)

        self._title_lbl = QLabel(title)
        self._title_lbl.setStyleSheet(f"color: {Color.TEXT_SECONDARY}; font-size: 14px;")
        self._title_lbl.setAlignment(Qt.AlignmentFlag.AlignLeft)

        self._value_lbl = QLabel(value)
        self._value_lbl.setStyleSheet(f"color: {Color.TEXT}; font-size: 28px; font-weight: bold;")
        self._value_lbl.setAlignment(Qt.AlignmentFlag.AlignLeft)

        self._sub_lbl = QLabel()
        self._sub_lbl.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self._sub_lbl.setStyleSheet("font-size: 13px;")

        layout.addWidget(self._title_lbl)
        layout.addWidget(self._value_lbl)
        layout.addWidget(self._sub_lbl)

        self.set_trend(trend)
        if subtitle:
            self._sub_lbl.setText(subtitle)

    # -- public API -----------------------------------------------------------
    def set_value(self, value: str) -> None:
        """Update the primary metric text."""
        self._value_lbl.setText(value)

    def set_trend(self, trend: str) -> None:
        """Set the trend indicator: ``'up'``, ``'down'``, or ``'neutral'``."""
        arrow, color = _TREND_ARROWS.get(trend, _TREND_ARROWS["neutral"])
        current = self._sub_lbl.text()
        prefix = current.lstrip("\u25B2\u25BC\u2014 ")
        self._sub_lbl.setText(f"{arrow} {prefix}" if prefix else arrow)
        self._sub_lbl.setStyleSheet(f"color: {color}; font-size: 13px;")
