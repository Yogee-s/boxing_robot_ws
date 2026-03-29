"""Animated punch-count display with scale-pulse on increment.

Shows a large green number with an optional label above it.
"""

from __future__ import annotations

import logging

from PySide6.QtCore import (
    QEasingCurve,
    QPropertyAnimation,
    Qt,
    Property,
)
from PySide6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

from boxbunny_gui.theme import Color, Size

log = logging.getLogger(__name__)


class PunchCounter(QWidget):
    """Large live punch-count display that pulses on every increment.

    Parameters
    ----------
    label : str
        Caption text shown above the number (default ``"PUNCHES"``).
    """

    def __init__(self, label: str = "PUNCHES", parent=None) -> None:
        super().__init__(parent)
        self._count: int = 0
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(0)

        self._label = QLabel(label)
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setStyleSheet(
            f"color: {Color.TEXT_SECONDARY}; font-size: 14px; font-weight: bold;"
            " background: transparent;"
        )

        self._number = QLabel("0")
        self._number.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._number.setStyleSheet(
            f"color: {Color.PRIMARY}; font-size: 56px; font-weight: bold;"
            " background: transparent;"
        )

        layout.addWidget(self._label)
        layout.addWidget(self._number)

        # -- pulse animation ---------------------------------------------------
        self._pulse_val: float = 1.0
        self._anim = QPropertyAnimation(self, b"pulse")
        self._anim.setDuration(200)
        self._anim.setKeyValueAt(0, 1.0)
        self._anim.setKeyValueAt(0.4, 1.18)
        self._anim.setKeyValueAt(1, 1.0)
        self._anim.setEasingCurve(QEasingCurve.Type.OutBack)

    # -- Qt property ----------------------------------------------------------
    def _get_pulse(self) -> float:
        return self._pulse_val

    def _set_pulse(self, v: float) -> None:
        self._pulse_val = v
        pt = int(56 * v)
        self._number.setStyleSheet(
            f"color: {Color.PRIMARY}; font-size: {pt}px; font-weight: bold;"
            " background: transparent;"
        )

    pulse = Property(float, _get_pulse, _set_pulse)

    # -- public API -----------------------------------------------------------
    def set_count(self, count: int) -> None:
        """Set the counter to an absolute value."""
        self._count = count
        self._number.setText(str(count))
        self._anim.start()

    def increment(self) -> None:
        """Increment by one and trigger the pulse animation."""
        self.set_count(self._count + 1)
