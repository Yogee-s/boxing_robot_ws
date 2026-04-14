"""Visual horizontal combo-sequence display with step indicators.

Each punch in the combo is shown as a coloured badge.  The current step
is enlarged and glowing; completed / missed steps get overlay marks.
"""

from __future__ import annotations

import logging

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QScrollArea,
    QScroller,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from boxbunny_gui.theme import Color, Size

log = logging.getLogger(__name__)

# Mapping from canonical punch name to (short label, colour hex).
PUNCH_PALETTE: dict[str, tuple[str, str]] = {
    "1":       ("1", Color.JAB),
    "jab":     ("1", Color.JAB),
    "2":       ("2", Color.CROSS),
    "cross":   ("2", Color.CROSS),
    "3":       ("3", Color.L_HOOK),
    "l.hook":  ("3", Color.L_HOOK),
    "4":       ("4", Color.R_HOOK),
    "r.hook":  ("4", Color.R_HOOK),
    "5":       ("5", Color.L_UPPERCUT),
    "l.uc":    ("5", Color.L_UPPERCUT),
    "6":       ("6", Color.R_UPPERCUT),
    "r.uc":    ("6", Color.R_UPPERCUT),
}

_BADGE_SIZE = 56
_BADGE_SIZE_ACTIVE = 68


class _PunchBadge(QWidget):
    """Single circular badge representing one punch in a combo."""

    def __init__(self, label: str, color_hex: str, parent=None) -> None:
        super().__init__(parent)
        self._label = label
        self._color = color_hex
        self._state: str = "pending"  # pending | current | correct | missed
        self.setFixedSize(_BADGE_SIZE, _BADGE_SIZE)

    def set_state(self, state: str) -> None:
        """Set visual state: ``pending``, ``current``, ``correct``, ``missed``."""
        self._state = state
        size = _BADGE_SIZE_ACTIVE if state == "current" else _BADGE_SIZE
        self.setFixedSize(size, size)

        # glow for current step
        if state == "current":
            glow = QGraphicsDropShadowEffect(self)
            glow.setColor(QColor(self._color))
            glow.setBlurRadius(20)
            glow.setOffset(0, 0)
            self.setGraphicsEffect(glow)
        else:
            self.setGraphicsEffect(None)  # type: ignore[arg-type]
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        r = min(w, h) // 2 - 2

        base_color = QColor(self._color)
        if self._state == "pending":
            base_color.setAlpha(100)
        elif self._state in ("correct", "missed"):
            base_color.setAlpha(160)

        # filled circle
        p.setBrush(base_color)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(w // 2 - r, h // 2 - r, 2 * r, 2 * r)

        # border for current
        if self._state == "current":
            pen = QPen(QColor(Color.TEXT), 3)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(w // 2 - r, h // 2 - r, 2 * r, 2 * r)

        # punch label
        p.setPen(QColor(Color.TEXT))
        p.setFont(QFont("Inter", 16, QFont.Weight.Bold))
        p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._label)

        # overlay for correct / missed
        if self._state == "correct":
            p.setPen(QPen(QColor(Color.PRIMARY), 3))
            p.setFont(QFont("Inter", 22, QFont.Weight.Bold))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "\u2713")
        elif self._state == "missed":
            p.setPen(QPen(QColor(Color.DANGER), 3))
            p.setFont(QFont("Inter", 22, QFont.Weight.Bold))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "\u2717")

        p.end()


class ComboDisplay(QWidget):
    """Horizontal row of punch-type badges with step tracking.

    The widget is scrollable when the combo exceeds ~8 punches.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._badges: list[_PunchBadge] = []
        self._step: int = 0
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(_BADGE_SIZE_ACTIVE + Size.SPACING * 2)

        # scrollable inner container
        self._scroll = QScrollArea(self)
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")

        self._container = QWidget()
        self._container.setStyleSheet("background: transparent;")
        self._row = QHBoxLayout(self._container)
        self._row.setContentsMargins(Size.SPACING, 4, Size.SPACING, 4)
        self._row.setSpacing(Size.SPACING_SM)
        self._row.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self._scroll.setWidget(self._container)
        QScroller.grabGesture(
            self._scroll.viewport(),
            QScroller.ScrollerGestureType.LeftMouseButtonGesture,
        )

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self._scroll)

    # -- public API -----------------------------------------------------------
    def set_combo(self, punches: list[str]) -> None:
        """Load a new combo sequence.  Each entry is a punch name or number."""
        self._clear()
        self._step = 0
        for name in punches:
            key = name.strip().lower()
            label, color = PUNCH_PALETTE.get(key, (key[:2], Color.TEXT_SECONDARY))
            badge = _PunchBadge(label, color, self._container)
            badge.set_state("pending")
            self._badges.append(badge)
            self._row.addWidget(badge)
        if self._badges:
            self._badges[0].set_state("current")

    def advance(self, detected_correct: bool) -> None:
        """Mark the current step and move to the next one.

        Parameters
        ----------
        detected_correct : bool
            ``True`` if the user threw the correct punch, ``False`` for a miss.
        """
        if self._step >= len(self._badges):
            return
        self._badges[self._step].set_state("correct" if detected_correct else "missed")
        self._step += 1
        if self._step < len(self._badges):
            self._badges[self._step].set_state("current")
            self._scroll.ensureWidgetVisible(self._badges[self._step])

    def reset(self) -> None:
        """Reset all badges to pending and rewind to step 0."""
        self._step = 0
        for i, badge in enumerate(self._badges):
            badge.set_state("current" if i == 0 else "pending")

    # -- helpers --------------------------------------------------------------
    def _clear(self) -> None:
        for badge in self._badges:
            self._row.removeWidget(badge)
            badge.deleteLater()
        self._badges.clear()
