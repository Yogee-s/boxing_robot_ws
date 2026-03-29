"""Android-style 3x3 pattern lock supporting both touch and IMU/keyboard input.

The user draws a path across 9 dots; on release the ``pattern_entered``
signal fires with the ordered list of dot indices (0-8, row-major).
"""

from __future__ import annotations

import logging
from typing import Optional

from PySide6.QtCore import QPointF, Qt, Signal, QTimer
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from boxbunny_gui.theme import Color, Size

log = logging.getLogger(__name__)

_ROWS = 3
_COLS = 3
_DOT_RADIUS = 22
_DOT_RADIUS_ACTIVE = 28
_HIT_RADIUS = 48  # generous touch hit-test area for gloved fingers


class PatternLock(QWidget):
    """A 3x3 dot-pattern lock widget.

    Supports two input modes:

    * **Touch / mouse** -- drag across dots to form a path.
    * **Keyboard / IMU** -- arrow keys move a cursor highlight;
      Enter / Space toggles a dot into the pattern.

    Signals
    -------
    pattern_entered(list)
        Emitted with an ordered ``list[int]`` of selected dot indices.
    """

    pattern_entered = Signal(list)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._enabled: bool = True
        self._selected: list[int] = []
        self._cursor: int = 4  # centre dot for keyboard nav
        self._dragging: bool = False
        self._drag_pos: Optional[QPointF] = None  # current finger position for trailing line

        self.setMinimumSize(220, 220)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)  # receive mouseMoveEvent even without button pressed
        self.setAttribute(Qt.WidgetAttribute.WA_AcceptTouchEvents, True)

        # pulse timer for cursor blink
        self._blink_on: bool = True
        self._blink_timer = QTimer(self)
        self._blink_timer.setInterval(500)
        self._blink_timer.timeout.connect(self._toggle_blink)
        self._blink_timer.start()

    # -- public API -----------------------------------------------------------
    def reset(self) -> None:
        """Clear the current pattern."""
        self._selected.clear()
        self._cursor = 4
        self.update()

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable interaction."""
        self._enabled = enabled
        self.update()

    # -- geometry helpers -----------------------------------------------------
    def _dot_center(self, index: int) -> QPointF:
        """Return the pixel centre of dot *index* (0-8, row-major)."""
        row, col = divmod(index, _COLS)
        cell_w = self.width() / _COLS
        cell_h = self.height() / _ROWS
        return QPointF(col * cell_w + cell_w / 2, row * cell_h + cell_h / 2)

    def _hit_test(self, pos: QPointF) -> Optional[int]:
        for i in range(_ROWS * _COLS):
            c = self._dot_center(i)
            if (pos - c).manhattanLength() < _HIT_RADIUS:
                return i
        return None

    # -- touch / mouse --------------------------------------------------------
    def mousePressEvent(self, event) -> None:  # noqa: N802
        if not self._enabled:
            return
        self._selected.clear()
        self._dragging = True
        self._drag_pos = event.position()
        self._try_add(event.position())
        self.update()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._dragging:
            self._drag_pos = event.position()
            self._try_add(event.position())
            self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._dragging:
            self._dragging = False
            self._drag_pos = None
            if self._selected:
                self.pattern_entered.emit(list(self._selected))
            self.update()

    def _try_add(self, pos: QPointF) -> None:
        idx = self._hit_test(pos)
        if idx is not None and idx not in self._selected:
            self._selected.append(idx)

    # -- keyboard / IMU -------------------------------------------------------
    def keyPressEvent(self, event) -> None:  # noqa: N802
        if not self._enabled:
            return
        key = event.key()
        row, col = divmod(self._cursor, _COLS)

        if key == Qt.Key.Key_Up and row > 0:
            self._cursor -= _COLS
        elif key == Qt.Key.Key_Down and row < _ROWS - 1:
            self._cursor += _COLS
        elif key == Qt.Key.Key_Left and col > 0:
            self._cursor -= 1
        elif key == Qt.Key.Key_Right and col < _COLS - 1:
            self._cursor += 1
        elif key in (Qt.Key.Key_Return, Qt.Key.Key_Space):
            if self._cursor not in self._selected:
                self._selected.append(self._cursor)
            else:
                # double-enter on last dot = submit
                if self._selected:
                    self.pattern_entered.emit(list(self._selected))
                return
        elif key == Qt.Key.Key_Escape:
            self.reset()
            return
        elif key == Qt.Key.Key_Backspace:
            if self._selected:
                self._selected.pop()
        else:
            super().keyPressEvent(event)
            return

        self.update()

    # -- blink ----------------------------------------------------------------
    def _toggle_blink(self) -> None:
        self._blink_on = not self._blink_on
        self.update()

    # -- painting -------------------------------------------------------------
    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        selected_set = set(self._selected)

        # draw path lines between selected dots
        if len(self._selected) > 1:
            pen = QPen(QColor(Color.PRIMARY), 5)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            p.setPen(pen)
            for i in range(len(self._selected) - 1):
                p.drawLine(self._dot_center(self._selected[i]),
                           self._dot_center(self._selected[i + 1]))

        # draw trailing line from last selected dot to current finger position
        if self._dragging and self._selected and self._drag_pos is not None:
            trail_pen = QPen(QColor(Color.PRIMARY), 3, Qt.PenStyle.DashLine)
            trail_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            p.setPen(trail_pen)
            p.drawLine(self._dot_center(self._selected[-1]), self._drag_pos)

        # draw dots
        for idx in range(_ROWS * _COLS):
            center = self._dot_center(idx)
            is_selected = idx in selected_set
            is_cursor = idx == self._cursor and not self._dragging

            # outer ring for cursor
            if is_cursor and self._blink_on:
                ring_pen = QPen(QColor(Color.PRIMARY), 2, Qt.PenStyle.DashLine)
                p.setPen(ring_pen)
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawEllipse(center, _DOT_RADIUS + 8, _DOT_RADIUS + 8)

            # dot fill
            radius = _DOT_RADIUS_ACTIVE if is_selected else _DOT_RADIUS
            color = QColor(Color.PRIMARY) if is_selected else QColor(Color.SURFACE_LIGHT)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(color)
            p.drawEllipse(center, radius, radius)

            # white border on selected dots
            if is_selected:
                border_pen = QPen(QColor(Color.TEXT), 2)
                p.setPen(border_pen)
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawEllipse(center, radius, radius)

        p.end()
