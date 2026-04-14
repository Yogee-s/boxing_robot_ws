"""Built-in virtual keyboard for touchscreen text input.

Renders a QWERTY keyboard as a Qt widget overlaid on the main window.
All keys use Qt.NoFocus so they never steal focus from the active
QLineEdit.  Text is inserted directly into the target field.
"""
from __future__ import annotations

import logging
from typing import Optional

from PySide6.QtCore import Qt, QEvent, QObject
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import (
    QApplication,
    QGraphicsProxyWidget,
    QGridLayout,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)

# ── Layout constants ────────────────────────────────────────────────────

_KEY_H = 44
_VGAP = 3
_HGAP = 3
_KB_H = (5 * _KEY_H) + (4 * _VGAP) + 14  # total keyboard height

# Each key: (label, col_span).  Every row sums to 20 grid columns.

_ROWS_LOWER = [
    [("1", 2), ("2", 2), ("3", 2), ("4", 2), ("5", 2),
     ("6", 2), ("7", 2), ("8", 2), ("9", 2), ("0", 2)],
    [("q", 2), ("w", 2), ("e", 2), ("r", 2), ("t", 2),
     ("y", 2), ("u", 2), ("i", 2), ("o", 2), ("p", 2)],
    [("a", 2), ("s", 2), ("d", 2), ("f", 2), ("g", 2),
     ("h", 2), ("j", 2), ("k", 2), ("l", 2), ("BKSP", 2)],
    [("SHIFT", 3), ("z", 2), ("x", 2), ("c", 2), ("v", 2),
     ("b", 2), ("n", 2), ("m", 2), ("ENTER", 3)],
    [("SYM", 3), (",", 2), ("SPACE", 10), (".", 2), ("HIDE", 3)],
]

_ROWS_UPPER = [
    [("1", 2), ("2", 2), ("3", 2), ("4", 2), ("5", 2),
     ("6", 2), ("7", 2), ("8", 2), ("9", 2), ("0", 2)],
    [("Q", 2), ("W", 2), ("E", 2), ("R", 2), ("T", 2),
     ("Y", 2), ("U", 2), ("I", 2), ("O", 2), ("P", 2)],
    [("A", 2), ("S", 2), ("D", 2), ("F", 2), ("G", 2),
     ("H", 2), ("J", 2), ("K", 2), ("L", 2), ("BKSP", 2)],
    [("SHIFT", 3), ("Z", 2), ("X", 2), ("C", 2), ("V", 2),
     ("B", 2), ("N", 2), ("M", 2), ("ENTER", 3)],
    [("SYM", 3), (",", 2), ("SPACE", 10), (".", 2), ("HIDE", 3)],
]

_ROWS_SYM = [
    [("!", 2), ("@", 2), ("#", 2), ("$", 2), ("%", 2),
     ("^", 2), ("&", 2), ("*", 2), ("(", 2), (")", 2)],
    [("-", 2), ("_", 2), ("=", 2), ("+", 2), ("[", 2),
     ("]", 2), ("{", 2), ("}", 2), ("|", 2), ("\\", 2)],
    [(";", 2), (":", 2), ("'", 2), ("\"", 2), ("<", 2),
     (">", 2), ("/", 2), ("?", 2), ("~", 2), ("BKSP", 2)],
    [("ABC", 3), ("`", 2), (",", 2), ("SPACE", 6), (".", 2),
     ("@", 2), ("ENTER", 3)],
    [("ABC", 3), (",", 2), ("SPACE", 10), (".", 2), ("HIDE", 3)],
]

# ── Styles ──────────────────────────────────────────────────────────────


def _style(bg: str, pressed: str, color: str = "#E6EDF3",
           size: int = 15, bold: bool = False) -> str:
    w = 600 if bold else 400
    return (
        f"QPushButton {{ background:{bg}; color:{color}; border:none;"
        f" border-radius:5px; font-size:{size}px; font-weight:{w}; }}"
        f"QPushButton:pressed {{ background:{pressed}; }}"
    )


_S_KEY = _style("#2A3340", "#3D4D60")
_S_SHIFT = _style("#2A3340", "#3D4D60", bold=True)
_S_SHIFT_ON = _style("#FF6B35", "#E85E2C", color="#FFF", bold=True)
_S_FN = _style("#1E2A38", "#2D3D50", size=13, bold=True)
_S_BKSP = _style("#3A2525", "#4D3333", size=17)
_S_ENTER = _style("#1A3A22", "#2A5533", color="#56D364", size=17, bold=True)
_S_SPACE = _style("#2A3340", "#3D4D60", size=13)
_S_HIDE = _style("#1E2A38", "#2D3D50", size=13, bold=True)


# ── Keyboard widget ────────────────────────────────────────────────────


class VirtualKeyboard(QWidget):
    """On-screen QWERTY keyboard widget."""

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self._shifted = False
        self._sym = False
        self._target_field: Optional[QLineEdit] = None
        self._grid: Optional[QGridLayout] = None
        self._buttons: list[QPushButton] = []
        self._build_ui()
        self.hide()

    def set_target(self, field: Optional[QLineEdit]) -> None:
        self._target_field = field

    def _build_ui(self) -> None:
        self.setStyleSheet(
            "VirtualKeyboard { background: #141A22;"
            " border-top: 2px solid #FF6B3560; }"
        )
        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 5, 4, 5)
        outer.setSpacing(0)
        self._grid = QGridLayout()
        self._grid.setSpacing(_HGAP)
        self._grid.setContentsMargins(0, 0, 0, 0)
        outer.addLayout(self._grid)
        self._populate(_ROWS_LOWER)

    def _clear(self) -> None:
        for btn in self._buttons:
            btn.deleteLater()
        self._buttons.clear()
        while self._grid.count():
            self._grid.takeAt(0)

    def _populate(self, rows: list[list[tuple[str, int]]]) -> None:
        self._clear()
        for r, row in enumerate(rows):
            col = 0
            for label, span in row:
                btn = self._make_btn(label)
                self._grid.addWidget(btn, r, col, 1, span)
                self._buttons.append(btn)
                col += span

    def _make_btn(self, label: str) -> QPushButton:
        text = label
        style = _S_KEY

        if label == "SPACE":
            text = "space"
            style = _S_SPACE
        elif label == "BKSP":
            text = "\u232b"
            style = _S_BKSP
        elif label == "ENTER":
            text = "\u21b5"
            style = _S_ENTER
        elif label == "SHIFT":
            text = "\u21e7"
            style = _S_SHIFT_ON if self._shifted else _S_SHIFT
        elif label in ("SYM", "ABC"):
            text = "?123" if label == "SYM" else "ABC"
            style = _S_FN
        elif label == "HIDE":
            text = "\u2715"
            style = _S_HIDE

        btn = QPushButton(text)
        btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        btn.setFixedHeight(_KEY_H)
        btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        btn.setStyleSheet(style)
        btn.pressed.connect(lambda k=label: self._on_key(k))
        return btn

    def _on_key(self, key: str) -> None:
        if key == "HIDE":
            self.hide()
            return
        if key == "SHIFT":
            self._shifted = not self._shifted
            self._sym = False
            self._populate(_ROWS_UPPER if self._shifted else _ROWS_LOWER)
            return
        if key in ("SYM", "ABC"):
            if key == "SYM":
                self._sym = True
                self._shifted = False
                self._populate(_ROWS_SYM)
            else:
                self._sym = False
                self._populate(_ROWS_UPPER if self._shifted else _ROWS_LOWER)
            return

        target = self._target_field
        if target is None:
            return

        if key == "BKSP":
            target.backspace()
        elif key == "ENTER":
            target.returnPressed.emit()
            self.hide()
        elif key == "SPACE":
            target.insert(" ")
        else:
            target.insert(key)
            if self._shifted and not self._sym:
                self._shifted = False
                self._populate(_ROWS_LOWER)


# ── Manager ─────────────────────────────────────────────────────────────


class VirtualKeyboardManager(QObject):
    """Show / hide the built-in keyboard when text fields are tapped.

    Detection strategy: on every MouseButtonPress the event filter
    checks whether ``obj`` (the event target) is a QLineEdit.  The
    QGraphicsProxy forwards synthetic mouse events to embedded widgets
    via QApplication.sendEvent, so the filter sees the real target.
    No focus-based detection needed.
    """

    def __init__(
        self,
        keyboard: VirtualKeyboard,
        proxy: QGraphicsProxyWidget,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._keyboard = keyboard
        self._proxy = proxy
        self._window = parent

    def install(self, app: QApplication) -> None:
        app.installEventFilter(self)

    def shutdown(self) -> None:
        self._dismiss()

    def on_page_changed(self, _name: str) -> None:
        """Hide keyboard whenever the page changes."""
        self._dismiss()

    # ── event filter ────────────────────────────────────────────────────

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: N802
        if event.type() != QEvent.Type.MouseButtonPress:
            return False

        mouse: QMouseEvent = event  # type: ignore[assignment]
        global_y = mouse.globalPosition().toPoint().y()

        if self._keyboard.isVisible():
            # Keyboard is showing -- check if click is ABOVE it
            kb_top = self._keyboard.mapToGlobal(
                self._keyboard.rect().topLeft()
            ).y()
            if global_y < kb_top:
                # Clicked above the keyboard area
                line_edit = self._resolve_line_edit(obj)
                if line_edit is not None:
                    # Tapped a different text field -- retarget
                    self._keyboard.set_target(line_edit)
                else:
                    # Tapped empty space above -- hide keyboard
                    self._dismiss()
            # Click is inside the keyboard area -- do nothing, let keys work
            return False

        # Keyboard is hidden -- check if a QLineEdit was tapped
        line_edit = self._resolve_line_edit(obj)
        if line_edit is not None:
            self._keyboard.set_target(line_edit)
            self._show()

        return False

    # ── helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_line_edit(obj: QObject) -> Optional[QLineEdit]:
        """Walk up from obj to find a QLineEdit (handles child widgets)."""
        widget = obj if isinstance(obj, QWidget) else None
        while widget is not None:
            if isinstance(widget, QLineEdit) and not widget.isReadOnly():
                return widget
            widget = widget.parentWidget()
        return None

    def _show(self) -> None:
        w = self._window
        if w is None:
            return
        pw = w.width()
        ph = w.height()
        self._keyboard.setGeometry(0, ph - _KB_H, pw, _KB_H)
        self._keyboard.raise_()
        self._keyboard.show()

    def _dismiss(self) -> None:
        self._keyboard.set_target(None)
        self._keyboard.hide()
