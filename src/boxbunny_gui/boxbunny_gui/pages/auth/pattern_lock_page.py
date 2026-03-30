"""Pattern lock screen for user authentication.

Supports both a 3x3 pattern lock (tap dots in sequence) and a
password text field as a fallback. User can toggle between modes.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, List

from PySide6.QtCore import Qt, QPoint, QRectF, QTimer
from PySide6.QtGui import QPainter, QPen, QColor
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from boxbunny_gui.theme import Color, Size, font, GHOST_BTN
from boxbunny_gui.widgets import BigButton

if TYPE_CHECKING:
    from boxbunny_gui.nav.router import PageRouter

logger = logging.getLogger(__name__)

_CORRECT_PATTERN = [0, 1, 2, 5, 8]
_CORRECT_PASSWORD = "boxing123"
_DOT_RADIUS = 22
_GRID_SIZE = 3
_CELL_SIZE = 80


class _PatternGrid(QWidget):
    """Custom-painted 3x3 pattern grid with touch/click support."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        total = _CELL_SIZE * _GRID_SIZE
        self.setFixedSize(total, total)
        self._entered: List[int] = []
        self._drawing = False
        self._current_pos: QPoint | None = None
        self._callback = None

    def set_callback(self, cb) -> None:
        self._callback = cb

    def reset(self) -> None:
        self._entered.clear()
        self._drawing = False
        self._current_pos = None
        self.update()

    @property
    def pattern(self) -> List[int]:
        return list(self._entered)

    def _dot_center(self, idx: int) -> QPoint:
        row, col = divmod(idx, _GRID_SIZE)
        x = col * _CELL_SIZE + _CELL_SIZE // 2
        y = row * _CELL_SIZE + _CELL_SIZE // 2
        return QPoint(x, y)

    def _hit_test(self, pos: QPoint) -> int:
        for i in range(9):
            center = self._dot_center(i)
            dx = pos.x() - center.x()
            dy = pos.y() - center.y()
            if dx * dx + dy * dy <= (_DOT_RADIUS + 10) ** 2:
                return i
        return -1

    def mousePressEvent(self, event) -> None:
        self._entered.clear()
        self._drawing = True
        self._handle_pos(event.position().toPoint())

    def mouseMoveEvent(self, event) -> None:
        if self._drawing:
            self._current_pos = event.position().toPoint()
            self._handle_pos(self._current_pos)

    def mouseReleaseEvent(self, event) -> None:
        self._drawing = False
        self._current_pos = None
        self.update()
        if self._callback and len(self._entered) >= 3:
            self._callback(self._entered)

    def _handle_pos(self, pos: QPoint) -> None:
        idx = self._hit_test(pos)
        if idx >= 0 and idx not in self._entered:
            self._entered.append(idx)
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        active_color = QColor(Color.PRIMARY)
        inactive_color = QColor(Color.SURFACE_HOVER)
        line_color = QColor(Color.PRIMARY)
        line_color.setAlpha(180)

        # Draw connecting lines
        if len(self._entered) > 1:
            pen = QPen(line_color, 4)
            painter.setPen(pen)
            for i in range(len(self._entered) - 1):
                p1 = self._dot_center(self._entered[i])
                p2 = self._dot_center(self._entered[i + 1])
                painter.drawLine(p1, p2)
            # Draw line to current mouse pos while dragging
            if self._drawing and self._current_pos:
                last = self._dot_center(self._entered[-1])
                painter.drawLine(last, self._current_pos)

        # Draw dots
        for i in range(9):
            center = self._dot_center(i)
            is_active = i in self._entered
            color = active_color if is_active else inactive_color
            painter.setPen(Qt.NoPen)
            painter.setBrush(color)
            r = _DOT_RADIUS if is_active else _DOT_RADIUS - 4
            painter.drawEllipse(center, r, r)

            # Inner dot for inactive
            if not is_active:
                painter.setBrush(QColor(Color.TEXT_DISABLED))
                painter.drawEllipse(center, 6, 6)

        painter.end()


class PatternLockPage(QWidget):
    """Authentication page with pattern lock and password fallback."""

    def __init__(self, router: PageRouter, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._router = router
        self._user_id: str = ""
        self._user_name: str = ""
        self._use_password: bool = False
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(40, 24, 40, 16)
        root.setSpacing(0)

        # User name with subtle icon
        self._name_lbl = QLabel()
        self._name_lbl.setFont(font(Size.TEXT_SUBHEADER, bold=True))
        self._name_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(self._name_lbl)

        root.addSpacing(6)

        # Status label in a subtle pill container
        self._status_lbl = QLabel("Draw your pattern")
        self._status_lbl.setStyleSheet(
            f"color: {Color.TEXT_SECONDARY}; font-size: 15px;"
            f" background-color: {Color.SURFACE}; border-radius: 12px;"
            " padding: 6px 20px;"
        )
        self._status_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_lbl.setWordWrap(True)
        self._status_lbl.setFixedHeight(36)
        root.addWidget(self._status_lbl, alignment=Qt.AlignmentFlag.AlignCenter)

        root.addStretch(2)

        # ── Pattern mode ─────────────────────────────────────────────────
        self._pattern_widget = QWidget()
        pattern_lay = QVBoxLayout(self._pattern_widget)
        pattern_lay.setContentsMargins(0, 0, 0, 0)
        pattern_lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pattern_lay.setSpacing(12)

        self._grid = _PatternGrid()
        self._grid.set_callback(self._on_pattern_complete)
        pattern_lay.addWidget(self._grid, alignment=Qt.AlignmentFlag.AlignCenter)

        hint = QLabel("Connect at least 3 dots")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint.setStyleSheet(
            f"color: {Color.TEXT_DISABLED}; font-size: 12px;"
            " letter-spacing: 0.3px;"
        )
        pattern_lay.addWidget(hint)

        root.addWidget(self._pattern_widget)

        # ── Password mode ────────────────────────────────────────────────
        self._password_widget = QWidget()
        self._password_widget.setObjectName("pwBox")
        self._password_widget.setStyleSheet(
            f"QWidget#pwBox {{ background-color: {Color.SURFACE};"
            f" border: 1px solid {Color.BORDER};"
            f" border-radius: 14px; }}"
        )
        self._password_widget.setFixedSize(360, 160)
        pw_lay = QVBoxLayout(self._password_widget)
        pw_lay.setContentsMargins(24, 20, 24, 20)
        pw_lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pw_lay.setSpacing(14)

        pw_icon_lbl = QLabel("Enter your password")
        pw_icon_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pw_icon_lbl.setStyleSheet(
            f"font-size: 13px; font-weight: 600; color: {Color.TEXT_SECONDARY};"
        )
        pw_lay.addWidget(pw_icon_lbl)

        self._pw_field = QLineEdit()
        self._pw_field.setPlaceholderText("Password")
        self._pw_field.setEchoMode(QLineEdit.EchoMode.Password)
        self._pw_field.setFixedHeight(46)
        self._pw_field.returnPressed.connect(self._check_password)
        pw_lay.addWidget(self._pw_field)

        pw_submit = QPushButton("Unlock")
        pw_submit.setCursor(Qt.CursorShape.PointingHandCursor)
        pw_submit.setFixedHeight(44)
        pw_submit.setStyleSheet(f"""
            QPushButton {{
                font-size: 16px; font-weight: 600;
                background-color: {Color.PRIMARY}; color: {Color.BG};
                border: none; border-radius: 12px;
            }}
            QPushButton:hover {{ background-color: {Color.PRIMARY_DARK}; }}
            QPushButton:pressed {{ background-color: {Color.PRIMARY_PRESSED}; }}
        """)
        pw_submit.clicked.connect(self._check_password)
        pw_lay.addWidget(pw_submit)

        self._password_widget.setVisible(False)
        root.addWidget(self._password_widget, alignment=Qt.AlignmentFlag.AlignCenter)

        root.addStretch(2)

        # ── Toggle + bottom row ──────────────────────────────────────────
        self._toggle_btn = QPushButton("Use password instead")
        self._toggle_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._toggle_btn.setStyleSheet(f"""
            QPushButton {{
                font-size: 13px; font-weight: 600;
                color: {Color.TEXT_SECONDARY};
                background-color: {Color.SURFACE}; border: 1px solid {Color.BORDER};
                border-radius: 8px; padding: 7px 18px;
                min-height: 0; min-width: 0;
            }}
            QPushButton:hover {{
                color: {Color.PRIMARY}; border-color: {Color.PRIMARY};
            }}
        """)
        self._toggle_btn.clicked.connect(self._toggle_mode)
        root.addWidget(self._toggle_btn, alignment=Qt.AlignmentFlag.AlignCenter)

        root.addSpacing(10)

        bottom = QHBoxLayout()
        self._btn_back = BigButton("Back", stylesheet=GHOST_BTN)
        self._btn_back.setFixedWidth(100)
        self._btn_back.clicked.connect(lambda: self._router.back())
        bottom.addWidget(self._btn_back)
        bottom.addStretch()
        root.addLayout(bottom)

    def _toggle_mode(self) -> None:
        self._use_password = not self._use_password
        self._pattern_widget.setVisible(not self._use_password)
        self._password_widget.setVisible(self._use_password)
        if self._use_password:
            self._status_lbl.setText("Enter your password")
            self._toggle_btn.setText("Use pattern instead")
            self._pw_field.clear()
            self._pw_field.setFocus()
        else:
            self._status_lbl.setText("Draw your pattern")
            self._toggle_btn.setText("Use password instead")
            self._grid.reset()
        self._status_lbl.setStyleSheet(
            f"color: {Color.TEXT_SECONDARY}; font-size: 15px;"
            f" background-color: {Color.SURFACE}; border-radius: 12px;"
            " padding: 6px 20px;"
        )

    def _on_pattern_complete(self, pattern: List[int]) -> None:
        if pattern == _CORRECT_PATTERN:
            logger.info("Pattern correct for user %s", self._user_id)
            self._status_lbl.setText("Unlocked!")
            self._status_lbl.setStyleSheet(
                f"color: {Color.PRIMARY}; font-size: 15px; font-weight: 600;"
                f" background-color: {Color.PRIMARY_MUTED}; border-radius: 12px;"
                " padding: 6px 20px;"
            )
            QTimer.singleShot(
                300,
                lambda: self._router.navigate(
                    "home", user_id=self._user_id,
                    username=self._user_name,
                ),
            )
        else:
            logger.info("Incorrect pattern for user %s", self._user_id)
            self._status_lbl.setText("Incorrect pattern \u2014 try again")
            self._status_lbl.setStyleSheet(
                f"color: {Color.DANGER}; font-size: 15px; font-weight: 600;"
                f" background-color: {Color.DANGER}18; border-radius: 12px;"
                " padding: 6px 20px;"
            )
            QTimer.singleShot(600, self._grid.reset)

    def _check_password(self) -> None:
        pw = self._pw_field.text()
        if pw == _CORRECT_PASSWORD:
            logger.info("Password correct for user %s", self._user_id)
            self._router.navigate(
                "home", user_id=self._user_id,
                username=self._user_name,
            )
        else:
            self._status_lbl.setText("Incorrect password \u2014 try again")
            self._status_lbl.setStyleSheet(
                f"color: {Color.DANGER}; font-size: 15px; font-weight: 600;"
                f" background-color: {Color.DANGER}18; border-radius: 12px;"
                " padding: 6px 20px;"
            )
            self._pw_field.clear()

    # ── Lifecycle ──────────────────────────────────────────────────────
    def on_enter(self, **kwargs: Any) -> None:
        self._user_id = kwargs.get("user_id", "")
        self._user_name = kwargs.get("user_name", "User")
        self._name_lbl.setText(self._user_name)
        self._use_password = False
        self._pattern_widget.setVisible(True)
        self._password_widget.setVisible(False)
        self._toggle_btn.setText("Use password instead")
        self._status_lbl.setText("Draw your pattern")
        self._status_lbl.setStyleSheet(
            f"color: {Color.TEXT_SECONDARY}; font-size: 15px;"
            f" background-color: {Color.SURFACE}; border-radius: 12px;"
            " padding: 6px 20px;"
        )
        self._grid.reset()
        self._pw_field.clear()
        logger.debug("PatternLockPage entered for user %s", self._user_id)

    def on_leave(self) -> None:
        self._grid.reset()
