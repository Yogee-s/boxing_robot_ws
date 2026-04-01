"""Pattern lock screen for user authentication.

Supports both a 3x3 pattern lock (tap dots in sequence) and a
password text field as a fallback. If the user has no pattern set,
only the password mode is shown.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, List, Optional

from PySide6.QtCore import Qt, QPoint, QRectF, QTimer, QPropertyAnimation, QEasingCurve
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


def _db_available() -> bool:
    """Check if the DB helper can reach the database."""
    try:
        from boxbunny_gui.db_helper import get_user_by_username
        return True
    except Exception:
        return False


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
        self._username: str = ""
        self._has_pattern: bool = True
        self._use_password: bool = False
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(40, 16, 40, 22)
        root.setSpacing(0)

        root.addStretch(1)

        # User name
        self._name_lbl = QLabel()
        self._name_lbl.setStyleSheet(
            f"font-size: 24px; font-weight: 700; color: {Color.TEXT};"
        )
        self._name_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(self._name_lbl)

        root.addSpacing(4)

        # Status label
        self._status_lbl = QLabel("Draw your pattern")
        self._status_lbl.setStyleSheet(
            f"color: {Color.TEXT_SECONDARY}; font-size: 14px;"
        )
        self._status_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(self._status_lbl, alignment=Qt.AlignmentFlag.AlignCenter)

        root.addSpacing(12)

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

        # ── Password mode (clean, no box) ────────────────────────────────
        self._password_widget = QWidget()
        pw_lay = QVBoxLayout(self._password_widget)
        pw_lay.setContentsMargins(0, 0, 0, 0)
        pw_lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pw_lay.setSpacing(12)

        self._pw_field = QLineEdit()
        self._pw_field.setPlaceholderText("Password")
        self._pw_field.setEchoMode(QLineEdit.EchoMode.Password)
        self._pw_field.setFixedSize(320, 48)
        self._pw_field.returnPressed.connect(self._check_password)
        pw_lay.addWidget(self._pw_field, alignment=Qt.AlignmentFlag.AlignCenter)

        pw_submit = QPushButton("Unlock")
        pw_submit.setCursor(Qt.CursorShape.PointingHandCursor)
        pw_submit.setFixedSize(320, 48)
        pw_submit.setStyleSheet(f"""
            QPushButton {{
                font-size: 16px; font-weight: 700;
                background-color: {Color.PRIMARY}; color: #FFFFFF;
                border: none; border-radius: 12px;
            }}
            QPushButton:hover {{ background-color: {Color.PRIMARY_DARK}; }}
            QPushButton:pressed {{ background-color: {Color.PRIMARY_PRESSED}; }}
        """)
        pw_submit.clicked.connect(self._check_password)
        pw_lay.addWidget(pw_submit, alignment=Qt.AlignmentFlag.AlignCenter)

        self._password_widget.setMaximumHeight(0)
        root.addWidget(self._password_widget, alignment=Qt.AlignmentFlag.AlignCenter)

        # Animations for smooth mode switching
        self._pat_anim = QPropertyAnimation(self._pattern_widget, b"maximumHeight")
        self._pat_anim.setDuration(250)
        self._pat_anim.setEasingCurve(QEasingCurve.Type.InOutCubic)

        self._pw_anim = QPropertyAnimation(self._password_widget, b"maximumHeight")
        self._pw_anim.setDuration(250)
        self._pw_anim.setEasingCurve(QEasingCurve.Type.InOutCubic)

        self._pat_full_h = 300  # pattern grid + hint
        self._pw_full_h = 120   # password field + unlock button

        root.addSpacing(12)

        # ── Toggle + bottom row ──────────────────────────────────────────
        self._toggle_btn = QPushButton("Use password instead")
        self._toggle_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._toggle_btn.setFixedHeight(48)
        self._toggle_btn.setStyleSheet(f"""
            QPushButton {{
                font-size: 15px; font-weight: 600;
                color: {Color.TEXT_SECONDARY};
                background-color: {Color.SURFACE}; border: 1px solid {Color.BORDER};
                border-radius: 10px; padding: 10px 24px;
            }}
            QPushButton:hover {{
                color: {Color.PRIMARY}; border-color: {Color.PRIMARY};
            }}
        """)
        self._toggle_btn.clicked.connect(self._toggle_mode)
        root.addWidget(self._toggle_btn, alignment=Qt.AlignmentFlag.AlignCenter)

        root.addStretch(1)

        bottom = QHBoxLayout()
        self._btn_back = BigButton("Back", stylesheet=GHOST_BTN)
        self._btn_back.setFixedWidth(120)
        self._btn_back.setFixedHeight(48)
        self._btn_back.clicked.connect(lambda: self._router.back())
        bottom.addWidget(self._btn_back)
        bottom.addStretch()
        root.addLayout(bottom)

    def _toggle_mode(self) -> None:
        self._use_password = not self._use_password
        self._pat_anim.stop()
        self._pw_anim.stop()

        if self._use_password:
            # Collapse pattern, expand password
            self._pat_anim.setStartValue(self._pattern_widget.height())
            self._pat_anim.setEndValue(0)
            self._pat_anim.start()
            self._pw_anim.setStartValue(0)
            self._pw_anim.setEndValue(self._pw_full_h)
            self._pw_anim.start()
            self._status_lbl.setText("Enter your password")
            self._toggle_btn.setText("Use pattern instead")
            self._pw_field.clear()
            QTimer.singleShot(260, self._pw_field.setFocus)
        else:
            # Collapse password, expand pattern
            self._pw_anim.setStartValue(self._password_widget.height())
            self._pw_anim.setEndValue(0)
            self._pw_anim.start()
            self._pat_anim.setStartValue(0)
            self._pat_anim.setEndValue(self._pat_full_h)
            self._pat_anim.start()
            self._status_lbl.setText("Draw your pattern")
            self._toggle_btn.setText("Use password instead")
            self._grid.reset()
        self._reset_status_style()

    def _reset_status_style(self) -> None:
        self._status_lbl.setStyleSheet(
            f"color: {Color.TEXT_SECONDARY}; font-size: 14px;"
        )

    def _on_pattern_complete(self, pattern: List[int]) -> None:
        verified = False
        try:
            from boxbunny_gui.db_helper import verify_pattern
            user_id_int = int(self._user_id)
            verified = verify_pattern(user_id_int, pattern)
        except Exception as exc:
            logger.debug("DB pattern verify failed: %s", exc)

        # Fall back to hardcoded check
        if not verified and not _db_available():
            verified = (pattern == _CORRECT_PATTERN)

        if verified:
            logger.info("Pattern correct for user %s", self._user_id)
            self._status_lbl.setText("Unlocked!")
            self._status_lbl.setStyleSheet(
                f"color: {Color.PRIMARY}; font-size: 14px; font-weight: 600;"
            )
            QTimer.singleShot(
                300,
                lambda: self._router.navigate(
                    "home", user_id=self._user_id,
                    username=self._username or self._user_name,
                ),
            )
        else:
            logger.info("Incorrect pattern for user %s", self._user_id)
            self._status_lbl.setText("Incorrect pattern \u2013 try again")
            self._status_lbl.setStyleSheet(
                f"color: {Color.DANGER}; font-size: 14px; font-weight: 600;"
            )
            QTimer.singleShot(600, self._grid.reset)

    def _check_password(self) -> None:
        pw = self._pw_field.text()

        verified = False
        if self._username:
            try:
                from boxbunny_gui.db_helper import verify_password
                result = verify_password(self._username, pw)
                verified = result is not None
            except Exception as exc:
                logger.debug("DB password verify failed: %s", exc)

        # Fall back to hardcoded check
        if not verified and not _db_available():
            verified = (pw == _CORRECT_PASSWORD)

        if verified:
            logger.info("Password correct for user %s", self._user_id)
            self._router.navigate(
                "home", user_id=self._user_id,
                username=self._username or self._user_name,
            )
        else:
            self._status_lbl.setText("Incorrect password \u2013 try again")
            self._status_lbl.setStyleSheet(
                f"color: {Color.DANGER}; font-size: 14px; font-weight: 600;"
            )
            self._pw_field.clear()

    # ── Lifecycle ──────────────────────────────────────────────────────
    def on_enter(self, **kwargs: Any) -> None:
        self._user_id = kwargs.get("user_id", "")
        self._user_name = kwargs.get("user_name", "User")
        self._username = kwargs.get("username", "")
        self._has_pattern = kwargs.get("has_pattern", True)
        self._name_lbl.setText(self._user_name)
        self._grid.reset()
        self._pw_field.clear()

        if self._has_pattern:
            # User has pattern — show pattern mode by default
            self._use_password = False
            self._pattern_widget.setMaximumHeight(self._pat_full_h)
            self._password_widget.setMaximumHeight(0)
            self._toggle_btn.setText("Use password instead")
            self._toggle_btn.setVisible(True)
            self._status_lbl.setText("Draw your pattern")
        else:
            # User has no pattern — go straight to password mode
            self._use_password = True
            self._pattern_widget.setMaximumHeight(0)
            self._password_widget.setMaximumHeight(self._pw_full_h)
            self._toggle_btn.setVisible(False)
            self._status_lbl.setText("Enter your password")
            self._pw_field.setFocus()

        self._reset_status_style()
        logger.debug("PatternLockPage entered for user %s (has_pattern=%s)",
                      self._user_id, self._has_pattern)

    def on_leave(self) -> None:
        self._grid.reset()
