"""Sign up page — create a new account.

Single centered column: fields, auth toggle, pattern/password, create button.
Pattern is default. Compact layout to fit 1024x600 screen.
"""
from __future__ import annotations

import logging
from typing import Any, List

from PySide6.QtCore import Qt, QPoint, QTimer, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QPainter, QPen, QColor
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget,
)

from boxbunny_gui.theme import Color, Size, back_link_style

logger = logging.getLogger(__name__)

_DOT_RADIUS = 18
_GRID_SIZE = 3
_CELL_SIZE = 66


class _PatternGrid(QWidget):
    """Compact pattern grid for signup."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        total = _CELL_SIZE * _GRID_SIZE
        self.setFixedSize(total, total)
        self._entered: List[int] = []
        self._drawing = False
        self._current_pos: QPoint | None = None

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
            c = self._dot_center(i)
            dx, dy = pos.x() - c.x(), pos.y() - c.y()
            if dx * dx + dy * dy <= (_DOT_RADIUS + 8) ** 2:
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

        if len(self._entered) > 1:
            line_color = QColor(Color.PRIMARY)
            line_color.setAlpha(180)
            painter.setPen(QPen(line_color, 3))
            for i in range(len(self._entered) - 1):
                painter.drawLine(
                    self._dot_center(self._entered[i]),
                    self._dot_center(self._entered[i + 1]),
                )
            if self._drawing and self._current_pos:
                painter.drawLine(
                    self._dot_center(self._entered[-1]), self._current_pos
                )

        for i in range(9):
            c = self._dot_center(i)
            is_active = i in self._entered
            painter.setPen(Qt.NoPen)
            painter.setBrush(active_color if is_active else inactive_color)
            r = _DOT_RADIUS if is_active else _DOT_RADIUS - 3
            painter.drawEllipse(c, r, r)
            if not is_active:
                painter.setBrush(QColor(Color.TEXT_DISABLED))
                painter.drawEllipse(c, 5, 5)

        painter.end()


class SignupPage(QWidget):
    """Account creation — single centered column."""

    def __init__(self, router=None, db=None, **kwargs):
        super().__init__()
        self._router = router
        self._db = db
        self._use_pattern: bool = True

        root = QVBoxLayout(self)
        root.setSpacing(0)
        root.setContentsMargins(60, 12, 60, 22)

        # ── Top bar ──────────────────────────────────────────────────────
        top = QHBoxLayout()
        back_btn = QPushButton("\u2190  Back")
        back_btn.setStyleSheet(back_link_style())
        back_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        back_btn.clicked.connect(lambda: self._nav("auth"))
        top.addWidget(back_btn)

        title = QLabel("Create Account")
        title.setStyleSheet(
            f"font-size: 22px; font-weight: 700; color: {Color.TEXT};"
        )
        top.addWidget(title)
        top.addStretch()
        root.addLayout(top)

        root.addStretch(1)

        # ── Centered form column ─────────────────────────────────────────
        self._username = self._make_field(root, "Username")
        root.addSpacing(8)
        self._display_name = self._make_field(root, "Display Name")
        root.addSpacing(12)

        # Auth toggle
        toggle_row = QHBoxLayout()
        toggle_row.setAlignment(Qt.AlignCenter)
        toggle_row.setSpacing(0)
        self._pat_toggle = QPushButton("Pattern")
        self._pw_toggle = QPushButton("Password")
        for btn in (self._pat_toggle, self._pw_toggle):
            btn.setFixedSize(150, 46)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._pat_toggle.clicked.connect(lambda: self._set_auth_mode(True))
        self._pw_toggle.clicked.connect(lambda: self._set_auth_mode(False))
        toggle_row.addWidget(self._pat_toggle)
        toggle_row.addWidget(self._pw_toggle)
        root.addLayout(toggle_row)
        root.addSpacing(10)

        # ── Pattern container ────────────────────────────────────────────
        self._pat_container = QWidget()
        pat_lay = QVBoxLayout(self._pat_container)
        pat_lay.setContentsMargins(0, 0, 0, 0)
        pat_lay.setAlignment(Qt.AlignCenter)
        pat_lay.setSpacing(6)
        self._pattern_grid = _PatternGrid()
        pat_lay.addWidget(self._pattern_grid, alignment=Qt.AlignCenter)
        pat_hint = QLabel("Connect at least 3 dots")
        pat_hint.setAlignment(Qt.AlignCenter)
        pat_hint.setStyleSheet(
            f"font-size: 11px; color: {Color.TEXT_DISABLED};"
        )
        pat_lay.addWidget(pat_hint)
        root.addWidget(self._pat_container)

        # ── Password container ───────────────────────────────────────────
        self._pw_container = QWidget()
        pw_lay = QVBoxLayout(self._pw_container)
        pw_lay.setContentsMargins(0, 0, 0, 0)
        pw_lay.setAlignment(Qt.AlignCenter)
        pw_lay.setSpacing(6)
        self._password = QLineEdit()
        self._password.setPlaceholderText("Password (min 4 chars)")
        self._password.setFixedSize(400, 44)
        self._password.setEchoMode(QLineEdit.EchoMode.Password)
        pw_lay.addWidget(self._password, alignment=Qt.AlignCenter)
        pw_hint = QLabel("You can set a pattern later in settings")
        pw_hint.setAlignment(Qt.AlignCenter)
        pw_hint.setStyleSheet(
            f"font-size: 11px; color: {Color.TEXT_DISABLED};"
        )
        pw_lay.addWidget(pw_hint)
        self._pw_container.setMaximumHeight(0)
        root.addWidget(self._pw_container)

        # Height animations for smooth switching
        self._pat_anim = QPropertyAnimation(self._pat_container, b"maximumHeight")
        self._pat_anim.setDuration(250)
        self._pat_anim.setEasingCurve(QEasingCurve.Type.InOutCubic)

        self._pw_anim = QPropertyAnimation(self._pw_container, b"maximumHeight")
        self._pw_anim.setDuration(250)
        self._pw_anim.setEasingCurve(QEasingCurve.Type.InOutCubic)

        _PAT_H = _CELL_SIZE * _GRID_SIZE + 30  # grid + hint
        _PW_H = 70  # field + hint
        self._pat_expanded = _PAT_H
        self._pw_expanded = _PW_H

        root.addSpacing(12)

        # ── Create button ────────────────────────────────────────────────
        create_btn = QPushButton("Create Account")
        create_btn.setFixedSize(400, 48)
        create_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        create_btn.setStyleSheet(f"""
            QPushButton {{
                font-size: 18px; font-weight: 700;
                background-color: {Color.PRIMARY}; color: #FFFFFF;
                border: none; border-radius: 14px;
            }}
            QPushButton:hover {{ background-color: {Color.PRIMARY_DARK}; }}
            QPushButton:pressed {{ background-color: {Color.PRIMARY_PRESSED}; }}
        """)
        create_btn.clicked.connect(self._on_create)
        root.addWidget(create_btn, alignment=Qt.AlignCenter)

        root.addSpacing(4)

        # ── Status ───────────────────────────────────────────────────────
        self._status = QLabel("")
        self._status.setAlignment(Qt.AlignCenter)
        self._status.setStyleSheet(f"font-size: 13px; color: {Color.DANGER};")
        self._status.setWordWrap(True)
        self._status.setMaximumWidth(400)
        root.addWidget(self._status, alignment=Qt.AlignCenter)

        root.addStretch(1)

        # Initial state
        self._apply_toggle_style()

    def _make_field(self, layout: QVBoxLayout, placeholder: str) -> QLineEdit:
        field = QLineEdit()
        field.setPlaceholderText(placeholder)
        field.setFixedSize(400, 44)
        layout.addWidget(field, alignment=Qt.AlignCenter)
        return field

    def _set_auth_mode(self, use_pattern: bool) -> None:
        if use_pattern == self._use_pattern:
            return
        self._use_pattern = use_pattern
        self._pattern_grid.reset()
        self._status.setText("")
        self._apply_toggle_style()

        # Animate: collapse one, expand the other
        self._pat_anim.stop()
        self._pw_anim.stop()

        if use_pattern:
            self._pw_anim.setStartValue(self._pw_container.height())
            self._pw_anim.setEndValue(0)
            self._pw_anim.start()
            self._pat_anim.setStartValue(0)
            self._pat_anim.setEndValue(self._pat_expanded)
            self._pat_anim.start()
        else:
            self._pat_anim.setStartValue(self._pat_container.height())
            self._pat_anim.setEndValue(0)
            self._pat_anim.start()
            self._pw_anim.setStartValue(0)
            self._pw_anim.setEndValue(self._pw_expanded)
            self._pw_anim.start()
            self._password.clear()
            QTimer.singleShot(260, self._password.setFocus)

    def _apply_toggle_style(self) -> None:
        active = f"""
            QPushButton {{
                font-size: 13px; font-weight: 600;
                background-color: {Color.PRIMARY}; color: #FFFFFF;
                border: none; border-radius: {Size.RADIUS}px;
            }}
            QPushButton:hover {{ background-color: {Color.PRIMARY_DARK}; }}
        """
        inactive = f"""
            QPushButton {{
                font-size: 13px; font-weight: 600;
                background-color: {Color.SURFACE}; color: {Color.TEXT_SECONDARY};
                border: 1px solid {Color.BORDER}; border-radius: {Size.RADIUS}px;
            }}
            QPushButton:hover {{
                color: {Color.TEXT}; border-color: {Color.PRIMARY};
                background-color: {Color.SURFACE_HOVER};
            }}
        """
        self._pat_toggle.setStyleSheet(
            active if self._use_pattern else inactive
        )
        self._pw_toggle.setStyleSheet(
            inactive if self._use_pattern else active
        )

    def _on_create(self) -> None:
        username = self._username.text().strip()
        display = self._display_name.text().strip() or username

        if not username:
            self._status.setText("Please enter a username")
            return

        if self._use_pattern:
            pattern = self._pattern_grid.pattern
            if len(pattern) < 3:
                self._status.setText("Pattern must connect at least 3 dots")
                return
            auth_data = ",".join(str(d) for d in pattern)
        else:
            password = self._password.text()
            if len(password) < 4:
                self._status.setText("Password must be at least 4 characters")
                return
            auth_data = password

        if self._db:
            user_id = self._db.create_user(
                username, auth_data, display, "individual", "beginner"
            )
            if user_id is None:
                self._status.setText("Username already taken")
                return

        self._status.setStyleSheet(f"font-size: 13px; color: {Color.PRIMARY};")
        self._status.setText(f"Account created! Welcome, {display}")
        logger.info("Created user: %s", username)
        QTimer.singleShot(
            800, lambda: self._nav("guest_assessment", username=username)
        )

    def _nav(self, page: str, **kwargs):
        if self._router:
            self._router.navigate(page, **kwargs)

    def on_enter(self, **kwargs: Any) -> None:
        self._username.clear()
        self._display_name.clear()
        self._password.clear()
        self._pattern_grid.reset()
        self._status.setText("")
        self._status.setStyleSheet(f"font-size: 13px; color: {Color.DANGER};")
        self._use_pattern = True
        self._apply_toggle_style()
        self._pat_container.setMaximumHeight(self._pat_expanded)
        self._pw_container.setMaximumHeight(0)

    def on_leave(self) -> None:
        pass
