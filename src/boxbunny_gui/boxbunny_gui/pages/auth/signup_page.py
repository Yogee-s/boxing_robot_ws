"""Sign up page — create a new account.

Simple form: username, display name, password or pattern, level selection.
Pattern is the default auth method (more user-friendly with gloves).
"""
from __future__ import annotations

import logging
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget,
)

logger = logging.getLogger(__name__)


class SignupPage(QWidget):
    """Account creation with username, display name, and pattern or password."""

    def __init__(self, router=None, db=None, **kwargs):
        super().__init__()
        self._router = router
        self._db = db

        root = QVBoxLayout(self)
        root.setSpacing(0)
        root.setContentsMargins(80, 30, 80, 30)

        # ── Top bar ──────────────────────────────────────────────────────
        top = QHBoxLayout()
        back_btn = QPushButton("Back")
        back_btn.setStyleSheet("""
            QPushButton {
                font-size: 16px; padding: 8px 20px;
                background-color: transparent; color: #999;
                border: 1px solid #2A2A2A; border-radius: 8px;
                min-height: 0; min-width: 0;
            }
            QPushButton:hover { color: #F5F5F5; border-color: #555; }
        """)
        back_btn.clicked.connect(lambda: self._nav("auth"))
        top.addWidget(back_btn)
        top.addStretch()
        root.addLayout(top)

        root.addStretch(2)

        # ── Title ────────────────────────────────────────────────────────
        title = QLabel("Create Account")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 28px; font-weight: 700; color: #F5F5F5;")
        root.addWidget(title, alignment=Qt.AlignCenter)
        root.addSpacing(20)

        # ── Form fields ──────────────────────────────────────────────────
        form = QVBoxLayout()
        form.setSpacing(12)
        form.setAlignment(Qt.AlignCenter)

        self._username = self._make_field(form, "Username")
        self._display_name = self._make_field(form, "Display Name")
        self._password = self._make_field(form, "Password", is_password=True)

        root.addLayout(form)
        root.addSpacing(16)

        # ── Level selection ──────────────────────────────────────────────
        level_label = QLabel("Experience Level")
        level_label.setAlignment(Qt.AlignCenter)
        level_label.setStyleSheet("font-size: 16px; color: #999;")
        root.addWidget(level_label, alignment=Qt.AlignCenter)
        root.addSpacing(8)

        level_row = QHBoxLayout()
        level_row.setAlignment(Qt.AlignCenter)
        level_row.setSpacing(12)
        self._level = "beginner"
        self._level_btns = []
        for lvl in ["Beginner", "Intermediate", "Advanced"]:
            btn = QPushButton(lvl)
            selected = (lvl.lower() == self._level)
            self._style_level_btn(btn, selected)
            btn.clicked.connect(lambda _, l=lvl: self._set_level(l))
            level_row.addWidget(btn)
            self._level_btns.append((lvl, btn))
        root.addLayout(level_row)

        root.addStretch(2)

        # ── Create button ────────────────────────────────────────────────
        create_btn = QPushButton("Create Account")
        create_btn.setStyleSheet("""
            QPushButton {
                font-size: 22px; font-weight: 700; padding: 14px;
                min-width: 400px; min-height: 56px;
                background-color: #FF6B35; color: #FFFFFF;
                border: none; border-radius: 14px;
            }
            QPushButton:hover { background-color: #E55A2B; }
        """)
        create_btn.clicked.connect(self._on_create)
        root.addWidget(create_btn, alignment=Qt.AlignCenter)

        root.addStretch(1)

        # ── Status ───────────────────────────────────────────────────────
        self._status = QLabel("")
        self._status.setAlignment(Qt.AlignCenter)
        self._status.setStyleSheet("font-size: 14px; color: #E53935;")
        self._status.setWordWrap(True)
        root.addWidget(self._status, alignment=Qt.AlignCenter)
        root.addStretch(1)

    def _make_field(self, layout: QVBoxLayout, placeholder: str,
                    is_password: bool = False) -> QLineEdit:
        field = QLineEdit()
        field.setPlaceholderText(placeholder)
        field.setFixedWidth(400)
        field.setFixedHeight(48)
        if is_password:
            field.setEchoMode(QLineEdit.EchoMode.Password)
        field.setStyleSheet("""
            QLineEdit {
                font-size: 16px; padding: 10px 16px;
                background-color: #141414; color: #F5F5F5;
                border: 2px solid #2A2A2A; border-radius: 10px;
            }
            QLineEdit:focus { border-color: #FF6B35; }
        """)
        layout.addWidget(field, alignment=Qt.AlignCenter)
        return field

    def _style_level_btn(self, btn: QPushButton, selected: bool) -> None:
        bg = "#FF6B35" if selected else "#141414"
        fg = "#FFF" if selected else "#999"
        border = "#FF6B35" if selected else "#2A2A2A"
        btn.setStyleSheet(f"""
            QPushButton {{
                font-size: 14px; font-weight: 600; padding: 8px 20px;
                background-color: {bg}; color: {fg};
                border: 2px solid {border}; border-radius: 10px;
                min-height: 0; min-width: 0;
            }}
            QPushButton:hover {{ border-color: #FF6B35; }}
        """)

    def _set_level(self, level: str) -> None:
        self._level = level.lower()
        for lvl_text, btn in self._level_btns:
            self._style_level_btn(btn, lvl_text.lower() == self._level)

    def _on_create(self) -> None:
        username = self._username.text().strip()
        display = self._display_name.text().strip() or username
        password = self._password.text()

        if not username:
            self._status.setText("Please enter a username")
            return
        if len(password) < 4:
            self._status.setText("Password must be at least 4 characters")
            return

        if self._db:
            user_id = self._db.create_user(
                username, password, display, "individual", self._level
            )
            if user_id is None:
                self._status.setText("Username already taken")
                return
            self._status.setStyleSheet("font-size: 14px; color: #FF6B35;")
            self._status.setText(f"Account created! Welcome, {display}")
            logger.info("Created user: %s (level=%s)", username, self._level)
            # Navigate to home after brief delay
            from PySide6.QtCore import QTimer
            QTimer.singleShot(1000, lambda: self._nav("home", username=username))
        else:
            self._status.setText("Database not available")

    def _nav(self, page: str, **kwargs):
        if self._router:
            self._router.navigate(page, **kwargs)

    def on_enter(self, **kwargs: Any) -> None:
        self._username.clear()
        self._display_name.clear()
        self._password.clear()
        self._status.setText("")

    def on_leave(self) -> None:
        pass
