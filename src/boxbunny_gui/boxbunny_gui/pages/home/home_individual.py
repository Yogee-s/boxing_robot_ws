"""Home page for logged-in users.

Clean modern layout with large mode buttons. Premium dark theme.
"""
from __future__ import annotations

import logging
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

logger = logging.getLogger(__name__)


def _mode_btn(text: str, accent: str) -> QPushButton:
    """Dark surface button with a subtle colored left border accent."""
    btn = QPushButton(text)
    btn.setStyleSheet(f"""
        QPushButton {{
            font-size: 22px; font-weight: 600; padding: 14px 20px;
            min-width: 480px; min-height: 60px;
            background-color: #161616; color: #E0E0E0;
            border: none; border-radius: 14px;
            border-left: 4px solid {accent};
            text-align: left; padding-left: 28px;
        }}
        QPushButton:hover {{
            background-color: #1E1E1E;
            color: #FFFFFF;
            border-left: 4px solid {accent};
        }}
        QPushButton:pressed {{ background-color: #252525; }}
    """)
    return btn


class HomeIndividualPage(QWidget):
    """Main menu for authenticated users."""

    def __init__(self, router=None, **kwargs):
        super().__init__()
        self._router = router

        root = QVBoxLayout(self)
        root.setSpacing(0)
        root.setContentsMargins(80, 25, 80, 25)

        # ── Top bar ──────────────────────────────────────────────────────
        top = QHBoxLayout()
        self._name_label = QLabel("Welcome back!")
        self._name_label.setStyleSheet("font-size: 24px; font-weight: 700; color: #FFFFFF;")
        top.addWidget(self._name_label)
        top.addStretch()

        settings_btn = QPushButton("Settings")
        settings_btn.setStyleSheet("""
            QPushButton {
                font-size: 14px; padding: 6px 16px;
                background-color: #1E1E1E; color: #B0B0B0;
                border: 1px solid #333; border-radius: 8px;
                min-height: 0; min-width: 0;
            }
            QPushButton:hover { color: #FFF; border-color: #FF6B35; }
        """)
        settings_btn.clicked.connect(lambda: self._nav("settings"))
        top.addWidget(settings_btn)

        close_btn = QPushButton("\u2715")
        close_btn.setFixedSize(38, 38)
        close_btn.setStyleSheet("""
            QPushButton {
                font-size: 14px; background-color: #1C1C1C; color: #666;
                border: 1px solid #2A2A2A; border-radius: 19px;
                min-height: 0; min-width: 0; padding: 0;
            }
            QPushButton:hover { background-color: #E53935; color: white; border-color: #E53935; }
        """)
        close_btn.clicked.connect(lambda: self.window().close())
        top.addSpacing(8)
        top.addWidget(close_btn)
        root.addLayout(top)

        root.addStretch(2)

        # ── Mode buttons — warm accent tones ─────────────────────────────
        training = _mode_btn("Training", "#FF6B35")       # orange
        training.clicked.connect(lambda: self._nav("training_select"))

        sparring = _mode_btn("Sparring", "#E53935")       # red
        sparring.clicked.connect(lambda: self._nav("sparring_select"))

        free = _mode_btn("Free Training", "#FF8A65")      # soft orange
        free.clicked.connect(lambda: self._nav("training_session"))

        perf = _mode_btn("Performance", "#FF5252")        # warm red
        perf.clicked.connect(lambda: self._nav("performance"))

        history = _mode_btn("History", "#555555")          # neutral grey
        history.clicked.connect(lambda: self._nav("history"))

        root.addWidget(training, alignment=Qt.AlignCenter)
        root.addStretch(1)
        root.addWidget(sparring, alignment=Qt.AlignCenter)
        root.addStretch(1)
        root.addWidget(free, alignment=Qt.AlignCenter)
        root.addStretch(1)
        root.addWidget(perf, alignment=Qt.AlignCenter)
        root.addStretch(1)
        root.addWidget(history, alignment=Qt.AlignCenter)

        root.addStretch(2)

        # ── Bottom ───────────────────────────────────────────────────────
        logout = QPushButton("Log Out")
        logout.setStyleSheet("""
            QPushButton {
                font-size: 16px; padding: 10px;
                min-width: 200px; min-height: 40px;
                background-color: transparent; color: #E53935;
                border: 1px solid #E53935; border-radius: 10px;
            }
            QPushButton:hover { background-color: #E53935; color: white; }
        """)
        logout.clicked.connect(lambda: self._nav("auth"))
        root.addWidget(logout, alignment=Qt.AlignCenter)
        root.addSpacing(8)

    def _nav(self, page: str):
        if self._router:
            self._router.navigate(page)

    def on_enter(self, username: str = "Guest", **kwargs: Any):
        self._username = username
        self._name_label.setText(f"Welcome, {username}!")

    def on_leave(self) -> None:
        pass
