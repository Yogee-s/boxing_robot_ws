"""Guest home page — shown after skill assessment.

Dark surface buttons with subtle colored left accent. Clean, minimal.
"""
from __future__ import annotations

import logging
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

logger = logging.getLogger(__name__)


def _mode_btn(text: str, accent: str) -> QPushButton:
    """Dark button with a thin colored left border accent."""
    btn = QPushButton(text)
    btn.setStyleSheet(f"""
        QPushButton {{
            font-size: 22px; font-weight: 600; padding: 14px 20px;
            min-width: 480px; min-height: 60px;
            background-color: #141414; color: #E0E0E0;
            border: none; border-radius: 14px;
            border-left: 4px solid {accent};
            text-align: left; padding-left: 28px;
        }}
        QPushButton:hover {{
            background-color: #1C1C1C; color: #FFFFFF;
            border-left: 4px solid {accent};
        }}
        QPushButton:pressed {{ background-color: #242424; }}
    """)
    return btn


class HomeGuestPage(QWidget):
    """Menu for guest (unauthenticated) users."""

    def __init__(self, router=None, **kwargs):
        super().__init__()
        self._router = router

        root = QVBoxLayout(self)
        root.setSpacing(0)
        root.setContentsMargins(80, 25, 80, 25)

        # ── Top bar ──────────────────────────────────────────────────────
        top = QHBoxLayout()
        title = QLabel("Guest Mode")
        title.setStyleSheet("font-size: 24px; font-weight: 700; color: #F5F5F5;")
        top.addWidget(title)
        top.addStretch()

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
        top.addWidget(close_btn)
        root.addLayout(top)

        root.addStretch(2)

        # ── Mode buttons ─────────────────────────────────────────────────
        training = _mode_btn("Training", "#FF6B35")
        training.clicked.connect(lambda: self._nav("training_select"))

        sparring = _mode_btn("Sparring", "#E53935")
        sparring.clicked.connect(lambda: self._nav("sparring_select"))

        free = _mode_btn("Free Training", "#FF8A65")
        free.clicked.connect(lambda: self._nav("training_session"))

        perf = _mode_btn("Performance", "#FF5252")
        perf.clicked.connect(lambda: self._nav("performance"))

        root.addWidget(training, alignment=Qt.AlignCenter)
        root.addStretch(1)
        root.addWidget(sparring, alignment=Qt.AlignCenter)
        root.addStretch(1)
        root.addWidget(free, alignment=Qt.AlignCenter)
        root.addStretch(1)
        root.addWidget(perf, alignment=Qt.AlignCenter)

        root.addStretch(3)

        # ── Back ─────────────────────────────────────────────────────────
        back = QPushButton("Back")
        back.setStyleSheet("""
            QPushButton {
                font-size: 16px; padding: 10px;
                min-width: 200px; min-height: 40px;
                background-color: transparent; color: #999;
                border: 1px solid #2A2A2A; border-radius: 10px;
            }
            QPushButton:hover { color: #F5F5F5; border-color: #555; }
        """)
        back.clicked.connect(lambda: self._nav("auth"))
        root.addWidget(back, alignment=Qt.AlignCenter)
        root.addSpacing(8)

    def _nav(self, page: str):
        if self._router:
            self._router.navigate(page)

    def on_enter(self, **kwargs: Any) -> None:
        pass

    def on_leave(self) -> None:
        pass
