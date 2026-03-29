"""Quick skill assessment — all questions on one page.

Collects experience, goal, and intensity in a single clean screen.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


def _option_btn(text: str, selected: bool = False) -> QPushButton:
    """Create a selectable option button."""
    btn = QPushButton(text)
    bg = "#FF6B35" if selected else "#1E1E1E"
    fg = "#0D0D0D" if selected else "#FFFFFF"
    border = "#FF6B35" if selected else "#333"
    btn.setStyleSheet(f"""
        QPushButton {{
            font-size: 16px; font-weight: 600; padding: 8px 16px;
            min-height: 40px; min-width: 120px;
            background-color: {bg}; color: {fg};
            border: 2px solid {border}; border-radius: 12px;
        }}
        QPushButton:hover {{ border-color: #FF6B35; }}
    """)
    return btn


class GuestAssessmentPage(QWidget):
    """Single-page skill assessment with three question rows."""

    def __init__(self, router=None, **kwargs):
        super().__init__()
        self._router = router
        self._answers: Dict[str, str] = {}
        self._btn_groups: Dict[str, list] = {}

        root = QVBoxLayout(self)
        root.setSpacing(0)
        root.setContentsMargins(80, 30, 80, 30)

        # ── Title ────────────────────────────────────────────────────────
        title = QLabel("Quick Assessment")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 28px; font-weight: 700; color: #FFFFFF;")

        sub = QLabel("Help us personalize your experience")
        sub.setAlignment(Qt.AlignCenter)
        sub.setStyleSheet("font-size: 16px; color: #B0B0B0;")

        root.addStretch(2)
        root.addWidget(title, alignment=Qt.AlignCenter)
        root.addSpacing(4)
        root.addWidget(sub, alignment=Qt.AlignCenter)
        root.addStretch(2)

        # ── Question 1: Experience ───────────────────────────────────────
        root.addLayout(self._make_question(
            "Have you boxed before?", "experience", ["No", "A Little", "Yes"]
        ))
        root.addStretch(1)

        # ── Question 2: Goal ─────────────────────────────────────────────
        root.addLayout(self._make_question(
            "What's your goal?", "goal", ["Fitness", "Learn Boxing", "Improve Skills"]
        ))
        root.addStretch(1)

        # ── Question 3: Intensity ────────────────────────────────────────
        root.addLayout(self._make_question(
            "Preferred intensity?", "intensity", ["Light", "Medium", "Hard"]
        ))
        root.addStretch(2)

        # ── Start button ─────────────────────────────────────────────────
        start_btn = QPushButton("Let's Go!")
        start_btn.setStyleSheet("""
            QPushButton {
                font-size: 22px; font-weight: 700; padding: 12px;
                min-width: 350px; min-height: 52px;
                background-color: #FF6B35; color: #FFFFFF;
                border: none; border-radius: 14px;
            }
            QPushButton:hover { background-color: #E55A2B; }
        """)
        start_btn.clicked.connect(self._on_start)
        root.addWidget(start_btn, alignment=Qt.AlignCenter)

        root.addStretch(1)

        # ── Back link ────────────────────────────────────────────────────
        back = QPushButton("Back")
        back.setStyleSheet("""
            QPushButton {
                font-size: 14px; color: #B0B0B0; background: transparent;
                border: none; min-height: 0; min-width: 0; padding: 4px;
            }
            QPushButton:hover { color: #FFF; }
        """)
        back.clicked.connect(lambda: self._router.navigate("auth") if self._router else None)
        root.addWidget(back, alignment=Qt.AlignCenter)
        root.addSpacing(8)

    def _make_question(self, prompt: str, key: str, options: list) -> QVBoxLayout:
        """Build a question row: label + horizontal option buttons."""
        col = QVBoxLayout()
        col.setSpacing(8)

        label = QLabel(prompt)
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("font-size: 16px; font-weight: 600; color: #FFFFFF;")
        col.addWidget(label, alignment=Qt.AlignCenter)

        row = QHBoxLayout()
        row.setSpacing(12)
        row.setAlignment(Qt.AlignCenter)
        btns = []
        for opt in options:
            btn = _option_btn(opt)
            btn.clicked.connect(lambda checked, k=key, o=opt, bl=btns: self._select(k, o, bl))
            row.addWidget(btn)
            btns.append((opt, btn))
        col.addLayout(row)

        self._btn_groups[key] = btns
        # Default to first option
        self._select(key, options[0], btns)
        return col

    def _select(self, key: str, value: str, btns: list) -> None:
        """Mark a button as selected and update answer."""
        self._answers[key] = value
        for opt_text, btn in btns:
            selected = (opt_text == value)
            bg = "#FF6B35" if selected else "#1E1E1E"
            fg = "#0D0D0D" if selected else "#FFFFFF"
            border = "#FF6B35" if selected else "#333"
            btn.setStyleSheet(f"""
                QPushButton {{
                    font-size: 16px; font-weight: 600; padding: 8px 16px;
                    min-height: 40px; min-width: 120px;
                    background-color: {bg}; color: {fg};
                    border: 2px solid {border}; border-radius: 12px;
                }}
                QPushButton:hover {{ border-color: #FF6B35; }}
            """)

    def _on_start(self) -> None:
        logger.info("Guest assessment: %s", self._answers)
        if self._router:
            self._router.navigate("home_guest", answers=self._answers)

    def on_enter(self, **kwargs: Any) -> None:
        pass

    def on_leave(self) -> None:
        pass
