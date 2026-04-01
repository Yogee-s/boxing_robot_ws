"""Skill assessment -- 6 questions + proficiency result with level override.

Two-step flow on a 1024x600 screen:
  Step 1 (questions): 2-column card grid, each card has the full question
         text and three option buttons.
  Step 2 (result):    Shows suggested level, lets user override, then starts.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QStackedWidget,
)

from boxbunny_gui.theme import Color, Icon, Size, back_link_style, subtle_btn_style

logger = logging.getLogger(__name__)

# Full question text matching the original proficiency checklist
_QUESTIONS = [
    ("Have you trained boxing before?", "experience",
     ["Never", "A few times", "Regularly"]),
    ("Do you know the basic punches\n(jab, cross, hook, uppercut)?", "punches",
     ["No", "Somewhat", "Yes"]),
    ("Can you throw a basic 1-2-3 combo?", "combo",
     ["No", "With help", "Yes"]),
    ("Have you done any sparring before?", "sparring",
     ["Never", "Once or twice", "Yes regularly"]),
    ("How would you describe\nyour fitness level?", "fitness",
     ["Low", "Moderate", "High"]),
    ("Have you used boxing equipment\nbefore (bag, pads)?", "equipment",
     ["Never", "Occasionally", "Often"]),
]

_LEVELS = ["Beginner", "Intermediate", "Advanced"]
_KW = f"color:{Color.PRIMARY}; font-weight:700"
_LEVEL_DESCRIPTIONS = {
    "Beginner": f'New to boxing. You\'ll start with <span style="{_KW}">fundamental '
                f'punches</span> and <span style="{_KW}">basic combos</span>.',
    "Intermediate": f'Some boxing experience. You\'ll work on '
                    f'<span style="{_KW}">combinations</span> and '
                    f'<span style="{_KW}">technique refinement</span>.',
    "Advanced": f'Experienced boxer. You\'ll tackle <span style="{_KW}">complex '
                f'combos</span> and <span style="{_KW}">sparring modes</span>.',
}
_LEVEL_COLORS = {
    "Beginner": Color.PRIMARY,
    "Intermediate": Color.WARNING,
    "Advanced": Color.DANGER,
}


# ── Shared style helpers ─────────────────────────────────────────────────

def _opt_style(selected: bool) -> str:
    if selected:
        return f"""
            QPushButton {{
                font-size: 13px; font-weight: 700; padding: 6px 10px;
                min-height: 36px;
                background-color: {Color.PRIMARY}; color: #FFFFFF;
                border: 2px solid {Color.PRIMARY_DARK}; border-radius: 8px;
            }}
            QPushButton:hover {{ background-color: {Color.PRIMARY_DARK}; }}
        """
    return f"""
        QPushButton {{
            font-size: 13px; font-weight: 600; padding: 6px 10px;
            min-height: 36px;
            background-color: {Color.SURFACE_LIGHT}; color: {Color.TEXT_SECONDARY};
            border: 1px solid {Color.BORDER}; border-radius: 8px;
        }}
        QPushButton:hover {{
            border-color: {Color.PRIMARY}; color: {Color.TEXT};
            background-color: {Color.SURFACE_HOVER};
        }}
    """


def _level_btn_style(selected: bool) -> str:
    if selected:
        return f"""
            QPushButton {{
                font-size: 18px; font-weight: 700; padding: 12px 20px;
                min-width: 170px; min-height: 54px;
                background-color: {Color.PRIMARY}; color: #FFFFFF;
                border: 2px solid {Color.PRIMARY_DARK};
                border-radius: {Size.RADIUS}px;
            }}
            QPushButton:hover {{ background-color: {Color.PRIMARY_DARK}; }}
        """
    return f"""
        QPushButton {{
            font-size: 18px; font-weight: 600; padding: 12px 20px;
            min-width: 170px; min-height: 54px;
            background-color: {Color.SURFACE}; color: {Color.TEXT_SECONDARY};
            border: 2px solid {Color.BORDER_LIGHT};
            border-radius: {Size.RADIUS}px;
        }}
        QPushButton:hover {{
            color: {Color.TEXT}; border-color: {Color.PRIMARY};
            background-color: {Color.SURFACE_HOVER};
        }}
    """


# ═══════════════════════════════════════════════════════════════════════════
# Step 1: Questions
# ═══════════════════════════════════════════════════════════════════════════

class _QuestionsWidget(QWidget):
    """2-column question grid with option buttons."""

    def __init__(self, parent: GuestAssessmentPage) -> None:
        super().__init__()
        self._page = parent
        self._answers: Dict[str, int] = {}
        self._btn_groups: Dict[str, list] = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(32, 14, 32, 22)
        root.setSpacing(0)

        # Push content to vertical center
        root.addStretch(1)

        # Header
        header = QVBoxLayout()
        header.setSpacing(4)
        title = QLabel("Proficiency Check")
        title.setStyleSheet(
            f"font-size: 22px; font-weight: 700; color: {Color.TEXT};"
        )
        header.addWidget(title)
        sub = QLabel("Select your experience level for each question")
        sub.setStyleSheet(f"font-size: 13px; color: {Color.TEXT_SECONDARY};")
        header.addWidget(sub)
        root.addLayout(header)

        root.addSpacing(12)

        # 2-column grid
        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(10)

        for idx, (prompt, key, options) in enumerate(_QUESTIONS):
            row = idx // 2
            col = idx % 2
            card = self._make_question(prompt, key, options)
            grid.addWidget(card, row, col)

        root.addLayout(grid)

        root.addStretch(1)
        root.addSpacing(4)

        # Bottom row: Skip + Back on left, Next on right
        bottom = QHBoxLayout()
        bottom.setSpacing(12)

        skip_btn = QPushButton("Skip")
        skip_btn.setFixedSize(110, 48)
        skip_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        skip_btn.setStyleSheet(subtle_btn_style())
        skip_btn.clicked.connect(self._on_skip)
        bottom.addWidget(skip_btn)

        back_btn = QPushButton("Back")
        back_btn.setFixedSize(110, 48)
        back_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        back_btn.setStyleSheet(subtle_btn_style())
        back_btn.clicked.connect(
            lambda: self._page._router.navigate("auth") if self._page._router else None
        )
        bottom.addWidget(back_btn)

        bottom.addStretch()

        self._next_btn = QPushButton(f"Next  {Icon.NEXT}")
        self._next_btn.setFixedSize(150, 50)
        self._next_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._next_btn.setStyleSheet(f"""
            QPushButton {{
                font-size: 15px; font-weight: 700;
                background-color: {Color.PRIMARY}; color: #FFFFFF;
                border: none; border-radius: {Size.RADIUS}px;
            }}
            QPushButton:hover {{ background-color: {Color.PRIMARY_DARK}; }}
            QPushButton:pressed {{ background-color: {Color.PRIMARY_PRESSED}; }}
        """)
        self._next_btn.clicked.connect(self._on_next)
        bottom.addWidget(self._next_btn)

        root.addLayout(bottom)

    def _make_question(self, prompt: str, key: str, options: list) -> QWidget:
        container = QWidget()
        container.setMinimumHeight(100)
        container.setMaximumHeight(120)
        container.setStyleSheet(f"""
            QWidget {{
                background-color: #131920;
                border: 1px solid #1E2832;
                border-left: 3px solid {Color.PRIMARY_MUTED};
                border-radius: {Size.RADIUS}px;
            }}
        """)
        col = QVBoxLayout(container)
        col.setContentsMargins(16, 12, 16, 12)
        col.setSpacing(10)

        label = QLabel(prompt.replace("\n", " "))
        label.setStyleSheet(
            f"font-size: 14px; font-weight: 600; color: {Color.TEXT};"
            " background: transparent; border: none;"
        )
        label.setWordWrap(True)
        col.addWidget(label)

        row = QHBoxLayout()
        row.setSpacing(6)
        btns = []
        for idx, opt in enumerate(options):
            btn = QPushButton(opt)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setStyleSheet(_opt_style(idx == 0))
            btn.clicked.connect(
                lambda _c, k=key, i=idx, bl=btns: self._select(k, i, bl)
            )
            row.addWidget(btn)
            btns.append((idx, btn))
        col.addLayout(row)

        # Pre-select first option as default
        self._answers[key] = 0
        self._btn_groups[key] = btns
        return container

    def _select(self, key: str, value: int, btns: list) -> None:
        self._answers[key] = value
        for opt_idx, btn in btns:
            btn.setStyleSheet(_opt_style(opt_idx == value))

    def _compute_level(self) -> str:
        total = sum(self._answers.values())
        if total <= 4:
            return "Beginner"
        if total <= 8:
            return "Intermediate"
        return "Advanced"

    def _on_skip(self) -> None:
        # Skip = unlock all levels
        self._page._go_to_home("Advanced")

    def _on_next(self) -> None:
        # Default unanswered to 0 (first option)
        for _, key, _ in _QUESTIONS:
            if key not in self._answers:
                self._answers[key] = 0
        level = self._compute_level()
        logger.info(
            "Assessment answers=%s -> %s (score=%d)",
            self._answers, level, sum(self._answers.values()),
        )
        self._page._show_result(level)

    def reset(self) -> None:
        self._answers.clear()
        for key, btns in self._btn_groups.items():
            for idx, btn in btns:
                btn.setStyleSheet(_opt_style(idx == 0))
            self._answers[key] = 0


# ═══════════════════════════════════════════════════════════════════════════
# Step 2: Result + level override
# ═══════════════════════════════════════════════════════════════════════════

class _ResultWidget(QWidget):
    """Shows suggested level, allows override, then confirms."""

    def __init__(self, parent: GuestAssessmentPage) -> None:
        super().__init__()
        self._page = parent
        self._chosen = "Beginner"
        self._level_btns: Dict[str, QPushButton] = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(60, 20, 60, 22)
        root.setSpacing(0)

        root.addStretch(1)

        # Title + suggestion — centered block
        title = QLabel("Your Proficiency Result")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(
            f"font-size: 22px; font-weight: 700; color: {Color.TEXT};"
        )
        root.addWidget(title)

        sub = QLabel("Based on your answers, we suggest:")
        sub.setAlignment(Qt.AlignCenter)
        sub.setStyleSheet(f"font-size: 14px; color: {Color.TEXT_SECONDARY};")
        root.addWidget(sub)

        root.addSpacing(12)

        self._suggestion_lbl = QLabel("Beginner")
        self._suggestion_lbl.setAlignment(Qt.AlignCenter)
        self._suggestion_lbl.setStyleSheet(
            f"font-size: 48px; font-weight: 700; color: {Color.PRIMARY};"
        )
        root.addWidget(self._suggestion_lbl)

        root.addSpacing(12)

        # Level override
        choose = QLabel("You can still choose your own level:")
        choose.setAlignment(Qt.AlignCenter)
        choose.setStyleSheet(f"font-size: 14px; color: {Color.TEXT_SECONDARY};")
        root.addWidget(choose)
        root.addSpacing(10)

        # Level selection buttons — centered
        level_row = QHBoxLayout()
        level_row.setSpacing(14)
        level_row.addStretch()
        for level in _LEVELS:
            btn = QPushButton(level)
            btn.setStyleSheet(_level_btn_style(False))
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(
                lambda _c=False, lv=level: self._select_level(lv)
            )
            level_row.addWidget(btn)
            self._level_btns[level] = btn
        level_row.addStretch()
        root.addLayout(level_row)
        root.addSpacing(12)

        # Description card — warm tint
        self._desc_lbl = QLabel("")
        self._desc_lbl.setAlignment(Qt.AlignCenter)
        self._desc_lbl.setWordWrap(True)
        self._desc_lbl.setStyleSheet(
            f"font-size: 16px; color: {Color.TEXT};"
            " background-color: #1A1510;"
            " border: 1px solid #3D2E1A;"
            f" border-radius: {Size.RADIUS}px;"
            " padding: 16px 28px;"
        )
        root.addWidget(self._desc_lbl)

        root.addStretch(1)

        # Bottom row — back left, confirm right
        bottom = QHBoxLayout()
        bottom.setSpacing(12)

        back_btn = QPushButton(f"{Icon.BACK}  Back")
        back_btn.setFixedSize(120, 48)
        back_btn.setStyleSheet(f"""
            QPushButton {{
                font-size: 13px; font-weight: 600;
                background-color: {Color.SURFACE}; color: {Color.TEXT_SECONDARY};
                border: 1px solid {Color.BORDER_LIGHT};
                border-radius: {Size.RADIUS}px;
            }}
            QPushButton:hover {{ color: {Color.TEXT}; border-color: {Color.PRIMARY}; }}
        """)
        back_btn.clicked.connect(self._page._show_questions)
        bottom.addWidget(back_btn)

        bottom.addStretch()

        confirm_btn = QPushButton(f"Confirm  {Icon.CHECK}")
        confirm_btn.setFixedSize(170, 54)
        confirm_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        confirm_btn.setStyleSheet(f"""
            QPushButton {{
                font-size: 16px; font-weight: 700;
                background-color: {Color.PRIMARY}; color: #FFFFFF;
                border: none; border-radius: {Size.RADIUS}px;
            }}
            QPushButton:hover {{ background-color: {Color.PRIMARY_DARK}; }}
            QPushButton:pressed {{ background-color: {Color.PRIMARY_PRESSED}; }}
        """)
        confirm_btn.clicked.connect(self._on_confirm)
        bottom.addWidget(confirm_btn)

        root.addLayout(bottom)

    def load(self, suggested: str) -> None:
        self._chosen = suggested
        color = _LEVEL_COLORS.get(suggested, Color.PRIMARY)
        self._suggestion_lbl.setText(suggested)
        self._suggestion_lbl.setStyleSheet(
            f"font-size: 48px; font-weight: 700; color: {color};"
        )
        self._refresh()

    def _select_level(self, level: str) -> None:
        self._chosen = level
        self._refresh()

    def _refresh(self) -> None:
        for lv, btn in self._level_btns.items():
            btn.setStyleSheet(_level_btn_style(lv == self._chosen))
        self._desc_lbl.setTextFormat(Qt.TextFormat.RichText)
        desc = _LEVEL_DESCRIPTIONS.get(self._chosen, "")
        self._desc_lbl.setText(
            f'<span style="color:{Color.TEXT}; font-size:16px;">{desc}</span>'
        )

    def _on_confirm(self) -> None:
        logger.info("Proficiency confirmed: %s", self._chosen)
        self._page._go_to_home(self._chosen.lower())


# ═══════════════════════════════════════════════════════════════════════════
# Container page
# ═══════════════════════════════════════════════════════════════════════════

class GuestAssessmentPage(QWidget):
    """Two-step assessment: questions -> result with level override."""

    def __init__(self, router=None, **kwargs: Any) -> None:
        super().__init__()
        self._router = router

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        self._stack = QStackedWidget()
        self._questions = _QuestionsWidget(self)
        self._result = _ResultWidget(self)
        self._stack.addWidget(self._questions)   # index 0
        self._stack.addWidget(self._result)       # index 1
        root.addWidget(self._stack)

    def _show_result(self, level: str) -> None:
        self._result.load(level)
        self._stack.setCurrentIndex(1)

    def _show_questions(self) -> None:
        self._stack.setCurrentIndex(0)

    def _go_to_home(self, level: str) -> None:
        if self._router:
            self._router.navigate("home_guest", level=level)

    # ── Lifecycle ──────────────────────────────────────────────────────
    def on_enter(self, **kwargs: Any) -> None:
        self._questions.reset()
        self._stack.setCurrentIndex(0)

    def on_leave(self) -> None:
        pass
