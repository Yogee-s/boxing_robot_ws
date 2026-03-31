"""Punch Combination training selector — matches old GUI flow.

Difficulty selection (Beginner/Intermediate/Advanced + Self-Select),
with locked levels based on user progression. Shows group progress
and recommended next combo via spaced repetition.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from boxbunny_gui.curriculum import (
    ComboCurriculum, GROUP_BOUNDARIES, MASTERY_THRESHOLDS,
    MIN_ATTEMPTS_FOR_MASTERY, _combo_index,
)
from boxbunny_gui.theme import (
    Color, Icon, Size, font, PRIMARY_BTN, back_link_style,
)
from boxbunny_gui.widgets import BigButton

if TYPE_CHECKING:
    from boxbunny_gui.nav.router import PageRouter

logger = logging.getLogger(__name__)

_PUNCH_COLORS = {
    "1": Color.JAB, "2": Color.CROSS, "3": Color.L_HOOK,
    "4": Color.R_HOOK, "5": Color.L_UPPERCUT, "6": Color.R_UPPERCUT,
}
_PUNCH_NAMES = {
    "1": "Jab", "2": "Cross", "3": "L Hook", "4": "R Hook",
    "5": "L Upper", "6": "R Upper",
}

_KW = f"color:{Color.PRIMARY_LIGHT}; font-weight:600"

_DIFFICULTIES = [
    {
        "name": "Beginner",
        "desc": f'<span style="{_KW}">15 foundational combos</span> — '
                f'jabs, crosses and <span style="{_KW}">basic sequences</span>',
        "accent": Color.SUCCESS,
        "bg": "#101A14", "border": "#1A3D22",
        "min_level": 0,
    },
    {
        "name": "Intermediate",
        "desc": f'<span style="{_KW}">20 combos</span> with body shots, '
                f'hooks and <span style="{_KW}">defensive counters</span>',
        "accent": Color.WARNING,
        "bg": "#1A1810", "border": "#3D351A",
        "min_level": 1,
    },
    {
        "name": "Advanced",
        "desc": f'<span style="{_KW}">15 complex combos</span> — '
                f'unlocked at <span style="{_KW}">Advanced level</span>',
        "accent": Color.DANGER,
        "bg": "#1A1214", "border": "#3D1A22",
        "min_level": 2,
    },
]

_LEVEL_HIERARCHY = ["Beginner", "Intermediate", "Advanced"]


def _progress_bar_style(accent: str = Color.PRIMARY) -> str:
    return f"""
        QProgressBar {{
            background-color: {Color.SURFACE_LIGHT};
            border: none; border-radius: 4px;
            height: 8px; text-align: center; font-size: 0px;
        }}
        QProgressBar::chunk {{
            background-color: {accent};
            border-radius: 4px;
        }}
    """


# ── Combo row (for the scrollable list) ──────────────────────────────────

class _ComboRow(QWidget):
    def __init__(self, combo: Dict[str, Any], threshold: float,
                 is_next: bool = False, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.combo = combo
        attempts = combo.get("total_attempts") or 0
        mastery = combo.get("mastery_score") or 0.0
        mastered = attempts >= MIN_ATTEMPTS_FOR_MASTERY and mastery >= threshold
        pct = min(int((mastery / threshold) * 100), 100) if threshold > 0 else 0

        if is_next:
            bg, border = "#1A1510", "#3D2E1A"
            left = f"border-left: 3px solid {Color.PRIMARY};"
        else:
            bg, border = "#131920", "#1E2832"
            left = ""

        self.setFixedHeight(48)
        self.setStyleSheet(f"""
            QWidget {{ background-color: {bg}; border: 1px solid {border};
                {left} border-radius: {Size.RADIUS_SM}px; }}
            QWidget QLabel {{ background: transparent; border: none; }}
        """)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 4, 12, 4)
        lay.setSpacing(8)

        if is_next:
            badge = QLabel("NEXT")
            badge.setFixedWidth(40)
            badge.setAlignment(Qt.AlignCenter)
            badge.setStyleSheet(
                f"background: {Color.PRIMARY}; color: #FFFFFF;"
                " font-size: 9px; font-weight: 700;"
                " border-radius: 4px; padding: 2px 4px;"
            )
            lay.addWidget(badge)

        name_lbl = QLabel(combo.get("combo_name", ""))
        name_lbl.setStyleSheet(
            f"font-size: 13px; font-weight: 600; color: {Color.TEXT};"
        )
        name_lbl.setFixedWidth(180)
        lay.addWidget(name_lbl)

        # Punch dots
        seq_str = combo.get("combo_sequence", "")
        for token in seq_str.split("-"):
            base = token.rstrip("b")
            if base in _PUNCH_COLORS:
                dot = QLabel("\u25CF")
                dot.setStyleSheet(
                    f"color: {_PUNCH_COLORS[base]}; font-size: 12px;"
                )
                dot.setFixedWidth(14)
                dot.setAlignment(Qt.AlignCenter)
                lay.addWidget(dot)
        lay.addStretch()

        if mastered:
            check = QLabel(Icon.CHECK)
            check.setStyleSheet(
                f"color: {Color.PRIMARY}; font-size: 16px; font-weight: 700;"
            )
            check.setFixedWidth(30)
            check.setAlignment(Qt.AlignCenter)
            lay.addWidget(check)
        else:
            pct_lbl = QLabel(f"{pct}%")
            color = Color.PRIMARY if pct >= 60 else (
                Color.WARNING if pct > 0 else Color.TEXT_DISABLED
            )
            pct_lbl.setStyleSheet(
                f"color: {color}; font-size: 13px; font-weight: 700;"
            )
            pct_lbl.setFixedWidth(30)
            pct_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            lay.addWidget(pct_lbl)


# ═══════════════════════════════════════════════════════════════════════════

class ComboSelectPage(QWidget):
    """Difficulty selection + combo progress — matches old GUI flow."""

    def __init__(self, router: PageRouter, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._router = router
        self._curriculum: Optional[ComboCurriculum] = None
        self._active_diff = "Beginner"
        self._user_level_idx: int = 0
        self._next_combo: Optional[Dict[str, Any]] = None
        self._combo_rows: list[QWidget] = []
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(32, 10, 32, 10)
        root.setSpacing(0)

        # ── Top bar ──────────────────────────────────────────────────────
        top = QHBoxLayout()
        btn_back = QPushButton(f"{Icon.BACK}  Back")
        btn_back.setStyleSheet(back_link_style())
        btn_back.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_back.clicked.connect(lambda: self._router.back())
        top.addWidget(btn_back)

        title = QLabel("Punch Combinations")
        title.setStyleSheet(
            f"font-size: 20px; font-weight: 700; color: {Color.TEXT};"
        )
        top.addWidget(title)
        top.addStretch()
        root.addLayout(top)

        root.addStretch(1)

        # ── Difficulty cards — 3 levels in a row ─────────────────────────
        diff_lbl = QLabel("Select Difficulty")
        diff_lbl.setStyleSheet(
            f"font-size: 13px; font-weight: 700; color: {Color.TEXT_SECONDARY};"
            " letter-spacing: 0.5px;"
        )
        root.addWidget(diff_lbl)
        root.addSpacing(6)

        self._diff_btns: list[QPushButton] = []
        diff_row = QHBoxLayout()
        diff_row.setSpacing(10)

        for diff in _DIFFICULTIES:
            btn = self._make_diff_card(diff)
            diff_row.addWidget(btn)
            self._diff_btns.append(btn)

        root.addLayout(diff_row)
        root.addSpacing(8)

        # Self-Select — full width below
        self._self_select_btn = self._make_self_select_card()
        self._self_select_btn.setFixedHeight(44)
        root.addWidget(self._self_select_btn)

        root.addSpacing(10)

        # ── Progress card ────────────────────────────────────────────────
        self._progress_card = QWidget()
        self._progress_card.setObjectName("prog")
        self._progress_card.setStyleSheet(f"""
            QWidget#prog {{
                background-color: #1A1510;
                border: 1px solid #3D2E1A;
                border-left: 3px solid {Color.PRIMARY};
                border-radius: {Size.RADIUS}px;
            }}
            QWidget#prog QLabel {{ background: transparent; border: none; }}
            QWidget#prog QProgressBar {{ border: none; }}
        """)
        pc_lay = QVBoxLayout(self._progress_card)
        pc_lay.setContentsMargins(18, 14, 18, 14)
        pc_lay.setSpacing(6)

        pc_top = QHBoxLayout()
        self._group_name_lbl = QLabel("Single Punches")
        self._group_name_lbl.setStyleSheet(
            f"font-size: 15px; font-weight: 600; color: {Color.TEXT};"
        )
        pc_top.addWidget(self._group_name_lbl)
        pc_top.addStretch()
        self._group_progress_lbl = QLabel("0/6 mastered")
        self._group_progress_lbl.setStyleSheet(
            f"font-size: 12px; color: {Color.TEXT_SECONDARY};"
        )
        pc_top.addWidget(self._group_progress_lbl)
        pc_lay.addLayout(pc_top)

        self._progress_bar = QProgressBar()
        self._progress_bar.setFixedHeight(8)
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setStyleSheet(_progress_bar_style())
        self._progress_bar.setTextVisible(False)
        pc_lay.addWidget(self._progress_bar)

        self._overall_lbl = QLabel("")
        self._overall_lbl.setStyleSheet(
            f"font-size: 11px; color: {Color.TEXT_DISABLED};"
        )
        pc_lay.addWidget(self._overall_lbl)

        root.addWidget(self._progress_card)

        root.addSpacing(8)

        # ── Combo list (scrollable) ──────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(
            "QScrollArea { border: none; background: transparent; }"
        )
        self._list_widget = QWidget()
        self._list_widget.setStyleSheet("background: transparent;")
        self._list_layout = QVBoxLayout(self._list_widget)
        self._list_layout.setSpacing(3)
        self._list_layout.setContentsMargins(0, 0, 4, 0)
        scroll.setWidget(self._list_widget)
        root.addWidget(scroll)

        root.addSpacing(8)

        # ── Train button ─────────────────────────────────────────────────
        self._btn_train = BigButton(
            f"{Icon.PLAY}  Start Training", stylesheet=PRIMARY_BTN
        )
        self._btn_train.setFixedHeight(52)
        self._btn_train.clicked.connect(self._on_train_next)
        root.addWidget(self._btn_train)

        root.addStretch(1)

    def _make_diff_card(self, diff: dict) -> QPushButton:
        accent = diff["accent"]
        bg = diff["bg"]
        border = diff["border"]
        btn = QPushButton()
        btn.setObjectName(diff["name"])
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setMinimumHeight(130)
        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {bg};
                border: 1px solid {border};
                border-bottom: 3px solid {accent};
                border-radius: {Size.RADIUS}px;
                padding: 14px 12px;
            }}
            QPushButton:hover {{
                background-color: {Color.SURFACE_HOVER};
                border: 1px solid {accent};
                border-bottom: 3px solid {accent};
            }}
            QPushButton:disabled {{
                background-color: {Color.SURFACE};
                border: 1px solid {Color.BORDER};
                border-bottom: 3px solid {Color.TEXT_DISABLED};
            }}
        """)

        lay = QVBoxLayout(btn)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(8)
        lay.setAlignment(Qt.AlignCenter)

        name = QLabel(diff["name"])
        name.setAlignment(Qt.AlignCenter)
        name.setStyleSheet(
            "background: transparent; border: none;"
            f" font-size: 22px; font-weight: 700; color: {Color.TEXT};"
        )
        name.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        lay.addWidget(name)

        btn.clicked.connect(
            lambda _c=False, d=diff["name"]: self._set_difficulty(d)
        )
        return btn

    def _make_self_select_card(self) -> QPushButton:
        btn = QPushButton()
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {Color.SURFACE};
                border: 1px solid {Color.BORDER};
                border-radius: {Size.RADIUS}px;
                padding: 8px 16px;
            }}
            QPushButton:hover {{
                background-color: {Color.SURFACE_HOVER};
                border-color: {Color.PURPLE};
            }}
        """)
        lay = QHBoxLayout(btn)
        lay.setContentsMargins(4, 0, 4, 0)
        lay.setSpacing(10)

        name = QLabel("Self-Select")
        name.setStyleSheet(
            "background: transparent; border: none;"
            f" font-size: 14px; font-weight: 600; color: {Color.TEXT};"
        )
        name.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        lay.addWidget(name)

        desc = QLabel("Build your own custom punch sequence")
        desc.setStyleSheet(
            "background: transparent; border: none;"
            f" font-size: 12px; color: {Color.TEXT_SECONDARY};"
        )
        desc.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        lay.addWidget(desc)

        lay.addStretch()

        btn.clicked.connect(self._on_self_select)
        return btn

    # ── Difficulty switching ──────────────────────────────────────────────

    def _set_difficulty(self, diff: str) -> None:
        self._active_diff = diff
        self._refresh()

    def _update_lock_state(self) -> None:
        """Enable/disable difficulty buttons based on user level."""
        for i, diff in enumerate(_DIFFICULTIES):
            btn = self._diff_btns[i]
            locked = diff["min_level"] > self._user_level_idx
            btn.setEnabled(not locked)
            if locked:
                btn.setToolTip(
                    f"Reach 80% progress at {_LEVEL_HIERARCHY[diff['min_level'] - 1]} "
                    "level to unlock"
                )
            else:
                btn.setToolTip("")

    # ── Data refresh ─────────────────────────────────────────────────────

    def _refresh(self) -> None:
        if not self._curriculum:
            return

        diff = self._active_diff
        threshold = MASTERY_THRESHOLDS.get(diff, 4.0)
        progress = self._curriculum.get_level_progress(diff)
        self._next_combo = self._curriculum.get_next_combo(diff)

        self._group_name_lbl.setText(
            f"Group {progress['current_group_number']}: "
            f"{progress['current_group_name']}"
            if progress["current_group_number"] > 0
            else "All Groups Mastered!"
        )
        self._group_progress_lbl.setText(progress["current_group_progress"])

        total = progress["total_combos"]
        mastered = progress["mastered_combos"]
        pct = int((mastered / total) * 100) if total > 0 else 0
        self._progress_bar.setValue(pct)
        self._overall_lbl.setText(
            f"{mastered}/{total} combos mastered  |  "
            f"Groups: {progress['groups_completed']}/{progress['total_groups']}"
        )

        if self._next_combo:
            self._btn_train.setText(
                f"{Icon.PLAY}  Start: {self._next_combo['combo_name']}"
            )
            self._btn_train.setEnabled(True)
        elif progress["can_level_up"]:
            nxt = ComboCurriculum.get_next_difficulty(diff)
            self._btn_train.setText(
                f"Level Up to {nxt}!" if nxt else f"{Icon.CHECK} All Mastered!"
            )
            self._btn_train.setEnabled(nxt is not None)
        else:
            self._btn_train.setText(f"{Icon.CHECK}  All Combos Mastered!")
            self._btn_train.setEnabled(False)

        self._populate_combos(diff, progress, threshold)

    def _populate_combos(self, diff: str, progress: Dict, threshold: float) -> None:
        for row in self._combo_rows:
            self._list_layout.removeWidget(row)
            row.deleteLater()
        self._combo_rows.clear()

        all_combos = self._curriculum.get_combos_by_difficulty(diff)
        next_id = self._next_combo["combo_id"] if self._next_combo else None

        for start, end, group_name in GROUP_BOUNDARIES.get(diff, []):
            group = [
                c for c in all_combos
                if start <= _combo_index(c["combo_id"]) <= end
            ]
            if not group:
                continue

            header = QLabel(f"  {group_name}")
            header.setStyleSheet(
                f"color: {Color.TEXT_SECONDARY}; font-size: 10px;"
                " font-weight: 700; letter-spacing: 0.5px;"
            )
            header.setFixedHeight(18)
            self._list_layout.addWidget(header)
            self._combo_rows.append(header)

            for combo in group:
                is_next = combo["combo_id"] == next_id
                row = _ComboRow(combo, threshold, is_next=is_next, parent=self)
                self._list_layout.addWidget(row)
                self._combo_rows.append(row)

        self._list_layout.addStretch()

    # ── Actions ──────────────────────────────────────────────────────────

    def _on_train_next(self) -> None:
        if self._next_combo:
            combo_data = {
                "id": self._next_combo["combo_id"],
                "name": self._next_combo["combo_name"],
                "seq": self._next_combo["combo_sequence"],
                "diff": self._active_diff.lower(),
            }
            self._router.navigate(
                "training_config",
                combo_id=combo_data["id"],
                combo=combo_data,
                difficulty=self._active_diff,
                curriculum=self._curriculum,
            )
        else:
            nxt = ComboCurriculum.get_next_difficulty(self._active_diff)
            if nxt:
                self._set_difficulty(nxt)

    def _on_self_select(self) -> None:
        # Go straight to training config with no preset combo (free select)
        self._router.navigate(
            "training_config",
            combo={},
            difficulty="Self-Select",
            curriculum=self._curriculum,
        )

    # ── Lifecycle ────────────────────────────────────────────────────────

    def on_enter(self, **kwargs: Any) -> None:
        level = kwargs.get("level", "Beginner")
        if level and level.title() in _LEVEL_HIERARCHY:
            self._active_diff = level.title()

        # Determine user level for lock/unlock
        user_level = kwargs.get("user_level", "Beginner")
        if user_level in _LEVEL_HIERARCHY:
            self._user_level_idx = _LEVEL_HIERARCHY.index(user_level)
        else:
            self._user_level_idx = 0

        if not self._curriculum:
            try:
                self._curriculum = ComboCurriculum()
            except Exception:
                logger.exception("Failed to create curriculum")
                self._curriculum = None

        self._update_lock_state()
        self._refresh()
        logger.info("ComboSelectPage entered (difficulty=%s)", self._active_diff)

    def on_leave(self) -> None:
        pass
