"""Anki-style combo training selector with difficulty tabs, group progress,
and automatic next-combo selection via spaced repetition.

Replaces the old manual combo browser.  Layout (1024x600):
  - Difficulty tabs (Beginner / Intermediate / Advanced)
  - Current group progress card
  - Recommended next combo (highlighted)
  - Scrollable list of combos in current group with mastery indicators
  - "Train Next" action button
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
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
    Color, Icon, Size, font, PRIMARY_BTN, back_link_style, tab_btn_style,
    accent_frame_style,
)
from boxbunny_gui.widgets import BigButton

if TYPE_CHECKING:
    from boxbunny_gui.nav.router import PageRouter

logger = logging.getLogger(__name__)

# Punch colour mapping
_PUNCH_COLORS = {
    "1": Color.JAB, "2": Color.CROSS, "3": Color.L_HOOK,
    "4": Color.R_HOOK, "5": Color.L_UPPERCUT, "6": Color.R_UPPERCUT,
}
_PUNCH_NAMES = {
    "1": "Jab", "2": "Cross", "3": "L Hook", "4": "R Hook",
    "5": "L Upper", "6": "R Upper",
}

_DIFFICULTIES = ["Beginner", "Intermediate", "Advanced"]


# ── Progress bar style ────────────────────────────────────────────────────

def _progress_bar_style(accent: str = Color.PRIMARY) -> str:
    return f"""
        QProgressBar {{
            background-color: {Color.SURFACE_LIGHT};
            border: none; border-radius: 4px;
            height: 8px; text-align: center;
            font-size: 0px;
        }}
        QProgressBar::chunk {{
            background-color: {accent};
            border-radius: 4px;
        }}
    """


# ── Combo card (row in scrollable list) ──────────────────────────────────

class _ComboRow(QWidget):
    """Single combo row showing name, sequence dots, mastery bar."""

    def __init__(
        self, combo: Dict[str, Any], threshold: float,
        is_next: bool = False, parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.combo = combo
        attempts = combo.get("total_attempts") or 0
        mastery = combo.get("mastery_score") or 0.0
        mastered = attempts >= MIN_ATTEMPTS_FOR_MASTERY and mastery >= threshold
        pct = min(int((mastery / threshold) * 100), 100) if threshold > 0 else 0

        border_color = Color.PRIMARY if is_next else Color.BORDER
        bg = Color.SURFACE_HOVER if is_next else Color.SURFACE

        self.setFixedHeight(60)
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {bg};
                border: 1px solid {border_color};
                {'border-left: ' + str(Size.ACCENT_BAR_W) + 'px solid ' + Color.PRIMARY + ';' if is_next else ''}
                border-radius: {Size.RADIUS_SM}px;
            }}
        """)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(14, 6, 14, 6)
        lay.setSpacing(10)

        # "NEXT" badge
        if is_next:
            badge = QLabel("NEXT")
            badge.setFixedWidth(42)
            badge.setAlignment(Qt.AlignCenter)
            badge.setStyleSheet(
                f"background: {Color.PRIMARY}; color: {Color.BG};"
                " font-size: 10px; font-weight: 700;"
                f" border-radius: 4px; padding: 2px 6px; border: none;"
            )
            lay.addWidget(badge)

        # Combo name
        name_lbl = QLabel(combo.get("combo_name", ""))
        name_lbl.setStyleSheet(
            f"color: {Color.TEXT}; font-size: 14px; font-weight: 600;"
            " background: transparent; border: none;"
        )
        name_lbl.setFixedWidth(200)
        lay.addWidget(name_lbl)

        # Punch sequence dots
        seq_str = combo.get("combo_sequence", "")
        dots = QHBoxLayout()
        dots.setSpacing(3)
        for token in seq_str.split("-"):
            base = token.rstrip("b")
            if base in _PUNCH_COLORS:
                dot = QLabel("\u25CF")
                dot.setStyleSheet(
                    f"color: {_PUNCH_COLORS[base]}; font-size: 14px;"
                    " background: transparent; border: none;"
                )
                dot.setAlignment(Qt.AlignCenter)
                dots.addWidget(dot)
            elif token in ("slip", "block"):
                lbl = QLabel(token[0].upper())
                lbl.setStyleSheet(
                    f"color: {Color.TEXT_DISABLED}; font-size: 11px;"
                    " font-weight: 700; background: transparent; border: none;"
                )
                lbl.setAlignment(Qt.AlignCenter)
                dots.addWidget(lbl)
        dots.addStretch()
        lay.addLayout(dots, stretch=1)

        # Mastery indicator
        if mastered:
            check = QLabel("\u2713")
            check.setStyleSheet(
                f"color: {Color.PRIMARY}; font-size: 18px; font-weight: 700;"
                " background: transparent; border: none;"
            )
            check.setFixedWidth(40)
            check.setAlignment(Qt.AlignCenter)
            lay.addWidget(check)
        else:
            pct_lbl = QLabel(f"{pct}%")
            color = Color.PRIMARY if pct >= 60 else (
                Color.WARNING if pct > 0 else Color.TEXT_DISABLED
            )
            pct_lbl.setStyleSheet(
                f"color: {color}; font-size: 14px; font-weight: 700;"
                " background: transparent; border: none;"
            )
            pct_lbl.setFixedWidth(40)
            pct_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            lay.addWidget(pct_lbl)


# ═══════════════════════════════════════════════════════════════════════════
# Main page
# ════════════════════════════════════════════════════��══════════════════════

class ComboSelectPage(QWidget):
    """Anki-style training selector with group-based progression."""

    def __init__(self, router: PageRouter, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._router = router
        self._curriculum: Optional[ComboCurriculum] = None
        self._active_diff = "Beginner"
        self._next_combo: Optional[Dict[str, Any]] = None
        self._tab_btns: list[QPushButton] = []
        self._combo_rows: list[_ComboRow] = []
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 12, 24, 14)
        root.setSpacing(10)

        # ── Top row: back + title ─────────────────────────────────────
        top = QHBoxLayout()
        btn_back = QPushButton(f"{Icon.BACK}  Back")
        btn_back.setStyleSheet(back_link_style())
        btn_back.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_back.clicked.connect(lambda: self._router.back())
        top.addWidget(btn_back)
        top.addStretch()
        title = QLabel("Combo Training")
        title.setFont(font(Size.TEXT_SUBHEADER, bold=True))
        top.addWidget(title)
        top.addStretch()
        # Tooltip
        tip = QLabel("Anki-style spaced repetition")
        tip.setStyleSheet(
            f"color: {Color.TEXT_DISABLED}; font-size: 11px;"
        )
        top.addWidget(tip)
        root.addLayout(top)

        # ── Difficulty tabs ────��──────────────────────────────────────
        tabs = QHBoxLayout()
        tabs.setSpacing(8)
        for diff in _DIFFICULTIES:
            btn = QPushButton(diff)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setStyleSheet(tab_btn_style(diff == self._active_diff))
            btn.clicked.connect(
                lambda _c=False, d=diff: self._set_difficulty(d)
            )
            tabs.addWidget(btn)
            self._tab_btns.append(btn)
        tabs.addStretch()
        root.addLayout(tabs)

        # ── Group progress card ─────────���─────────────────────────────
        self._progress_card = QFrame()
        self._progress_card.setStyleSheet(accent_frame_style(Color.PRIMARY))
        pc_lay = QVBoxLayout(self._progress_card)
        pc_lay.setContentsMargins(18, 12, 18, 12)
        pc_lay.setSpacing(6)

        pc_top = QHBoxLayout()
        self._group_name_lbl = QLabel("Single Punches")
        self._group_name_lbl.setStyleSheet(
            f"color: {Color.TEXT}; font-size: 16px; font-weight: 600;"
            " border: none;"
        )
        pc_top.addWidget(self._group_name_lbl)
        pc_top.addStretch()
        self._group_progress_lbl = QLabel("0/6 mastered")
        self._group_progress_lbl.setStyleSheet(
            f"color: {Color.TEXT_SECONDARY}; font-size: 13px; border: none;"
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

        pc_bottom = QHBoxLayout()
        self._overall_lbl = QLabel("")
        self._overall_lbl.setStyleSheet(
            f"color: {Color.TEXT_DISABLED}; font-size: 11px; border: none;"
        )
        pc_bottom.addWidget(self._overall_lbl)
        pc_bottom.addStretch()
        self._groups_lbl = QLabel("")
        self._groups_lbl.setStyleSheet(
            f"color: {Color.TEXT_DISABLED}; font-size: 11px; border: none;"
        )
        pc_bottom.addWidget(self._groups_lbl)
        pc_lay.addLayout(pc_bottom)

        # Center the progress card vertically
        root.addStretch(1)

        root.addWidget(self._progress_card)

        root.addSpacing(8)

        # ── Scrollable combo list ─────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("border: none;")
        scroll.setMaximumHeight(260)
        self._list_widget = QWidget()
        self._list_layout = QVBoxLayout(self._list_widget)
        self._list_layout.setSpacing(6)
        self._list_layout.setContentsMargins(0, 0, 4, 0)
        scroll.setWidget(self._list_widget)
        root.addWidget(scroll)

        root.addStretch(1)

        # ── Bottom: Train Next button ─────────────────────────────────
        self._btn_train = BigButton(f"{Icon.PLAY}  Train Next Combo", stylesheet=PRIMARY_BTN)
        self._btn_train.setFixedHeight(54)
        self._btn_train.clicked.connect(self._on_train_next)
        root.addWidget(self._btn_train)

    # ── Tab switching ─────────────────────────────────────────────────

    def _set_difficulty(self, diff: str) -> None:
        self._active_diff = diff
        for i, d in enumerate(_DIFFICULTIES):
            self._tab_btns[i].setStyleSheet(tab_btn_style(d == diff))
        self._refresh()

    # ── Data refresh ──────────���───────────────────────────────────────

    def _refresh(self) -> None:
        if not self._curriculum:
            return

        diff = self._active_diff
        threshold = MASTERY_THRESHOLDS.get(diff, 4.0)
        progress = self._curriculum.get_level_progress(diff)
        self._next_combo = self._curriculum.get_next_combo(diff)

        # Update progress card
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
            f"{mastered}/{total} combos mastered overall"
        )
        self._groups_lbl.setText(
            f"Groups: {progress['groups_completed']}/{progress['total_groups']}"
        )

        # Update button
        if self._next_combo:
            self._btn_train.setText(
                f"\u25B6  Train: {self._next_combo['combo_name']}"
            )
            self._btn_train.setEnabled(True)
        elif progress["can_level_up"]:
            nxt = ComboCurriculum.get_next_difficulty(diff)
            self._btn_train.setText(
                f"\u2B06  Level Up to {nxt}!" if nxt else "\u2713 All Mastered!"
            )
            self._btn_train.setEnabled(nxt is not None)
        else:
            self._btn_train.setText("\u2713  All Combos Mastered!")
            self._btn_train.setEnabled(False)

        # Populate combo list for current group
        self._populate_combos(diff, progress, threshold)

    def _populate_combos(
        self, diff: str, progress: Dict[str, Any], threshold: float,
    ) -> None:
        # Clear old
        for row in self._combo_rows:
            self._list_layout.removeWidget(row)
            row.deleteLater()
        self._combo_rows.clear()

        all_combos = self._curriculum.get_combos_by_difficulty(diff)
        next_id = self._next_combo["combo_id"] if self._next_combo else None

        # Show combos grouped
        for start, end, group_name in GROUP_BOUNDARIES.get(diff, []):
            group = [
                c for c in all_combos
                if start <= _combo_index(c["combo_id"]) <= end
            ]
            if not group:
                continue

            # Group header
            header = QLabel(f"  {group_name}")
            header.setStyleSheet(
                f"color: {Color.TEXT_DISABLED}; font-size: 11px;"
                " font-weight: 700; letter-spacing: 0.6px;"
            )
            header.setFixedHeight(20)
            self._list_layout.addWidget(header)
            self._combo_rows.append(header)

            for combo in group:
                is_next = combo["combo_id"] == next_id
                row = _ComboRow(combo, threshold, is_next=is_next, parent=self)
                self._list_layout.addWidget(row)
                self._combo_rows.append(row)

        self._list_layout.addStretch()

    # ── Actions ─────────��─────────────────────────────────────────────

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
            # Level up
            nxt = ComboCurriculum.get_next_difficulty(self._active_diff)
            if nxt:
                self._set_difficulty(nxt)

    # ── Lifecycle ─────────────────────────────────────────────────────

    def on_enter(self, **kwargs: Any) -> None:
        level = kwargs.get("level", "Beginner")
        if level and level.title() in _DIFFICULTIES:
            self._active_diff = level.title()

        # Create curriculum (guest mode = temp db)
        if not self._curriculum:
            try:
                self._curriculum = ComboCurriculum()
            except Exception:
                logger.exception("Failed to create curriculum (sqlite3 unavailable?)")
                self._curriculum = None

        # Sync tab styles
        for i, d in enumerate(_DIFFICULTIES):
            self._tab_btns[i].setStyleSheet(
                tab_btn_style(d == self._active_diff)
            )
        self._refresh()
        logger.info("ComboSelectPage entered (difficulty=%s)", self._active_diff)

    def on_leave(self) -> None:
        pass
