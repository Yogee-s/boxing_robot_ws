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
    "slip": Color.BLOCK, "slipr": Color.BLOCK,
    "block": Color.BLOCK, "blockr": Color.BLOCK,
}
_PUNCH_NAMES = {
    "1": "Jab", "2": "Cross", "3": "L Hook", "4": "R Hook",
    "5": "L Upper", "6": "R Upper",
    "slip": "Slip-L", "slipr": "Slip-R",
    "block": "Block-L", "blockr": "Block-R",
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
            height: 8px; text-align: center; font-size: 1px;
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
            bg = "#1E1610"
        else:
            bg = "#131920"

        self.setFixedHeight(26)
        self.setStyleSheet(f"""
            QWidget {{ background-color: {bg}; border: none;
                border-radius: 3px; }}
            QWidget QLabel {{ background: transparent; border: none; }}
        """)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 0, 8, 0)
        lay.setSpacing(4)

        name_lbl = QLabel(combo.get("combo_name", ""))
        name_color = Color.PRIMARY if is_next else Color.TEXT
        name_lbl.setStyleSheet(
            f"font-size: 12px; font-weight: {'700' if is_next else '500'};"
            f" color: {name_color};"
        )
        name_lbl.setFixedWidth(120)
        lay.addWidget(name_lbl)

        # Punch dots
        seq_str = combo.get("combo_sequence", "")
        for token in seq_str.split("-"):
            base = token.rstrip("b")
            if base in _PUNCH_COLORS:
                dot = QLabel("\u25CF")
                dot.setStyleSheet(
                    f"color: {_PUNCH_COLORS[base]}; font-size: 8px;"
                )
                dot.setFixedWidth(10)
                dot.setAlignment(Qt.AlignCenter)
                lay.addWidget(dot)
        lay.addStretch()

        if mastered:
            check = QLabel(Icon.CHECK)
            check.setStyleSheet(
                f"color: {Color.PRIMARY}; font-size: 14px; font-weight: 700;"
            )
            check.setFixedWidth(30)
            check.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            lay.addWidget(check)
        else:
            pct_lbl = QLabel(f"{pct}%")
            color = Color.PRIMARY if pct >= 60 else (
                Color.WARNING if pct > 0 else Color.TEXT_DISABLED
            )
            pct_lbl.setStyleSheet(
                f"color: {color}; font-size: 12px; font-weight: 600;"
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
        self._username: str = ""
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(32, 10, 32, 22)
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
        self._self_select_btn.setFixedHeight(56)
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

        bottom_row = QHBoxLayout()
        self._overall_lbl = QLabel("")
        self._overall_lbl.setStyleSheet(
            f"font-size: 11px; color: {Color.TEXT_DISABLED};"
        )
        bottom_row.addWidget(self._overall_lbl)
        bottom_row.addStretch()

        self._btn_view = QPushButton("View All Combos")
        self._btn_view.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_view.setFixedHeight(32)
        self._btn_view.setStyleSheet(f"""
            QPushButton {{
                font-size: 12px; font-weight: 700;
                background-color: {Color.SURFACE_LIGHT};
                color: {Color.TEXT};
                border: 1px solid {Color.BORDER_LIGHT};
                border-radius: 6px;
                padding: 0 14px;
            }}
            QPushButton:hover {{
                color: #FFFFFF;
                background-color: {Color.PRIMARY};
                border-color: {Color.PRIMARY};
            }}
        """)
        self._btn_view.clicked.connect(self._show_combo_popup)
        bottom_row.addWidget(self._btn_view)
        pc_lay.addLayout(bottom_row)

        root.addWidget(self._progress_card)

        root.addSpacing(10)

        # ── Train button ─────────────────────────────────────────────────
        self._btn_train = BigButton(
            f"{Icon.PLAY}  Continue to Training Setup", stylesheet=PRIMARY_BTN
        )
        self._btn_train.setFixedHeight(70)
        self._btn_train.clicked.connect(self._on_train_next)
        root.addWidget(self._btn_train)

        root.addStretch(1)

        # Hidden combo list data — populated on refresh, shown in popup
        self._list_widget = QWidget()
        self._list_widget.setStyleSheet("background: transparent;")
        self._list_layout = QVBoxLayout(self._list_widget)
        self._list_layout.setSpacing(3)
        self._list_layout.setContentsMargins(0, 0, 4, 0)
        self._combo_popup: QWidget | None = None

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
            lambda _c=False, d=diff["name"]: self._select_and_go(d)
        )
        return btn

    def _make_self_select_card(self) -> QPushButton:
        btn = QPushButton()
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {Color.SURFACE};
                border: none;
                border-radius: {Size.RADIUS}px;
                padding: 10px 16px;
            }}
            QPushButton:hover {{
                background-color: {Color.SURFACE_HOVER};
            }}
        """)
        lay = QVBoxLayout(btn)
        lay.setContentsMargins(0, 4, 0, 4)
        lay.setSpacing(2)
        lay.setAlignment(Qt.AlignCenter)

        name = QLabel("Self-Select")
        name.setAlignment(Qt.AlignCenter)
        name.setStyleSheet(
            "background: transparent; border: none;"
            f" font-size: 16px; font-weight: 700; color: {Color.TEXT};"
        )
        name.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        lay.addWidget(name)

        desc = QLabel("Build your own custom punch sequence")
        desc.setAlignment(Qt.AlignCenter)
        desc.setStyleSheet(
            "background: transparent; border: none;"
            f" font-size: 12px; color: {Color.TEXT_SECONDARY};"
        )
        desc.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        lay.addWidget(desc)

        btn.clicked.connect(self._on_self_select)
        return btn

    # ── Combo popup overlay ────────────────────────────────────────────

    def _show_combo_popup(self) -> None:
        """Show a centred popup overlay with the combo list."""
        win = self.window()
        if win is None:
            return

        # Semi-transparent backdrop
        self._combo_popup = QWidget(win)
        self._combo_popup.setStyleSheet("background-color: rgba(0, 0, 0, 160);")
        self._combo_popup.setGeometry(0, 0, win.width(), win.height())

        # Inner panel
        margin_x, margin_y = 50, 30
        panel = QWidget(self._combo_popup)
        panel.setGeometry(
            margin_x, margin_y,
            win.width() - margin_x * 2, win.height() - margin_y * 2,
        )
        panel.setStyleSheet(f"""
            QWidget {{
                background-color: {Color.BG};
                border: 1px solid {Color.BORDER_LIGHT};
                border-radius: {Size.RADIUS_LG}px;
            }}
        """)

        lay = QVBoxLayout(panel)
        lay.setContentsMargins(20, 12, 20, 14)
        lay.setSpacing(6)

        # Header
        header = QHBoxLayout()
        title = QLabel(f"{self._active_diff} Combos")
        title.setStyleSheet(
            f"font-size: 22px; font-weight: 700; color: {Color.PRIMARY};"
            " border: none; background: transparent;"
        )
        header.addWidget(title)
        header.addStretch()
        close_btn = QPushButton(f"{Icon.CLOSE}  Close")
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn.setFixedHeight(44)
        close_btn.setStyleSheet(f"""
            QPushButton {{
                font-size: 14px; font-weight: 600;
                background-color: {Color.SURFACE}; color: {Color.TEXT_SECONDARY};
                border: 1px solid {Color.BORDER_LIGHT}; border-radius: 8px;
                padding: 0 16px;
            }}
            QPushButton:hover {{
                color: {Color.TEXT}; border-color: {Color.PRIMARY};
            }}
        """)
        close_btn.clicked.connect(self._close_combo_popup)
        header.addWidget(close_btn)
        lay.addLayout(header)

        # Scrollable grouped columns
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(
            "QScrollArea { border: none; background: transparent; }"
        )
        content_w = QWidget()
        content_w.setStyleSheet("background: transparent;")
        content_lay = QVBoxLayout(content_w)
        content_lay.setContentsMargins(0, 0, 0, 0)
        content_lay.setSpacing(12)

        if self._curriculum:
            diff = self._active_diff
            threshold = MASTERY_THRESHOLDS.get(diff, 4.0)
            next_id = self._next_combo["combo_id"] if self._next_combo else None
            all_combos = self._curriculum.get_combos_by_difficulty(diff)

            for start, end, group_name in GROUP_BOUNDARIES.get(diff, []):
                group = [
                    c for c in all_combos
                    if start <= _combo_index(c["combo_id"]) <= end
                ]
                if not group:
                    continue

                # Group section
                card = QWidget()
                card.setStyleSheet(
                    f"QWidget {{ background-color: transparent;"
                    f" border: none; }}"
                    f" QWidget QLabel {{ border: none; background: transparent; }}"
                )
                card_lay = QVBoxLayout(card)
                card_lay.setContentsMargins(4, 0, 4, 0)
                card_lay.setSpacing(4)

                hdr = QLabel(group_name.upper())
                hdr.setStyleSheet(
                    f"color: {Color.TEXT_DISABLED}; font-size: 10px;"
                    " font-weight: 700; letter-spacing: 1px;"
                )
                card_lay.addWidget(hdr)

                div = QFrame()
                div.setFixedHeight(1)
                div.setStyleSheet(
                    f"background-color: {Color.BORDER}; border: none;"
                )
                card_lay.addWidget(div)

                # 2-column grid inside the card
                from PySide6.QtWidgets import QGridLayout as _Grid
                g = _Grid()
                g.setContentsMargins(0, 0, 0, 0)
                g.setHorizontalSpacing(6)
                g.setVerticalSpacing(1)
                g.setColumnStretch(0, 1)
                g.setColumnStretch(1, 0)

                for gi, combo in enumerate(group):
                    is_next = combo["combo_id"] == next_id
                    attempts = combo.get("total_attempts") or 0
                    ms = combo.get("mastery_score") or 0.0
                    mastered = (
                        attempts >= MIN_ATTEMPTS_FOR_MASTERY
                        and ms >= threshold
                    )
                    pct = min(int((ms / threshold) * 100), 100) if threshold > 0 else 0

                    nc = Color.PRIMARY if is_next else Color.TEXT
                    nw = "700" if is_next else "500"
                    n = QLabel(combo.get("combo_name", ""))
                    n.setFixedHeight(20)
                    n.setStyleSheet(
                        f"font-size: 12px; font-weight: {nw}; color: {nc};"
                    )
                    g.addWidget(n, gi, 0)

                    if mastered:
                        v = QLabel(Icon.CHECK)
                        v.setStyleSheet(
                            f"color: {Color.PRIMARY}; font-size: 12px;"
                            " font-weight: 700;"
                        )
                    else:
                        pc = Color.PRIMARY if pct >= 60 else (
                            Color.WARNING if pct > 0 else Color.TEXT_DISABLED
                        )
                        v = QLabel(f"{pct}%")
                        v.setStyleSheet(
                            f"color: {pc}; font-size: 11px; font-weight: 600;"
                        )
                    v.setFixedHeight(20)
                    v.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    g.addWidget(v, gi, 1)

                card_lay.addLayout(g)
                content_lay.addWidget(card)

        content_lay.addStretch()
        scroll.setWidget(content_w)
        lay.addWidget(scroll, stretch=1)

        self._combo_popup.raise_()
        self._combo_popup.show()

    def _close_combo_popup(self) -> None:
        if self._combo_popup is not None:
            self._combo_popup.close()
            self._combo_popup = None

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
                f"{Icon.PLAY}  Continue: {self._next_combo['combo_name']}"
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
        # Combo list is now shown in a popup — update the button text
        total = progress.get("total_combos", 0)
        mastered = progress.get("mastered_combos", 0)
        self._btn_view.setText(
            f"View All Combos  ({mastered}/{total} mastered)"
        )

    # ── Actions ──────────────────────────────────────────────────────────

    def _select_and_go(self, diff: str) -> None:
        """Select difficulty and go straight to config — like old GUI."""
        self._active_diff = diff
        if not self._curriculum:
            try:
                self._curriculum = ComboCurriculum()
            except Exception:
                logger.exception("Failed to create curriculum")
        self._next_combo = None
        if self._curriculum:
            self._next_combo = self._curriculum.get_next_combo(diff)
            # Fallback: grab the first combo from this difficulty
            if not self._next_combo:
                all_combos = self._curriculum.get_combos_by_difficulty(diff)
                if all_combos:
                    self._next_combo = all_combos[0]
                    logger.info(
                        "No unmastered combo — falling back to first: %s",
                        self._next_combo["combo_name"],
                    )

        combo_data = {}
        if self._next_combo:
            combo_data = {
                "id": self._next_combo["combo_id"],
                "name": self._next_combo["combo_name"],
                "seq": self._next_combo["combo_sequence"],
                "diff": diff.lower(),
            }
            logger.info("Selected combo: %s seq=%s", combo_data["name"], combo_data["seq"])
        else:
            # No combo found at all — use difficulty name as label
            combo_data = {
                "id": None,
                "name": f"{diff} Training",
                "seq": "",
                "diff": diff.lower(),
            }
            logger.warning("No combo found for difficulty %s", diff)

        self._router.navigate(
            "training_config",
            combo_id=combo_data.get("id"),
            combo=combo_data,
            difficulty=diff,
            curriculum=self._curriculum,
            username=self._username,
        )

    def _on_train_next(self) -> None:
        """Start button — go to config with the current difficulty."""
        self._select_and_go(self._active_diff)

    def _on_self_select(self) -> None:
        """Self-Select: open the custom sequence builder."""
        self._router.navigate("self_select", reset=True)

    # ── Lifecycle ────────────────────────────────────────────────────────

    def on_enter(self, **kwargs: Any) -> None:
        self._username = kwargs.get("username", "")
        level = kwargs.get("level", "Beginner")
        if level and level.title() in _LEVEL_HIERARCHY:
            self._active_diff = level.title()

        # Determine user level for lock/unlock
        user_level = kwargs.get("user_level", "Beginner")
        if user_level in _LEVEL_HIERARCHY:
            self._user_level_idx = _LEVEL_HIERARCHY.index(user_level)
        else:
            self._user_level_idx = 0

        # Guest mode: always start fresh (no username = guest)
        # Logged-in: reuse curriculum if it exists
        if not self._username:
            self._curriculum = None
        if not self._curriculum:
            try:
                self._curriculum = ComboCurriculum()
            except Exception:
                logger.exception("Failed to create curriculum")
                self._curriculum = None

        self._close_combo_popup()
        self._update_lock_state()
        self._refresh()
        logger.info("ComboSelectPage entered (difficulty=%s)", self._active_diff)

    def on_leave(self) -> None:
        self._close_combo_popup()
