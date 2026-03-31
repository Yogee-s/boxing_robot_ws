"""Session configuration page — selected combo + parameter tiles.

Matches the old GUI's config flow (rounds, work time, rest time, speed)
but consolidated into a single clean page.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from boxbunny_gui.theme import (
    Color, Icon, Size, font, PRIMARY_BTN,
    back_link_style, badge_style,
)
from boxbunny_gui.widgets import BigButton

if TYPE_CHECKING:
    from boxbunny_gui.nav.router import PageRouter

logger = logging.getLogger(__name__)

# Punch display helpers
_PUNCH_NAMES = {
    "1": "Jab", "2": "Cross", "3": "L Hook", "4": "R Hook",
    "5": "L Upper", "6": "R Upper",
}

_PARAMS: Dict[str, Dict] = {
    "Rounds":    {"opts": ["1", "2", "3", "5", "8"], "accent": "#4A90D9",
                  "default": 2},
    "Work Time": {"opts": ["60s", "90s", "120s", "180s"], "accent": "#56B886",
                  "default": 1},
    "Rest Time": {"opts": ["30s", "45s", "60s", "90s"], "accent": "#8B7EC8",
                  "default": 1},
    "Speed":     {"opts": ["Slow", "Medium", "Fast"], "accent": "#C88D2E",
                  "default": 1},
}


class _ParamTile(QPushButton):
    """Tappable tile with left accent that cycles through values."""

    def __init__(self, label: str, options: List[str], accent: str,
                 default: int = 0, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._label = label
        self._options = options
        self._index: int = default
        self._accent = accent
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(70)
        self._apply_style()
        self._update_text()
        self.clicked.connect(self._cycle)

    def _apply_style(self) -> None:
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {Color.SURFACE}; color: {Color.TEXT};
                border: 1px solid {Color.BORDER};
                border-left: 3px solid {self._accent};
                border-radius: {Size.RADIUS}px;
                font-size: 15px; font-weight: 600; padding: 10px 14px;
            }}
            QPushButton:hover {{
                background-color: {Color.SURFACE_HOVER};
                border-color: {self._accent};
                border-left: 3px solid {self._accent};
            }}
            QPushButton:pressed {{
                background-color: {self._accent}; color: #FFFFFF;
                border-color: {self._accent};
                border-left: 3px solid {self._accent};
            }}
        """)

    def _cycle(self) -> None:
        self._index = (self._index + 1) % len(self._options)
        self._update_text()

    def _update_text(self) -> None:
        self.setText(f"{self._label}\n{self._options[self._index]}")

    @property
    def value(self) -> str:
        return self._options[self._index]


class TrainingConfigPage(QWidget):
    """Configure training session parameters and start."""

    def __init__(self, router: PageRouter, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._router = router
        self._combo: Dict[str, Any] = {}
        self._tiles: Dict[str, _ParamTile] = {}
        self._curriculum = None
        self._difficulty: str = ""
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(32, 12, 32, 14)
        root.setSpacing(0)

        # ── Top bar ──────────────────────────────────────────────────────
        top = QHBoxLayout()
        btn_back = QPushButton(f"{Icon.BACK}  Back")
        btn_back.setStyleSheet(back_link_style())
        btn_back.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_back.clicked.connect(lambda: self._router.back())
        top.addWidget(btn_back)
        self._title = QLabel("Training Setup")
        self._title.setFont(font(Size.TEXT_SUBHEADER, bold=True))
        top.addWidget(self._title)
        top.addStretch()
        self._diff_badge = QLabel("TRAINING")
        self._diff_badge.setStyleSheet(badge_style(Color.PRIMARY))
        top.addWidget(self._diff_badge)
        root.addLayout(top)

        root.addSpacing(14)

        # ── Selected combo display ───────────────────────────────────────
        combo_lbl = QLabel("Selected Combo")
        combo_lbl.setStyleSheet(
            f"font-size: 13px; font-weight: 700; color: {Color.TEXT_SECONDARY};"
            " letter-spacing: 0.5px;"
        )
        root.addWidget(combo_lbl)
        root.addSpacing(6)

        # Combo info card — warm tint like sparring description
        self._combo_card = QWidget()
        self._combo_card.setStyleSheet(f"""
            QWidget {{
                background-color: #1A1510;
                border: 1px solid #3D2E1A;
                border-left: 3px solid {Color.PRIMARY};
                border-radius: {Size.RADIUS}px;
            }}
        """)
        card_lay = QHBoxLayout(self._combo_card)
        card_lay.setContentsMargins(16, 12, 16, 12)
        card_lay.setSpacing(16)

        self._combo_name_lbl = QLabel("Free Training")
        self._combo_name_lbl.setStyleSheet(
            f"font-size: 18px; font-weight: 700; color: {Color.TEXT};"
            " background: transparent; border: none;"
        )
        card_lay.addWidget(self._combo_name_lbl)

        self._combo_seq_lbl = QLabel("")
        self._combo_seq_lbl.setStyleSheet(
            f"font-size: 13px; color: {Color.TEXT_SECONDARY};"
            " background: transparent; border: none;"
        )
        card_lay.addWidget(self._combo_seq_lbl)
        card_lay.addStretch()

        root.addWidget(self._combo_card)

        root.addSpacing(18)

        # ── Parameters ───────────────────────────────────────────────────
        params_header = QHBoxLayout()
        params_lbl = QLabel("Parameters")
        params_lbl.setStyleSheet(
            f"font-size: 13px; font-weight: 700; color: {Color.TEXT_SECONDARY};"
            " letter-spacing: 0.5px;"
        )
        params_header.addWidget(params_lbl)
        params_header.addStretch()
        tap_hint = QLabel("Tap to cycle")
        tap_hint.setStyleSheet(f"font-size: 11px; color: {Color.TEXT_DISABLED};")
        params_header.addWidget(tap_hint)
        root.addLayout(params_header)
        root.addSpacing(8)

        # Row 1: Rounds, Work Time
        row1 = QHBoxLayout()
        row1.setSpacing(10)
        for key in ["Rounds", "Work Time"]:
            p = _PARAMS[key]
            tile = _ParamTile(key, p["opts"], p["accent"], p["default"], self)
            row1.addWidget(tile)
            self._tiles[key] = tile
        root.addLayout(row1)

        root.addSpacing(10)

        # Row 2: Rest Time, Speed
        row2 = QHBoxLayout()
        row2.setSpacing(10)
        for key in ["Rest Time", "Speed"]:
            p = _PARAMS[key]
            tile = _ParamTile(key, p["opts"], p["accent"], p["default"], self)
            row2.addWidget(tile)
            self._tiles[key] = tile
        root.addLayout(row2)

        root.addStretch(1)

        # ── Start button ─────────────────────────────────────────────────
        self._btn_start = BigButton(
            f"{Icon.PLAY}  Start Training", stylesheet=PRIMARY_BTN
        )
        self._btn_start.setFixedHeight(54)
        self._btn_start.clicked.connect(self._on_start)
        root.addWidget(self._btn_start)

    def _on_start(self) -> None:
        config = {k: t.value for k, t in self._tiles.items()}
        config["combo"] = self._combo
        logger.info("Starting training with config: %s", config)
        self._router.navigate(
            "training_session", config=config,
            curriculum=self._curriculum,
            combo_id=self._combo.get("id"),
            difficulty=self._difficulty,
        )

    def on_enter(self, **kwargs: Any) -> None:
        self._combo = kwargs.get("combo", {})
        self._curriculum = kwargs.get("curriculum")
        self._difficulty = kwargs.get("difficulty", "")

        combo_name = self._combo.get("name", "Free Training")
        self._combo_name_lbl.setText(combo_name)

        # Build readable sequence
        seq = self._combo.get("seq", "")
        if seq:
            parts = []
            for token in seq.split("-"):
                base = token.rstrip("b")
                name = _PUNCH_NAMES.get(base, token)
                if token.endswith("b"):
                    name += " (body)"
                parts.append(name)
            self._combo_seq_lbl.setText(" \u2192 ".join(parts))
        else:
            self._combo_seq_lbl.setText("")

        # Update badge
        if self._difficulty:
            self._diff_badge.setText(self._difficulty.upper())

        logger.debug("TrainingConfigPage entered (combo=%s)", combo_name)

    def on_leave(self) -> None:
        pass
