"""Session configuration page (single page replaces old 5-page flow).

Shows selected combo and tappable parameter tiles for rounds, work time,
rest time, and speed. Enhanced with colored tile accents.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from boxbunny_gui.theme import (
    Color, Icon, Size, font, GHOST_BTN, PRIMARY_BTN,
    config_tile_style_v2, back_link_style, badge_style,
)
from boxbunny_gui.widgets import BigButton

if TYPE_CHECKING:
    from boxbunny_gui.nav.router import PageRouter

logger = logging.getLogger(__name__)

_PARAMS: Dict[str, Dict[str, Any]] = {
    "Rounds": {"options": ["1", "2", "3", "5"], "accent": Color.PRIMARY},
    "Work Time": {"options": ["60s", "90s", "120s", "180s"], "accent": Color.DANGER},
    "Rest Time": {"options": ["30s", "45s", "60s"], "accent": Color.INFO},
    "Speed": {"options": ["Slow", "Medium", "Fast"], "accent": Color.WARNING},
}


class _ParamTile(QPushButton):
    """Tappable tile with colored top accent that cycles through values."""

    def __init__(
        self, label: str, options: List[str],
        accent: str = "", parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._label_text = label
        self._options = options
        self._index: int = 0
        self._accent = accent or Color.PRIMARY
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(220, 110)
        self.setText("")
        self.setStyleSheet(config_tile_style_v2(self._accent))
        self.clicked.connect(self._cycle)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 12, 12, 8)
        lay.setSpacing(2)

        self._label_lbl = QLabel(label.upper())
        self._label_lbl.setStyleSheet(
            "background: transparent;"
            f" color: {Color.TEXT_DISABLED}; font-size: 10px;"
            " font-weight: 700; letter-spacing: 1px;"
        )
        self._label_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self._label_lbl)

        self._value_lbl = QLabel(self._options[0])
        self._value_lbl.setStyleSheet(
            "background: transparent;"
            f" color: {Color.TEXT}; font-size: 28px; font-weight: 700;"
        )
        self._value_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self._value_lbl, stretch=1)

        self._hint_lbl = QLabel("tap to change")
        self._hint_lbl.setStyleSheet(
            "background: transparent;"
            f" color: {Color.TEXT_DISABLED}; font-size: 10px;"
        )
        self._hint_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self._hint_lbl)

    def _cycle(self) -> None:
        self._index = (self._index + 1) % len(self._options)
        self._value_lbl.setText(self._options[self._index])

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
        root.setContentsMargins(24, 16, 24, 20)
        root.setSpacing(14)

        # Back link + title row
        top = QHBoxLayout()
        btn_back = QPushButton(f"{Icon.BACK}  Back")
        btn_back.setStyleSheet(back_link_style())
        btn_back.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_back.clicked.connect(lambda: self._router.back())
        top.addWidget(btn_back)
        top.addStretch()
        self._title = QLabel("Configure Session")
        self._title.setFont(font(Size.TEXT_SUBHEADER, bold=True))
        top.addWidget(self._title)
        top.addStretch()
        spacer = QLabel()
        spacer.setFixedWidth(80)
        top.addWidget(spacer)
        root.addLayout(top)

        # Combo badge display
        combo_row = QHBoxLayout()
        combo_row.addStretch()
        self._combo_lbl = QLabel()
        self._combo_lbl.setStyleSheet(
            f"color: {Color.PRIMARY}; font-size: 16px; font-weight: 600;"
            f" background-color: {Color.SURFACE};"
            f" border: 1px solid {Color.PRIMARY}30;"
            f" border-left: {Size.ACCENT_BAR_W}px solid {Color.PRIMARY};"
            f" border-radius: {Size.RADIUS}px;"
            " padding: 8px 24px;"
        )
        self._combo_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        combo_row.addWidget(self._combo_lbl)
        combo_row.addStretch()
        root.addLayout(combo_row)

        # Parameter grid (2x2)
        grid = QGridLayout()
        grid.setSpacing(14)
        for i, (label, cfg) in enumerate(_PARAMS.items()):
            tile = _ParamTile(
                label, cfg["options"],
                accent=cfg.get("accent", Color.PRIMARY),
                parent=self,
            )
            grid.addWidget(
                tile, i // 2, i % 2,
                alignment=Qt.AlignmentFlag.AlignCenter,
            )
            self._tiles[label] = tile
        root.addLayout(grid, stretch=1)

        # Start button
        self._btn_start = BigButton(
            f"{Icon.PLAY}  Start Training", stylesheet=PRIMARY_BTN
        )
        self._btn_start.setFixedHeight(64)
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
        self._combo_lbl.setText(combo_name)
        logger.debug("TrainingConfigPage entered (combo=%s)", combo_name)

    def on_leave(self) -> None:
        pass
