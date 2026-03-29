"""Session configuration page (single page replaces old 5-page flow).

Shows selected combo and tappable parameter tiles for rounds, work time,
rest time, and speed.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from boxbunny_gui.theme import Color, Size, font, GHOST_BTN, PRIMARY_BTN
from boxbunny_gui.widgets import BigButton

if TYPE_CHECKING:
    from boxbunny_gui.nav.router import PageRouter

logger = logging.getLogger(__name__)

_PARAMS: Dict[str, List[str]] = {
    "Rounds": ["1", "2", "3", "5"],
    "Work Time": ["60s", "90s", "120s", "180s"],
    "Rest Time": ["30s", "45s", "60s"],
    "Speed": ["Slow", "Medium", "Fast"],
}


class _ParamTile(QFrame):
    """Tappable tile that cycles through values on each tap."""

    def __init__(self, label: str, options: List[str], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._label = label
        self._options = options
        self._index: int = 0
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(220, 100)
        self.setStyleSheet(
            f"QFrame {{ background-color: {Color.SURFACE};"
            f" border-radius: {Size.RADIUS}px; }}"
            f" QFrame:hover {{ background-color: {Color.SURFACE_HOVER}; }}"
        )
        lay = QVBoxLayout(self)
        lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._title_lbl = QLabel(label)
        self._title_lbl.setStyleSheet(f"color: {Color.TEXT_SECONDARY}; font-size: 14px;")
        self._title_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._value_lbl = QLabel(options[0])
        self._value_lbl.setFont(font(24, bold=True))
        self._value_lbl.setStyleSheet(f"color: {Color.PRIMARY};")
        self._value_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self._title_lbl)
        lay.addWidget(self._value_lbl)

    def mousePressEvent(self, event: Any) -> None:  # noqa: N802
        self._index = (self._index + 1) % len(self._options)
        self._value_lbl.setText(self._options[self._index])
        super().mousePressEvent(event)

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
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(Size.SPACING, Size.SPACING_SM, Size.SPACING, Size.SPACING_SM)
        root.setSpacing(Size.SPACING)

        # Back + title
        top = QHBoxLayout()
        btn_back = BigButton("Back", stylesheet=GHOST_BTN)
        btn_back.setFixedWidth(100)
        btn_back.clicked.connect(lambda: self._router.back())
        top.addWidget(btn_back)
        self._title = QLabel("Configure Session")
        self._title.setFont(font(Size.TEXT_SUBHEADER, bold=True))
        top.addWidget(self._title)
        top.addStretch()
        root.addLayout(top)

        # Combo display area
        self._combo_lbl = QLabel()
        self._combo_lbl.setFont(font(20, bold=True))
        self._combo_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._combo_lbl.setWordWrap(True)
        root.addWidget(self._combo_lbl)

        # TODO: add ComboDisplay widget showing visual punch sequence

        # Parameter grid (2x2)
        grid = QGridLayout()
        grid.setSpacing(Size.SPACING)
        for i, (label, options) in enumerate(_PARAMS.items()):
            tile = _ParamTile(label, options, self)
            grid.addWidget(tile, i // 2, i % 2, alignment=Qt.AlignmentFlag.AlignCenter)
            self._tiles[label] = tile
        root.addLayout(grid)

        root.addStretch()

        # Start button
        self._btn_start = BigButton("Start", stylesheet=PRIMARY_BTN)
        self._btn_start.setFixedHeight(70)
        self._btn_start.clicked.connect(self._on_start)
        root.addWidget(self._btn_start)

    def _on_start(self) -> None:
        config = {k: t.value for k, t in self._tiles.items()}
        config["combo"] = self._combo
        logger.info("Starting training with config: %s", config)
        self._router.navigate("training_session", config=config)

    # ── Lifecycle ──────────────────────────────────────────────────────
    def on_enter(self, **kwargs: Any) -> None:
        self._combo = kwargs.get("combo", {})
        combo_name = self._combo.get("name", "Free Training")
        self._combo_lbl.setText(combo_name)
        logger.debug("TrainingConfigPage entered (combo=%s)", combo_name)

    def on_leave(self) -> None:
        pass
