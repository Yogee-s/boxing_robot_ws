"""Sparring style selection and parameter configuration."""
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

from boxbunny_gui.theme import Color, Size, font, GHOST_BTN, PRIMARY_BTN, SURFACE_BTN
from boxbunny_gui.widgets import BigButton

if TYPE_CHECKING:
    from boxbunny_gui.nav.router import PageRouter

logger = logging.getLogger(__name__)

_STYLES = [
    ("Boxer", "Classic out-fighter style"),
    ("Brawler", "Aggressive pressure fighter"),
    ("Counter-Puncher", "Wait and exploit openings"),
    ("Pressure", "Relentless forward movement"),
    ("Switch", "Alternates stance and rhythm"),
]

_DIFFICULTIES = ["Easy", "Medium", "Hard"]

_PARAMS: Dict[str, List[str]] = {
    "Rounds": ["1", "2", "3", "5"],
    "Work Time": ["60s", "90s", "120s", "180s"],
    "Rest Time": ["30s", "45s", "60s"],
    "Speed": ["Slow", "Medium", "Fast"],
}


class _StyleCard(QFrame):
    """Selectable style card with name and description."""

    def __init__(self, name: str, desc: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.style_name = name
        self._selected = False
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(180, 70)
        self._update_style()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 4, 8, 4)
        lbl = QLabel(name)
        lbl.setFont(font(16, bold=True))
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc_lbl = QLabel(desc)
        desc_lbl.setStyleSheet(f"color: {Color.TEXT_SECONDARY}; font-size: 11px;")
        desc_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc_lbl.setWordWrap(True)
        lay.addWidget(lbl)
        lay.addWidget(desc_lbl)

    def set_selected(self, selected: bool) -> None:
        self._selected = selected
        self._update_style()

    def _update_style(self) -> None:
        border = Color.PRIMARY if self._selected else Color.BORDER
        bg = Color.SURFACE_LIGHT if self._selected else Color.SURFACE
        self.setStyleSheet(
            f"QFrame {{ background-color: {bg}; border: 2px solid {border};"
            f" border-radius: {Size.RADIUS}px; }}"
        )


class _ParamTile(QFrame):
    """Tappable tile cycling through values."""

    def __init__(self, label: str, options: List[str], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._options = options
        self._index: int = 0
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(120, 64)
        self.setStyleSheet(
            f"QFrame {{ background-color: {Color.SURFACE};"
            f" border-radius: {Size.RADIUS_SM}px; }}"
        )
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 2, 4, 2)
        lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        t = QLabel(label)
        t.setStyleSheet(f"color: {Color.TEXT_SECONDARY}; font-size: 11px;")
        t.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._val = QLabel(options[0])
        self._val.setFont(font(18, bold=True))
        self._val.setStyleSheet(f"color: {Color.PRIMARY};")
        self._val.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(t)
        lay.addWidget(self._val)

    def mousePressEvent(self, event: Any) -> None:  # noqa: N802
        self._index = (self._index + 1) % len(self._options)
        self._val.setText(self._options[self._index])

    @property
    def value(self) -> str:
        return self._options[self._index]


class SparringConfigPage(QWidget):
    """Sparring mode configuration: style + parameters."""

    def __init__(self, router: PageRouter, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._router = router
        self._selected_style: str = _STYLES[0][0]
        self._diff_index: int = 1
        self._style_cards: list[_StyleCard] = []
        self._tiles: Dict[str, _ParamTile] = {}
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(Size.SPACING, Size.SPACING_SM, Size.SPACING, Size.SPACING_SM)
        root.setSpacing(Size.SPACING_SM)

        # Back
        top = QHBoxLayout()
        btn_back = BigButton("Back", stylesheet=GHOST_BTN)
        btn_back.setFixedWidth(100)
        btn_back.clicked.connect(lambda: self._router.back())
        top.addWidget(btn_back)
        title = QLabel("Sparring Setup")
        title.setFont(font(Size.TEXT_SUBHEADER, bold=True))
        top.addWidget(title)
        top.addStretch()
        root.addLayout(top)

        # Style cards row
        styles_row = QHBoxLayout()
        styles_row.setSpacing(Size.SPACING_SM)
        for name, desc in _STYLES:
            card = _StyleCard(name, desc, self)
            card.mousePressEvent = lambda _e, n=name: self._pick_style(n)
            styles_row.addWidget(card)
            self._style_cards.append(card)
        root.addLayout(styles_row)
        self._refresh_style_selection()

        # Param tiles + difficulty
        params_row = QHBoxLayout()
        params_row.setSpacing(Size.SPACING_SM)
        for label, opts in _PARAMS.items():
            tile = _ParamTile(label, opts, self)
            params_row.addWidget(tile)
            self._tiles[label] = tile

        self._diff_btn = BigButton(_DIFFICULTIES[self._diff_index], stylesheet=SURFACE_BTN)
        self._diff_btn.setFixedSize(120, 64)
        self._diff_btn.clicked.connect(self._cycle_difficulty)
        params_row.addWidget(self._diff_btn)
        root.addLayout(params_row)

        root.addStretch()

        # Start button
        btn_start = BigButton("Start Sparring", stylesheet=PRIMARY_BTN)
        btn_start.setFixedHeight(70)
        btn_start.clicked.connect(self._on_start)
        root.addWidget(btn_start)

    def _pick_style(self, name: str) -> None:
        self._selected_style = name
        self._refresh_style_selection()

    def _refresh_style_selection(self) -> None:
        for card in self._style_cards:
            card.set_selected(card.style_name == self._selected_style)

    def _cycle_difficulty(self) -> None:
        self._diff_index = (self._diff_index + 1) % len(_DIFFICULTIES)
        self._diff_btn.setText(_DIFFICULTIES[self._diff_index])

    def _on_start(self) -> None:
        config = {k: t.value for k, t in self._tiles.items()}
        config["style"] = self._selected_style
        config["difficulty"] = _DIFFICULTIES[self._diff_index]
        logger.info("Starting sparring: %s", config)
        self._router.navigate("sparring_session", config=config)

    def on_enter(self, **kwargs: Any) -> None:
        logger.debug("SparringConfigPage entered")

    def on_leave(self) -> None:
        pass
