"""Sparring style selection and parameter configuration — premium layout."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from boxbunny_gui.theme import (
    Color, Icon, Size, font, PRIMARY_BTN,
    config_tile_style_v2, section_title_style, badge_style, back_link_style,
)
from boxbunny_gui.widgets import BigButton

if TYPE_CHECKING:
    from boxbunny_gui.nav.router import PageRouter

logger = logging.getLogger(__name__)

_STYLES = [
    ("Boxer", "Out-fighter"),
    ("Brawler", "Pressure"),
    ("Counter", "Exploit"),
    ("Pressure", "Forward"),
    ("Switch", "Rhythm"),
]

_DIFFICULTIES = ["Easy", "Medium", "Hard"]

_PARAMS: Dict[str, List[str]] = {
    "Rounds": ["1", "2", "3", "5"],
    "Work": ["60s", "90s", "120s", "180s"],
    "Rest": ["30s", "45s", "60s"],
    "Speed": ["Slow", "Medium", "Fast"],
}


class _StyleCard(QPushButton):
    """Selectable style card with premium selected state."""

    def __init__(self, name: str, desc: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.style_name = name
        self._desc = desc
        self._selected = False
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._update_style()

    def set_selected(self, selected: bool) -> None:
        self._selected = selected
        self._update_style()

    def _update_style(self) -> None:
        if self._selected:
            self.setStyleSheet(f"""
                QPushButton {{
                    background-color: {Color.PRIMARY_MUTED};
                    color: {Color.PRIMARY_LIGHT};
                    border: 2px solid {Color.PRIMARY};
                    border-radius: {Size.RADIUS_LG}px;
                    font-size: 14px; font-weight: 700;
                    padding: 8px;
                }}
            """)
        else:
            self.setStyleSheet(f"""
                QPushButton {{
                    background-color: {Color.SURFACE};
                    color: {Color.TEXT_SECONDARY};
                    border: 1px solid {Color.BORDER};
                    border-radius: {Size.RADIUS_LG}px;
                    font-size: 14px; font-weight: 600;
                    padding: 8px;
                }}
                QPushButton:hover {{
                    background-color: {Color.SURFACE_HOVER};
                    border-color: {Color.PRIMARY};
                    color: {Color.TEXT};
                }}
            """)
        self.setText(f"{self.style_name}\n{self._desc}")


class _ParamTile(QPushButton):
    """Tappable tile cycling through values, with accent top border."""

    def __init__(self, label: str, options: List[str], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._label = label
        self._options = options
        self._index: int = 0
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._update_text()
        self.setStyleSheet(config_tile_style_v2(Color.DANGER))
        self.clicked.connect(self._cycle)

    def _cycle(self) -> None:
        self._index = (self._index + 1) % len(self._options)
        self._update_text()

    def _update_text(self) -> None:
        self.setText(f"{self._label}\n{self._options[self._index]}")

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
        root.setContentsMargins(24, 12, 24, 12)
        root.setSpacing(10)

        # Back + title
        top = QHBoxLayout()
        btn_back = QPushButton(f"{Icon.BACK}  Back")
        btn_back.setStyleSheet(back_link_style())
        btn_back.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_back.clicked.connect(lambda: self._router.back())
        top.addWidget(btn_back)
        title = QLabel("Sparring Setup")
        title.setFont(font(Size.TEXT_SUBHEADER, bold=True))
        top.addWidget(title)
        top.addStretch()

        mode_badge = QLabel("SPARRING")
        mode_badge.setStyleSheet(badge_style(Color.DANGER))
        top.addWidget(mode_badge)
        root.addLayout(top)

        # Style section
        style_lbl = QLabel("FIGHTING STYLE")
        style_lbl.setStyleSheet(section_title_style(Color.DANGER))
        root.addWidget(style_lbl)

        styles_row = QHBoxLayout()
        styles_row.setSpacing(8)
        for name, desc in _STYLES:
            card = _StyleCard(name, desc, self)
            card.clicked.connect(
                lambda _c=False, n=name: self._pick_style(n)
            )
            styles_row.addWidget(card)
            self._style_cards.append(card)
        root.addLayout(styles_row, stretch=2)
        self._refresh_style_selection()

        # Parameters section
        params_lbl = QLabel("PARAMETERS")
        params_lbl.setStyleSheet(section_title_style(Color.DANGER))
        root.addWidget(params_lbl)

        params_grid = QGridLayout()
        params_grid.setSpacing(8)
        param_items = list(_PARAMS.items())
        for i, (label, opts) in enumerate(param_items):
            tile = _ParamTile(label, opts, self)
            params_grid.addWidget(tile, i // 3, i % 3)
            self._tiles[label] = tile

        self._diff_btn = QPushButton(
            f"Difficulty\n{_DIFFICULTIES[self._diff_index]}"
        )
        self._diff_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._diff_btn.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._diff_btn.setStyleSheet(config_tile_style_v2(Color.WARNING))
        self._diff_btn.clicked.connect(self._cycle_difficulty)
        params_grid.addWidget(self._diff_btn, 1, 1)
        root.addLayout(params_grid, stretch=3)

        # Start button
        btn_start = BigButton(
            f"{Icon.PLAY}  Start Sparring", stylesheet=PRIMARY_BTN
        )
        btn_start.setFixedHeight(60)
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
        self._diff_btn.setText(
            f"Difficulty\n{_DIFFICULTIES[self._diff_index]}"
        )

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
