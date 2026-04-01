"""Sparring style selection and parameter configuration."""
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
    badge_style, back_link_style,
)
from boxbunny_gui.widgets import BigButton

if TYPE_CHECKING:
    from boxbunny_gui.nav.router import PageRouter

logger = logging.getLogger(__name__)

_KW = f"color:{Color.PRIMARY}; font-weight:700"
_STYLES: Dict[str, Dict[str, str]] = {
    "Boxer": {
        "sub": "Out-fighter",
        "desc": f'<span style="{_KW}">Technical</span> style using '
                f'<span style="{_KW}">footwork</span> and '
                f'<span style="{_KW}">reach advantage</span>. '
                'Maintains distance with jabs and straights, avoiding close exchanges.',
        "range": "Long",
        "punches": "Jab / Cross",
    },
    "Brawler": {
        "sub": "Pressure",
        "desc": f'<span style="{_KW}">Aggressive</span> power puncher favouring '
                f'<span style="{_KW}">hooks</span> and '
                f'<span style="{_KW}">uppercuts</span> at close range. '
                f'<span style="{_KW}">High volume</span>, high risk, wears down opponents.',
        "range": "Close",
        "punches": "Hooks / Uppercuts",
    },
    "Counter": {
        "sub": "Exploit",
        "desc": f'<span style="{_KW}">Defensive</span> specialist. Waits for opponents '
                f'to attack, then <span style="{_KW}">counters</span> with '
                f'<span style="{_KW}">precise punches</span>. '
                f'Relies on <span style="{_KW}">timing</span> and reflexes.',
        "range": "Mid",
        "punches": "Cross / Counter",
    },
    "Pressure": {
        "sub": "Forward",
        "desc": f'<span style="{_KW}">Relentless</span> pressure fighter. '
                f'Constantly <span style="{_KW}">moves forward</span> cutting off the ring '
                f'with <span style="{_KW}">high-volume combinations</span>.',
        "range": "Close",
        "punches": "All / Body shots",
    },
    "Switch": {
        "sub": "Rhythm",
        "desc": f'<span style="{_KW}">Unpredictable</span> switch-hitter. '
                f'<span style="{_KW}">Alternates stances</span> and punch selections. '
                f'<span style="{_KW}">Hard to read</span> and prepare for.',
        "range": "Mid",
        "punches": "Mixed",
    },
}
_STYLE_ORDER = ["Boxer", "Brawler", "Counter", "Pressure", "Switch"]

_DIFFICULTIES = ["Easy", "Medium", "Hard"]
_DIFF_COLORS = {"Easy": "#3B9A6D", "Medium": "#C88D2E", "Hard": "#C0453A"}

_PARAMS: Dict[str, Dict] = {
    "Rounds":   {"opts": ["1", "2", "3", "5", "8", "12"], "accent": "#4A90D9",
                 "default": 2},
    "Duration": {"opts": ["30s", "60s", "90s", "120s", "150s", "180s"],
                 "accent": "#56B886", "default": 3},
    "Rest":     {"opts": ["30s", "60s", "90s", "120s"], "accent": "#8B7EC8",
                 "default": 1},
}


# ── Widgets ──────────────────────────────────────────────────────────────

class _StyleCard(QPushButton):
    def __init__(self, name: str, sub: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.style_name = name
        self._sub = sub
        self._selected = False
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(58)
        self._update_style()

    def set_selected(self, sel: bool) -> None:
        self._selected = sel
        self._update_style()

    def _update_style(self) -> None:
        if self._selected:
            self.setStyleSheet(f"""
                QPushButton {{
                    background-color: {Color.PRIMARY}; color: #FFFFFF;
                    border: 2px solid {Color.PRIMARY};
                    border-radius: {Size.RADIUS}px;
                    font-size: 14px; font-weight: 700; padding: 8px;
                }}
                QPushButton:hover {{ background-color: {Color.PRIMARY_DARK}; }}
            """)
        else:
            self.setStyleSheet(f"""
                QPushButton {{
                    background-color: {Color.SURFACE};
                    color: {Color.TEXT_SECONDARY};
                    border: 1px solid {Color.BORDER};
                    border-radius: {Size.RADIUS}px;
                    font-size: 14px; font-weight: 600; padding: 8px;
                }}
                QPushButton:hover {{
                    background-color: {Color.SURFACE_HOVER};
                    border-color: {Color.PRIMARY}; color: {Color.TEXT};
                }}
            """)
        self.setText(f"{self.style_name}\n{self._sub}")


class _ParamTile(QPushButton):
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


class _DiffTile(QPushButton):
    """Difficulty tile — color changes with value."""
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._index: int = 1
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(70)
        self.clicked.connect(self._cycle)
        self._refresh()

    def _cycle(self) -> None:
        self._index = (self._index + 1) % len(_DIFFICULTIES)
        self._refresh()

    def _refresh(self) -> None:
        name = _DIFFICULTIES[self._index]
        accent = _DIFF_COLORS[name]
        self.setText(f"Difficulty\n{name}")
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {Color.SURFACE}; color: {Color.TEXT};
                border: 1px solid {Color.BORDER};
                border-left: 3px solid {accent};
                border-radius: {Size.RADIUS}px;
                font-size: 15px; font-weight: 600; padding: 10px 14px;
            }}
            QPushButton:hover {{
                background-color: {Color.SURFACE_HOVER};
                border-color: {accent};
                border-left: 3px solid {accent};
            }}
            QPushButton:pressed {{
                background-color: {accent}; color: #FFFFFF;
                border-color: {accent};
                border-left: 3px solid {accent};
            }}
        """)

    @property
    def value(self) -> str:
        return _DIFFICULTIES[self._index]


# ═══════════════════════════════════════════════════════════════════════════

class SparringConfigPage(QWidget):
    def __init__(self, router: PageRouter, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._router = router
        self._selected_style: str = _STYLE_ORDER[0]
        self._style_cards: list[_StyleCard] = []
        self._tiles: Dict[str, _ParamTile] = {}
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
        title = QLabel("Sparring Setup")
        title.setStyleSheet(
            f"font-size: 20px; font-weight: 700; color: {Color.TEXT};"
        )
        top.addWidget(title)
        top.addStretch()
        mode_badge = QLabel("SPARRING")
        mode_badge.setStyleSheet(badge_style(Color.DANGER))
        top.addWidget(mode_badge)
        root.addLayout(top)

        root.addStretch(1)

        # ── Fighting Style ───────────────────────────────────────────────
        style_lbl = QLabel("Fighting Style")
        style_lbl.setStyleSheet(
            f"font-size: 13px; font-weight: 700; color: {Color.TEXT_SECONDARY};"
            " letter-spacing: 0.5px;"
        )
        root.addWidget(style_lbl)
        root.addSpacing(6)

        styles_row = QHBoxLayout()
        styles_row.setSpacing(8)
        for name in _STYLE_ORDER:
            card = _StyleCard(name, _STYLES[name]["sub"], self)
            card.clicked.connect(lambda _c=False, n=name: self._pick_style(n))
            styles_row.addWidget(card)
            self._style_cards.append(card)
        root.addLayout(styles_row)
        self._refresh_style_selection()

        root.addSpacing(16)

        # ── Description + Range/Punches side by side ─────────────────────
        desc_row = QHBoxLayout()
        desc_row.setSpacing(10)

        # Left: description text
        self._desc_text = QLabel()
        self._desc_text.setWordWrap(True)
        self._desc_text.setStyleSheet(f"""
            font-size: 14px; color: {Color.TEXT};
            background-color: #1A1510;
            border: 1px solid #3D2E1A;
            border-left: 3px solid {Color.PRIMARY};
            border-radius: {Size.RADIUS}px;
            padding: 16px 20px;
        """)
        desc_row.addWidget(self._desc_text, stretch=3)

        # Right: Range + Punches
        info_col = QVBoxLayout()
        info_col.setSpacing(8)
        self._range_box = self._make_info_box("Range", "", "#162030", "#2A4A6B")
        info_col.addWidget(self._range_box)
        self._punches_box = self._make_info_box("Punches", "", "#1C1628", "#4A3570")
        info_col.addWidget(self._punches_box)
        desc_row.addLayout(info_col, stretch=1)
        root.addLayout(desc_row)

        root.addStretch(2)

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
        root.addSpacing(6)

        # All params in one row: Rounds, Duration, Rest, Difficulty
        params_row = QHBoxLayout()
        params_row.setSpacing(10)
        for key in ["Rounds", "Duration", "Rest"]:
            p = _PARAMS[key]
            tile = _ParamTile(key, p["opts"], p["accent"], p.get("default", 0), self)
            params_row.addWidget(tile)
            self._tiles[key] = tile
        self._diff_tile = _DiffTile(self)
        params_row.addWidget(self._diff_tile)
        root.addLayout(params_row)

        root.addStretch(2)

        # ── Start button ─────────────────────────────────────────────────
        self._update_description()

        btn_start = BigButton(
            f"{Icon.PLAY}  Start Sparring", stylesheet=PRIMARY_BTN
        )
        btn_start.setFixedHeight(70)
        btn_start.clicked.connect(self._on_start)
        root.addWidget(btn_start)

    def _make_info_box(self, label: str, value: str,
                       bg: str = "", border: str = "") -> QWidget:
        """Small info card with label on top and value below."""
        _bg = bg or Color.SURFACE
        _border = border or Color.BORDER
        box = QWidget()
        box.setStyleSheet(f"""
            QWidget {{
                background-color: {_bg};
                border: 1px solid {_border};
                border-radius: {Size.RADIUS_SM}px;
            }}
        """)
        lay = QVBoxLayout(box)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(2)

        lbl = QLabel(label.upper())
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet(
            f"font-size: 9px; font-weight: 700; color: {Color.TEXT_DISABLED};"
            " letter-spacing: 1px; background: transparent; border: none;"
        )
        lay.addWidget(lbl)

        val = QLabel(value)
        val.setObjectName("val")
        val.setAlignment(Qt.AlignCenter)
        val.setStyleSheet(
            f"font-size: 15px; font-weight: 700; color: {Color.TEXT};"
            " background: transparent; border: none;"
        )
        lay.addWidget(val)
        return box

    def _pick_style(self, name: str) -> None:
        self._selected_style = name
        self._refresh_style_selection()
        self._update_description()

    def _refresh_style_selection(self) -> None:
        for card in self._style_cards:
            card.set_selected(card.style_name == self._selected_style)

    def _update_description(self) -> None:
        info = _STYLES[self._selected_style]
        self._desc_text.setTextFormat(Qt.TextFormat.RichText)
        self._desc_text.setText(
            f'<span style="color:{Color.TEXT}; font-size:14px;">'
            f'{info["desc"]}</span>'
        )
        self._range_box.findChild(QLabel, "val").setText(info["range"])
        self._punches_box.findChild(QLabel, "val").setText(info["punches"])

    def _on_start(self) -> None:
        config = {k: t.value for k, t in self._tiles.items()}
        config["style"] = self._selected_style
        config["difficulty"] = self._diff_tile.value
        logger.info("Starting sparring: %s", config)
        self._router.navigate("sparring_session", config=config)

    def on_enter(self, **kwargs: Any) -> None:
        logger.debug("SparringConfigPage entered")

    def on_leave(self) -> None:
        pass
