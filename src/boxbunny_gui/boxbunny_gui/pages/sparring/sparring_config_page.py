"""Sparring setup — two-step: style selection then parameters.

Step 1: Fighting style + description (centered, clean)
Step 2: Parameters slide in, description hides, Start button appears
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from PySide6.QtCore import Qt, QPropertyAnimation, QEasingCurve, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from boxbunny_gui.theme import (
    Color, Icon, Size, font, PRIMARY_BTN,
    badge_style, back_link_style,
)
from boxbunny_gui.widgets import BigButton

if TYPE_CHECKING:
    from boxbunny_gui.gui_bridge import GuiBridge
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


class _StyleCard(QPushButton):
    def __init__(self, name: str, sub: str, parent=None) -> None:
        super().__init__(parent)
        self.style_name = name
        self._sub = sub
        self._selected = False
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(76)
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
                    font-size: 16px; font-weight: 700; padding: 8px;
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
                    font-size: 16px; font-weight: 600; padding: 8px;
                }}
                QPushButton:hover {{
                    background-color: {Color.SURFACE_HOVER};
                    border-color: {Color.PRIMARY}; color: {Color.TEXT};
                }}
            """)
        self.setText(f"{self.style_name}\n{self._sub}")


_SPARRING_RANGES: Dict[str, Dict] = {
    "Rounds":   {"min": 1, "max": 12, "suffix": "", "step": 1},
    "Duration": {"min": 30, "max": 300, "suffix": "s", "step": 5},
    "Rest":     {"min": 15, "max": 180, "suffix": "s", "step": 5},
}


class _ParamTile(QPushButton):
    """Tappable tile: short tap cycles presets, long press opens custom slider."""

    def __init__(self, label: str, options: List[str], accent: str,
                 default: int = 0, parent=None) -> None:
        super().__init__(parent)
        self._label = label
        self._options = options
        self._index: int = default
        self._accent = accent
        self._custom_value: str = ""
        self._hold_timer = QTimer(self)
        self._hold_timer.setSingleShot(True)
        self._hold_timer.setInterval(600)
        self._hold_timer.timeout.connect(self._on_long_press)
        self._held = False
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(70)
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
            }}
        """)
        self._update_text()

    def mousePressEvent(self, event: Any) -> None:
        self._held = False
        self._hold_timer.start()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: Any) -> None:
        self._hold_timer.stop()
        if not self._held:
            self._cycle()
        super().mouseReleaseEvent(event)

    def _on_long_press(self) -> None:
        self._held = True
        rng = _SPARRING_RANGES.get(self._label)
        if rng is None:
            return
        dlg = _CustomParamDialog(
            self._label, rng["min"], rng["max"],
            rng["suffix"], rng["step"], self._accent,
        )
        from PySide6.QtWidgets import QApplication
        for w in QApplication.topLevelWidgets():
            if hasattr(w, '_boxbunny_app'):
                geo = w.frameGeometry()
                dlg.move(
                    geo.x() + (geo.width() - dlg.width()) // 2,
                    geo.y() + (geo.height() - dlg.height()) // 2,
                )
                break
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._custom_value = dlg.result_str
            self._update_text()

    def _cycle(self) -> None:
        self._custom_value = ""
        self._index = (self._index + 1) % len(self._options)
        self._update_text()

    def _update_text(self) -> None:
        if self._custom_value:
            self.setText(f"{self._label}\n{self._custom_value}")
        else:
            self.setText(f"{self._label}\n{self._options[self._index]}")

    @property
    def value(self) -> str:
        return self._custom_value if self._custom_value else self._options[self._index]


class _CustomParamDialog(QDialog):
    """Slider popup for setting a custom parameter value."""

    def __init__(self, label: str, min_val: int, max_val: int,
                 suffix: str, step: int, accent: str,
                 parent: QWidget | None = None) -> None:
        super().__init__()
        self.setObjectName("customParam")
        self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setFixedSize(320, 200)
        self.setStyleSheet(f"""
            QDialog#customParam {{
                background-color: {Color.BG}; color: {Color.TEXT};
                border: 1px solid {accent}; border-radius: 12px;
            }}
            QDialog#customParam QLabel {{ border: none; background: transparent; }}
        """)
        self._suffix = suffix
        self.result_str = f"{min_val}{suffix}"

        lay = QVBoxLayout(self)
        lay.setSpacing(12)
        lay.setContentsMargins(20, 16, 20, 16)

        title = QLabel(f"Set custom {label.lower()}")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(
            f"font-size: 14px; font-weight: 700; color: {Color.TEXT_SECONDARY};")
        lay.addWidget(title)

        self._val_lbl = QLabel(f"{min_val}{suffix}")
        self._val_lbl.setAlignment(Qt.AlignCenter)
        self._val_lbl.setStyleSheet(
            f"font-size: 28px; font-weight: 800; color: {accent};")
        lay.addWidget(self._val_lbl)

        slider = QSlider(Qt.Horizontal)
        slider.setRange(min_val // step, max_val // step)
        slider.setValue(min_val // step)
        slider.setMinimumHeight(40)
        slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                background: {Color.SURFACE}; height: 10px; border-radius: 5px;
            }}
            QSlider::handle:horizontal {{
                background: {accent}; width: 28px; height: 28px;
                margin: -9px 0; border-radius: 14px;
            }}
        """)
        slider.valueChanged.connect(
            lambda v: (
                self._val_lbl.setText(f"{v * step}{suffix}"),
                setattr(self, 'result_str', f"{v * step}{suffix}"),
            )
        )
        lay.addWidget(slider)

        btn = QPushButton("Apply")
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setFixedHeight(40)
        btn.setStyleSheet(
            f"background: {accent}; color: #fff; font-size: 14px;"
            f" font-weight: 700; border: none; border-radius: {Size.RADIUS}px;"
        )
        btn.clicked.connect(self.accept)
        lay.addWidget(btn)


class _DiffTile(QPushButton):
    def __init__(self, parent=None) -> None:
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

    _COUNTER_PROB = {"Easy": "30%", "Medium": "50%", "Hard": "80%"}

    def _refresh(self) -> None:
        name = _DIFFICULTIES[self._index]
        accent = _DIFF_COLORS[name]
        prob = self._COUNTER_PROB.get(name, "50%")
        self.setText(f"Difficulty\n{name} ({prob} counter)")
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
            }}
        """)

    @property
    def value(self) -> str:
        return _DIFFICULTIES[self._index]


def _make_info_box(label: str, value: str, bg: str, border: str) -> QWidget:
    box = QWidget()
    box.setStyleSheet(f"""
        QWidget {{ background-color: {bg}; border: 1px solid {border};
            border-radius: {Size.RADIUS_SM}px; }}
        QWidget QLabel {{ background: transparent; border: none; }}
    """)
    lay = QVBoxLayout(box)
    lay.setContentsMargins(12, 10, 12, 10)
    lay.setSpacing(2)
    lbl = QLabel(label.upper())
    lbl.setAlignment(Qt.AlignCenter)
    lbl.setStyleSheet(
        f"font-size: 10px; font-weight: 700; color: {Color.TEXT_DISABLED};"
        " letter-spacing: 1px;"
    )
    lay.addWidget(lbl)
    val = QLabel(value)
    val.setObjectName("val")
    val.setAlignment(Qt.AlignCenter)
    val.setStyleSheet(
        f"font-size: 20px; font-weight: 700; color: {Color.TEXT};"
    )
    lay.addWidget(val)
    return box


class SparringConfigPage(QWidget):
    def __init__(
        self,
        router: PageRouter,
        bridge: Optional["GuiBridge"] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._router = router
        self._bridge = bridge
        self._selected_style: str = _STYLE_ORDER[0]
        self._style_cards: list[_StyleCard] = []
        self._tiles: Dict[str, _ParamTile] = {}
        self._step = 1  # 1 = style selection, 2 = parameters
        self._username: str = ""
        # Base-tracking state — must exist before _build_ui/_connect_bridge
        # because ble_status_changed can fire immediately once wired.
        self._tracking_active: bool = False
        self._ble_connected: bool = False
        self._build_ui()
        if self._bridge is not None:
            self._bridge.ble_status_changed.connect(self._on_ble_status)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(32, 10, 32, 22)
        root.setSpacing(0)

        # ── Top bar ──────────────────────────────────────────────────────
        top = QHBoxLayout()
        btn_back = QPushButton(f"{Icon.BACK}  Back")
        btn_back.setStyleSheet(back_link_style())
        btn_back.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_back.clicked.connect(self._on_back)
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
            f"font-size: 15px; font-weight: 700; color: {Color.TEXT_SECONDARY};"
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

        root.addSpacing(12)

        # ── Description + info (collapsible) ─────────────────────────────
        self._desc_section = QWidget()
        desc_outer = QVBoxLayout(self._desc_section)
        desc_outer.setContentsMargins(0, 0, 0, 0)
        desc_outer.setSpacing(10)

        desc_row = QHBoxLayout()
        desc_row.setSpacing(10)

        self._desc_text = QLabel()
        self._desc_text.setWordWrap(True)
        self._desc_text.setTextFormat(Qt.TextFormat.RichText)
        self._desc_text.setStyleSheet(f"""
            font-size: 24px; color: {Color.TEXT};
            background-color: #1A1510;
            border: 1px solid #3D2E1A;
            border-left: 3px solid {Color.PRIMARY};
            border-radius: {Size.RADIUS}px;
            padding: 28px 32px;
        """)
        desc_row.addWidget(self._desc_text, stretch=3)

        info_col = QVBoxLayout()
        info_col.setSpacing(8)
        self._range_box = _make_info_box("Range", "", "#162030", "#2A4A6B")
        info_col.addWidget(self._range_box)
        self._punches_box = _make_info_box("Punches", "", "#1C1628", "#4A3570")
        info_col.addWidget(self._punches_box)
        desc_row.addLayout(info_col, stretch=1)

        desc_outer.addLayout(desc_row)
        root.addWidget(self._desc_section)
        self._update_description()

        # ── Parameters (hidden, appears below description) ───────────────
        self._params_section = QWidget()
        self._params_section.setMaximumHeight(0)
        params_lay = QVBoxLayout(self._params_section)
        params_lay.setContentsMargins(0, 8, 0, 0)
        params_lay.setSpacing(6)

        params_header = QHBoxLayout()
        params_lbl = QLabel("Parameters")
        params_lbl.setStyleSheet(
            f"font-size: 14px; font-weight: 700; color: {Color.TEXT_SECONDARY};"
            " letter-spacing: 0.5px;"
        )
        params_header.addWidget(params_lbl)
        params_header.addStretch()
        tap_hint = QLabel("Tap to cycle")
        tap_hint.setStyleSheet(f"font-size: 11px; color: {Color.TEXT_DISABLED};")
        params_header.addWidget(tap_hint)
        params_lay.addLayout(params_header)

        params_row = QHBoxLayout()
        params_row.setSpacing(10)
        for key in ["Rounds", "Duration", "Rest"]:
            p = _PARAMS[key]
            tile = _ParamTile(key, p["opts"], p["accent"], p.get("default", 0), self)
            params_row.addWidget(tile)
            self._tiles[key] = tile
        self._diff_tile = _DiffTile(self)
        params_row.addWidget(self._diff_tile)
        params_lay.addLayout(params_row)

        params_lay.addSpacing(10)

        # Dynamic counters toggle (robot counter-punches when user hits pads)
        counter_box = QWidget()
        counter_box.setStyleSheet(f"""
            QWidget {{
                background-color: {Color.SURFACE};
                border: 1px solid {Color.BORDER};
                border-radius: {Size.RADIUS}px;
            }}
        """)
        counter_box_lay = QVBoxLayout(counter_box)
        counter_box_lay.setContentsMargins(14, 10, 14, 14)
        counter_box_lay.setSpacing(8)

        counter_row = QHBoxLayout()
        counter_row.setSpacing(8)
        counter_hdr = QLabel("DYNAMIC COUNTERS")
        counter_hdr.setStyleSheet(
            f"font-size: 10px; font-weight: 700; color: {Color.TEXT_DISABLED};"
            " letter-spacing: 0.8px; background: transparent; border: none;")
        counter_row.addWidget(counter_hdr)
        counter_row.addStretch()
        self._counters_cb = QCheckBox()
        self._counters_cb.setChecked(True)
        self._counters_cb.setStyleSheet(f"""
            QCheckBox {{
                background: transparent; border: none;
                spacing: 6px;
            }}
            QCheckBox::indicator {{
                width: 36px; height: 20px; border-radius: 10px;
                background: {Color.BORDER};
            }}
            QCheckBox::indicator:checked {{
                background: {Color.PRIMARY};
            }}
        """)
        self._counters_cb.toggled.connect(self._on_counter_toggle)
        self._counter_status = QLabel("ON")
        self._counter_status.setStyleSheet(
            f"font-size: 11px; font-weight: 700; color: {Color.PRIMARY};"
            " background: transparent; border: none;")
        counter_row.addWidget(self._counters_cb)
        counter_row.addWidget(self._counter_status)
        counter_box_lay.addLayout(counter_row)

        # Counter speed selector (slides in when counters ON)
        # Uses a tap-to-cycle button: Slow → Medium → Fast → Custom
        # At Custom, a slider slides out beside it
        self._counter_speed_section = QWidget()
        self._counter_speed_section.setMaximumHeight(60)
        self._counter_speed_section.setStyleSheet("background: transparent; border: none;")
        cs_lay = QHBoxLayout(self._counter_speed_section)
        cs_lay.setContentsMargins(0, 6, 0, 0)
        cs_lay.setSpacing(6)

        _SPEED_ACCENT = "#FF6B35"
        self._cs_options = ["Slow", "Normal", "Fast", "Custom"]
        self._cs_idx = 1  # Normal
        self._cs_custom_rads: float = 15.0
        self._counter_speed = "medium"

        self._cs_btn = QPushButton("Normal Counter")
        self._cs_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._cs_btn.setFixedHeight(44)
        self._cs_btn.setStyleSheet(f"""
            QPushButton {{
                background: #1A2540; color: {Color.TEXT};
                font-size: 14px; font-weight: 600;
                border: 1px solid #2A3A5A;
                border-left: 3px solid {_SPEED_ACCENT};
                border-radius: {Size.RADIUS}px; padding: 6px 16px;
            }}
            QPushButton:hover {{
                background: #223050;
                border-color: {_SPEED_ACCENT};
                border-left: 3px solid {_SPEED_ACCENT};
            }}
            QPushButton:pressed {{ background: {_SPEED_ACCENT}; color: #fff; }}
        """)
        self._cs_btn.clicked.connect(self._cycle_counter_speed)
        cs_lay.addWidget(self._cs_btn)

        # Slider (hidden, slides in at Custom)
        from PySide6.QtWidgets import QSlider
        self._cs_slider_w = QWidget()
        self._cs_slider_w.setMaximumWidth(0)
        self._cs_slider_w.setStyleSheet("background: transparent; border: none;")
        sl_lay = QHBoxLayout(self._cs_slider_w)
        sl_lay.setContentsMargins(0, 0, 0, 0)
        sl_lay.setSpacing(4)
        self._cs_val_lbl = QLabel("15")
        self._cs_val_lbl.setFixedWidth(28)
        self._cs_val_lbl.setAlignment(Qt.AlignCenter)
        self._cs_val_lbl.setStyleSheet(
            f"font-size: 12px; font-weight: 700; color: {_SPEED_ACCENT};"
            " background: transparent; border: none;")
        sl_lay.addWidget(self._cs_val_lbl)
        self._cs_slider = QSlider(Qt.Horizontal)
        self._cs_slider.setRange(1, 30)
        self._cs_slider.setValue(15)
        self._cs_slider.setMinimumWidth(100)
        self._cs_slider.setMinimumHeight(30)
        self._cs_slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                background: {Color.BG}; height: 8px; border-radius: 4px;
            }}
            QSlider::handle:horizontal {{
                background: {_SPEED_ACCENT}; width: 22px; height: 22px;
                margin: -8px 0; border-radius: 11px;
                border: 2px solid {Color.SURFACE};
            }}
            QSlider::sub-page:horizontal {{
                background: {_SPEED_ACCENT}; border-radius: 4px;
            }}
        """)
        self._cs_slider.valueChanged.connect(self._on_cs_slide)
        sl_lay.addWidget(self._cs_slider)
        cs_lay.addWidget(self._cs_slider_w)

        counter_box_lay.addWidget(self._counter_speed_section)
        params_lay.addWidget(counter_box)

        # ── Base-tracking card ───────────────────────────────────────────
        # Pre-configure person tracking before entering the session. The
        # session page has the same toggle so the user can stop/restart
        # mid-session. Safety: session end always halts tracking.
        track_box = QWidget()
        track_box.setStyleSheet(f"""
            QWidget {{
                background-color: {Color.SURFACE};
                border: 1px solid {Color.BORDER};
                border-radius: {Size.RADIUS}px;
            }}
        """)
        track_lay = QHBoxLayout(track_box)
        track_lay.setContentsMargins(14, 10, 14, 10)
        track_lay.setSpacing(10)
        track_title = QLabel("BASE TRACKING")
        track_title.setStyleSheet(
            f"font-size: 10px; font-weight: 700; color: {Color.TEXT_DISABLED};"
            " letter-spacing: 0.8px; background: transparent; border: none;"
        )
        track_lay.addWidget(track_title)
        self._track_status_lbl = QLabel("BLE: …")
        self._track_status_lbl.setStyleSheet(
            f"font-size: 11px; color: {Color.TEXT_SECONDARY};"
            " background: transparent; border: none;"
        )
        track_lay.addWidget(self._track_status_lbl)
        track_lay.addStretch()
        self._btn_track = QPushButton("Start Tracking")
        self._btn_track.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_track.setFixedHeight(40)
        self._btn_track.setMinimumWidth(160)
        self._btn_track.setEnabled(False)  # locked until BLE connects
        self._btn_track.clicked.connect(self._on_track_toggle)
        track_lay.addWidget(self._btn_track)
        self._apply_track_button_style(active=False)
        params_lay.addWidget(track_box)

        root.addWidget(self._params_section)

        # Animations
        self._desc_anim = QPropertyAnimation(self._desc_section, b"maximumHeight")
        self._desc_anim.setDuration(300)
        self._desc_anim.setEasingCurve(QEasingCurve.Type.InOutCubic)

        self._params_anim = QPropertyAnimation(self._params_section, b"maximumHeight")
        self._params_anim.setDuration(300)
        self._params_anim.setEasingCurve(QEasingCurve.Type.InOutCubic)

        self._cs_anim = QPropertyAnimation(self._counter_speed_section, b"maximumHeight")
        self._cs_anim.setDuration(250)
        self._cs_anim.setEasingCurve(QEasingCurve.Type.InOutCubic)

        self._cs_sl_anim = QPropertyAnimation(self._cs_slider_w, b"maximumWidth")
        self._cs_sl_anim.setDuration(250)
        self._cs_sl_anim.setEasingCurve(QEasingCurve.Type.InOutQuad)

        root.addStretch(2)

        # ── Action button ────────────────────────────────────────────────
        self._btn_action = BigButton(
            f"{Icon.PLAY}  Choose Parameters", stylesheet=PRIMARY_BTN
        )
        self._btn_action.setFixedHeight(70)
        self._btn_action.clicked.connect(self._on_action)
        root.addWidget(self._btn_action)

    def _pick_style(self, name: str) -> None:
        self._selected_style = name
        self._refresh_style_selection()
        self._update_description()
        # If on step 2, go back to step 1
        if self._step == 2:
            self._show_step1()

    def _refresh_style_selection(self) -> None:
        for card in self._style_cards:
            card.set_selected(card.style_name == self._selected_style)

    def _update_description(self) -> None:
        info = _STYLES[self._selected_style]
        self._desc_text.setText(
            f'<span style="color:{Color.TEXT}; font-size:15px;">'
            f'{info["desc"]}</span>'
        )
        self._range_box.findChild(QLabel, "val").setText(info["range"])
        self._punches_box.findChild(QLabel, "val").setText(info["punches"])

    def _show_step1(self) -> None:
        """Show description, hide parameters."""
        self._step = 1
        self._btn_action.setText(f"{Icon.PLAY}  Choose Parameters")

        # Expand description
        self._desc_anim.stop()
        self._desc_anim.setStartValue(self._desc_section.maximumHeight())
        self._desc_anim.setEndValue(200)
        self._desc_anim.start()

        # Collapse parameters
        self._params_anim.stop()
        self._params_anim.setStartValue(self._params_section.height())
        self._params_anim.setEndValue(0)
        self._params_anim.start()

    def _show_step2(self) -> None:
        """Hide description, show parameters."""
        self._step = 2
        self._btn_action.setText(f"{Icon.PLAY}  Start Sparring")

        # Collapse description
        self._desc_anim.stop()
        self._desc_anim.setStartValue(self._desc_section.height())
        self._desc_anim.setEndValue(0)
        self._desc_anim.start()

        # Expand parameters
        self._params_anim.stop()
        self._params_anim.setStartValue(0)
        self._params_anim.setEndValue(250)
        self._params_anim.start()

    def imu_start(self) -> None:
        """Called by centre pad IMU."""
        self._on_action()

    def _on_action(self) -> None:
        if self._step == 1:
            self._show_step2()
        else:
            self._on_start()

    def _on_back(self) -> None:
        if self._step == 2:
            self._show_step1()
        else:
            self._router.back()

    def _on_counter_toggle(self, checked: bool) -> None:
        """Slide the counter speed section in/out."""
        self._counter_status.setText("ON" if checked else "OFF")
        self._counter_status.setStyleSheet(
            f"font-size: 11px; font-weight: 700;"
            f" color: {Color.PRIMARY if checked else Color.TEXT_DISABLED};"
            " background: transparent; border: none;")
        self._cs_anim.stop()
        self._cs_anim.setStartValue(self._counter_speed_section.maximumHeight())
        self._cs_anim.setEndValue(60 if checked else 0)
        self._cs_anim.start()
        if not checked:
            self._cs_sl_anim.stop()
            self._cs_sl_anim.setStartValue(self._cs_slider_w.maximumWidth())
            self._cs_sl_anim.setEndValue(0)
            self._cs_sl_anim.start()

    def _cycle_counter_speed(self) -> None:
        """Cycle Slow → Medium → Fast → Custom."""
        _SPEED_ACCENT = "#FF6B35"
        self._cs_idx = (self._cs_idx + 1) % len(self._cs_options)
        name = self._cs_options[self._cs_idx]
        _normal = f"""
            QPushButton {{
                background: #1A2540; color: {Color.TEXT};
                font-size: 13px; font-weight: 600;
                border: 1px solid #2A3A5A;
                border-left: 3px solid {_SPEED_ACCENT};
                border-radius: 6px; padding: 4px 12px;
            }}
            QPushButton:hover {{
                background: #223050;
                border-color: {_SPEED_ACCENT};
                border-left: 3px solid {_SPEED_ACCENT};
            }}
            QPushButton:pressed {{ background: {_SPEED_ACCENT}; color: #fff; }}
        """
        _custom = f"""
            QPushButton {{
                background: #1A2540; color: {Color.TEXT};
                font-size: 13px; font-weight: 600;
                border: 2px solid {_SPEED_ACCENT};
                border-radius: 6px; padding: 4px 12px;
            }}
            QPushButton:hover {{ background: #223050; }}
            QPushButton:pressed {{ background: {_SPEED_ACCENT}; color: #fff; }}
        """
        if name == "Custom":
            self._counter_speed = str(self._cs_custom_rads)
            self._cs_btn.setText(f"Custom ({int(self._cs_custom_rads)} rad/s)")
            self._cs_btn.setStyleSheet(_custom)
        else:
            self._counter_speed = {"Slow": "slow", "Normal": "medium", "Fast": "fast"}[name]
            self._cs_btn.setText(f"{name} Counter")
            self._cs_btn.setStyleSheet(_normal)
        cur = self._cs_slider_w.width()
        self._cs_sl_anim.stop()
        if name == "Custom":
            self._cs_sl_anim.setStartValue(cur)
            self._cs_sl_anim.setEndValue(16777215)
        else:
            self._cs_slider_w.setMaximumWidth(cur)
            self._cs_sl_anim.setStartValue(cur)
            self._cs_sl_anim.setEndValue(0)
        self._cs_sl_anim.start()

    def _on_cs_slide(self, val: int) -> None:
        self._cs_val_lbl.setText(str(val))
        self._cs_custom_rads = float(val)
        self._counter_speed = str(float(val))
        self._cs_btn.setText(f"Counter Speed: {val} rad/s")

    def _on_start(self) -> None:
        config = {k: t.value for k, t in self._tiles.items()}
        config["style"] = self._selected_style
        config["difficulty"] = self._diff_tile.value
        config["counters_enabled"] = self._counters_cb.isChecked()
        config["counter_speed"] = self._counter_speed if self._counters_cb.isChecked() else "medium"
        logger.info("Starting sparring: %s", config)
        self._router.navigate(
            "sparring_session", config=config, username=self._username,
            tracking_active=self._tracking_active,
        )

    def _save_as_preset(self) -> None:
        """Save sparring config as a preset."""
        if not self._username:
            return
        import json, sqlite3
        from pathlib import Path
        config = {k: t.value for k, t in self._tiles.items()}
        cfg_json = json.dumps({
            "rounds": int(config.get("Rounds", "3")),
            "work_sec": int(config.get("Duration", "90s").rstrip("s")),
            "rest_sec": int(config.get("Rest", "60s").rstrip("s")),
            "speed": "medium",
            "difficulty": self._diff_tile.value,
            "style": self._selected_style,
            "route": "sparring_session",
        })
        try:
            db_path = Path(__file__).resolve().parents[5] / "data" / "boxbunny_main.db"
            conn = sqlite3.connect(str(db_path))
            user_row = conn.execute("SELECT id FROM users WHERE username = ?", (self._username,)).fetchone()
            if user_row:
                conn.execute(
                    "INSERT INTO presets (user_id, name, description, preset_type, config_json) VALUES (?, ?, ?, 'sparring', ?)",
                    (user_row[0], f"Sparring: {self._selected_style}", f"{self._diff_tile.value} {self._selected_style}", cfg_json),
                )
                conn.commit()
                logger.info("Sparring preset saved for %s", self._username)
            conn.close()
        except Exception as exc:
            logger.warning("Failed to save preset: %s", exc)

    def on_enter(self, **kwargs: Any) -> None:
        self._username = kwargs.get("username", "")
        self._step = 1
        self._desc_section.setMaximumHeight(200)
        self._params_section.setMaximumHeight(0)
        self._btn_action.setText(f"{Icon.PLAY}  Choose Parameters")
        # Refresh tracking-button state in case BLE status or tracking
        # changed while the page was not shown.
        self._apply_track_button_style(active=self._tracking_active)
        logger.debug("SparringConfigPage entered")

    def on_leave(self) -> None:
        pass

    # ── Base tracking controls ──────────────────────────────────────────

    def _on_ble_status(self, status: Dict[str, Any]) -> None:
        self._ble_connected = bool(status.get("connected"))
        self._track_status_lbl.setText(
            "BLE: connected" if self._ble_connected else "BLE: disconnected"
        )
        # Sync with the bridge's authoritative tracking state — e.g. if
        # the user already enabled it from the session page.
        remote_tracking = bool(status.get("tracking"))
        if self._ble_connected and remote_tracking:
            self._tracking_active = True
        if not self._ble_connected and self._tracking_active:
            self._tracking_active = False
        self._apply_track_button_style(active=self._tracking_active)

    def _apply_track_button_style(self, *, active: bool) -> None:
        self._btn_track.setEnabled(self._ble_connected)
        if active:
            self._btn_track.setText("Stop Tracking")
            self._btn_track.setStyleSheet(f"""
                QPushButton {{
                    background-color: {Color.WARNING}; color: #FFFFFF;
                    font-size: 13px; font-weight: 700;
                    border: none; border-radius: {Size.RADIUS}px;
                    padding: 0 16px;
                }}
                QPushButton:hover {{ background-color: {Color.WARNING_DARK}; }}
                QPushButton:disabled {{
                    background-color: {Color.SURFACE};
                    color: {Color.TEXT_DISABLED};
                }}
            """)
        else:
            self._btn_track.setText("Start Tracking")
            self._btn_track.setStyleSheet(f"""
                QPushButton {{
                    background-color: {Color.PRIMARY}; color: #FFFFFF;
                    font-size: 13px; font-weight: 700;
                    border: none; border-radius: {Size.RADIUS}px;
                    padding: 0 16px;
                }}
                QPushButton:hover {{ background-color: {Color.PRIMARY_DARK}; }}
                QPushButton:disabled {{
                    background-color: {Color.SURFACE};
                    color: {Color.TEXT_DISABLED};
                }}
            """)

    def _on_track_toggle(self) -> None:
        if self._bridge is None:
            logger.warning("Tracking toggled but GUI bridge is offline")
            return
        if self._tracking_active:
            self._bridge.call_tracking_stop(self._on_track_stop_result)
        else:
            self._bridge.call_tracking_start(self._on_track_start_result)

    def _on_track_start_result(self, success: bool, message: str) -> None:
        if success:
            self._tracking_active = True
            self._apply_track_button_style(active=True)
            logger.info("Tracking started: %s", message)
        else:
            logger.warning("Tracking start failed: %s", message)

    def _on_track_stop_result(self, success: bool, message: str) -> None:
        self._tracking_active = False
        self._apply_track_button_style(active=False)
        if success:
            logger.info("Tracking stopped: %s", message)
        else:
            logger.warning("Tracking stop reported failure: %s", message)
