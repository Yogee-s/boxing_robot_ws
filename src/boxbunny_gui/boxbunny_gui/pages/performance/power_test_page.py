"""Power test — 3 punches per pad (left, centre, right) = 9 total.

Shows 3 pad columns with checkmarks. Records acceleration, shows peak.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from boxbunny_gui.theme import Color, Icon, Size, PRIMARY_BTN, back_link_style
from boxbunny_gui.widgets import BigButton, TimerDisplay

if TYPE_CHECKING:
    from boxbunny_gui.gui_bridge import GuiBridge
    from boxbunny_gui.nav.router import PageRouter

logger = logging.getLogger(__name__)

_PUNCHES_PER_PAD = 3
_PADS = ["left", "centre", "right"]
_PAD_LABELS = {"left": "Left Pad", "centre": "Centre Pad", "right": "Right Pad"}
_PAD_COLORS = {"left": Color.INFO, "centre": Color.PRIMARY, "right": Color.PURPLE}

_STATE_INSTRUCTIONS = "instructions"
_STATE_COUNTDOWN = "countdown"
_STATE_ACTIVE = "active"
_STATE_RESULTS = "results"

_KW = f"color:{Color.PRIMARY_LIGHT}; font-weight:600"


def _stat_tile(title: str, value: str, accent: str) -> QWidget:
    w = QWidget()
    w.setObjectName("tile")
    w.setStyleSheet(f"""
        QWidget#tile {{
            background-color: #131920;
            border: 1px solid #1E2832;
            border-left: 3px solid {accent};
            border-radius: {Size.RADIUS}px;
        }}
        QWidget#tile QLabel {{ background: transparent; border: none; }}
    """)
    lay = QVBoxLayout(w)
    lay.setContentsMargins(14, 10, 14, 10)
    lay.setSpacing(2)
    hdr = QLabel(title.upper())
    hdr.setStyleSheet(
        f"font-size: 10px; font-weight: 700; color: {Color.TEXT_DISABLED};"
        " letter-spacing: 0.8px;"
    )
    lay.addWidget(hdr)
    val = QLabel(value)
    val.setObjectName("val")
    val.setStyleSheet(f"font-size: 22px; font-weight: 700; color: {Color.TEXT};")
    lay.addWidget(val)
    return w


class _PadCard(QWidget):
    """Pad card with coloured top border, peak value, and 3 checkboxes."""

    def __init__(self, pad: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.pad = pad
        self.punches: List[float] = []
        self._color = _PAD_COLORS.get(pad, Color.TEXT)

        self.setObjectName(f"pc_{pad}")
        self.setStyleSheet(f"""
            QWidget#pc_{pad} {{
                background-color: transparent;
                border: none;
            }}
            QWidget#pc_{pad} QLabel {{ background: transparent; border: none; }}
        """)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 18, 16, 18)
        lay.setSpacing(8)
        lay.setAlignment(Qt.AlignCenter)

        # Pad name
        name = QLabel(_PAD_LABELS.get(pad, pad))
        name.setAlignment(Qt.AlignCenter)
        name.setStyleSheet(
            f"font-size: 20px; font-weight: 700; color: {self._color};"
        )
        lay.addWidget(name)

        # Peak value — big
        self._peak_lbl = QLabel("--")
        self._peak_lbl.setAlignment(Qt.AlignCenter)
        self._peak_lbl.setStyleSheet(
            f"font-size: 44px; font-weight: 700; color: {self._color};"
        )
        lay.addWidget(self._peak_lbl)

        unit_lbl = QLabel("m/s\u00B2")
        unit_lbl.setAlignment(Qt.AlignCenter)
        unit_lbl.setStyleSheet(f"font-size: 12px; color: {Color.TEXT_DISABLED};")
        lay.addWidget(unit_lbl)

        lay.addSpacing(8)

        # 3 checkboxes
        checks_row = QHBoxLayout()
        checks_row.setAlignment(Qt.AlignCenter)
        checks_row.setSpacing(12)
        self._checks: list[QLabel] = []
        for i in range(_PUNCHES_PER_PAD):
            check = QLabel("")
            check.setFixedSize(44, 44)
            check.setAlignment(Qt.AlignCenter)
            check.setStyleSheet(f"""
                font-size: 18px; font-weight: 700;
                color: {Color.TEXT_DISABLED};
                background-color: {Color.SURFACE};
                border: 2px solid {Color.BORDER};
                border-radius: 8px;
            """)
            checks_row.addWidget(check)
            self._checks.append(check)
        lay.addLayout(checks_row)

    def record_punch(self, accel: float) -> bool:
        if len(self.punches) >= _PUNCHES_PER_PAD:
            return True
        self.punches.append(accel)
        idx = len(self.punches) - 1

        self._checks[idx].setText(Icon.CHECK)
        self._checks[idx].setStyleSheet(f"""
            font-size: 20px; font-weight: 700;
            color: #FFFFFF;
            background-color: {self._color};
            border: 2px solid {self._color};
            border-radius: 8px;
        """)

        peak = max(self.punches)
        self._peak_lbl.setText(f"{peak:.1f}")
        return len(self.punches) >= _PUNCHES_PER_PAD

    def reset(self) -> None:
        self.punches.clear()
        self._peak_lbl.setText("--")
        for check in self._checks:
            check.setText("")
            check.setStyleSheet(f"""
                font-size: 18px; font-weight: 700;
                color: {Color.TEXT_DISABLED};
                background-color: {Color.SURFACE};
                border: 2px solid {Color.BORDER};
                border-radius: 8px;
            """)

    @property
    def is_complete(self) -> bool:
        return len(self.punches) >= _PUNCHES_PER_PAD

    @property
    def peak(self) -> float:
        return max(self.punches) if self.punches else 0.0


class PowerTestPage(QWidget):
    def __init__(self, router: PageRouter, bridge: Optional[GuiBridge] = None,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._router = router
        self._bridge = bridge
        self._state: str = _STATE_INSTRUCTIONS
        self._current_pad_idx: int = 0
        self._pad_cards: list[_PadCard] = []
        self._build_ui()
        if self._bridge:
            self._bridge.punch_confirmed.connect(self._on_punch)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(32, 10, 32, 22)
        root.setSpacing(0)

        # Top bar
        top = QHBoxLayout()
        btn_back = QPushButton(f"{Icon.BACK}  Back")
        btn_back.setStyleSheet(back_link_style())
        btn_back.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_back.clicked.connect(self._on_back)
        top.addWidget(btn_back)
        title = QLabel("Power Test")
        title.setStyleSheet(
            f"font-size: 20px; font-weight: 700; color: {Color.TEXT};"
        )
        top.addWidget(title)
        top.addStretch()
        root.addLayout(top)

        # ── Instructions ─────────────────────────────────────────────────
        self._instr_widget = QWidget()
        self._instr_widget.setObjectName("instr")
        self._instr_widget.setStyleSheet(f"""
            QWidget#instr {{
                background-color: #131920;
                border: 1px solid #1E2832;
                border-radius: {Size.RADIUS_LG}px;
            }}
            QWidget#instr QLabel {{ background: transparent; border: none; }}
        """)
        instr_lay = QVBoxLayout(self._instr_widget)
        instr_lay.setAlignment(Qt.AlignCenter)
        instr_lay.setContentsMargins(30, 24, 30, 24)
        instr_lay.setSpacing(14)

        instr_title = QLabel("Max Power Test")
        instr_title.setAlignment(Qt.AlignCenter)
        instr_title.setStyleSheet(
            f"font-size: 32px; font-weight: 700; color: {Color.TEXT};"
        )
        instr_lay.addWidget(instr_title)

        instr_text = QLabel(
            f'Punch each pad <span style="{_KW}">3 times</span> as '
            f'<span style="{_KW}">hard as you can</span>. '
            f'We will measure your <span style="{_KW}">peak acceleration</span>.'
        )
        instr_text.setTextFormat(Qt.TextFormat.RichText)
        instr_text.setAlignment(Qt.AlignCenter)
        instr_text.setWordWrap(False)
        instr_text.setStyleSheet(
            f"font-size: 16px; color: {Color.TEXT};"
        )
        instr_lay.addWidget(instr_text)

        from boxbunny_gui.theme import hero_btn_style
        self._btn_begin = QPushButton("Begin Test")
        self._btn_begin.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_begin.setFixedHeight(76)
        self._btn_begin.setFixedWidth(350)
        self._btn_begin.setStyleSheet(hero_btn_style(size=22))
        self._btn_begin.clicked.connect(self._start_countdown)
        btn_wrap = QHBoxLayout()
        btn_wrap.setAlignment(Qt.AlignCenter)
        btn_wrap.addWidget(self._btn_begin)
        instr_lay.addLayout(btn_wrap)

        root.addSpacing(10)
        root.addWidget(self._instr_widget, stretch=1)
        root.addSpacing(10)

        # ── Countdown ────────────────────────────────────────────────────
        self._countdown = TimerDisplay(font_size=Size.TEXT_TIMER, show_ring=True)
        self._countdown.finished.connect(self._start_active)
        self._countdown.setVisible(False)
        root.addWidget(self._countdown, stretch=1)

        # ── Active phase ─────────────────────────────────────────────────
        self._active_widget = QWidget()
        active_lay = QVBoxLayout(self._active_widget)
        active_lay.setSpacing(0)
        active_lay.setContentsMargins(20, 0, 20, 0)

        active_lay.addStretch(1)

        active_title = QLabel("Throw 3 power punches to each pad")
        active_title.setAlignment(Qt.AlignCenter)
        active_title.setStyleSheet(
            f"font-size: 26px; font-weight: 700; color: {Color.TEXT};"
        )
        active_lay.addWidget(active_title)

        active_sub = QLabel("Hit as hard as you can!")
        active_sub.setAlignment(Qt.AlignCenter)
        active_sub.setStyleSheet(
            f"font-size: 18px; font-weight: 600; color: {Color.PRIMARY_LIGHT};"
        )
        active_lay.addWidget(active_sub)

        active_lay.addSpacing(32)

        # 3 pad cards — each inside a tinted wrapper card
        cols = QHBoxLayout()
        cols.setSpacing(14)
        for pad in _PADS:
            color = _PAD_COLORS.get(pad, Color.TEXT)
            wrapper = QWidget()
            wrapper.setObjectName(f"pw_{pad}")
            wrapper.setStyleSheet(f"""
                QWidget#pw_{pad} {{
                    background-color: #131920;
                    border: 1px solid #1E2832;
                    border-top: 3px solid {color};
                    border-radius: {Size.RADIUS}px;
                }}
                QWidget#pw_{pad} QLabel {{
                    background: transparent; border: none;
                }}
            """)
            w_lay = QVBoxLayout(wrapper)
            w_lay.setContentsMargins(6, 8, 6, 8)
            card = _PadCard(pad, wrapper)
            w_lay.addWidget(card)
            cols.addWidget(wrapper)
            self._pad_cards.append(card)
        active_lay.addLayout(cols)

        active_lay.addStretch(1)

        self._active_widget.setVisible(False)
        root.addWidget(self._active_widget, stretch=1)

        # ── Results ──────────────────────────────────────────────────────
        self._results_widget = QWidget()
        res_lay = QVBoxLayout(self._results_widget)
        res_lay.setSpacing(12)
        res_lay.setContentsMargins(0, 0, 0, 0)

        res_lay.addStretch(1)

        res_title = QLabel("Power Test Results")
        res_title.setAlignment(Qt.AlignCenter)
        res_title.setStyleSheet(
            f"font-size: 26px; font-weight: 700; color: {Color.TEXT};"
        )
        res_lay.addWidget(res_title)

        res_lay.addSpacing(16)

        # 3 pad results in a row
        res_row = QHBoxLayout()
        res_row.setSpacing(10)
        self._res_left = _stat_tile("Left Peak", "--", Color.INFO)
        self._res_centre = _stat_tile("Centre Peak", "--", Color.PRIMARY)
        self._res_right = _stat_tile("Right Peak", "--", Color.PURPLE)
        res_row.addWidget(self._res_left)
        res_row.addWidget(self._res_centre)
        res_row.addWidget(self._res_right)
        res_lay.addLayout(res_row)

        res_lay.addSpacing(14)

        # Overall peak — big, centered
        self._res_overall_lbl = QLabel("--")
        self._res_overall_lbl.setAlignment(Qt.AlignCenter)
        self._res_overall_lbl.setStyleSheet(
            f"font-size: 52px; font-weight: 700; color: {Color.PRIMARY};"
        )
        res_lay.addWidget(self._res_overall_lbl)

        overall_sub = QLabel("Overall Peak (m/s\u00B2)")
        overall_sub.setAlignment(Qt.AlignCenter)
        overall_sub.setStyleSheet(
            f"font-size: 14px; color: {Color.TEXT_SECONDARY};"
        )
        res_lay.addWidget(overall_sub)

        res_lay.addStretch(1)

        btn_done = BigButton("Done", stylesheet=PRIMARY_BTN)
        btn_done.setFixedHeight(70)
        btn_done.clicked.connect(lambda: self._router.navigate("performance"))
        res_lay.addWidget(btn_done)

        self._results_widget.setVisible(False)
        root.addWidget(self._results_widget, stretch=1)

    def _set_state(self, state: str) -> None:
        self._state = state
        self._instr_widget.setVisible(state == _STATE_INSTRUCTIONS)
        self._countdown.setVisible(state == _STATE_COUNTDOWN)
        self._active_widget.setVisible(state == _STATE_ACTIVE)
        self._results_widget.setVisible(state == _STATE_RESULTS)

    def imu_start(self) -> None:
        """Called by centre pad IMU to begin the test."""
        if self._state == _STATE_INSTRUCTIONS:
            self._start_countdown()

    def _start_countdown(self) -> None:
        self._set_state(_STATE_COUNTDOWN)
        self._countdown.set_overlay("Get Ready")
        self._cd_remaining = 3
        QTimer.singleShot(1000, self._cd_tick)

    def _cd_tick(self) -> None:
        if self._cd_remaining > 0:
            self._countdown.set_overlay(str(self._cd_remaining))
            self._cd_remaining -= 1
            QTimer.singleShot(1000, self._cd_tick)
        else:
            self._countdown.set_overlay("GO!")
            QTimer.singleShot(500, self._start_active)

    def _start_active(self) -> None:
        self._countdown.clear_overlay()
        for card in self._pad_cards:
            card.reset()
        self._set_state(_STATE_ACTIVE)

    def _on_punch(self, data: Dict[str, Any]) -> None:
        if self._state != _STATE_ACTIVE:
            return
        force = data.get("force", 0.0)
        accel = force * 60.0
        pad = data.get("pad", "")

        # Only update the card that matches the punch pad — no fallback
        target = None
        for card in self._pad_cards:
            if card.pad == pad and not card.is_complete:
                target = card
                break

        if target is None:
            # Ignore punches from unrecognised or already-complete pads
            return

        target.record_punch(accel)
        if all(c.is_complete for c in self._pad_cards):
            QTimer.singleShot(800, self._show_results)

    def _show_results(self) -> None:
        peaks = [c.peak for c in self._pad_cards]
        self._res_left.findChild(QLabel, "val").setText(
            f"{peaks[0]:.1f} m/s\u00B2" if peaks[0] > 0 else "--"
        )
        self._res_centre.findChild(QLabel, "val").setText(
            f"{peaks[1]:.1f} m/s\u00B2" if peaks[1] > 0 else "--"
        )
        self._res_right.findChild(QLabel, "val").setText(
            f"{peaks[2]:.1f} m/s\u00B2" if peaks[2] > 0 else "--"
        )
        overall = max(peaks) if peaks else 0
        self._res_overall_lbl.setText(
            f"{overall:.1f}" if overall > 0 else "--"
        )

        from boxbunny_gui.session_tracker import get_tracker
        get_tracker().add_session(
            mode="Performance",
            duration="Power Test",
            punches="9",
            score=f"{overall:.1f} m/s\u00B2",
        )
        self._set_state(_STATE_RESULTS)

    def _on_back(self) -> None:
        self._countdown.reset()
        self._router.back()

    def on_enter(self, **kwargs: Any) -> None:
        for card in self._pad_cards:
            card.reset()
        self._set_state(_STATE_INSTRUCTIONS)
        logger.debug("PowerTestPage entered")

    def on_leave(self) -> None:
        self._countdown.reset()
