"""Full-screen sparring session — clean dark layout with live stats."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from boxbunny_gui.theme import Color, Icon, Size, font, badge_style
from boxbunny_gui.widgets import PunchCounter, TimerDisplay

if TYPE_CHECKING:
    from boxbunny_gui.gui_bridge import GuiBridge
    from boxbunny_gui.nav.router import PageRouter

logger = logging.getLogger(__name__)


def _stat_box(title: str, value: str, accent: str) -> tuple:
    """Compact stat display. Returns (widget, value_label)."""
    box = QWidget()
    box.setFixedHeight(70)
    box.setStyleSheet(f"""
        QWidget {{
            background-color: #131920;
            border: 1px solid #1E2832;
            border-left: 3px solid {accent};
            border-radius: {Size.RADIUS}px;
        }}
    """)
    lay = QVBoxLayout(box)
    lay.setContentsMargins(14, 6, 14, 6)
    lay.setSpacing(0)

    hdr = QLabel(title)
    hdr.setAlignment(Qt.AlignCenter)
    hdr.setStyleSheet(
        f"font-size: 10px; font-weight: 700; color: {Color.TEXT_DISABLED};"
        " letter-spacing: 0.8px; background: transparent; border: none;"
    )
    lay.addWidget(hdr)

    val = QLabel(value)
    val.setAlignment(Qt.AlignCenter)
    val.setStyleSheet(
        f"font-size: 24px; font-weight: 700; color: {Color.TEXT};"
        " background: transparent; border: none;"
    )
    lay.addWidget(val)
    return box, val


class SparringSessionPage(QWidget):
    """Live sparring session with offense/defense tracking."""

    def __init__(
        self,
        router: PageRouter,
        bridge: Optional[GuiBridge] = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._router = router
        self._bridge = bridge
        self._config: Dict[str, Any] = {}
        self._hits_taken: int = 0
        self._total_attacks: int = 0
        self._build_ui()
        self._connect_bridge()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(28, 10, 28, 12)
        root.setSpacing(0)

        # ── Top: round + mode ────────────────────────────────────────────
        top = QHBoxLayout()
        self._round_lbl = QLabel("Round 1/3")
        self._round_lbl.setStyleSheet(
            f"font-size: 15px; font-weight: 700; color: {Color.TEXT};"
            " background-color: #1A1214;"
            " border: 1px solid #3D1A22;"
            f" border-left: 3px solid {Color.DANGER};"
            f" border-radius: {Size.RADIUS_SM}px;"
            " padding: 6px 16px;"
        )
        top.addWidget(self._round_lbl)
        top.addStretch()
        mode_lbl = QLabel("SPARRING")
        mode_lbl.setStyleSheet(badge_style(Color.DANGER))
        top.addWidget(mode_lbl)
        root.addLayout(top)

        # ── Timer (main visual) ──────────────────────────────────────────
        self._timer = TimerDisplay(
            font_size=Size.TEXT_TIMER_SM, show_ring=True
        )
        self._timer.finished.connect(self._on_timer_done)
        root.addWidget(self._timer, stretch=1)

        root.addSpacing(6)

        # ── Incoming attack indicator ────────────────────────────────────
        attack_row = QHBoxLayout()
        attack_tag = QLabel("INCOMING")
        attack_tag.setStyleSheet(
            f"font-size: 10px; font-weight: 700; color: {Color.TEXT_DISABLED};"
            " letter-spacing: 1px;"
        )
        attack_row.addWidget(attack_tag)
        self._attack_lbl = QLabel("Waiting...")
        self._attack_lbl.setStyleSheet(
            f"font-size: 16px; font-weight: 700; color: {Color.WARNING};"
        )
        self._attack_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        attack_row.addWidget(self._attack_lbl, stretch=1)
        root.addLayout(attack_row)

        root.addSpacing(10)

        # ── Stats row ────────────────────────────────────────────────────
        stats_row = QHBoxLayout()
        stats_row.setSpacing(10)

        self._punch_counter = PunchCounter(label="YOUR PUNCHES")
        stats_row.addWidget(self._punch_counter)

        hits_box, self._hits_lbl = _stat_box("HITS TAKEN", "0", Color.DANGER)
        stats_row.addWidget(hits_box)

        def_box, self._def_lbl = _stat_box("DEFENSE", "--%", Color.INFO)
        stats_row.addWidget(def_box)

        root.addLayout(stats_row)

        root.addSpacing(10)

        # ── Stop button ──────────────────────────────────────────────────
        bottom = QHBoxLayout()
        bottom.addStretch()
        self._btn_stop = QPushButton(f"{Icon.STOP}  STOP")
        self._btn_stop.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_stop.setFixedSize(160, 44)
        self._btn_stop.setStyleSheet(f"""
            QPushButton {{
                background-color: {Color.DANGER}; color: white;
                font-size: 15px; font-weight: 700;
                border: none; border-radius: {Size.RADIUS}px;
            }}
            QPushButton:hover {{ background-color: {Color.DANGER_DARK}; }}
        """)
        self._btn_stop.clicked.connect(self._on_stop)
        bottom.addWidget(self._btn_stop)
        bottom.addStretch()
        root.addLayout(bottom)

    def _connect_bridge(self) -> None:
        if self._bridge is None:
            return
        self._bridge.punch_confirmed.connect(self._on_punch)
        self._bridge.defense_event.connect(self._on_defense)

    def _on_punch(self, data: Dict[str, Any]) -> None:
        self._punch_counter.increment()

    def _on_defense(self, data: Dict[str, Any]) -> None:
        self._total_attacks += 1
        if data.get("struck", False):
            self._hits_taken += 1
            self._hits_lbl.setText(str(self._hits_taken))
        self._attack_lbl.setText(data.get("robot_punch_code", ""))
        blocked = self._total_attacks - self._hits_taken
        rate = (
            int(100 * blocked / self._total_attacks)
            if self._total_attacks else 0
        )
        self._def_lbl.setText(f"{rate}%")

    def _on_timer_done(self) -> None:
        self._router.replace("sparring_results", config=self._config)

    def _on_stop(self) -> None:
        self._timer.pause()
        logger.info("Sparring stopped by user")
        self._router.replace("sparring_results", config=self._config)

    def _parse_seconds(self, val: str) -> int:
        return int(val.rstrip("s")) if val.rstrip("s").isdigit() else 90

    def on_enter(self, **kwargs: Any) -> None:
        self._config = kwargs.get("config", {})
        rounds = self._config.get("Rounds", "3")
        work_time = self._parse_seconds(
            self._config.get("Duration", self._config.get("Work", "90s"))
        )
        self._round_lbl.setText(f"Round 1/{rounds}")
        self._hits_taken = 0
        self._total_attacks = 0
        self._hits_lbl.setText("0")
        self._def_lbl.setText("--%")
        self._attack_lbl.setText("Waiting...")
        self._punch_counter.set_count(0)
        self._timer.start(work_time)
        logger.debug("SparringSessionPage entered")

    def on_leave(self) -> None:
        self._timer.pause()
