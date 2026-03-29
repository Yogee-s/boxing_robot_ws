"""Full-screen sparring session page.

Timer, round counter, robot attack indicator, live offense/defense
stats, and stop button. Connects to gui_bridge signals.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from boxbunny_gui.theme import Color, Size, font, DANGER_BTN
from boxbunny_gui.widgets import BigButton, PunchCounter, TimerDisplay

if TYPE_CHECKING:
    from boxbunny_gui.gui_bridge import GuiBridge
    from boxbunny_gui.nav.router import PageRouter

logger = logging.getLogger(__name__)


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
        root.setContentsMargins(Size.SPACING, Size.SPACING_SM, Size.SPACING, Size.SPACING_SM)
        root.setSpacing(Size.SPACING_SM)

        # Top: timer + round counter
        top = QHBoxLayout()
        self._timer = TimerDisplay(font_size=Size.TEXT_TIMER_SM, show_ring=True)
        self._timer.finished.connect(self._on_timer_done)
        top.addWidget(self._timer, stretch=1)

        self._round_lbl = QLabel("Round 1/3")
        self._round_lbl.setFont(font(22, bold=True))
        self._round_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop)
        top.addWidget(self._round_lbl)
        root.addLayout(top)

        # Robot attack indicator
        self._attack_lbl = QLabel("Waiting...")
        self._attack_lbl.setFont(font(28, bold=True))
        self._attack_lbl.setStyleSheet(f"color: {Color.WARNING};")
        self._attack_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(self._attack_lbl)

        # Stats row
        stats = QHBoxLayout()
        stats.setSpacing(Size.SPACING)

        self._punch_counter = PunchCounter(label="YOUR PUNCHES")
        stats.addWidget(self._punch_counter)

        # Hits taken
        hits_col = QVBoxLayout()
        hits_col.setAlignment(Qt.AlignmentFlag.AlignCenter)
        h_label = QLabel("HITS TAKEN")
        h_label.setStyleSheet(f"color: {Color.TEXT_SECONDARY}; font-size: 14px; font-weight: bold;")
        h_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._hits_lbl = QLabel("0")
        self._hits_lbl.setFont(font(36, bold=True))
        self._hits_lbl.setStyleSheet(f"color: {Color.DANGER};")
        self._hits_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hits_col.addWidget(h_label)
        hits_col.addWidget(self._hits_lbl)
        stats.addLayout(hits_col)

        # Defense rate
        def_col = QVBoxLayout()
        def_col.setAlignment(Qt.AlignmentFlag.AlignCenter)
        d_label = QLabel("DEFENSE RATE")
        d_label.setStyleSheet(f"color: {Color.TEXT_SECONDARY}; font-size: 14px; font-weight: bold;")
        d_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._def_lbl = QLabel("--%")
        self._def_lbl.setFont(font(36, bold=True))
        self._def_lbl.setStyleSheet(f"color: {Color.PRIMARY};")
        self._def_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        def_col.addWidget(d_label)
        def_col.addWidget(self._def_lbl)
        stats.addLayout(def_col)

        root.addLayout(stats, stretch=1)

        # Stop button
        bottom = QHBoxLayout()
        bottom.addStretch()
        self._btn_stop = BigButton("STOP", stylesheet=DANGER_BTN)
        self._btn_stop.setFixedSize(80, 80)
        self._btn_stop.clicked.connect(self._on_stop)
        bottom.addWidget(self._btn_stop)
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
        # Update defense rate
        blocked = self._total_attacks - self._hits_taken
        rate = int(100 * blocked / self._total_attacks) if self._total_attacks else 0
        self._def_lbl.setText(f"{rate}%")

    def _on_timer_done(self) -> None:
        self._router.replace("sparring_results", config=self._config)

    def _on_stop(self) -> None:
        self._timer.pause()
        logger.info("Sparring stopped by user")
        self._router.replace("sparring_results", config=self._config)

    def _parse_seconds(self, val: str) -> int:
        return int(val.rstrip("s")) if val.rstrip("s").isdigit() else 90

    # ── Lifecycle ──────────────────────────────────────────────────────
    def on_enter(self, **kwargs: Any) -> None:
        self._config = kwargs.get("config", {})
        rounds = self._config.get("Rounds", "3")
        work_time = self._parse_seconds(self._config.get("Work Time", "90s"))
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
