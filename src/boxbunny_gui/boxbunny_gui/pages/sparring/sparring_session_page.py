"""Full-screen sparring session page — red-themed treatment.

Timer, round counter, robot attack indicator, live offense/defense
stats, and stop button. Connects to gui_bridge signals.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from boxbunny_gui.theme import Color, Icon, Size, font, badge_style
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
        root.setContentsMargins(24, 10, 24, 10)
        root.setSpacing(8)

        # ── Top: round counter + mode badge ──────────────────────────────
        top = QHBoxLayout()
        self._round_lbl = QLabel("Round 1/3")
        self._round_lbl.setFont(font(18, bold=True))
        self._round_lbl.setStyleSheet(
            f"color: {Color.TEXT};"
            f" background-color: {Color.SURFACE};"
            f" border-left: {Size.ACCENT_BAR_W}px solid {Color.DANGER};"
            f" border-radius: {Size.RADIUS_SM}px;"
            " padding: 6px 16px;"
        )
        top.addWidget(self._round_lbl)
        top.addStretch()

        mode_lbl = QLabel("SPARRING")
        mode_lbl.setStyleSheet(badge_style(Color.DANGER))
        top.addWidget(mode_lbl)
        root.addLayout(top)

        # ── Timer ────────────────────────────────────────────────────────
        self._timer = TimerDisplay(
            font_size=Size.TEXT_TIMER_SM, show_ring=True
        )
        self._timer.finished.connect(self._on_timer_done)
        root.addWidget(self._timer, stretch=1)

        # ── Robot attack indicator ───────────────────────────────────────
        attack_frame = QFrame()
        attack_frame.setFixedHeight(50)
        attack_frame.setStyleSheet(
            f"QFrame {{ background-color: {Color.SURFACE};"
            f" border: 1px solid {Color.BORDER};"
            f" border-left: {Size.ACCENT_BAR_W}px solid {Color.WARNING};"
            f" border-radius: {Size.RADIUS}px; }}"
        )
        attack_lay = QHBoxLayout(attack_frame)
        attack_lay.setContentsMargins(16, 0, 16, 0)
        attack_tag = QLabel("INCOMING")
        attack_tag.setStyleSheet(
            "background: transparent;"
            f" color: {Color.TEXT_DISABLED}; font-size: 11px;"
            " font-weight: 700; letter-spacing: 1px;"
        )
        attack_lay.addWidget(attack_tag)
        self._attack_lbl = QLabel("Waiting...")
        self._attack_lbl.setFont(font(18, bold=True))
        self._attack_lbl.setStyleSheet(
            f"background: transparent; color: {Color.WARNING};"
        )
        self._attack_lbl.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        attack_lay.addWidget(self._attack_lbl, stretch=1)
        root.addWidget(attack_frame)

        # ── Stats row: punches | hits taken | defense rate ───────────────
        stats = QHBoxLayout()
        stats.setSpacing(10)

        self._punch_counter = PunchCounter(label="YOUR PUNCHES")
        stats.addWidget(self._punch_counter)

        stats.addStretch()

        # Hits taken
        hits_col = QVBoxLayout()
        hits_col.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hits_col.setSpacing(2)
        hits_hdr = QLabel("HITS TAKEN")
        hits_hdr.setStyleSheet(
            f"color: {Color.TEXT_DISABLED}; font-size: 11px;"
            " font-weight: 700; letter-spacing: 0.5px;"
        )
        hits_hdr.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._hits_lbl = QLabel("0")
        self._hits_lbl.setFont(font(28, bold=True))
        self._hits_lbl.setStyleSheet(f"color: {Color.DANGER};")
        self._hits_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hits_col.addWidget(hits_hdr)
        hits_col.addWidget(self._hits_lbl)
        stats.addLayout(hits_col)

        stats.addStretch()

        # Defense rate
        def_col = QVBoxLayout()
        def_col.setAlignment(Qt.AlignmentFlag.AlignCenter)
        def_col.setSpacing(2)
        def_hdr = QLabel("DEFENSE")
        def_hdr.setStyleSheet(
            f"color: {Color.TEXT_DISABLED}; font-size: 11px;"
            " font-weight: 700; letter-spacing: 0.5px;"
        )
        def_hdr.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._def_lbl = QLabel("--%")
        self._def_lbl.setFont(font(28, bold=True))
        self._def_lbl.setStyleSheet(f"color: {Color.PRIMARY};")
        self._def_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        def_col.addWidget(def_hdr)
        def_col.addWidget(self._def_lbl)
        stats.addLayout(def_col)

        root.addLayout(stats)

        # ── Stop button ──────────────────────────────────────────────────
        bottom = QHBoxLayout()
        bottom.addStretch()
        self._btn_stop = QPushButton(f"{Icon.STOP}  STOP")
        self._btn_stop.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_stop.setFixedSize(140, 46)
        self._btn_stop.setStyleSheet(f"""
            QPushButton {{
                background-color: {Color.DANGER}; color: white;
                font-size: 16px; font-weight: 700;
                border: none; border-radius: {Size.RADIUS_LG}px;
            }}
            QPushButton:hover {{ background-color: {Color.DANGER_DARK}; }}
            QPushButton:pressed {{ background-color: #C33C3C; }}
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

    # ── Lifecycle ──────────────────────────────────────────────────────
    def on_enter(self, **kwargs: Any) -> None:
        self._config = kwargs.get("config", {})
        rounds = self._config.get("Rounds", "3")
        work_time = self._parse_seconds(self._config.get("Work", "90s"))
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
