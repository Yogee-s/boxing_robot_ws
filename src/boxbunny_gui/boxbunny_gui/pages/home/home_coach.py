"""Coach home page with station management.

Idle state: Start Station button + preset browser.
Active state: current config, participant count, session timer, controls.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from boxbunny_gui.theme import Color, Size, font, DANGER_BTN, GHOST_BTN, PRIMARY_BTN, SURFACE_BTN
from boxbunny_gui.widgets import BigButton, PresetCard, TimerDisplay

if TYPE_CHECKING:
    from boxbunny_gui.nav.router import PageRouter

logger = logging.getLogger(__name__)


class HomeCoachPage(QWidget):
    """Coach dashboard with session management."""

    def __init__(self, router: PageRouter, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._router = router
        self._session_active: bool = False
        self._participant_num: int = 0
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(Size.SPACING, Size.SPACING_SM, Size.SPACING, Size.SPACING_SM)
        root.setSpacing(Size.SPACING)

        # Top bar
        top = QHBoxLayout()
        self._name_lbl = QLabel("Coach")
        self._name_lbl.setFont(font(22, bold=True))
        top.addWidget(self._name_lbl)
        self._status_badge = QLabel("Idle")
        self._status_badge.setStyleSheet(
            f"color: {Color.TEXT_SECONDARY}; font-size: 14px;"
            f" background-color: {Color.SURFACE}; border-radius: 8px; padding: 4px 10px;"
        )
        top.addWidget(self._status_badge)
        top.addStretch()
        self._btn_back = BigButton("Log Out", stylesheet=GHOST_BTN)
        self._btn_back.setFixedWidth(100)
        self._btn_back.clicked.connect(lambda: self._router.navigate("startup"))
        top.addWidget(self._btn_back)
        root.addLayout(top)

        # Idle state widgets
        self._idle_widget = QWidget()
        idle_lay = QVBoxLayout(self._idle_widget)
        idle_lay.setContentsMargins(0, 0, 0, 0)
        idle_lay.setSpacing(Size.SPACING)

        self._btn_start_station = BigButton("Start Station", stylesheet=PRIMARY_BTN)
        self._btn_start_station.setFixedHeight(80)
        self._btn_start_station.clicked.connect(self._start_session)
        idle_lay.addWidget(self._btn_start_station)

        presets_lbl = QLabel("Presets")
        presets_lbl.setFont(font(18, bold=True))
        idle_lay.addWidget(presets_lbl)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        preset_container = QWidget()
        self._preset_layout = QVBoxLayout(preset_container)
        self._preset_layout.setSpacing(Size.SPACING_SM)
        for i in range(4):
            card = PresetCard(parent=self)
            card.set_preset({"id": i, "name": f"Station Preset {i+1}", "mode": "training", "summary": ""})
            self._preset_layout.addWidget(card)
        self._preset_layout.addStretch()
        scroll.setWidget(preset_container)
        idle_lay.addWidget(scroll, stretch=1)
        root.addWidget(self._idle_widget)

        # Active state widgets
        self._active_widget = QWidget()
        active_lay = QVBoxLayout(self._active_widget)
        active_lay.setContentsMargins(0, 0, 0, 0)
        active_lay.setSpacing(Size.SPACING)

        self._config_lbl = QLabel("Config: Jab-Cross Drill")
        self._config_lbl.setFont(font(18, bold=True))
        active_lay.addWidget(self._config_lbl)

        self._participant_lbl = QLabel("Participant #1")
        self._participant_lbl.setStyleSheet(f"color: {Color.PRIMARY}; font-size: 20px;")
        active_lay.addWidget(self._participant_lbl)

        self._timer = TimerDisplay(font_size=Size.TEXT_TIMER_SM, show_ring=True)
        active_lay.addWidget(self._timer, stretch=1)

        btn_row = QHBoxLayout()
        self._btn_pause = BigButton("Pause", stylesheet=SURFACE_BTN)
        self._btn_pause.clicked.connect(self._toggle_pause)
        btn_row.addWidget(self._btn_pause)
        self._btn_switch = BigButton("Switch Config", stylesheet=SURFACE_BTN)
        self._btn_switch.clicked.connect(lambda: self._router.navigate("presets"))
        btn_row.addWidget(self._btn_switch)
        self._btn_end = BigButton("End Session", stylesheet=DANGER_BTN)
        self._btn_end.clicked.connect(self._end_session)
        btn_row.addWidget(self._btn_end)
        active_lay.addLayout(btn_row)

        self._active_widget.setVisible(False)
        root.addWidget(self._active_widget)

    def _start_session(self) -> None:
        self._session_active = True
        self._participant_num = 1
        self._update_state()
        self._timer.start(180)
        logger.info("Coach station started")

    def _end_session(self) -> None:
        self._session_active = False
        self._timer.reset()
        self._update_state()
        logger.info("Coach station ended")

    def _toggle_pause(self) -> None:
        if self._timer._running:
            self._timer.pause()
            self._btn_pause.setText("Resume")
        else:
            self._timer.resume()
            self._btn_pause.setText("Pause")

    def _update_state(self) -> None:
        self._idle_widget.setVisible(not self._session_active)
        self._active_widget.setVisible(self._session_active)
        badge_text = "Active" if self._session_active else "Idle"
        badge_color = Color.PRIMARY if self._session_active else Color.TEXT_SECONDARY
        self._status_badge.setText(badge_text)
        self._status_badge.setStyleSheet(
            f"color: {badge_color}; font-size: 14px;"
            f" background-color: {Color.SURFACE}; border-radius: 8px; padding: 4px 10px;"
        )
        self._participant_lbl.setText(f"Participant #{self._participant_num}")

    # ── Lifecycle ──────────────────────────────────────────────────────
    def on_enter(self, **kwargs: Any) -> None:
        self._update_state()
        logger.debug("HomeCoachPage entered")

    def on_leave(self) -> None:
        pass
