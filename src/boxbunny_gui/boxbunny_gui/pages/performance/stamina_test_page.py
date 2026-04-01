"""Stamina test page: throw as many punches as possible in a timed period.

Large timer countdown, live punch count, punches-per-minute display,
and tappable target selector.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

from PySide6.QtCore import QPoint, Qt, QTimer
from PySide6.QtWidgets import QScroller
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtGui import QColor

from boxbunny_gui.theme import (
    Color, Icon, Size, font, badge_style, back_link_style,
    DANGER_BTN, PRIMARY_BTN,
)
from boxbunny_gui.widgets import BigButton, PunchCounter, StatCard, TimerDisplay

if TYPE_CHECKING:
    from boxbunny_gui.gui_bridge import GuiBridge
    from boxbunny_gui.nav.router import PageRouter

logger = logging.getLogger(__name__)

_DEFAULT_DURATION = 120  # seconds
_TARGET_OPTIONS = [None, 20, 40, 60, 80, 100, 150, 200, 250, 300]
_STATE_READY = "ready"
_STATE_COUNTDOWN = "countdown"
_STATE_ACTIVE = "active"
_STATE_RESULTS = "results"


def _stat_col(label: str, value: str, color: str) -> tuple:
    """Clean stat tile — no child border artifacts."""
    frame = QWidget()
    frame.setObjectName("stile")
    frame.setFixedHeight(70)
    frame.setStyleSheet(f"""
        QWidget#stile {{
            background-color: #131920;
            border: 1px solid #1E2832;
            border-left: 3px solid {color};
            border-radius: {Size.RADIUS}px;
        }}
        QWidget#stile QLabel {{
            background: transparent; border: none;
        }}
    """)
    col = QVBoxLayout(frame)
    col.setAlignment(Qt.AlignmentFlag.AlignCenter)
    col.setContentsMargins(14, 6, 14, 6)
    col.setSpacing(0)
    h = QLabel(label)
    h.setStyleSheet(
        f"font-size: 10px; font-weight: 700; color: {Color.TEXT_DISABLED};"
        " letter-spacing: 0.8px;"
    )
    h.setAlignment(Qt.AlignmentFlag.AlignCenter)
    v = QLabel(value)
    v.setStyleSheet(f"font-size: 24px; font-weight: 700; color: {Color.TEXT};")
    v.setAlignment(Qt.AlignmentFlag.AlignCenter)
    col.addWidget(h)
    col.addWidget(v)
    return frame, v


class StaminaTestPage(QWidget):
    """Timed stamina test with live punch rate tracking."""

    def __init__(
        self,
        router: PageRouter,
        bridge: Optional[GuiBridge] = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._router = router
        self._bridge = bridge
        self._state: str = _STATE_READY
        self._punch_count: int = 0
        self._elapsed: int = 0
        self._peak_rate: float = 0.0
        self._target_idx: int = 0  # index into _TARGET_OPTIONS
        self._build_ui()
        if self._bridge:
            self._bridge.punch_confirmed.connect(self._on_punch)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(30, Size.SPACING_SM, 30, 22)
        root.setSpacing(Size.SPACING_SM)

        # Top bar
        top = QHBoxLayout()
        btn_back = QPushButton(f"{Icon.BACK}  Back")
        btn_back.setStyleSheet(back_link_style())
        btn_back.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_back.clicked.connect(self._on_back)
        top.addWidget(btn_back)
        title = QLabel("Stamina Test")
        title.setStyleSheet(f"font-size: 20px; font-weight: 700; color: {Color.TEXT};")
        top.addWidget(title)
        top.addStretch()

        self._duration_badge = QLabel("2:00")
        self._duration_badge.setStyleSheet(badge_style(Color.PRIMARY))
        top.addWidget(self._duration_badge)
        root.addLayout(top)

        # Timer
        self._timer = TimerDisplay(font_size=Size.TEXT_TIMER, show_ring=True)
        self._timer.finished.connect(self._on_done)
        self._timer.tick.connect(self._on_tick)
        root.addWidget(self._timer, stretch=1)

        # Live stats row
        stats = QHBoxLayout()
        stats.setSpacing(12)
        self._punch_counter = PunchCounter(label="PUNCHES")
        stats.addWidget(self._punch_counter)

        rate_widget, self._rate_lbl = _stat_col(
            "PUNCHES/MIN", "0", Color.PRIMARY
        )
        stats.addWidget(rate_widget)

        # Target — tappable tile with upward popup
        self._target_btn = QPushButton()
        self._target_btn.setObjectName("target_tile")
        self._target_btn.setFixedHeight(70)
        self._target_btn.setMinimumWidth(130)
        self._target_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._target_btn.clicked.connect(self._show_target_popup)
        self._target_lbl_header = QLabel("TARGET")
        self._target_lbl_header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._target_lbl_header.setStyleSheet(
            f"font-size: 10px; font-weight: 700; color: {Color.TEXT_DISABLED};"
            " letter-spacing: 0.8px; background: transparent; border: none;"
        )
        self._target_lbl_header.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents
        )
        self._target_val = QLabel("None")
        self._target_val.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._target_val.setStyleSheet(
            f"font-size: 24px; font-weight: 700; color: {Color.TEXT};"
            " background: transparent; border: none;"
        )
        self._target_val.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents
        )
        target_lay = QVBoxLayout(self._target_btn)
        target_lay.setContentsMargins(14, 6, 14, 6)
        target_lay.setSpacing(0)
        target_lay.addWidget(self._target_lbl_header)
        target_lay.addWidget(self._target_val)
        self._update_target_style()
        stats.addWidget(self._target_btn)
        root.addLayout(stats)

        # Start / Stop button
        self._btn_action = BigButton("Start", stylesheet=PRIMARY_BTN)
        self._btn_action.setFixedHeight(70)
        self._btn_action.clicked.connect(self._toggle)
        root.addWidget(self._btn_action)

        # Results overlay
        self._results_widget = QWidget()
        res_lay = QVBoxLayout(self._results_widget)
        res_lay.setSpacing(16)
        res_lay.setContentsMargins(0, 8, 0, 0)

        res_row = QHBoxLayout()
        res_row.setSpacing(12)
        self._stat_total = StatCard(
            "Total Punches", "--", accent=Color.PRIMARY,
        )
        self._stat_peak = StatCard(
            "Peak Rate", "--/min", accent=Color.WARNING,
        )
        self._stat_fatigue = StatCard(
            "Fatigue", "--", accent=Color.DANGER,
        )
        res_row.addWidget(self._stat_total)
        res_row.addWidget(self._stat_peak)
        res_row.addWidget(self._stat_fatigue)
        res_lay.addLayout(res_row)

        btn_done = BigButton("Done", stylesheet=PRIMARY_BTN)
        btn_done.setFixedHeight(70)
        btn_done.clicked.connect(
            lambda: self._router.navigate("performance")
        )
        res_lay.addWidget(btn_done)
        self._results_widget.setVisible(False)
        root.addWidget(self._results_widget)

    # ── Target selector ───────────────────────────────────────────────

    def _show_target_popup(self) -> None:
        """Show upward scrollable popup menu with target options."""
        if self._state != _STATE_READY:
            return

        _MAX_HEIGHT = 280
        _BTN_H = 40
        _BTN_W = 120

        popup = QWidget(self, Qt.WindowType.Popup)
        popup.setObjectName("target_popup")
        popup.setStyleSheet(f"""
            QWidget#target_popup {{
                background-color: #1A2030;
                border: 1px solid #2A3848;
                border-radius: {Size.RADIUS}px;
            }}
        """)

        # Inner content widget for the scroll area
        content = QWidget()
        content_lay = QVBoxLayout(content)
        content_lay.setContentsMargins(4, 4, 4, 4)
        content_lay.setSpacing(2)

        selected_btn = None
        for opt in reversed(_TARGET_OPTIONS):
            label = "None" if opt is None else str(opt)
            btn = QPushButton(label)
            btn.setFixedHeight(_BTN_H)
            btn.setMinimumWidth(_BTN_W)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            is_selected = (_TARGET_OPTIONS[self._target_idx] == opt)
            if is_selected:
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background-color: {Color.WARNING};
                        color: #FFFFFF;
                        font-size: 16px; font-weight: 700;
                        border: none; border-radius: 6px;
                    }}
                """)
                selected_btn = btn
            else:
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background-color: transparent;
                        color: {Color.TEXT};
                        font-size: 16px; font-weight: 500;
                        border: none; border-radius: 6px;
                    }}
                    QPushButton:hover {{
                        background-color: #242E3E;
                    }}
                """)
            btn.clicked.connect(
                lambda checked=False, v=opt, p=popup: self._pick_target(v, p)
            )
            content_lay.addWidget(btn)

        scroll = QScrollArea()
        scroll.setWidget(content)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        scroll.setStyleSheet("""
            QScrollArea { background: transparent; border: none; }
            QScrollBar:vertical {
                background: #1A2030; width: 6px; border: none;
            }
            QScrollBar::handle:vertical {
                background: #3A4858; border-radius: 3px; min-height: 20px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)
        # Enable finger/touch drag scrolling
        QScroller.grabGesture(
            scroll.viewport(),
            QScroller.ScrollerGestureType.LeftMouseButtonGesture,
        )

        popup_lay = QVBoxLayout(popup)
        popup_lay.setContentsMargins(0, 4, 0, 4)
        popup_lay.addWidget(scroll)

        # Size: fit content up to max height
        content_h = len(_TARGET_OPTIONS) * (_BTN_H + 2) + 12
        popup_h = min(content_h, _MAX_HEIGHT)
        popup.setFixedSize(_BTN_W + 20, popup_h)

        # Position above the target button
        btn_pos = self._target_btn.mapToGlobal(QPoint(0, 0))
        popup_x = btn_pos.x() + (self._target_btn.width() - popup.width()) // 2
        popup_y = btn_pos.y() - popup_h - 4
        popup.move(popup_x, popup_y)
        popup.show()

        # Scroll to selected item
        if selected_btn is not None:
            scroll.ensureWidgetVisible(selected_btn, 0, 40)

    def _pick_target(self, value: int | None, popup: QWidget) -> None:
        """Select a target value from the popup."""
        popup.close()
        try:
            self._target_idx = _TARGET_OPTIONS.index(value)
        except ValueError:
            self._target_idx = 0
        self._update_target_display()

    def _update_target_display(self) -> None:
        target = _TARGET_OPTIONS[self._target_idx]
        if target is None:
            self._target_val.setText("None")
        else:
            self._target_val.setText(str(target))
        self._update_target_style()

    def _update_target_style(self) -> None:
        target = _TARGET_OPTIONS[self._target_idx]
        active = target is not None
        accent = Color.WARNING if active else Color.BORDER
        self._target_btn.setStyleSheet(f"""
            QPushButton#target_tile {{
                background-color: #131920;
                border: 1px solid #1E2832;
                border-left: 3px solid {accent};
                border-radius: {Size.RADIUS}px;
            }}
            QPushButton#target_tile:hover {{
                background-color: #1A2030;
            }}
        """)

    # ── Countdown ─────────────────────────────────────────────────────

    def imu_start(self) -> None:
        """Called by centre pad IMU to begin the test."""
        if self._state == _STATE_READY:
            self._start_countdown()

    def _toggle(self) -> None:
        if self._state == _STATE_READY:
            self._start_countdown()
        elif self._state == _STATE_ACTIVE:
            self._on_done()

    def _start_countdown(self) -> None:
        self._state = _STATE_COUNTDOWN
        self._punch_count = 0
        self._elapsed = 0
        self._peak_rate = 0.0
        self._punch_counter.set_count(0)
        self._rate_lbl.setText("0")
        self._btn_action.setVisible(False)
        self._results_widget.setVisible(False)
        self._target_btn.setEnabled(False)
        self._timer.set_overlay("Get Ready")
        self._cd_remaining = 3
        QTimer.singleShot(1000, self._cd_tick)

    def _cd_tick(self) -> None:
        if self._state != _STATE_COUNTDOWN:
            return
        if self._cd_remaining > 0:
            self._timer.set_overlay(str(self._cd_remaining))
            self._cd_remaining -= 1
            QTimer.singleShot(1000, self._cd_tick)
        else:
            self._timer.set_overlay("GO!")
            QTimer.singleShot(500, self._start_test)

    def _start_test(self) -> None:
        self._state = _STATE_ACTIVE
        self._timer.clear_overlay()
        self._timer.start(_DEFAULT_DURATION)
        self._btn_action.setText("Stop")
        self._btn_action.setStyleSheet(DANGER_BTN)
        self._btn_action.setVisible(True)
        self._duration_badge.setStyleSheet(badge_style(Color.DANGER))

    def _on_tick(self, remaining: int) -> None:
        self._elapsed = _DEFAULT_DURATION - remaining
        mins, secs = divmod(remaining, 60)
        self._duration_badge.setText(f"{mins}:{secs:02d}")
        if self._elapsed > 0:
            rate = self._punch_count / (self._elapsed / 60.0)
            self._rate_lbl.setText(str(int(rate)))
            self._peak_rate = max(self._peak_rate, rate)

        # Update target display during active — show progress
        target = _TARGET_OPTIONS[self._target_idx]
        if target is not None:
            self._target_val.setText(f"{self._punch_count}/{target}")

    def _on_punch(self, data: Dict[str, Any]) -> None:
        if self._state != _STATE_ACTIVE:
            return
        self._punch_count += 1
        self._punch_counter.set_count(self._punch_count)

        # Check if target reached
        target = _TARGET_OPTIONS[self._target_idx]
        if target is not None:
            self._target_val.setText(f"{self._punch_count}/{target}")

    def _on_done(self) -> None:
        self._timer.pause()
        self._state = _STATE_RESULTS
        self._btn_action.setVisible(False)
        self._target_btn.setEnabled(False)
        self._stat_total.set_value(str(self._punch_count))
        self._stat_peak.set_value(f"{self._peak_rate:.0f}/min")

        # Target result
        target = _TARGET_OPTIONS[self._target_idx]
        if target is not None:
            if self._punch_count >= target:
                self._stat_fatigue.set_value("Target Hit!")
            else:
                remaining = target - self._punch_count
                self._stat_fatigue.set_value(f"-{remaining}")
        else:
            self._stat_fatigue.set_value("--")

        from boxbunny_gui.session_tracker import get_tracker
        score = f"{self._punch_count} punches"
        if target is not None:
            score += f" (target: {target})"
        get_tracker().add_session(
            mode="Performance",
            duration=f"Stamina {_DEFAULT_DURATION}s",
            punches=str(self._punch_count),
            score=score,
        )
        self._results_widget.setVisible(True)

    def _on_back(self) -> None:
        self._timer.reset()
        self._router.back()

    # ── Lifecycle ──────────────────────────────────────────────────────
    def on_enter(self, **kwargs: Any) -> None:
        self._state = _STATE_READY
        self._timer.set_time(_DEFAULT_DURATION)
        self._timer.clear_overlay()
        self._btn_action.setText("Start")
        self._btn_action.setStyleSheet(PRIMARY_BTN)
        self._btn_action.setVisible(True)
        self._results_widget.setVisible(False)
        self._punch_counter.set_count(0)
        self._rate_lbl.setText("0")
        self._duration_badge.setText("2:00")
        self._duration_badge.setStyleSheet(badge_style(Color.PRIMARY))
        self._target_btn.setEnabled(True)
        self._update_target_display()
        logger.debug("StaminaTestPage entered")

    def on_leave(self) -> None:
        self._timer.reset()
