"""Full-page debug detection panel for BoxBunny.

Shows CV detection metadata as styled text (no camera feed to avoid
slowing down the voxelflow inference). Displays punch type, confidence,
frame persistence, IMU match status, FPS, and a scrolling punch log.
"""
from __future__ import annotations

import logging
import time
from collections import deque
from typing import Any, Dict, Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QScroller,
    QVBoxLayout,
    QWidget,
)

from boxbunny_gui.theme import Color, Size

logger = logging.getLogger(__name__)

_PUNCH_COLORS = {
    "jab": Color.JAB, "cross": Color.CROSS,
    "left_hook": Color.L_HOOK, "right_hook": Color.R_HOOK,
    "left_uppercut": Color.L_UPPERCUT, "right_uppercut": Color.R_UPPERCUT,
    "block": Color.BLOCK, "idle": Color.IDLE,
}

_PAD_COLORS = {
    "left": Color.INFO, "centre": Color.PRIMARY,
    "right": Color.PURPLE, "head": Color.WARNING,
}


def _card_style(accent: str = Color.BORDER_LIGHT) -> str:
    return (
        f"background-color: {Color.SURFACE};"
        f" border: 1px solid {Color.BORDER};"
        f" border-left: 3px solid {accent};"
        f" border-radius: {Size.RADIUS}px;"
        " padding: 10px 14px;"
    )


class DebugDetectionPanel(QWidget):
    """Full-page debug panel showing CV detection data as text."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._active = False
        self._punch_log: deque = deque(maxlen=30)
        self._last_update = 0.0
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 10, 20, 10)
        root.setSpacing(8)

        # Title bar
        title = QLabel("DEBUG  DETECTION PANEL")
        title.setStyleSheet(
            f"font-size: 12px; font-weight: 700; color: {Color.WARNING};"
            " letter-spacing: 2px;"
        )
        root.addWidget(title)

        # Main content: left = big prediction, right = stats
        content = QHBoxLayout()
        content.setSpacing(12)

        # ── Left: Current Prediction (big) ───────────────────────────────
        left = QVBoxLayout()
        left.setSpacing(6)

        self._pred_label = QLabel("idle")
        self._pred_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._pred_label.setStyleSheet(
            f"font-size: 56px; font-weight: 800; color: {Color.IDLE};"
            " letter-spacing: 3px;"
            f" background-color: {Color.SURFACE};"
            f" border: 1px solid {Color.BORDER};"
            f" border-radius: {Size.RADIUS_LG}px;"
            " padding: 20px;"
        )
        left.addWidget(self._pred_label, stretch=2)

        # Confidence bar
        conf_row = QHBoxLayout()
        conf_row.setSpacing(8)
        self._conf_label = QLabel("0%")
        self._conf_label.setStyleSheet(
            f"font-size: 28px; font-weight: 700; color: {Color.TEXT};"
        )
        conf_row.addWidget(self._conf_label)
        self._conf_bar = QWidget()
        self._conf_bar.setFixedHeight(20)
        self._conf_bar.setStyleSheet(
            f"background-color: {Color.SURFACE};"
            f" border-radius: 10px;"
        )
        conf_row.addWidget(self._conf_bar, stretch=1)
        left.addLayout(conf_row)

        # Frame persistence + movement delta
        persist_row = QHBoxLayout()
        self._frames_label = QLabel("Frames: 0")
        self._frames_label.setStyleSheet(
            f"font-size: 14px; font-weight: 600; color: {Color.TEXT_SECONDARY};"
            f" {_card_style(Color.INFO)}"
        )
        persist_row.addWidget(self._frames_label)
        self._delta_label = QLabel("Delta: 0.0")
        self._delta_label.setStyleSheet(
            f"font-size: 14px; font-weight: 600; color: {Color.TEXT_SECONDARY};"
            f" {_card_style(Color.SUCCESS)}"
        )
        persist_row.addWidget(self._delta_label)
        self._fps_label = QLabel("FPS: --")
        self._fps_label.setStyleSheet(
            f"font-size: 14px; font-weight: 600; color: {Color.TEXT_SECONDARY};"
            f" {_card_style(Color.WARNING)}"
        )
        persist_row.addWidget(self._fps_label)
        left.addLayout(persist_row)

        content.addLayout(left, stretch=3)

        # ── Right: Stats + Punch Log ─────────────────────────────────────
        right = QVBoxLayout()
        right.setSpacing(6)

        # Pad activity indicators
        pad_title = QLabel("PAD ACTIVITY")
        pad_title.setStyleSheet(
            f"font-size: 10px; font-weight: 700; color: {Color.TEXT_DISABLED};"
            " letter-spacing: 1px;"
        )
        right.addWidget(pad_title)

        pad_row = QHBoxLayout()
        pad_row.setSpacing(4)
        self._pad_indicators: Dict[str, QLabel] = {}
        for pad_name in ["left", "centre", "right", "head"]:
            lbl = QLabel(pad_name.upper())
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setFixedHeight(36)
            lbl.setStyleSheet(
                f"font-size: 11px; font-weight: 700;"
                f" color: {Color.TEXT_DISABLED};"
                f" background-color: {Color.SURFACE};"
                f" border: 1px solid {Color.BORDER};"
                f" border-radius: {Size.RADIUS_SM}px;"
            )
            pad_row.addWidget(lbl)
            self._pad_indicators[pad_name] = lbl
        right.addLayout(pad_row)

        # Punch log title
        log_title = QLabel("CONFIRMED PUNCHES")
        log_title.setStyleSheet(
            f"font-size: 10px; font-weight: 700; color: {Color.TEXT_DISABLED};"
            " letter-spacing: 1px;"
        )
        right.addWidget(log_title)

        # Scrollable punch log
        self._log_widget = QWidget()
        self._log_layout = QVBoxLayout(self._log_widget)
        self._log_layout.setContentsMargins(0, 0, 0, 0)
        self._log_layout.setSpacing(2)
        self._log_layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidget(self._log_widget)
        scroll.setWidgetResizable(True)
        QScroller.grabGesture(
            scroll.viewport(),
            QScroller.ScrollerGestureType.LeftMouseButtonGesture,
        )
        scroll.setStyleSheet(
            f"QScrollArea {{ background: {Color.SURFACE};"
            f" border: 1px solid {Color.BORDER};"
            f" border-radius: {Size.RADIUS}px; }}"
            f" QScrollBar {{ background: {Color.SURFACE}; width: 6px; }}"
            f" QScrollBar::handle {{ background: {Color.BORDER_LIGHT};"
            f" border-radius: 3px; }}"
        )
        right.addWidget(scroll, stretch=1)

        content.addLayout(right, stretch=2)
        root.addLayout(content, stretch=1)

    # ── Public API ──────────────────────────────────────────────────────

    def set_active(self, active: bool) -> None:
        self._active = active

    def on_debug_info(self, data: Dict[str, Any]) -> None:
        """Update display from cv_node debug_info JSON."""
        if not self._active:
            return

        action = data.get("action", "idle")
        confidence = data.get("confidence", 0.0)
        consecutive = data.get("consecutive", 0)
        fps = data.get("fps", 0.0)
        delta = data.get("movement_delta", 0.0)

        color = _PUNCH_COLORS.get(action, Color.TEXT)

        self._pred_label.setText(action.replace("_", " ").upper())
        self._pred_label.setStyleSheet(
            f"font-size: 56px; font-weight: 800; color: {color};"
            " letter-spacing: 3px;"
            f" background-color: {Color.SURFACE};"
            f" border: 1px solid {color};"
            f" border-radius: {Size.RADIUS_LG}px;"
            " padding: 20px;"
        )

        pct = int(confidence * 100)
        self._conf_label.setText(f"{pct}%")
        bar_color = Color.SUCCESS if pct >= 80 else (Color.WARNING if pct >= 50 else Color.DANGER)
        self._conf_bar.setStyleSheet(
            f"background: qlineargradient(x1:0, y1:0, x2:1, y2:0,"
            f" stop:0 {bar_color}, stop:{min(confidence, 1.0)} {bar_color},"
            f" stop:{min(confidence + 0.01, 1.0)} {Color.SURFACE},"
            f" stop:1 {Color.SURFACE});"
            " border-radius: 10px;"
        )

        self._frames_label.setText(f"Frames: {consecutive}")
        self._delta_label.setText(f"Delta: {delta:.1f}")
        self._fps_label.setText(f"FPS: {fps:.0f}")

    def on_punch_confirmed(self, data: Dict[str, Any]) -> None:
        """Add a confirmed punch to the log."""
        if not self._active:
            return

        punch_type = data.get("punch_type", "?")
        pad = data.get("pad", "")
        conf = data.get("cv_confidence", 0.0)
        imu = data.get("imu_confirmed", False)
        accel = data.get("accel_magnitude", 0.0)

        color = _PUNCH_COLORS.get(punch_type, Color.TEXT)
        imu_tag = "IMU" if imu else "CV"
        text = f"{punch_type:<15s}  pad={pad:<7s}  conf={conf:.0%}  {imu_tag}"
        if accel > 0:
            text += f"  {accel:.0f}m/s\u00B2"

        entry = QLabel(text)
        entry.setStyleSheet(
            f"font-family: monospace; font-size: 11px; color: {color};"
            f" background: transparent; padding: 2px 6px;"
        )

        # Insert before the stretch at the end
        count = self._log_layout.count()
        self._log_layout.insertWidget(max(0, count - 1), entry)

        # Limit log entries
        while self._log_layout.count() > 31:  # 30 entries + 1 stretch
            item = self._log_layout.takeAt(0)
            if item and item.widget():
                item.widget().deleteLater()

        # Flash pad indicator
        if pad in self._pad_indicators:
            pad_color = _PAD_COLORS.get(pad, Color.PRIMARY)
            lbl = self._pad_indicators[pad]
            lbl.setStyleSheet(
                f"font-size: 11px; font-weight: 700;"
                f" color: #FFFFFF;"
                f" background-color: {pad_color};"
                f" border: 1px solid {pad_color};"
                f" border-radius: {Size.RADIUS_SM}px;"
            )
            QTimer.singleShot(300, lambda p=pad: self._reset_pad(p))

    def _reset_pad(self, pad: str) -> None:
        if pad in self._pad_indicators:
            self._pad_indicators[pad].setStyleSheet(
                f"font-size: 11px; font-weight: 700;"
                f" color: {Color.TEXT_DISABLED};"
                f" background-color: {Color.SURFACE};"
                f" border: 1px solid {Color.BORDER};"
                f" border-radius: {Size.RADIUS_SM}px;"
            )
