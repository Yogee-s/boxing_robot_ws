"""Large timer display with rounded-rectangle progress bar.

Clean, modern design — time text centered over a horizontal progress track.
"""
from __future__ import annotations

import logging

from PySide6.QtCore import QRectF, QTimer, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QLinearGradient
from PySide6.QtWidgets import QSizePolicy, QWidget

from boxbunny_gui.theme import Color, Size

log = logging.getLogger(__name__)


class TimerDisplay(QWidget):
    """Countdown timer with a rounded-rectangle progress bar.

    Signals
    -------
    finished
        Emitted when the countdown reaches zero.
    tick(int)
        Emitted every second with the remaining seconds.
    """

    finished = Signal()
    tick = Signal(int)

    def __init__(
        self,
        font_size: int = Size.TEXT_TIMER,
        show_ring: bool = True,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._font_size = font_size
        self._show_ring = show_ring

        self._total: int = 0
        self._remaining: int = 0
        self._running: bool = False

        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._on_tick)

        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setMinimumSize(160, 100)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

    # -- public API -----------------------------------------------------------
    def start(self, total_seconds: int) -> None:
        self._total = total_seconds
        self._remaining = total_seconds
        self._running = True
        self._timer.start()
        self.update()

    def pause(self) -> None:
        self._running = False
        self._timer.stop()

    def resume(self) -> None:
        if self._remaining > 0:
            self._running = True
            self._timer.start()

    def reset(self) -> None:
        self._timer.stop()
        self._running = False
        self._remaining = 0
        self._total = 0
        self.update()

    def set_time(self, seconds: int) -> None:
        self._remaining = max(0, seconds)
        self._total = max(self._total, self._remaining)
        self.update()

    # -- internal -------------------------------------------------------------
    def _on_tick(self) -> None:
        self._remaining = max(0, self._remaining - 1)
        self.tick.emit(self._remaining)
        self.update()
        if self._remaining <= 0:
            self._timer.stop()
            self._running = False
            self.finished.emit()

    def _current_color(self) -> str:
        if self._remaining <= 10:
            return Color.DANGER
        if self._remaining <= 30:
            return Color.WARNING
        return Color.PRIMARY

    @staticmethod
    def _format_time(seconds: int) -> str:
        if seconds >= 60:
            m, s = divmod(seconds, 60)
            return f"{m:02d}:{s:02d}"
        return f"{seconds}"

    # -- painting -------------------------------------------------------------
    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()
        color_hex = self._current_color()
        color = QColor(color_hex)

        # Scale font to widget
        time_font_size = max(28, int(h * 0.35))
        time_font = QFont("Inter", time_font_size, QFont.Weight.Bold)

        # ── Background card ──────────────────────────────────────────────
        card_margin = 16
        card_rect = QRectF(
            card_margin, card_margin,
            w - card_margin * 2, h - card_margin * 2,
        )
        bg = QColor("#131920")
        p.setPen(QPen(QColor("#1E2832"), 1))
        p.setBrush(bg)
        p.drawRoundedRect(card_rect, Size.RADIUS_LG, Size.RADIUS_LG)

        # ── Time text ────────────────────────────────────────────────────
        p.setPen(color)
        p.setFont(time_font)
        text_rect = QRectF(
            card_margin, card_margin,
            w - card_margin * 2, h - card_margin * 2 - 30,
        )
        p.drawText(text_rect, Qt.AlignCenter, self._format_time(self._remaining))

        # ── Progress bar at bottom of card ───────────────────────────────
        if self._show_ring and self._total > 0:
            bar_h = 6
            bar_margin = 28
            bar_y = card_rect.bottom() - bar_h - 14
            bar_w = w - card_margin * 2 - bar_margin * 2
            bar_x = card_margin + bar_margin

            # Track
            track_rect = QRectF(bar_x, bar_y, bar_w, bar_h)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(Color.SURFACE_LIGHT))
            p.drawRoundedRect(track_rect, bar_h / 2, bar_h / 2)

            # Fill
            fraction = self._remaining / self._total if self._total else 0
            fill_w = max(bar_h, bar_w * fraction)
            fill_rect = QRectF(bar_x, bar_y, fill_w, bar_h)

            # Gradient fill
            grad = QLinearGradient(bar_x, 0, bar_x + fill_w, 0)
            grad.setColorAt(0.0, color)
            dim = QColor(color_hex)
            dim.setAlpha(160)
            grad.setColorAt(1.0, dim)
            p.setBrush(grad)
            p.drawRoundedRect(fill_rect, bar_h / 2, bar_h / 2)

            # Glow when critical
            if self._remaining <= 10 and self._running:
                glow = QColor(color_hex)
                glow.setAlpha(40)
                glow_rect = QRectF(bar_x - 2, bar_y - 2, fill_w + 4, bar_h + 4)
                p.setBrush(glow)
                p.drawRoundedRect(glow_rect, bar_h, bar_h)

        p.end()
