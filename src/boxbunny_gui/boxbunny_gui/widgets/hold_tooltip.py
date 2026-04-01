"""Press-and-hold tooltip card.

Tap to navigate, hold 400ms to reveal a bottom info bar that slides up.
Releasing slides it back down and dismisses without triggering navigation.
"""
from __future__ import annotations

from PySide6.QtCore import QEasingCurve, QPoint, QPropertyAnimation, QTimer, Qt
from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

from boxbunny_gui.theme import Color, Size

_HOLD_MS = 400
_ANIM_MS = 250


class HoldTooltipCard(QPushButton):
    """Card button with slide-up info bar on long-press."""

    def __init__(self, desc_html: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._desc_html = desc_html
        self._hold_timer = QTimer(self)
        self._hold_timer.setSingleShot(True)
        self._hold_timer.setInterval(_HOLD_MS)
        self._hold_timer.timeout.connect(self._on_hold)
        self._tooltip_shown: bool = False
        self._popup: QWidget | None = None
        self._anim: QPropertyAnimation | None = None

    # ── Qt event overrides ──────────────────────────────────────────────

    def mousePressEvent(self, event) -> None:
        self._tooltip_shown = False
        if self._desc_html:
            self._hold_timer.start()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        self._hold_timer.stop()
        if self._tooltip_shown:
            self._slide_out()
            self.setDown(False)
            return
        super().mouseReleaseEvent(event)

    # ── Slide-up info bar ───────────────────────────────────────────────

    def _on_hold(self) -> None:
        self._tooltip_shown = True
        win = self.window()
        if win is None:
            return

        self._popup = QWidget(win)
        self._popup.setStyleSheet(f"""
            QWidget {{
                background-color: {Color.SURFACE};
                border-top: 2px solid {Color.PRIMARY};
            }}
        """)
        lay = QVBoxLayout(self._popup)
        lay.setContentsMargins(40, 22, 40, 26)

        lbl = QLabel(self._desc_html)
        lbl.setTextFormat(Qt.TextFormat.RichText)
        lbl.setWordWrap(True)
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet(
            f"color: {Color.TEXT}; font-size: 22px; font-weight: 600;"
            " border: none; background: transparent;"
        )
        lay.addWidget(lbl)

        self._popup.setFixedWidth(win.width())
        self._popup.adjustSize()
        popup_h = self._popup.height()
        win_h = win.height()

        # Start below the window, slide up into view
        self._popup.move(0, win_h)
        self._popup.raise_()
        self._popup.show()

        self._anim = QPropertyAnimation(self._popup, b"pos")
        self._anim.setDuration(_ANIM_MS)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._anim.setStartValue(QPoint(0, win_h))
        self._anim.setEndValue(QPoint(0, win_h - popup_h))
        self._anim.start()

    def _slide_out(self) -> None:
        if self._popup is None:
            self._tooltip_shown = False
            return

        win = self.window()
        win_h = win.height() if win else Size.SCREEN_H

        anim = QPropertyAnimation(self._popup, b"pos")
        anim.setDuration(_ANIM_MS)
        anim.setEasingCurve(QEasingCurve.Type.InCubic)
        anim.setStartValue(self._popup.pos())
        anim.setEndValue(QPoint(0, win_h))
        anim.finished.connect(self._cleanup)
        # Keep a reference so it doesn't get garbage-collected
        self._anim = anim
        anim.start()

    def _cleanup(self) -> None:
        if self._popup is not None:
            self._popup.close()
            self._popup = None
        self._anim = None
        self._tooltip_shown = False
