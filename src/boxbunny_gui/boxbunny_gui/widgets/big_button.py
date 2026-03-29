"""Touch-friendly button with glow focus, press animation, and IMU support.

Minimum 60x60 px touch target with optional left-side icon.
"""

from __future__ import annotations

import logging
from typing import Optional

from PySide6.QtCore import (
    QEasingCurve,
    QPropertyAnimation,
    QSize,
    Property,
)
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QGraphicsDropShadowEffect,
    QPushButton,
    QSizePolicy,
)

from boxbunny_gui.theme import Color, Size, PRIMARY_BTN

log = logging.getLogger(__name__)


class BigButton(QPushButton):
    """A large, dark-themed push-button designed for touchscreen use.

    Features
    --------
    * Enforces ``MIN_TOUCH`` minimum dimensions.
    * Optional ``QIcon`` rendered on the left side.
    * Green drop-shadow glow when focused (IMU or keyboard).
    * Subtle scale-down animation on press.
    """

    def __init__(
        self,
        text: str = "",
        icon: Optional[QIcon] = None,
        stylesheet: str = PRIMARY_BTN,
        parent=None,
    ) -> None:
        super().__init__(text, parent)
        self.setMinimumSize(Size.MIN_TOUCH, 36)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setStyleSheet(stylesheet)

        if icon is not None:
            self.setIcon(icon)
            self.setIconSize(QSize(28, 28))

        # -- glow effect (hidden until focused) --------------------------------
        self._glow = QGraphicsDropShadowEffect(self)
        self._glow.setColor(Color.PRIMARY)
        self._glow.setBlurRadius(24)
        self._glow.setOffset(0, 0)
        self._glow.setEnabled(False)
        self.setGraphicsEffect(self._glow)

        # -- scale animation property ------------------------------------------
        self._scale: float = 1.0
        self._anim = QPropertyAnimation(self, b"btn_scale")
        self._anim.setDuration(80)
        self._anim.setEasingCurve(QEasingCurve.Type.InOutQuad)

    # -- Qt property for animation target -------------------------------------
    def _get_scale(self) -> float:
        return self._scale

    def _set_scale(self, value: float) -> None:
        self._scale = value
        self.update()

    btn_scale = Property(float, _get_scale, _set_scale)

    # -- IMU focus cycling ----------------------------------------------------
    def set_focused(self, focused: bool) -> None:
        """Highlight the button for IMU-based focus cycling."""
        self._glow.setEnabled(focused)
        if focused:
            self.setFocus()

    # -- press / release animation --------------------------------------------
    def mousePressEvent(self, event) -> None:  # noqa: N802
        self._animate_scale(0.95)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        self._animate_scale(1.0)
        super().mouseReleaseEvent(event)

    def _animate_scale(self, target: float) -> None:
        self._anim.stop()
        self._anim.setStartValue(self._scale)
        self._anim.setEndValue(target)
        self._anim.start()
