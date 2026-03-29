"""Settings page: hardware, sound, display, AI, network, system sections."""
from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Any, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from boxbunny_gui.theme import Color, Size, font, GHOST_BTN, SURFACE_BTN
from boxbunny_gui.widgets import BigButton

if TYPE_CHECKING:
    from boxbunny_gui.gui_bridge import GuiBridge
    from boxbunny_gui.nav.router import PageRouter

logger = logging.getLogger(__name__)


class _StatusDot(QLabel):
    """Simple colored circle status indicator."""

    def __init__(
        self, connected: bool = False, parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setFixedSize(16, 16)
        self.set_connected(connected)

    def set_connected(self, connected: bool) -> None:
        color = Color.PRIMARY if connected else Color.DANGER
        self.setStyleSheet(
            f"background-color: {color}; border-radius: 8px; border: none;"
        )


class _Section(QFrame):
    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("section")
        self.setStyleSheet(
            f"QFrame#section {{ background-color: {Color.SURFACE};"
            f" border: 1px solid {Color.BORDER};"
            f" border-radius: 14px; }}"
        )
        self._root = QVBoxLayout(self)
        self._root.setContentsMargins(20, 14, 20, 14)
        self._root.setSpacing(10)

        header = QLabel(title.upper())
        header.setStyleSheet(
            "background: transparent;"
            f" color: {Color.PRIMARY}; font-size: 12px; font-weight: 700;"
            " letter-spacing: 1px;"
        )
        self._root.addWidget(header)

        divider = QFrame()
        divider.setFixedHeight(1)
        divider.setStyleSheet(
            f"background-color: {Color.BORDER}; border: none;"
        )
        self._root.addWidget(divider)

        self._content = QWidget()
        self._content.setStyleSheet("background: transparent;")
        self._content_lay = QVBoxLayout(self._content)
        self._content_lay.setContentsMargins(0, 2, 0, 2)
        self._content_lay.setSpacing(8)
        self._root.addWidget(self._content)

    @property
    def content_layout(self) -> QVBoxLayout:
        return self._content_lay


def _setting_row(label_text: str) -> tuple:
    """Create a standard settings row with label and return (layout, label)."""
    row = QHBoxLayout()
    row.setContentsMargins(0, 2, 0, 2)
    lbl = QLabel(label_text)
    lbl.setStyleSheet(
        f"background: transparent; font-size: 14px; color: {Color.TEXT};"
    )
    row.addWidget(lbl)
    row.addStretch()
    return row, lbl


class SettingsPage(QWidget):
    def __init__(
        self,
        router: PageRouter,
        bridge: Optional[GuiBridge] = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._router = router
        self._bridge = bridge
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 16, 24, 12)
        root.setSpacing(14)

        # Top bar
        top = QHBoxLayout()
        btn_back = BigButton("Back", stylesheet=GHOST_BTN)
        btn_back.setFixedWidth(100)
        btn_back.clicked.connect(lambda: self._router.back())
        top.addWidget(btn_back)
        title = QLabel("Settings")
        title.setFont(font(Size.TEXT_SUBHEADER, bold=True))
        top.addWidget(title)
        top.addStretch()
        root.addLayout(top)

        # Scrollable sections
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        container.setStyleSheet("background: transparent;")
        sections = QVBoxLayout(container)
        sections.setSpacing(12)
        sections.setContentsMargins(2, 2, 2, 2)

        # Hardware
        hw = _Section("Hardware")
        for device in ["Camera", "Robot", "IMU Left", "IMU Right"]:
            row, _ = _setting_row(device)
            dot = _StatusDot(connected=False)
            row.addWidget(dot)
            hw.content_layout.addLayout(row)
        sections.addWidget(hw)

        # Sound
        snd = _Section("Sound")
        vol_row, _ = _setting_row("Master Volume")
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, 100)
        slider.setValue(80)
        slider.setMinimumHeight(44)
        slider.setMinimumWidth(200)
        vol_row.addWidget(slider)
        snd.content_layout.addLayout(vol_row)
        for toggle_name in ["Punch Sounds", "Timer Beeps", "Coach Voice"]:
            row, _ = _setting_row(toggle_name)
            cb = QCheckBox()
            cb.setChecked(True)
            row.addWidget(cb)
            snd.content_layout.addLayout(row)
        sections.addWidget(snd)

        # Display
        disp = _Section("Display")
        row, _ = _setting_row("Gesture Control")
        cb = QCheckBox()
        row.addWidget(cb)
        disp.content_layout.addLayout(row)
        sections.addWidget(disp)

        # AI Coach
        ai = _Section("AI Coach")
        row, _ = _setting_row("Enable AI Coach")
        self._ai_cb = QCheckBox()
        self._ai_cb.setChecked(True)
        row.addWidget(self._ai_cb)
        ai.content_layout.addLayout(row)
        row2, _ = _setting_row("LLM Status")
        self._llm_dot = _StatusDot(connected=False)
        row2.addWidget(self._llm_dot)
        ai.content_layout.addLayout(row2)
        sections.addWidget(ai)

        # Network
        net = _Section("Network")
        net_row, _ = _setting_row("WiFi AP Status")
        self._wifi_dot = _StatusDot(connected=True)
        net_row.addWidget(self._wifi_dot)
        net.content_layout.addLayout(net_row)

        url_row, _ = _setting_row("Dashboard")
        url_val = QLabel("boxbunny.local")
        url_val.setStyleSheet(
            f"background: transparent; color: {Color.TEXT_SECONDARY};"
            " font-size: 13px;"
        )
        url_row.addWidget(url_val)
        net.content_layout.addLayout(url_row)
        sections.addWidget(net)

        # System
        sys_sec = _Section("System")
        ver_row, _ = _setting_row("Version")
        ver_val = QLabel("v1.0.0")
        ver_val.setStyleSheet(
            f"background: transparent; color: {Color.TEXT_SECONDARY};"
            " font-size: 13px;"
        )
        ver_row.addWidget(ver_val)
        sys_sec.content_layout.addLayout(ver_row)

        btn_maint = BigButton("Database Maintenance", stylesheet=SURFACE_BTN)
        btn_maint.setFixedHeight(44)
        sys_sec.content_layout.addWidget(btn_maint)
        sections.addWidget(sys_sec)

        sections.addStretch()
        scroll.setWidget(container)
        root.addWidget(scroll, stretch=1)

    def on_enter(self, **kwargs: Any) -> None:
        logger.debug("SettingsPage entered")

    def on_leave(self) -> None:
        pass
