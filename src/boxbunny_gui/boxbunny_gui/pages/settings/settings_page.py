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
from boxbunny_gui.widgets import BigButton, QRWidget

if TYPE_CHECKING:
    from boxbunny_gui.gui_bridge import GuiBridge
    from boxbunny_gui.nav.router import PageRouter

logger = logging.getLogger(__name__)


class _StatusDot(QLabel):
    def __init__(self, connected: bool = False, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(14, 14)
        self.set_connected(connected)

    def set_connected(self, connected: bool) -> None:
        color = Color.PRIMARY if connected else Color.DANGER
        self.setStyleSheet(f"background-color: {color}; border-radius: 7px;")


class _Section(QFrame):
    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setStyleSheet(
            f"QFrame {{ background-color: {Color.SURFACE};"
            f" border-radius: {Size.RADIUS}px; }}"
        )
        self._root = QVBoxLayout(self)
        self._root.setContentsMargins(Size.SPACING, Size.SPACING_SM,
                                      Size.SPACING, Size.SPACING_SM)
        self._root.setSpacing(Size.SPACING_SM)

        header = QLabel(title)
        header.setFont(font(18, bold=True))
        self._root.addWidget(header)

        self._content = QWidget()
        self._content_lay = QVBoxLayout(self._content)
        self._content_lay.setContentsMargins(0, 0, 0, 0)
        self._content_lay.setSpacing(Size.SPACING_SM)
        self._root.addWidget(self._content)

    @property
    def content_layout(self) -> QVBoxLayout:
        return self._content_lay


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
        root.setContentsMargins(Size.SPACING, Size.SPACING_SM, Size.SPACING, Size.SPACING_SM)
        root.setSpacing(Size.SPACING_SM)

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
        sections = QVBoxLayout(container)
        sections.setSpacing(Size.SPACING_SM)

        # Hardware
        hw = _Section("Hardware")
        for device in ["Camera", "Robot", "IMU Left", "IMU Right"]:
            row = QHBoxLayout()
            row.addWidget(QLabel(device))
            row.addStretch()
            dot = _StatusDot(connected=False)
            row.addWidget(dot)
            hw.content_layout.addLayout(row)
        sections.addWidget(hw)

        # Sound
        snd = _Section("Sound")
        vol_row = QHBoxLayout()
        vol_row.addWidget(QLabel("Master Volume"))
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, 100)
        slider.setValue(80)
        slider.setMinimumHeight(Size.MIN_TOUCH)
        vol_row.addWidget(slider)
        snd.content_layout.addLayout(vol_row)
        for toggle_name in ["Punch Sounds", "Timer Beeps", "Coach Voice"]:
            row = QHBoxLayout()
            row.addWidget(QLabel(toggle_name))
            row.addStretch()
            cb = QCheckBox()
            cb.setChecked(True)
            row.addWidget(cb)
            snd.content_layout.addLayout(row)
        sections.addWidget(snd)

        # Display
        disp = _Section("Display")
        row = QHBoxLayout()
        row.addWidget(QLabel("Gesture Control"))
        row.addStretch()
        cb = QCheckBox()
        row.addWidget(cb)
        disp.content_layout.addLayout(row)
        sections.addWidget(disp)

        # AI Coach
        ai = _Section("AI Coach")
        row = QHBoxLayout()
        row.addWidget(QLabel("Enable AI Coach"))
        row.addStretch()
        self._ai_cb = QCheckBox()
        self._ai_cb.setChecked(True)
        row.addWidget(self._ai_cb)
        ai.content_layout.addLayout(row)
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("LLM Status"))
        row2.addStretch()
        self._llm_dot = _StatusDot(connected=False)
        row2.addWidget(self._llm_dot)
        ai.content_layout.addLayout(row2)
        sections.addWidget(ai)

        # Network
        net = _Section("Network")
        net_row = QHBoxLayout()
        net_row.addWidget(QLabel("WiFi AP Status"))
        net_row.addStretch()
        self._wifi_dot = _StatusDot(connected=True)
        net_row.addWidget(self._wifi_dot)
        net.content_layout.addLayout(net_row)
        qr_row = QHBoxLayout()
        self._net_qr = QRWidget(size=64)
        self._net_qr.set_text("https://boxbunny.local")
        qr_row.addWidget(self._net_qr)
        url_lbl = QLabel("boxbunny.local")
        url_lbl.setStyleSheet(f"color: {Color.TEXT_SECONDARY}; font-size: 14px;")
        qr_row.addWidget(url_lbl)
        qr_row.addStretch()
        net.content_layout.addLayout(qr_row)
        sections.addWidget(net)

        # System
        sys_sec = _Section("System")
        ver = QLabel("BoxBunny v1.0.0")
        ver.setStyleSheet(f"color: {Color.TEXT_SECONDARY}; font-size: 14px;")
        sys_sec.content_layout.addWidget(ver)
        db_lbl = QLabel("Database: ~/.boxbunny/sessions.db")
        db_lbl.setStyleSheet(f"color: {Color.TEXT_DISABLED}; font-size: 13px;")
        sys_sec.content_layout.addWidget(db_lbl)
        btn_maint = BigButton("Database Maintenance", stylesheet=SURFACE_BTN)
        btn_maint.setFixedHeight(48)
        # TODO: connect to maintenance routine
        sys_sec.content_layout.addWidget(btn_maint)
        sections.addWidget(sys_sec)

        sections.addStretch()
        scroll.setWidget(container)
        root.addWidget(scroll, stretch=1)

    def on_enter(self, **kwargs: Any) -> None:
        logger.debug("SettingsPage entered")  # TODO: refresh hardware status

    def on_leave(self) -> None:
        pass
