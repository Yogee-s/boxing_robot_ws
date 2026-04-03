"""Settings page: account, hardware, sound, display, AI, network, system."""
from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Any, List, Optional

from PySide6.QtCore import Qt, QPoint, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QPainter, QPen, QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from boxbunny_gui.theme import Color, Size, font, GHOST_BTN, SURFACE_BTN, back_link_style
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
        color = Color.SUCCESS if connected else Color.DANGER
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


class _SettingsPatternGrid(QWidget):
    """Compact 3x3 pattern grid for settings."""

    _CELL = 50
    _DOT_R = 14

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        total = self._CELL * 3
        self.setFixedSize(total, total)
        self._entered: List[int] = []
        self._drawing = False
        self._cur: QPoint | None = None

    def reset(self) -> None:
        self._entered.clear()
        self._drawing = False
        self._cur = None
        self.update()

    @property
    def pattern(self) -> List[int]:
        return list(self._entered)

    def _center(self, idx: int) -> QPoint:
        r, c = divmod(idx, 3)
        return QPoint(c * self._CELL + self._CELL // 2,
                       r * self._CELL + self._CELL // 2)

    def _hit(self, pos: QPoint) -> int:
        for i in range(9):
            c = self._center(i)
            dx, dy = pos.x() - c.x(), pos.y() - c.y()
            if dx * dx + dy * dy <= (self._DOT_R + 8) ** 2:
                return i
        return -1

    def mousePressEvent(self, e) -> None:
        self._entered.clear()
        self._drawing = True
        self._handle(e.position().toPoint())

    def mouseMoveEvent(self, e) -> None:
        if self._drawing:
            self._cur = e.position().toPoint()
            self._handle(self._cur)

    def mouseReleaseEvent(self, e) -> None:
        self._drawing = False
        self._cur = None
        self.update()

    def _handle(self, pos: QPoint) -> None:
        idx = self._hit(pos)
        if idx >= 0 and idx not in self._entered:
            self._entered.append(idx)
        self.update()

    def paintEvent(self, e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        active = QColor(Color.PRIMARY)
        inactive = QColor(Color.SURFACE_HOVER)
        if len(self._entered) > 1:
            lc = QColor(Color.PRIMARY)
            lc.setAlpha(180)
            p.setPen(QPen(lc, 3))
            for i in range(len(self._entered) - 1):
                p.drawLine(self._center(self._entered[i]),
                           self._center(self._entered[i + 1]))
            if self._drawing and self._cur:
                p.drawLine(self._center(self._entered[-1]), self._cur)
        for i in range(9):
            c = self._center(i)
            is_on = i in self._entered
            p.setPen(Qt.NoPen)
            p.setBrush(active if is_on else inactive)
            r = self._DOT_R if is_on else self._DOT_R - 3
            p.drawEllipse(c, r, r)
            if not is_on:
                p.setBrush(QColor(Color.TEXT_DISABLED))
                p.drawEllipse(c, 4, 4)
        p.end()


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
        self._username: str = ""

        # Panel lives as a free-positioned child (no layout on self)
        # so we can animate its position for the slide-down effect.
        self._panel = QWidget(self)
        self._slide_anim = QPropertyAnimation(self._panel, b"pos")
        self._slide_anim.setDuration(350)
        self._slide_anim.setEasingCurve(QEasingCurve.Type.OutCubic)

        self._build_ui()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._panel.resize(self.size())

    def _build_ui(self) -> None:
        root = QVBoxLayout(self._panel)
        root.setContentsMargins(24, 16, 24, 22)
        root.setSpacing(14)

        # Top bar — consistent with other pages
        top = QHBoxLayout()
        btn_back = QPushButton(f"\u2190  Back")
        btn_back.setStyleSheet(back_link_style())
        btn_back.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_back.clicked.connect(lambda: self._router.back())
        top.addWidget(btn_back)
        title = QLabel("Settings")
        title.setStyleSheet(f"font-size: 20px; font-weight: 700; color: {Color.TEXT};")
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

        # Account section
        acct = _Section("Account")
        self._acct_user_lbl = QLabel("Not logged in")
        self._acct_user_lbl.setStyleSheet(
            f"background: transparent; font-size: 14px; color: {Color.TEXT};"
        )
        acct.content_layout.addWidget(self._acct_user_lbl)

        # ── Guest mode: Create Account prompt ────────────────────────────
        self._guest_section = QWidget()
        guest_lay = QVBoxLayout(self._guest_section)
        guest_lay.setContentsMargins(0, 0, 0, 0)
        guest_lay.setSpacing(8)

        guest_info = QLabel(
            "You're in guest mode. Create an account to save your progress."
        )
        guest_info.setWordWrap(True)
        guest_info.setStyleSheet(
            f"background: transparent; font-size: 13px; color: {Color.TEXT_SECONDARY};"
        )
        guest_lay.addWidget(guest_info)

        create_btn = QPushButton("Create Account")
        create_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        create_btn.setFixedSize(200, 48)
        create_btn.setStyleSheet(self._action_btn_style())
        create_btn.clicked.connect(
            lambda: self._router.navigate("signup") if self._router else None
        )
        guest_lay.addWidget(create_btn)
        acct.content_layout.addWidget(self._guest_section)

        # ── Logged-in mode: password & pattern controls ──────────────────
        self._auth_section = QWidget()
        auth_lay = QVBoxLayout(self._auth_section)
        auth_lay.setContentsMargins(0, 0, 0, 0)
        auth_lay.setSpacing(8)

        # Password change row
        pw_lbl = QLabel("Change Password")
        pw_lbl.setStyleSheet(
            f"background: transparent; font-size: 12px; font-weight: 600;"
            f" color: {Color.TEXT_SECONDARY};"
        )
        auth_lay.addWidget(pw_lbl)

        pw_row = QHBoxLayout()
        pw_row.setSpacing(8)
        self._new_pw = QLineEdit()
        self._new_pw.setPlaceholderText("New password")
        self._new_pw.setEchoMode(QLineEdit.EchoMode.Password)
        self._new_pw.setFixedHeight(44)
        self._new_pw.setMinimumWidth(160)
        pw_row.addWidget(self._new_pw)

        self._confirm_pw = QLineEdit()
        self._confirm_pw.setPlaceholderText("Confirm password")
        self._confirm_pw.setEchoMode(QLineEdit.EchoMode.Password)
        self._confirm_pw.setFixedHeight(44)
        self._confirm_pw.setMinimumWidth(160)
        pw_row.addWidget(self._confirm_pw)

        pw_save = QPushButton("Update Password")
        pw_save.setCursor(Qt.CursorShape.PointingHandCursor)
        pw_save.setFixedSize(160, 44)
        pw_save.setStyleSheet(self._action_btn_style())
        pw_save.clicked.connect(self._on_change_password)
        pw_row.addWidget(pw_save)
        auth_lay.addLayout(pw_row)

        # Pattern — collapsible behind a button
        pat_toggle_row = QHBoxLayout()
        pat_lbl = QLabel("Pattern Lock")
        pat_lbl.setStyleSheet(
            f"background: transparent; font-size: 12px; font-weight: 600;"
            f" color: {Color.TEXT_SECONDARY};"
        )
        pat_toggle_row.addWidget(pat_lbl)
        pat_toggle_row.addStretch()

        self._pat_toggle_btn = QPushButton("Set / Change Pattern")
        self._pat_toggle_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._pat_toggle_btn.setFixedSize(190, 44)
        self._pat_toggle_btn.setStyleSheet(f"""
            QPushButton {{
                font-size: 12px; font-weight: 600;
                background-color: {Color.SURFACE_LIGHT}; color: {Color.TEXT_SECONDARY};
                border: 1px solid {Color.BORDER}; border-radius: 6px;
            }}
            QPushButton:hover {{
                color: {Color.PRIMARY}; border-color: {Color.PRIMARY};
            }}
        """)
        self._pat_toggle_btn.clicked.connect(self._toggle_pattern_panel)
        pat_toggle_row.addWidget(self._pat_toggle_btn)
        auth_lay.addLayout(pat_toggle_row)

        # Hidden pattern panel — centered grid + buttons below
        self._pat_panel = QWidget()
        self._pat_panel.setMaximumHeight(0)
        pat_panel_lay = QVBoxLayout(self._pat_panel)
        pat_panel_lay.setContentsMargins(0, 6, 0, 0)
        pat_panel_lay.setSpacing(8)
        pat_panel_lay.setAlignment(Qt.AlignCenter)

        self._pattern_grid = _SettingsPatternGrid()
        pat_panel_lay.addWidget(self._pattern_grid, alignment=Qt.AlignCenter)

        pat_btn_row = QHBoxLayout()
        pat_btn_row.setAlignment(Qt.AlignCenter)
        pat_btn_row.setSpacing(8)

        pat_save = QPushButton("Save Pattern")
        pat_save.setCursor(Qt.CursorShape.PointingHandCursor)
        pat_save.setFixedSize(140, 44)
        pat_save.setStyleSheet(self._action_btn_style())
        pat_save.clicked.connect(self._on_set_pattern)
        pat_btn_row.addWidget(pat_save)

        pat_clear = QPushButton("Reset")
        pat_clear.setCursor(Qt.CursorShape.PointingHandCursor)
        pat_clear.setFixedSize(80, 32)
        pat_clear.setStyleSheet(f"""
            QPushButton {{
                font-size: 12px; font-weight: 600;
                background-color: transparent; color: {Color.TEXT_SECONDARY};
                border: 1px solid {Color.BORDER}; border-radius: 6px;
            }}
            QPushButton:hover {{ color: {Color.TEXT}; border-color: {Color.PRIMARY}; }}
        """)
        pat_clear.clicked.connect(self._pattern_grid.reset)
        pat_btn_row.addWidget(pat_clear)
        pat_panel_lay.addLayout(pat_btn_row)

        # Animation for smooth expand/collapse
        self._pat_anim = QPropertyAnimation(self._pat_panel, b"maximumHeight")
        self._pat_anim.setDuration(250)
        self._pat_anim.setEasingCurve(QEasingCurve.Type.InOutCubic)

        auth_lay.addWidget(self._pat_panel)

        acct.content_layout.addWidget(self._auth_section)

        self._pw_status = QLabel("")
        self._pw_status.setStyleSheet(
            f"background: transparent; font-size: 12px; color: {Color.DANGER};"
        )
        auth_lay.addWidget(self._pw_status)
        sections.addWidget(acct)

        # Hardware
        hw = _Section("Hardware")
        self._hw_dots = {}
        for device in ["Camera", "Robot", "IMU Left", "IMU Right"]:
            row, _ = _setting_row(device)
            dot = _StatusDot(connected=False)
            self._hw_dots[device] = dot
            row.addWidget(dot)
            hw.content_layout.addLayout(row)
        # Height adjustment
        height_row = QHBoxLayout()
        height_lbl = QLabel("Height Adjust")
        height_lbl.setFont(font(14))
        height_lbl.setStyleSheet(f"color: {Color.TEXT_SECONDARY};")
        height_row.addWidget(height_lbl)
        height_row.addStretch()
        btn_up = QPushButton("UP")
        btn_up.setMinimumSize(80, 50)
        btn_up.setFont(font(14, bold=True))
        btn_up.setStyleSheet(SURFACE_BTN)
        btn_down = QPushButton("DOWN")
        btn_down.setMinimumSize(80, 50)
        btn_down.setFont(font(14, bold=True))
        btn_down.setStyleSheet(SURFACE_BTN)

        self._height_timer = None

        def _height_start(action: str) -> None:
            if self._bridge:
                self._bridge.publish_height_command(action)
            from PySide6.QtCore import QTimer
            self._height_timer = QTimer()
            self._height_timer.timeout.connect(
                lambda: self._bridge.publish_height_command(action) if self._bridge else None
            )
            self._height_timer.start(100)  # 10Hz continuous publish

        def _height_stop() -> None:
            if self._height_timer:
                self._height_timer.stop()
                self._height_timer = None
            if self._bridge:
                self._bridge.publish_height_command("stop")

        btn_up.pressed.connect(lambda: _height_start("manual_up"))
        btn_up.released.connect(_height_stop)
        btn_down.pressed.connect(lambda: _height_start("manual_down"))
        btn_down.released.connect(_height_stop)
        height_row.addWidget(btn_up)
        height_row.addWidget(btn_down)
        hw.content_layout.addLayout(height_row)
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

        # Phone login QR button
        phone_btn = QPushButton("Phone Login / Dashboard")
        phone_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        phone_btn.setFixedHeight(48)
        phone_btn.setStyleSheet(self._action_btn_style())
        phone_btn.clicked.connect(self._on_phone_login)
        net.content_layout.addWidget(phone_btn)

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

    def _toggle_pattern_panel(self) -> None:
        expanding = self._pat_panel.maximumHeight() == 0
        self._pat_anim.stop()
        if expanding:
            self._pat_anim.setStartValue(0)
            self._pat_anim.setEndValue(220)
            self._pat_toggle_btn.setText("Cancel")
        else:
            self._pat_anim.setStartValue(self._pat_panel.height())
            self._pat_anim.setEndValue(0)
            self._pat_toggle_btn.setText("Set / Change Pattern")
            self._pattern_grid.reset()
        self._pat_anim.start()

    @staticmethod
    def _action_btn_style() -> str:
        return f"""
            QPushButton {{
                font-size: 13px; font-weight: 600;
                background-color: {Color.PRIMARY}; color: #FFFFFF;
                border: none; border-radius: 8px;
            }}
            QPushButton:hover {{ background-color: {Color.PRIMARY_DARK}; }}
        """

    def _set_status(self, text: str, success: bool = False) -> None:
        color = Color.PRIMARY if success else Color.DANGER
        self._pw_status.setStyleSheet(
            f"background: transparent; font-size: 12px; color: {color};"
        )
        self._pw_status.setText(text)

    def _on_set_pattern(self) -> None:
        pattern = self._pattern_grid.pattern
        if len(pattern) < 3:
            self._set_status("Pattern must connect at least 3 dots")
            return
        try:
            from boxbunny_gui.db_helper import update_pattern
            if update_pattern(self._username, pattern):
                self._set_status("Pattern updated successfully", success=True)
                self._pattern_grid.reset()
                self._pat_anim.stop()
                self._pat_anim.setStartValue(self._pat_panel.height())
                self._pat_anim.setEndValue(0)
                self._pat_anim.start()
                self._pat_toggle_btn.setText("Set / Change Pattern")
                logger.info("Pattern updated for user: %s", self._username)
            else:
                self._set_status("Failed to update pattern")
        except Exception as exc:
            logger.warning("Pattern change failed: %s", exc)
            self._set_status("Failed to update pattern")

    def _on_change_password(self) -> None:
        new_pw = self._new_pw.text()
        confirm = self._confirm_pw.text()

        if len(new_pw) < 4:
            self._set_status("Password must be at least 4 characters")
            return
        if new_pw != confirm:
            self._set_status("Passwords do not match")
            return

        try:
            from boxbunny_gui.db_helper import update_password
            if update_password(self._username, new_pw):
                self._set_status("Password updated successfully", success=True)
                self._new_pw.clear()
                self._confirm_pw.clear()
                logger.info("Password updated for user: %s", self._username)
            else:
                self._set_status("Failed to update password")
        except Exception as exc:
            logger.warning("Password change failed: %s", exc)
            self._set_status("Failed to update password")

    def _on_phone_login(self) -> None:
        """Open the QR code popup for phone dashboard login."""
        try:
            from boxbunny_gui.pages.auth.startup_page import _QrPopup
            popup = _QrPopup(self._panel)
            result = popup.exec()
            if result and popup.login_info:
                username = popup.login_info.get("username", "")
                if username and self._router:
                    self._router.navigate(
                        "home", username=username,
                    )
        except Exception as exc:
            logger.warning("Phone login popup failed: %s", exc)

    def on_enter(self, **kwargs: Any) -> None:
        self._username = kwargs.get("username", "")
        is_guest = not self._username
        if self._username:
            self._acct_user_lbl.setText(f"Logged in as: {self._username}")
        else:
            self._acct_user_lbl.setText("Guest Session")
        self._guest_section.setVisible(is_guest)
        self._auth_section.setVisible(not is_guest)
        self._new_pw.clear()
        self._confirm_pw.clear()
        self._pattern_grid.reset()
        self._pat_panel.setMaximumHeight(0)
        self._pat_toggle_btn.setText("Set / Change Pattern")
        self._pw_status.setText("")

        # Update live status indicators
        self._update_status()

        # Slide-down entrance animation
        h = self.height() or Size.SCREEN_H
        self._panel.move(0, -h)
        self._slide_anim.stop()
        self._slide_anim.setStartValue(QPoint(0, -h))
        self._slide_anim.setEndValue(QPoint(0, 0))
        self._slide_anim.start()
        logger.debug("SettingsPage entered (user=%s)", self._username)

    def _update_status(self) -> None:
        """Check live status of hardware, LLM, and network."""
        bridge_online = self._bridge.online if self._bridge else False

        # Camera — check if CV node is publishing (use service discovery)
        cam_ok = False
        if bridge_online and self._bridge._worker and self._bridge._worker.node:
            try:
                topics = self._bridge._worker.node.get_topic_names_and_types()
                topic_names = [t[0] for t in topics]
                cam_ok = "/boxbunny/cv/detection" in topic_names
            except Exception:
                pass
        self._hw_dots["Camera"].set_connected(cam_ok)

        # Robot — check if robot node topic exists
        robot_ok = False
        if bridge_online and self._bridge._worker and self._bridge._worker.node:
            try:
                topics = self._bridge._worker.node.get_topic_names_and_types()
                topic_names = [t[0] for t in topics]
                robot_ok = "/boxbunny/robot/command" in topic_names
            except Exception:
                pass
        self._hw_dots["Robot"].set_connected(robot_ok)

        # IMU — based on last IMU status message
        self._hw_dots["IMU Left"].set_connected(bridge_online)
        self._hw_dots["IMU Right"].set_connected(bridge_online)

        # LLM — check if the service is available
        llm_ok = False
        if bridge_online and self._bridge._worker and self._bridge._worker._cli_llm:
            try:
                llm_ok = self._bridge._worker._cli_llm.service_is_ready()
            except Exception:
                pass
        self._llm_dot.set_connected(llm_ok)

        # WiFi
        self._wifi_dot.set_connected(bridge_online)

    def on_leave(self) -> None:
        pass
