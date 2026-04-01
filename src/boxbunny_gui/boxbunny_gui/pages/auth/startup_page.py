"""Startup page — first screen users see.

Premium dark landing page with gradient branding, animated CTA button,
and clear entry points. QR popup for phone dashboard access.
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
import urllib.request
from io import BytesIO
from pathlib import Path
from threading import Thread
from typing import Any

from PySide6.QtCore import Qt, Signal, QObject, QTimer, QPropertyAnimation, QEasingCurve, Property
from PySide6.QtGui import QPixmap, QColor, QPainter, QLinearGradient, QFont
from PySide6.QtWidgets import (
    QDialog, QFrame, QGraphicsDropShadowEffect,
    QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget,
)

from boxbunny_gui.theme import (
    Color, Icon, Size, close_btn_style,
    hero_btn_style, secondary_btn_style, subtle_btn_style,
)

logger = logging.getLogger(__name__)

_URL_FILE = "/tmp/boxbunny_dashboard_url.txt"
_GUI_LOGIN_FILE = Path("/tmp/boxbunny_gui_login.json")
_WS_ROOT = Path(__file__).resolve().parents[5]  # boxing_robot_ws/
_DASHBOARD_SCRIPT = _WS_ROOT / "tools" / "dashboard_server.py"

# Module-level refs so subprocesses outlive the popup dialog
_dashboard_proc: subprocess.Popen | None = None
_tunnel_proc: subprocess.Popen | None = None


def _server_is_up() -> bool:
    try:
        urllib.request.urlopen("http://localhost:8080/api/health", timeout=2)
        return True
    except Exception:
        return False


def _ensure_server() -> None:
    """Start the dashboard server if it isn't already running."""
    global _dashboard_proc  # noqa: PLW0603
    if _server_is_up():
        return
    logger.info("Starting dashboard server: %s", _DASHBOARD_SCRIPT)
    _dashboard_proc = subprocess.Popen(
        [sys.executable, str(_DASHBOARD_SCRIPT)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(20):
        time.sleep(0.5)
        if _server_is_up():
            return
    logger.warning("Dashboard server did not respond in time")


def _start_tunnel() -> str | None:
    """Open a localhost.run SSH tunnel. Returns public https URL or None."""
    global _tunnel_proc  # noqa: PLW0603
    for attempt in range(3):
        if attempt > 0:
            logger.info("Tunnel retry %d/3", attempt + 1)
            time.sleep(2)
        _tunnel_proc = subprocess.Popen(
            [
                "ssh", "-o", "StrictHostKeyChecking=no",
                "-o", "ServerAliveInterval=30",
                "-o", "ConnectTimeout=10",
                "-R", "80:localhost:8080", "nokey@localhost.run",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        deadline = time.time() + 20
        while time.time() < deadline:
            line = _tunnel_proc.stdout.readline().decode("utf-8", errors="ignore")
            if not line:
                break
            if ".lhr.life" in line:
                for word in line.split():
                    if word.startswith("https://"):
                        url = word.strip().rstrip(",")
                        logger.info("Tunnel URL: %s", url)
                        return url
        _tunnel_proc.terminate()
        try:
            _tunnel_proc.wait(timeout=3)
        except Exception:
            _tunnel_proc.kill()
    return None


class _TunnelSignals(QObject):
    """Signals emitted from the background tunnel thread."""
    url_ready = Signal(str)


class _QrPopup(QDialog):
    """QR code popup that auto-starts dashboard + tunnel."""

    login_info: dict | None = None

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Scan QR Code")
        self.setFixedSize(500, 520)
        self.setStyleSheet(f"background-color: {Color.BG};")
        self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(16)
        layout.setContentsMargins(40, 30, 40, 30)

        title = QLabel("Scan with your phone")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(
            f"font-size: 22px; font-weight: 700; color: {Color.TEXT};"
        )
        layout.addWidget(title)

        # QR code area
        self._qr_label = QLabel("Connecting...")
        self._qr_label.setAlignment(Qt.AlignCenter)
        self._qr_label.setFixedSize(300, 300)
        self._qr_label.setStyleSheet(
            f"color: {Color.TEXT_SECONDARY}; font-size: 18px;"
        )
        layout.addWidget(self._qr_label, alignment=Qt.AlignCenter)

        self._url_label = QLabel("Setting up public URL...")
        self._url_label.setAlignment(Qt.AlignCenter)
        self._url_label.setStyleSheet(
            f"font-size: 14px; color: {Color.TEXT_SECONDARY}; font-weight: 600;"
        )
        self._url_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self._url_label)

        self._hint = QLabel("Starting dashboard server and tunnel...")
        self._hint.setAlignment(Qt.AlignCenter)
        self._hint.setStyleSheet(
            f"font-size: 14px; color: {Color.TEXT_SECONDARY};"
        )
        self._hint.setWordWrap(True)
        layout.addWidget(self._hint)

        close_btn = QPushButton("Close")
        close_btn.setFixedSize(140, 44)
        close_btn.setStyleSheet(f"""
            QPushButton {{
                font-size: 16px; font-weight: 600;
                background-color: {Color.SURFACE}; color: {Color.TEXT_SECONDARY};
                border: 1px solid {Color.BORDER_LIGHT}; border-radius: 10px;
            }}
            QPushButton:hover {{ color: {Color.TEXT}; border-color: {Color.PRIMARY}; }}
        """)
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn, alignment=Qt.AlignCenter)

        # Clear any stale login file
        try:
            _GUI_LOGIN_FILE.unlink(missing_ok=True)
        except OSError:
            pass

        self._signals = _TunnelSignals()
        self._signals.url_ready.connect(self._on_url_ready)
        self._worker = Thread(target=self._setup_tunnel, daemon=True)
        self._worker.start()

        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self._check_phone_login)
        self._poll_timer.start(1000)

    def _setup_tunnel(self) -> None:
        _ensure_server()
        try:
            with open(_URL_FILE) as f:
                existing = f.read().strip()
            if existing.startswith("https://"):
                self._signals.url_ready.emit(existing)
                return
        except OSError:
            pass
        url = _start_tunnel()
        if url:
            try:
                with open(_URL_FILE, "w") as f:
                    f.write(url)
            except OSError:
                pass
            self._signals.url_ready.emit(url)
        else:
            fallback = "http://localhost:8080"
            try:
                with open(_URL_FILE) as f:
                    fallback = f.read().strip() or fallback
            except OSError:
                pass
            self._signals.url_ready.emit(fallback)

    def _on_url_ready(self, url: str) -> None:
        self._url_label.setText(url)
        self._url_label.setStyleSheet(
            f"font-size: 16px; color: {Color.PRIMARY}; font-weight: 600;"
        )
        self._hint.setText("Scan the QR code or visit the URL to log in")
        try:
            import qrcode
            qr = qrcode.QRCode(version=1, box_size=8, border=2)
            qr.add_data(url)
            qr.make(fit=True)
            img = qr.make_image(fill_color="white", back_color=Color.BG)
            buf = BytesIO()
            img.save(buf, format="PNG")
            pix = QPixmap()
            pix.loadFromData(buf.getvalue())
            self._qr_label.setPixmap(
                pix.scaled(280, 280, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
            self._qr_label.setStyleSheet("")
        except ImportError:
            self._qr_label.setText(
                "QR code library not installed\npip install qrcode pillow"
            )
            self._qr_label.setStyleSheet(
                f"color: {Color.DANGER}; font-size: 14px;"
            )

    def _check_phone_login(self) -> None:
        try:
            data = json.loads(_GUI_LOGIN_FILE.read_text())
            username = data.get("username")
            if not username:
                return
            logger.info("Phone login detected: %s", username)
            self._poll_timer.stop()
            self.login_info = data
            _GUI_LOGIN_FILE.unlink(missing_ok=True)
            self.accept()
        except (OSError, json.JSONDecodeError):
            pass


# ── Gradient text label ──────────────────────────────────────────────────────

class _GradientLabel(QWidget):
    """Custom-painted label with a horizontal gradient fill on the text."""

    def __init__(
        self,
        text: str,
        font_size: int = 54,
        color_left: str = "#FF6B35",
        color_right: str = "#FFAB40",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._text = text
        self._font = QFont("Inter", font_size, QFont.Weight.ExtraBold)
        self._color_left = QColor(color_left)
        self._color_right = QColor(color_right)
        self.setFixedHeight(font_size + 36)
        from PySide6.QtGui import QFontMetrics
        fm = QFontMetrics(self._font)
        self.setMinimumWidth(fm.horizontalAdvance(text) + 60)

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setFont(self._font)

        # Measure text bounding rect to center it
        fm = p.fontMetrics()
        text_rect = fm.boundingRect(self._text)
        x = (self.width() - text_rect.width()) // 2
        y = (self.height() + fm.ascent() - fm.descent()) // 2

        # Create horizontal gradient across the text
        gradient = QLinearGradient(x, 0, x + text_rect.width(), 0)
        gradient.setColorAt(0.0, self._color_left)
        gradient.setColorAt(1.0, self._color_right)

        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(gradient)

        # Draw text path for gradient fill
        from PySide6.QtGui import QPainterPath
        path = QPainterPath()
        path.addText(x, y, self._font, self._text)
        p.drawPath(path)
        p.end()


# ── Animated glow wrapper ────────────────────────────────────────────────────

class _GlowButton(QPushButton):
    """QPushButton with a pulsating drop-shadow glow animation."""

    def __init__(self, text: str, parent=None) -> None:
        super().__init__(text, parent)
        self._glow_radius = 12.0

        self._shadow = QGraphicsDropShadowEffect(self)
        self._shadow.setColor(QColor(Color.PRIMARY))
        self._shadow.setBlurRadius(int(self._glow_radius))
        self._shadow.setOffset(0, 2)
        self.setGraphicsEffect(self._shadow)

        # Pulse animation
        self._anim = QPropertyAnimation(self, b"glowRadius")
        self._anim.setDuration(2000)
        self._anim.setStartValue(10.0)
        self._anim.setKeyValueAt(0.5, 28.0)
        self._anim.setEndValue(10.0)
        self._anim.setEasingCurve(QEasingCurve.Type.InOutSine)
        self._anim.setLoopCount(-1)
        self._anim.start()

    def _get_glow(self) -> float:
        return self._glow_radius

    def _set_glow(self, v: float) -> None:
        self._glow_radius = v
        self._shadow.setBlurRadius(int(v))

    glowRadius = Property(float, _get_glow, _set_glow)


# ═══════════════════════════════════════════════════════════════════════════════
# Startup page
# ═══════════════════════════════════════════════════════════════════════════════

class StartupPage(QWidget):
    """Landing screen — premium branding + clear entry points."""

    def __init__(self, router=None, **kwargs):
        super().__init__()
        self._router = router

        root = QVBoxLayout(self)
        root.setSpacing(0)
        root.setContentsMargins(24, 10, 24, 10)

        # ── Top bar: close button flush to top-right ─────────────────────
        top = QHBoxLayout()
        top.addStretch()
        close_btn = QPushButton("Close")
        close_btn.setFixedSize(100, 44)
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn.setStyleSheet(f"""
            QPushButton {{
                font-size: 15px; font-weight: 600;
                background-color: {Color.SURFACE}; color: {Color.TEXT_SECONDARY};
                border: 1px solid {Color.BORDER_LIGHT}; border-radius: 10px;
            }}
            QPushButton:hover {{
                background-color: {Color.DANGER}; color: white;
                border-color: {Color.DANGER};
            }}
        """)
        close_btn.clicked.connect(lambda: self.window().close())
        top.addWidget(close_btn)
        root.addLayout(top)

        root.addStretch(3)

        # ── Branding ─────────────────────────────────────────────────────
        title = _GradientLabel("BoxBunny", font_size=72)
        root.addWidget(title, alignment=Qt.AlignCenter)

        subtitle = QLabel("AI  BOXING  TRAINING  ROBOT")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet(
            f"font-size: 15px; color: {Color.TEXT_DISABLED};"
            " letter-spacing: 4px; font-weight: 600;"
        )
        root.addWidget(subtitle, alignment=Qt.AlignCenter)

        root.addStretch(2)

        # ── Quick Start ──────────────────────────────────────────────────
        start_btn = _GlowButton("Quick Start  \u2192")
        start_btn.setFixedSize(500, 76)
        start_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        start_btn.setStyleSheet(hero_btn_style(size=24))
        start_btn.clicked.connect(lambda: self._nav("guest_assessment"))
        root.addWidget(start_btn, alignment=Qt.AlignCenter)

        root.addSpacing(6)

        guest_hint = QLabel("No account needed \u2014 start training right away")
        guest_hint.setAlignment(Qt.AlignCenter)
        guest_hint.setStyleSheet(
            f"font-size: 14px; color: {Color.TEXT};"
        )
        root.addWidget(guest_hint, alignment=Qt.AlignCenter)

        root.addSpacing(20)

        # ── Log In / Sign Up ─────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.setAlignment(Qt.AlignCenter)
        btn_row.setSpacing(16)

        login_btn = QPushButton("Log In")
        login_btn.setFixedSize(240, 62)
        login_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        login_btn.setStyleSheet(secondary_btn_style())
        login_btn.clicked.connect(lambda: self._nav("account_picker"))

        signup_btn = QPushButton("Sign Up")
        signup_btn.setFixedSize(240, 62)
        signup_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        signup_btn.setStyleSheet(secondary_btn_style())
        signup_btn.clicked.connect(lambda: self._nav("signup"))

        btn_row.addWidget(login_btn)
        btn_row.addWidget(signup_btn)
        root.addLayout(btn_row)

        root.addStretch(2)

        # ── Phone Login — flush to bottom-right ──────────────────────────
        bottom = QHBoxLayout()
        bottom.addStretch()
        qr_btn = QPushButton("Phone Login")
        qr_btn.setFixedSize(170, 48)
        qr_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        qr_btn.setStyleSheet(f"""
            QPushButton {{
                font-size: 14px; font-weight: 600;
                background-color: {Color.SURFACE}; color: {Color.TEXT_SECONDARY};
                border: 1px solid {Color.BORDER_LIGHT}; border-radius: 10px;
            }}
            QPushButton:hover {{
                color: {Color.TEXT}; border-color: {Color.PRIMARY};
                background-color: {Color.SURFACE_HOVER};
            }}
        """)
        qr_btn.clicked.connect(self._show_qr)
        bottom.addWidget(qr_btn)
        root.addLayout(bottom)

    def _show_qr(self) -> None:
        popup = _QrPopup(self)
        if popup.exec() and popup.login_info:
            info = popup.login_info
            logger.info(
                "Auto-login from phone: %s (%s)",
                info.get("username"), info.get("user_type"),
            )
            if self._router:
                self._router.navigate(
                    "home",
                    user_id=info.get("user_id"),
                    username=info.get("username", "Guest"),
                )

    def _nav(self, page: str):
        if self._router:
            self._router.navigate(page)

    def on_enter(self, **kwargs: Any) -> None:
        pass

    def on_leave(self) -> None:
        pass
