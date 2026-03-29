"""Startup page — first screen users see.

Clean, modern dark UI. Large buttons, clear hierarchy, no clutter.
QR popup for phone dashboard access.
"""
from __future__ import annotations

import logging
import socket
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget,
)

logger = logging.getLogger(__name__)


def _btn(text: str, bg: str, hover: str, size: int = 22, w: int = 480, h: int = 60) -> QPushButton:
    b = QPushButton(text)
    b.setStyleSheet(f"""
        QPushButton {{
            font-size: {size}px; font-weight: 600; padding: 12px;
            min-width: {w}px; min-height: {h}px;
            background-color: {bg}; color: #FFFFFF;
            border: none; border-radius: 14px;
        }}
        QPushButton:hover {{ background-color: {hover}; }}
    """)
    return b


class _QrPopup(QDialog):
    """Full-screen QR code popup for phone dashboard scanning."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Scan QR Code")
        self.setFixedSize(500, 500)
        self.setStyleSheet("background-color: #0A0A0A;")
        self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(16)

        title = QLabel("Scan with your phone")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 22px; font-weight: 700; color: #F5F5F5;")
        layout.addWidget(title)

        # Get local IP
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
        except Exception:
            ip = "localhost"

        url = f"http://{ip}:8080"

        # Generate QR code
        qr_label = QLabel()
        qr_label.setAlignment(Qt.AlignCenter)
        qr_label.setFixedSize(300, 300)
        try:
            import qrcode
            from io import BytesIO
            from PySide6.QtGui import QPixmap

            qr = qrcode.QRCode(version=1, box_size=8, border=2)
            qr.add_data(url)
            qr.make(fit=True)
            img = qr.make_image(fill_color="white", back_color="#0A0A0A")
            buf = BytesIO()
            img.save(buf, format="PNG")
            pix = QPixmap()
            pix.loadFromData(buf.getvalue())
            qr_label.setPixmap(pix.scaled(280, 280, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        except ImportError:
            qr_label.setText("QR code library not installed\npip install qrcode pillow")
            qr_label.setStyleSheet("color: #E53935; font-size: 14px;")
        layout.addWidget(qr_label, alignment=Qt.AlignCenter)

        url_label = QLabel(url)
        url_label.setAlignment(Qt.AlignCenter)
        url_label.setStyleSheet("font-size: 16px; color: #FF6B35; font-weight: 600;")
        url_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(url_label)

        hint = QLabel("Connect your phone to the same network\nthen scan or visit the URL above")
        hint.setAlignment(Qt.AlignCenter)
        hint.setStyleSheet("font-size: 13px; color: #888;")
        layout.addWidget(hint)

        close_btn = QPushButton("Close")
        close_btn.setStyleSheet("""
            QPushButton {
                font-size: 16px; padding: 10px 40px;
                background-color: #1C1C1C; color: #999;
                border: 1px solid #2A2A2A; border-radius: 10px;
                min-height: 0; min-width: 0;
            }
            QPushButton:hover { color: #F5F5F5; border-color: #555; }
        """)
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn, alignment=Qt.AlignCenter)


class StartupPage(QWidget):
    """Landing screen — branding + clear entry points."""

    def __init__(self, router=None, **kwargs):
        super().__init__()
        self._router = router

        root = QVBoxLayout(self)
        root.setSpacing(0)
        root.setContentsMargins(80, 30, 80, 30)

        # ── Close button (top-right) ─────────────────────────────────────
        close_btn = QPushButton("\u2715")
        close_btn.setFixedSize(38, 38)
        close_btn.setStyleSheet("""
            QPushButton {
                font-size: 14px; background-color: #1C1C1C; color: #666;
                border: 1px solid #2A2A2A; border-radius: 19px;
                min-height: 0; min-width: 0; padding: 0;
            }
            QPushButton:hover { background-color: #E53935; color: white; border-color: #E53935; }
        """)
        close_btn.clicked.connect(lambda: self.window().close())

        top = QHBoxLayout()
        top.addStretch()
        top.addWidget(close_btn)
        root.addLayout(top)

        root.addStretch(4)

        # ── Branding ─────────────────────────────────────────────────────
        title = QLabel("BoxBunny")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 54px; font-weight: 800; color: #FF6B35; letter-spacing: 2px;")

        subtitle = QLabel("AI Boxing Training Robot")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("font-size: 18px; color: #888; letter-spacing: 1px;")

        root.addWidget(title, alignment=Qt.AlignCenter)
        root.addSpacing(6)
        root.addWidget(subtitle, alignment=Qt.AlignCenter)

        root.addStretch(3)

        # ── Primary action ───────────────────────────────────────────────
        start_btn = _btn("Start Training", "#FF6B35", "#E55A2B")
        start_btn.clicked.connect(lambda: self._nav("guest_assessment"))
        root.addWidget(start_btn, alignment=Qt.AlignCenter)

        root.addStretch(1)

        # ── Log In / Sign Up side by side ────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.setAlignment(Qt.AlignCenter)
        btn_row.setSpacing(16)

        login_btn = _btn("Log In", "#1C1C1C", "#242424", size=20, w=230, h=54)
        login_btn.setStyleSheet(login_btn.styleSheet() + """
            QPushButton { border: 2px solid #2A2A2A; }
            QPushButton:hover { border-color: #FF6B35; }
        """)
        login_btn.clicked.connect(lambda: self._nav("account_picker"))

        signup_btn = _btn("Sign Up", "#1C1C1C", "#242424", size=20, w=230, h=54)
        signup_btn.setStyleSheet(signup_btn.styleSheet() + """
            QPushButton { border: 2px solid #2A2A2A; }
            QPushButton:hover { border-color: #FF6B35; }
        """)
        signup_btn.clicked.connect(lambda: self._nav("signup"))

        btn_row.addWidget(login_btn)
        btn_row.addWidget(signup_btn)
        root.addLayout(btn_row)

        root.addStretch(1)

        # ── QR code link (opens full popup) ──────────────────────────────
        qr_link = QPushButton("Open Phone Dashboard (QR)")
        qr_link.setStyleSheet("""
            QPushButton {
                font-size: 14px; color: #FF6B35; background: transparent;
                border: none; min-height: 0; min-width: 0; padding: 4px;
                text-decoration: underline;
            }
            QPushButton:hover { color: #FF8A65; }
        """)
        qr_link.clicked.connect(self._show_qr)
        root.addWidget(qr_link, alignment=Qt.AlignCenter)

        root.addStretch(2)

    def _show_qr(self) -> None:
        popup = _QrPopup(self)
        popup.exec()

    def _nav(self, page: str):
        if self._router:
            self._router.navigate(page)

    def on_enter(self, **kwargs: Any) -> None:
        pass

    def on_leave(self) -> None:
        pass
