"""BoxBunny GUI Theme -- Dark theme with green accent.

All colors, sizes, fonts, and stylesheet factories in one place.
No inline hex colors anywhere else in the codebase.
"""
from __future__ import annotations

from PySide6.QtGui import QColor, QFont


class Color:
    """Canonical color palette."""

    BG = "#0D0D0D"
    SURFACE = "#1A1A1A"
    SURFACE_LIGHT = "#2A2A2A"
    SURFACE_HOVER = "#333333"
    PRIMARY = "#00E676"        # green accent
    PRIMARY_DARK = "#00C853"
    PRIMARY_LIGHT = "#69F0AE"
    WARNING = "#FF9800"
    WARNING_DARK = "#F57C00"
    DANGER = "#FF1744"
    DANGER_DARK = "#D50000"
    TEXT = "#FFFFFF"
    TEXT_SECONDARY = "#9E9E9E"
    TEXT_DISABLED = "#616161"
    BORDER = "#333333"
    BORDER_LIGHT = "#444444"
    TRANSPARENT = "transparent"


class Size:
    """Canonical dimension constants (pixels)."""

    MIN_TOUCH = 60
    SPACING = 16
    SPACING_SM = 8
    SPACING_LG = 24
    RADIUS = 12
    RADIUS_SM = 8
    RADIUS_LG = 16
    TEXT_BODY = 18
    TEXT_HEADER = 32
    TEXT_SUBHEADER = 24
    TEXT_TIMER = 96
    TEXT_TIMER_SM = 72
    TEXT_TIMER_XL = 120
    TEXT_LABEL = 14
    SCREEN_W = 1024
    SCREEN_H = 600
    SIDEBAR_W = 200
    TOP_BAR_H = 50
    BUTTON_H = 60
    BUTTON_W_SM = 120
    BUTTON_W_MD = 200
    BUTTON_W_LG = 300


def font(size: int = 18, bold: bool = False) -> QFont:
    """Create a QFont with the BoxBunny standard font."""
    f = QFont("Inter", size)
    if bold:
        f.setBold(True)
    return f


def button_style(
    bg: str,
    hover: str,
    pressed: str,
    text: str = Color.TEXT,
    font_size: int = 20,
    min_h: int = 60,
    radius: int = 12,
) -> str:
    """Generate a QPushButton stylesheet."""
    return f"""
        QPushButton {{
            background-color: {bg};
            color: {text};
            font-size: {font_size}px;
            font-weight: bold;
            border: none;
            border-radius: {radius}px;
            min-height: {min_h}px;
            padding: 8px 24px;
        }}
        QPushButton:hover {{ background-color: {hover}; }}
        QPushButton:pressed {{ background-color: {pressed}; }}
        QPushButton:disabled {{
            background-color: {Color.SURFACE_LIGHT};
            color: {Color.TEXT_DISABLED};
        }}
    """


# ── Pre-built button styles ────────────────────────────────────────────────
PRIMARY_BTN = button_style(Color.PRIMARY, Color.PRIMARY_DARK, "#009624", Color.BG)
DANGER_BTN = button_style(Color.DANGER, Color.DANGER_DARK, "#9B0000")
WARNING_BTN = button_style(Color.WARNING, Color.WARNING_DARK, "#E65100")
SURFACE_BTN = button_style(Color.SURFACE_LIGHT, Color.SURFACE_HOVER, Color.SURFACE)
GHOST_BTN = button_style(Color.TRANSPARENT, Color.SURFACE, Color.SURFACE_LIGHT)


# ── Global application stylesheet ──────────────────────────────────────────
GLOBAL_STYLESHEET = f"""
    QWidget {{
        background-color: {Color.BG};
        color: {Color.TEXT};
        font-family: "Inter", "Segoe UI", "Helvetica Neue", sans-serif;
        font-size: {Size.TEXT_BODY}px;
    }}
    QLabel {{
        background-color: transparent;
    }}
    QScrollArea {{
        border: none;
        background-color: {Color.BG};
    }}
    QScrollBar:vertical {{
        background-color: {Color.SURFACE};
        width: 8px;
        border-radius: 4px;
    }}
    QScrollBar::handle:vertical {{
        background-color: {Color.SURFACE_HOVER};
        border-radius: 4px;
        min-height: 30px;
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0px;
    }}
    QLineEdit {{
        background-color: {Color.SURFACE};
        color: {Color.TEXT};
        border: 2px solid {Color.BORDER};
        border-radius: {Size.RADIUS_SM}px;
        padding: 8px 12px;
        font-size: {Size.TEXT_BODY}px;
        min-height: 40px;
    }}
    QLineEdit:focus {{
        border-color: {Color.PRIMARY};
    }}
"""
