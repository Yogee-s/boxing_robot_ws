"""BoxBunny GUI Theme -- Dark theme matching the original main_gui.py style.

All colors, sizes, fonts, and stylesheet factories in one place.
No inline hex colors anywhere else in the codebase.
"""
from __future__ import annotations

from PySide6.QtGui import QColor, QFont


class Color:
    """Canonical color palette."""

    # ── Black & Orange/Red theme — minimalist, premium ──
    BG = "#0A0A0A"               # deep black
    SURFACE = "#141414"          # cards, panels
    SURFACE_LIGHT = "#1C1C1C"    # raised elements
    SURFACE_HOVER = "#242424"    # hover state
    PRIMARY = "#FF6B35"          # warm orange — primary accent
    PRIMARY_DARK = "#E55A2B"     # pressed
    PRIMARY_LIGHT = "#FF8A5C"    # highlights
    WARNING = "#FF9800"          # amber
    WARNING_DARK = "#F57C00"
    DANGER = "#E53935"           # muted red — stop, back
    DANGER_DARK = "#C62828"
    INFO = "#FF8A65"             # soft orange — secondary info
    INFO_DARK = "#FF7043"
    PURPLE = "#FF5252"           # red-ish accent
    TEXT = "#F5F5F5"             # off-white — easier on eyes
    TEXT_SECONDARY = "#999999"
    TEXT_DISABLED = "#444444"
    BORDER = "#1E1E1E"
    BORDER_LIGHT = "#2A2A2A"
    TRANSPARENT = "transparent"

    # Punch type colors (used by combo_display, dev_overlay, charts)
    JAB = "#42A5F5"            # blue
    CROSS = "#EF5350"          # red
    L_HOOK = "#66BB6A"         # green
    R_HOOK = "#FFA726"         # orange
    L_UPPERCUT = "#AB47BC"     # purple
    R_UPPERCUT = "#FFEE58"     # yellow
    BLOCK = "#78909C"          # grey
    IDLE = "#424242"           # dark grey


class Size:
    """Canonical dimension constants (pixels)."""

    MIN_TOUCH = 60
    SPACING = 20
    SPACING_SM = 8
    SPACING_LG = 24
    RADIUS = 10
    RADIUS_SM = 8
    RADIUS_LG = 16
    TEXT_BODY = 16
    TEXT_HEADER = 28
    TEXT_SUBHEADER = 24
    TEXT_TIMER = 96
    TEXT_TIMER_SM = 72
    TEXT_TIMER_XL = 120
    TEXT_LABEL = 14
    SCREEN_W = 1024
    SCREEN_H = 600
    SIDEBAR_W = 200
    TOP_BAR_H = 50
    BUTTON_H = 65
    BUTTON_W_SM = 120
    BUTTON_W_MD = 300
    BUTTON_W_LG = 500
    LAYOUT_MARGINS = (60, 40, 60, 40)


def font(size: int = 16, bold: bool = False) -> QFont:
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
    min_h: int = 65,
    radius: int = 10,
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


# -- Pre-built button styles ------------------------------------------------
PRIMARY_BTN = button_style(Color.PRIMARY, Color.PRIMARY_DARK, "#CC4A1A", Color.TEXT)
DANGER_BTN = button_style(Color.DANGER, Color.DANGER_DARK, "#9B0000")
WARNING_BTN = button_style(Color.WARNING, Color.WARNING_DARK, "#E65100")
SURFACE_BTN = button_style(Color.SURFACE_LIGHT, Color.SURFACE_HOVER, Color.SURFACE)
GHOST_BTN = button_style(Color.TRANSPARENT, Color.SURFACE, Color.SURFACE_LIGHT)


# -- Global application stylesheet (minimal — pages style buttons individually)
GLOBAL_STYLESHEET = f"""
    QWidget {{
        background-color: {Color.BG};
        color: {Color.TEXT};
        font-family: "Inter", "Segoe UI", "Helvetica Neue", sans-serif;
        font-size: 16px;
    }}
    QLabel {{
        background-color: transparent;
    }}
    QFrame {{
        background-color: transparent;
    }}
"""
