"""BoxBunny GUI Theme -- Modern dark theme with teal accent.

All colors, sizes, fonts, and stylesheet factories in one place.
No inline hex colors anywhere else in the codebase.
"""
from __future__ import annotations

from PySide6.QtGui import QColor, QFont


class Color:
    """Canonical color palette — dark navy + warm orange accent."""

    # ── Background layers ──
    BG = "#0B0F14"               # deep navy-black
    SURFACE = "#131920"          # cards, panels
    SURFACE_LIGHT = "#1A2029"    # raised elements
    SURFACE_HOVER = "#222B37"    # hover state

    # ── Accent colors ──
    PRIMARY = "#FF6B35"          # warm orange — bold, energetic
    PRIMARY_DARK = "#E85E2C"     # hover
    PRIMARY_PRESSED = "#CC5025"  # pressed
    PRIMARY_LIGHT = "#FF8C5E"    # highlights, glow
    PRIMARY_MUTED = "#FF6B3518"  # very subtle orange tint
    WARNING = "#FFAB40"          # warm amber
    WARNING_DARK = "#FF9100"
    DANGER = "#FF5C5C"           # vibrant coral-red
    DANGER_DARK = "#E84545"
    INFO = "#58A6FF"             # soft blue
    INFO_DARK = "#388BFD"
    PURPLE = "#BC8CFF"           # lavender accent

    # ── Text ──
    TEXT = "#E6EDF3"             # bright off-white
    TEXT_SECONDARY = "#8B949E"   # muted grey
    TEXT_DISABLED = "#484F58"    # very dim

    # ── Borders ──
    BORDER = "#1C222A"
    BORDER_LIGHT = "#2A3340"
    TRANSPARENT = "transparent"

    # Punch type colors (used by combo_display, dev_overlay, charts)
    JAB = "#58A6FF"            # blue
    CROSS = "#FF5C5C"          # coral
    L_HOOK = "#56D364"         # green
    R_HOOK = "#FFAB40"         # amber
    L_UPPERCUT = "#BC8CFF"     # purple
    R_UPPERCUT = "#F8E45C"     # yellow
    BLOCK = "#8B949E"          # grey
    IDLE = "#484F58"           # dark grey


class Size:
    """Canonical dimension constants (pixels)."""

    MIN_TOUCH = 60
    SPACING = 20
    SPACING_SM = 10
    SPACING_LG = 24
    RADIUS = 12
    RADIUS_SM = 8
    RADIUS_LG = 16
    TEXT_BODY = 16
    TEXT_HEADER = 28
    TEXT_SUBHEADER = 22
    TEXT_TIMER = 80
    TEXT_TIMER_SM = 60
    TEXT_TIMER_XL = 96
    TEXT_LABEL = 14
    SCREEN_W = 1024
    SCREEN_H = 600
    SIDEBAR_W = 200
    TOP_BAR_H = 50
    BUTTON_H = 60
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
    font_size: int = 16,
    min_h: int = 44,
    radius: int = 12,
) -> str:
    """Generate a QPushButton stylesheet."""
    return f"""
        QPushButton {{
            background-color: {bg};
            color: {text};
            font-size: {font_size}px;
            font-weight: 600;
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
PRIMARY_BTN = button_style(
    Color.PRIMARY, Color.PRIMARY_DARK, Color.PRIMARY_PRESSED, Color.BG
)
DANGER_BTN = button_style(Color.DANGER, Color.DANGER_DARK, "#C33C3C")
WARNING_BTN = button_style(Color.WARNING, Color.WARNING_DARK, "#E07800")
SURFACE_BTN = button_style(
    Color.SURFACE_LIGHT, Color.SURFACE_HOVER, Color.SURFACE,
    text=Color.TEXT_SECONDARY,
)
GHOST_BTN = button_style(
    Color.TRANSPARENT, Color.SURFACE, Color.SURFACE_LIGHT,
    text=Color.TEXT_SECONDARY,
)


# -- Global application stylesheet -----------------------------------------
GLOBAL_STYLESHEET = f"""
    QWidget {{
        background-color: {Color.BG};
        color: {Color.TEXT};
        font-family: "Inter", "Segoe UI", "Helvetica Neue", sans-serif;
        font-size: 15px;
    }}
    QLabel {{
        background-color: transparent;
        border: none;
    }}
    QFrame {{
        background-color: transparent;
    }}
    QPushButton QLabel {{
        background-color: transparent;
        border: none;
    }}
    QFrame QLabel {{
        border: none;
    }}
    QScrollArea {{
        border: none;
        background-color: transparent;
    }}
    QScrollArea > QWidget > QWidget {{
        background-color: transparent;
    }}
    QScrollBar:vertical {{
        background: transparent;
        width: 5px;
        border-radius: 2px;
        margin: 4px 1px;
    }}
    QScrollBar::handle:vertical {{
        background: {Color.BORDER_LIGHT};
        border-radius: 2px;
        min-height: 30px;
    }}
    QScrollBar::handle:vertical:hover {{
        background: {Color.TEXT_DISABLED};
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0;
    }}
    QScrollBar:horizontal {{
        background: transparent;
        height: 5px;
        border-radius: 2px;
        margin: 1px 4px;
    }}
    QScrollBar::handle:horizontal {{
        background: {Color.BORDER_LIGHT};
        border-radius: 2px;
        min-width: 30px;
    }}
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
        width: 0;
    }}
    QCheckBox {{
        spacing: 8px;
        font-size: 15px;
    }}
    QCheckBox::indicator {{
        width: 26px;
        height: 26px;
        border-radius: 7px;
        border: 2px solid {Color.BORDER_LIGHT};
        background-color: {Color.SURFACE};
    }}
    QCheckBox::indicator:checked {{
        background-color: {Color.PRIMARY};
        border-color: {Color.PRIMARY};
    }}
    QCheckBox::indicator:hover {{
        border-color: {Color.PRIMARY_LIGHT};
    }}
    QSlider::groove:horizontal {{
        height: 6px;
        background: {Color.SURFACE_LIGHT};
        border-radius: 3px;
    }}
    QSlider::handle:horizontal {{
        width: 22px;
        height: 22px;
        margin: -8px 0;
        background: {Color.PRIMARY};
        border-radius: 11px;
    }}
    QSlider::handle:horizontal:hover {{
        background: {Color.PRIMARY_LIGHT};
    }}
    QSlider::sub-page:horizontal {{
        background: {Color.PRIMARY};
        border-radius: 3px;
    }}
    QLineEdit {{
        font-size: 15px;
        padding: 10px 16px;
        background-color: {Color.SURFACE};
        color: {Color.TEXT};
        border: 1px solid {Color.BORDER_LIGHT};
        border-radius: {Size.RADIUS}px;
        selection-background-color: {Color.PRIMARY};
    }}
    QLineEdit:focus {{
        border-color: {Color.PRIMARY};
    }}
    QLineEdit::placeholder {{
        color: {Color.TEXT_DISABLED};
    }}
"""


# -- Reusable style snippets ------------------------------------------------
def top_bar_btn_style() -> str:
    """Small ghost button for top-bar actions (Settings, Close, etc.)."""
    return f"""
        QPushButton {{
            font-size: 13px; font-weight: 600; padding: 6px 16px;
            background-color: {Color.SURFACE}; color: {Color.TEXT_SECONDARY};
            border: 1px solid {Color.BORDER_LIGHT}; border-radius: 8px;
            min-height: 0; min-width: 0;
        }}
        QPushButton:hover {{ color: {Color.TEXT}; border-color: {Color.PRIMARY}; }}
        QPushButton:pressed {{ background-color: {Color.SURFACE_HOVER}; }}
    """


def close_btn_style() -> str:
    """Round close (X) button style."""
    return f"""
        QPushButton {{
            font-size: 14px; background-color: {Color.SURFACE};
            color: {Color.TEXT_DISABLED};
            border: 1px solid {Color.BORDER_LIGHT}; border-radius: 18px;
            min-height: 0; min-width: 0; padding: 0;
        }}
        QPushButton:hover {{
            background-color: {Color.DANGER}; color: white;
            border-color: {Color.DANGER};
        }}
    """


def section_title_style() -> str:
    """Section header label style."""
    return f"color: {Color.PRIMARY}; font-size: 15px; font-weight: 600;"


def tab_btn_style(active: bool = False) -> str:
    """Filter/tab pill button style with active state."""
    if active:
        return f"""
            QPushButton {{
                font-size: 13px; font-weight: 600; padding: 7px 18px;
                background-color: {Color.PRIMARY}; color: {Color.BG};
                border: none; border-radius: 8px;
                min-height: 0; min-width: 0;
            }}
            QPushButton:hover {{ background-color: {Color.PRIMARY_DARK}; }}
        """
    return f"""
        QPushButton {{
            font-size: 13px; font-weight: 600; padding: 7px 18px;
            background-color: {Color.SURFACE}; color: {Color.TEXT_SECONDARY};
            border: 1px solid {Color.BORDER}; border-radius: 8px;
            min-height: 0; min-width: 0;
        }}
        QPushButton:hover {{ color: {Color.TEXT}; border-color: {Color.PRIMARY}; }}
    """


def back_link_style() -> str:
    """Subtle text-only back link style."""
    return f"""
        QPushButton {{
            font-size: 13px; color: {Color.TEXT_SECONDARY};
            background: transparent; border: none;
            min-height: 0; min-width: 0; padding: 6px 12px;
        }}
        QPushButton:hover {{ color: {Color.TEXT}; }}
    """


def mode_card_style(accent: str) -> str:
    """Large mode selection card with colored left accent."""
    return f"""
        QPushButton {{
            font-size: 16px; font-weight: 600;
            padding: 20px 24px;
            background-color: {Color.SURFACE};
            color: {Color.TEXT};
            border: 1px solid {Color.BORDER};
            border-left: 4px solid {accent};
            border-radius: 14px;
            text-align: left;
        }}
        QPushButton:hover {{
            background-color: {Color.SURFACE_HOVER};
            border-color: {accent}50;
            border-left: 4px solid {accent};
        }}
        QPushButton:pressed {{
            background-color: {Color.SURFACE_LIGHT};
        }}
    """


def config_tile_style() -> str:
    """Tappable configuration tile that cycles through values."""
    return f"""
        QPushButton {{
            background-color: {Color.SURFACE};
            color: {Color.TEXT};
            border: 1px solid {Color.BORDER};
            border-radius: {Size.RADIUS_LG}px;
            font-size: 15px; font-weight: 600;
            padding: 8px;
        }}
        QPushButton:hover {{
            background-color: {Color.SURFACE_HOVER};
            border-color: {Color.PRIMARY};
        }}
        QPushButton:pressed {{ background-color: {Color.SURFACE_LIGHT}; }}
    """


def badge_style(color: str = "") -> str:
    """Small inline badge / pill label."""
    c = color or Color.TEXT_SECONDARY
    return (
        f"font-size: 12px; font-weight: 600; color: {c};"
        f" background-color: {Color.SURFACE}; border-radius: 8px;"
        " padding: 4px 12px;"
    )
