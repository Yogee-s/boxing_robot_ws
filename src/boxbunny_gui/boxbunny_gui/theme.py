"""BoxBunny GUI Theme -- Premium dark theme with warm orange accent.

All colors, sizes, fonts, and stylesheet factories in one place.
No inline hex colors anywhere else in the codebase.
"""
from __future__ import annotations

from PySide6.QtGui import QColor, QFont


class Color:
    """Canonical color palette — dark navy + warm orange accent."""

    # ── Background layers ──
    BG = "#0B0F14"               # deep navy-black
    BG_GRADIENT_TOP = "#0E1319"  # slightly lighter for gradient
    BG_GRADIENT_BTM = "#080B10"  # slightly darker for gradient
    SURFACE = "#131920"          # cards, panels
    SURFACE_LIGHT = "#1A2029"    # raised elements
    SURFACE_HOVER = "#222B37"    # hover state
    SURFACE_GLASS = "rgba(19, 25, 32, 0.85)"  # glassmorphism

    # ── Accent colors ──
    PRIMARY = "#FF6B35"          # warm orange — bold, energetic
    PRIMARY_DARK = "#E85E2C"     # hover
    PRIMARY_PRESSED = "#CC5025"  # pressed
    PRIMARY_LIGHT = "#FF8C5E"    # highlights, glow
    PRIMARY_MUTED = "#FF6B3518"  # very subtle orange tint
    PRIMARY_GLOW = "#FF6B3530"   # subtle glow background
    WARNING = "#FFAB40"          # warm amber
    WARNING_DARK = "#FF9100"
    DANGER = "#FF5C5C"           # vibrant coral-red
    DANGER_DARK = "#E84545"
    SUCCESS = "#56D364"          # fresh green
    SUCCESS_DARK = "#3FB950"
    INFO = "#58A6FF"             # soft blue
    INFO_DARK = "#388BFD"
    PURPLE = "#BC8CFF"           # lavender accent

    # ── Text ──
    TEXT = "#E6EDF3"             # bright off-white
    TEXT_SECONDARY = "#8B949E"   # muted grey
    TEXT_DISABLED = "#484F58"    # very dim
    TEXT_ACCENT = "#FFB088"      # warm accent text (titles, highlights)

    # ── Borders ──
    BORDER = "#1C222A"
    BORDER_LIGHT = "#2A3340"
    BORDER_ACCENT = "#FF6B3540"  # subtle orange border
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


class Icon:
    """Minimal text symbols — no emojis, clean and professional."""
    CHECK = "✓"
    CLOSE = "✕"
    BACK = "←"
    NEXT = "→"
    PLAY = "▶"
    STOP = "■"


class Size:
    """Canonical dimension constants (pixels)."""

    MIN_TOUCH = 60
    SPACING = 20
    SPACING_SM = 10
    SPACING_XS = 6
    SPACING_LG = 24
    SPACING_XL = 32
    RADIUS = 12
    RADIUS_SM = 8
    RADIUS_LG = 16
    RADIUS_XL = 20
    TEXT_BODY = 16
    TEXT_HEADER = 28
    TEXT_SUBHEADER = 22
    TEXT_TIMER = 80
    TEXT_TIMER_SM = 60
    TEXT_TIMER_XL = 96
    TEXT_LABEL = 14
    TEXT_CAPTION = 12
    TEXT_OVERLINE = 10
    SCREEN_W = 1024
    SCREEN_H = 600
    SIDEBAR_W = 200
    TOP_BAR_H = 50
    BUTTON_H = 60
    BUTTON_H_SM = 44
    BUTTON_H_LG = 64
    BUTTON_W_SM = 120
    BUTTON_W_MD = 300
    BUTTON_W_LG = 500
    LAYOUT_MARGINS = (60, 40, 60, 40)
    SHADOW_BLUR = 20
    SHADOW_BLUR_LG = 32
    ACCENT_BAR_W = 4
    RING_THICKNESS = 6


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
    border: str = "",
) -> str:
    """Generate a QPushButton stylesheet with premium feel."""
    border_css = f"border: 1px solid {border};" if border else "border: none;"
    return f"""
        QPushButton {{
            background-color: {bg};
            color: {text};
            font-size: {font_size}px;
            font-weight: 600;
            {border_css}
            border-radius: {radius}px;
            min-height: {min_h}px;
            padding: 8px 24px;
            letter-spacing: 0.3px;
        }}
        QPushButton:hover {{
            background-color: {hover};
        }}
        QPushButton:pressed {{
            background-color: {pressed};
        }}
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
SUCCESS_BTN = button_style(Color.SUCCESS, Color.SUCCESS_DARK, "#2EA043")
SURFACE_BTN = button_style(
    Color.SURFACE_LIGHT, Color.SURFACE_HOVER, Color.SURFACE,
    text=Color.TEXT_SECONDARY,
    border=Color.BORDER_LIGHT,
)
GHOST_BTN = button_style(
    Color.TRANSPARENT, Color.SURFACE, Color.SURFACE_LIGHT,
    text=Color.TEXT_SECONDARY,
)
INFO_BTN = button_style(
    Color.SURFACE, Color.SURFACE_HOVER, Color.SURFACE_LIGHT,
    text=Color.INFO,
    border=Color.BORDER_LIGHT,
)


# -- Global application stylesheet -----------------------------------------
GLOBAL_STYLESHEET = f"""
    QWidget {{
        background-color: {Color.BG};
        color: {Color.TEXT};
        font-family: "Inter", "Segoe UI", "Helvetica Neue", sans-serif;
        font-size: 15px;
        font-weight: 500;
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
    QProgressBar {{
        background-color: {Color.SURFACE_LIGHT};
        border: none;
        border-radius: 4px;
        height: 8px;
        text-align: center;
        font-size: 0px;
    }}
    QProgressBar::chunk {{
        background-color: {Color.PRIMARY};
        border-radius: 4px;
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
        QPushButton:hover {{
            color: {Color.TEXT}; border-color: {Color.PRIMARY};
            background-color: {Color.SURFACE_LIGHT};
        }}
        QPushButton:pressed {{ background-color: {Color.SURFACE_HOVER}; }}
    """


def close_btn_style() -> str:
    """Subtle close button style matching top bar buttons."""
    return f"""
        QPushButton {{
            font-size: 13px; font-weight: 600; padding: 6px 14px;
            background-color: {Color.SURFACE}; color: {Color.TEXT_SECONDARY};
            border: 1px solid {Color.BORDER_LIGHT}; border-radius: 8px;
            min-height: 0; min-width: 0;
        }}
        QPushButton:hover {{
            background-color: {Color.DANGER}; color: white;
            border-color: {Color.DANGER};
        }}
        QPushButton:pressed {{ background-color: {Color.DANGER_DARK}; color: white; }}
    """


def section_title_style(color: str = "") -> str:
    """Section header label style."""
    c = color or Color.PRIMARY
    return (
        f"color: {c}; font-size: 13px; font-weight: 700;"
        " letter-spacing: 1.2px; text-transform: uppercase;"
    )


def tab_btn_style(active: bool = False) -> str:
    """Filter/tab pill button style with active state."""
    if active:
        return f"""
            QPushButton {{
                font-size: 13px; font-weight: 700; padding: 8px 20px;
                background-color: {Color.PRIMARY}; color: #FFFFFF;
                border: none; border-radius: 10px;
                min-height: 0; min-width: 0;
                letter-spacing: 0.3px;
            }}
            QPushButton:hover {{ background-color: {Color.PRIMARY_DARK}; }}
        """
    return f"""
        QPushButton {{
            font-size: 13px; font-weight: 600; padding: 8px 20px;
            background-color: {Color.SURFACE}; color: {Color.TEXT_SECONDARY};
            border: 1px solid {Color.BORDER}; border-radius: 10px;
            min-height: 0; min-width: 0;
        }}
        QPushButton:hover {{
            color: {Color.TEXT}; border-color: {Color.PRIMARY};
            background-color: {Color.SURFACE_LIGHT};
        }}
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
    """Premium mode selection card with colored left accent bar and glow hover."""
    return f"""
        QPushButton {{
            font-size: 16px; font-weight: 600;
            padding: 16px 20px;
            background-color: {Color.SURFACE};
            color: {Color.TEXT};
            border: 1px solid {Color.BORDER};
            border-left: {Size.ACCENT_BAR_W}px solid {accent};
            border-radius: {Size.RADIUS_LG}px;
            text-align: left;
        }}
        QPushButton:hover {{
            background-color: {Color.SURFACE_HOVER};
            border-color: {accent}40;
            border-left: {Size.ACCENT_BAR_W}px solid {accent};
        }}
        QPushButton:pressed {{
            background-color: {Color.SURFACE_LIGHT};
        }}
    """


def mode_card_style_v2(accent: str) -> str:
    """Premium elevated mode card with thicker accent and hover glow effect."""
    return f"""
        QPushButton {{
            font-size: 16px; font-weight: 600;
            padding: 0px;
            background-color: {Color.SURFACE};
            color: {Color.TEXT};
            border: 1px solid {Color.BORDER};
            border-left: 5px solid {accent};
            border-radius: {Size.RADIUS_LG}px;
            text-align: left;
        }}
        QPushButton:hover {{
            background-color: {Color.SURFACE_HOVER};
            border-color: {accent}50;
            border-left: 5px solid {accent};
        }}
        QPushButton:pressed {{
            background-color: {Color.SURFACE_LIGHT};
            border-left: 5px solid {accent};
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


def config_tile_style_v2(accent: str = "") -> str:
    """Enhanced config tile with colored top accent."""
    c = accent or Color.PRIMARY
    return f"""
        QPushButton {{
            background-color: {Color.SURFACE};
            color: {Color.TEXT};
            border: 1px solid {Color.BORDER};
            border-top: 3px solid {c};
            border-radius: {Size.RADIUS_LG}px;
            font-size: 15px; font-weight: 600;
            padding: 8px;
        }}
        QPushButton:hover {{
            background-color: {Color.SURFACE_HOVER};
            border-color: {c}50;
            border-top: 3px solid {c};
        }}
        QPushButton:pressed {{
            background-color: {Color.SURFACE_LIGHT};
            border-top: 3px solid {c};
        }}
    """


def badge_style(color: str = "") -> str:
    """Small inline badge / pill label."""
    c = color or Color.TEXT_SECONDARY
    return (
        f"font-size: 11px; font-weight: 700; color: {c};"
        f" background-color: {Color.SURFACE}; border-radius: 8px;"
        " padding: 4px 12px; letter-spacing: 0.6px;"
    )


def elevated_card_style(accent: str = "") -> str:
    """Card with subtle hover elevation and optional accent."""
    border_top = f"border-top: 3px solid {accent};" if accent else ""
    return f"""
        QFrame {{
            background-color: {Color.SURFACE};
            border: 1px solid {Color.BORDER};
            {border_top}
            border-radius: {Size.RADIUS_LG}px;
        }}
        QFrame:hover {{
            background-color: {Color.SURFACE_LIGHT};
            border-color: {Color.BORDER_LIGHT};
        }}
    """


def glass_card_style() -> str:
    """Glassmorphism-inspired card (translucent dark surface)."""
    return f"""
        QFrame {{
            background-color: rgba(19, 25, 32, 0.82);
            border: 1px solid {Color.BORDER_LIGHT};
            border-radius: {Size.RADIUS_LG}px;
        }}
    """


def accent_frame_style(accent: str) -> str:
    """Frame with left accent bar — used for stat cards, info panels."""
    return f"""
        QFrame {{
            background-color: {Color.SURFACE};
            border: 1px solid {Color.BORDER};
            border-left: {Size.ACCENT_BAR_W}px solid {accent};
            border-radius: {Size.RADIUS}px;
        }}
    """


def hero_btn_style(bg: str = "", hover: str = "", size: int = 22) -> str:
    """Large hero CTA button with rounded corners and weight."""
    _bg = bg or Color.PRIMARY
    _hover = hover or Color.PRIMARY_DARK
    return f"""
        QPushButton {{
            font-size: {size}px; font-weight: 700;
            background-color: {_bg}; color: #FFFFFF;
            border: none; border-radius: {Size.RADIUS_LG}px;
            letter-spacing: 0.5px;
        }}
        QPushButton:hover {{ background-color: {_hover}; }}
        QPushButton:pressed {{ background-color: {Color.PRIMARY_PRESSED}; }}
    """


def secondary_btn_style() -> str:
    """Secondary button with border — used for Log In, Sign Up, etc."""
    return f"""
        QPushButton {{
            font-size: 18px; font-weight: 600;
            background-color: {Color.SURFACE}; color: {Color.TEXT};
            border: 2px solid {Color.BORDER_LIGHT}; border-radius: {Size.RADIUS_LG}px;
        }}
        QPushButton:hover {{
            background-color: {Color.SURFACE_HOVER};
            border-color: {Color.PRIMARY}; color: {Color.PRIMARY_LIGHT};
        }}
        QPushButton:pressed {{ background-color: {Color.SURFACE_LIGHT}; }}
    """


def outline_btn_style(accent: str = "") -> str:
    """Outline button that fills on hover."""
    c = accent or Color.PRIMARY
    return f"""
        QPushButton {{
            font-size: 14px; font-weight: 600;
            background-color: transparent; color: {c};
            border: 1px solid {c}; border-radius: {Size.RADIUS_SM}px;
        }}
        QPushButton:hover {{
            background-color: {c}; color: #FFFFFF;
        }}
        QPushButton:pressed {{
            background-color: {c}; color: #FFFFFF;
        }}
    """


def subtle_btn_style() -> str:
    """Very subtle text-like button for secondary actions."""
    return f"""
        QPushButton {{
            font-size: 13px; font-weight: 600;
            background-color: transparent; color: {Color.TEXT_SECONDARY};
            border: 1px solid {Color.BORDER_LIGHT}; border-radius: 8px;
            min-height: 0; min-width: 0; padding: 7px 16px;
        }}
        QPushButton:hover {{
            color: {Color.TEXT}; border-color: {Color.PRIMARY};
            background-color: {Color.SURFACE};
        }}
        QPushButton:pressed {{
            background-color: {Color.SURFACE_HOVER};
        }}
    """


def pill_toggle_style(active: bool) -> str:
    """Segmented-control pill toggle button."""
    if active:
        return f"""
            QPushButton {{
                font-size: 14px; font-weight: 700;
                background-color: {Color.PRIMARY}; color: #FFFFFF;
                border: 2px solid {Color.PRIMARY}; border-radius: {Size.RADIUS}px;
                padding: 8px 16px;
            }}
            QPushButton:hover {{ background-color: {Color.PRIMARY_DARK}; }}
        """
    return f"""
        QPushButton {{
            font-size: 14px; font-weight: 600;
            background-color: {Color.SURFACE}; color: {Color.TEXT_SECONDARY};
            border: 2px solid {Color.BORDER}; border-radius: {Size.RADIUS}px;
            padding: 8px 16px;
        }}
        QPushButton:hover {{
            color: {Color.TEXT}; border-color: {Color.PRIMARY};
            background-color: {Color.SURFACE_HOVER};
        }}
    """
