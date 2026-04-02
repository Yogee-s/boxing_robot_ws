"""Quick-launch preset overlay — triggered by head pad IMU.

Shows a centred popup card with horizontal preset cards and glow selection.
Left/right pads cycle, centre confirms and starts training,
head pad again dismisses.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from PySide6.QtCore import (
    QEasingCurve, QPoint, QPropertyAnimation, QTimer, Qt,
)
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QGraphicsDropShadowEffect,
    QHBoxLayout, QLabel, QVBoxLayout, QWidget,
)

from boxbunny_gui.theme import Color, Size

logger = logging.getLogger(__name__)

# Default guest presets
GUEST_PRESETS: List[Dict[str, Any]] = [
    {
        "name": "Free Training",
        "tag": "OPEN SESSION",
        "desc": "Punch freely with no combos",
        "route": "training_session",
        "combo": {"id": None, "name": "Free Training", "seq": ""},
        "config": {
            "Rounds": "1", "Work Time": "120s",
            "Rest Time": "30s", "Speed": "Medium (2s)",
        },
        "difficulty": "Beginner",
        "accent": Color.INFO,
    },
    {
        "name": "Jab-Cross Drill",
        "tag": "TECHNIQUE",
        "desc": "Classic 1-2 combo drill",
        "route": "training_session",
        "combo": {"id": "beginner_007", "name": "Jab-Cross", "seq": "1-2"},
        "config": {
            "Rounds": "2", "Work Time": "60s",
            "Rest Time": "30s", "Speed": "Medium (2s)",
        },
        "difficulty": "Beginner",
        "accent": Color.PRIMARY,
    },
    {
        "name": "Power Test",
        "tag": "PERFORMANCE",
        "desc": "Test your max punch force",
        "route": "power_test",
        "combo": {},
        "config": {},
        "difficulty": "",
        "accent": Color.DANGER,
    },
    {
        "name": "Reaction Time",
        "tag": "PERFORMANCE",
        "desc": "Test your reflexes",
        "route": "reaction_test",
        "combo": {},
        "config": {},
        "difficulty": "",
        "accent": Color.WARNING,
    },
]

_CARD_W = 220
_CARD_H = 180
_ANIM_MS = 300


class _PresetCard(QWidget):
    """Single preset card."""

    def __init__(
        self, preset: Dict[str, Any], parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.preset = preset
        self.setFixedSize(_CARD_W, _CARD_H)

        accent = preset.get("accent", Color.PRIMARY)
        self._accent = accent

        self.setObjectName("pcard")
        self._default_style = f"""
            QWidget#pcard {{
                background-color: {Color.SURFACE};
                border: 2px solid {Color.BORDER};
                border-bottom: 3px solid {accent};
                border-radius: {Size.RADIUS_LG}px;
            }}
            QWidget#pcard QLabel {{
                background: transparent; border: none;
            }}
        """
        self._glow_style = f"""
            QWidget#pcard {{
                background-color: {Color.SURFACE_LIGHT};
                border: 2px solid {accent};
                border-bottom: 3px solid {accent};
                border-radius: {Size.RADIUS_LG}px;
            }}
            QWidget#pcard QLabel {{
                background: transparent; border: none;
            }}
        """
        self.setStyleSheet(self._default_style)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 16, 18, 16)
        lay.setSpacing(6)

        # Tag
        tag = QLabel(preset.get("tag", ""))
        tag.setStyleSheet(
            f"font-size: 11px; font-weight: 700; color: {accent};"
            " letter-spacing: 1px;"
        )
        lay.addWidget(tag)

        # Name — big and bold
        name = QLabel(preset["name"])
        name.setStyleSheet(
            f"font-size: 24px; font-weight: 700; color: {Color.TEXT};"
        )
        name.setWordWrap(True)
        lay.addWidget(name)

        lay.addStretch()

        # Description
        desc = preset.get("desc", "")
        if desc:
            desc_lbl = QLabel(desc)
            desc_lbl.setStyleSheet(
                f"font-size: 14px; color: {Color.TEXT_SECONDARY};"
            )
            desc_lbl.setWordWrap(True)
            lay.addWidget(desc_lbl)

    def set_selected(self, selected: bool) -> None:
        self.setStyleSheet(self._glow_style if selected else self._default_style)
        if selected:
            shadow = QGraphicsDropShadowEffect(self)
            shadow.setBlurRadius(30)
            shadow.setColor(QColor(self._accent))
            shadow.setOffset(0, 0)
            self.setGraphicsEffect(shadow)
        else:
            self.setGraphicsEffect(None)


class PresetOverlay(QWidget):
    """Full-screen dark backdrop with a centred popup panel."""

    def __init__(
        self,
        on_select: Callable[[Dict[str, Any]], None],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._on_select = on_select
        self._presets: List[Dict[str, Any]] = []
        self._cards: List[_PresetCard] = []
        self._current_idx: int = 0
        self._visible: bool = False
        self._animating: bool = False
        self._username: str = ""

        self.setStyleSheet("background-color: rgba(0, 0, 0, 160);")
        # Allow clicks to pass through to widgets below (like Quick Start button)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.hide()

        # ── Popup panel ──────────────────────────────────────────────────
        self._panel = QWidget(self)
        self._panel.setStyleSheet(f"""
            QWidget {{
                background-color: {Color.BG};
                border: 1px solid {Color.BORDER_LIGHT};
                border-radius: {Size.RADIUS_LG}px;
            }}
        """)

        panel_lay = QVBoxLayout(self._panel)
        panel_lay.setContentsMargins(24, 16, 24, 20)
        panel_lay.setSpacing(12)

        # Header
        header = QHBoxLayout()
        title = QLabel("QUICK START")
        title.setStyleSheet(
            f"font-size: 16px; font-weight: 700; color: {Color.PRIMARY};"
            " letter-spacing: 2px; border: none; background: transparent;"
        )
        header.addWidget(title)
        header.addStretch()
        panel_lay.addLayout(header)

        # Cards row
        self._cards_container = QWidget(self._panel)
        self._cards_container.setStyleSheet("background: transparent; border: none;")
        cards_lay = QHBoxLayout(self._cards_container)
        cards_lay.setSpacing(14)
        cards_lay.setContentsMargins(8, 8, 8, 8)
        cards_lay.setAlignment(Qt.AlignCenter)
        self._cards_layout = cards_lay
        panel_lay.addWidget(self._cards_container)

        # Selection highlight — slides behind the selected card
        self._highlight = QWidget(self._cards_container)
        self._highlight.setFixedSize(_CARD_W + 10, _CARD_H + 10)
        self._highlight.setStyleSheet(f"""
            background-color: transparent;
            border: 3px solid {Color.PRIMARY};
            border-radius: {Size.RADIUS_LG + 3}px;
        """)
        self._highlight.lower()
        self._highlight.hide()

        self._highlight_anim = QPropertyAnimation(self._highlight, b"pos")
        self._highlight_anim.setDuration(200)
        self._highlight_anim.setEasingCurve(QEasingCurve.Type.OutCubic)

        # Footer instructions
        footer = QHBoxLayout()
        footer.setAlignment(Qt.AlignCenter)
        footer.setSpacing(24)
        for icon, text in [
            ("\u2190 \u2192", "Left / Right pad to browse"),
            ("\u25CF", "Centre pad to start"),
            ("HEAD", "Head pad to close"),
        ]:
            icon_lbl = QLabel(icon)
            icon_lbl.setStyleSheet(
                f"font-size: 13px; font-weight: 700; color: {Color.PRIMARY};"
                " border: none; background: transparent;"
            )
            text_lbl = QLabel(text)
            text_lbl.setStyleSheet(
                f"font-size: 12px; color: {Color.TEXT_SECONDARY};"
                " border: none; background: transparent;"
            )
            pair = QHBoxLayout()
            pair.setSpacing(6)
            pair.addWidget(icon_lbl)
            pair.addWidget(text_lbl)
            footer.addLayout(pair)
        panel_lay.addLayout(footer)

        # Panel animation
        self._panel_anim = QPropertyAnimation(self._panel, b"pos")
        self._panel_anim.setDuration(_ANIM_MS)
        self._panel_anim.setEasingCurve(QEasingCurve.Type.OutCubic)

    # ── Public API ───────────────────────────────────────────────────────

    def set_username(self, username: str) -> None:
        self._username = username

    def toggle(self) -> None:
        if self._animating:
            return
        if self._visible:
            self.slide_out()
        else:
            self.slide_in()

    def slide_in(self) -> None:
        self._load_presets()
        self._build_cards()
        self._visible = True
        self._animating = True

        win = self.parent()
        if win is None:
            return
        win_w, win_h = win.width(), win.height()

        # Start below the top bar (60px) so Quick Start button stays clickable
        top_bar_h = 60
        self.setGeometry(0, top_bar_h, win_w, win_h - top_bar_h)

        n = len(self._presets)
        panel_w = min(win_w - 30, _CARD_W * n + 14 * (n - 1) + 52)
        panel_h = _CARD_H + 150  # header + cards + footer with padding
        panel_x = (win_w - panel_w) // 2
        overlay_h = win_h - top_bar_h
        panel_y_end = (overlay_h - panel_h) // 2

        self._panel.setFixedSize(panel_w, panel_h)
        self._panel.move(panel_x, overlay_h)

        self.raise_()
        self.show()

        self._panel_anim.stop()
        self._panel_anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._panel_anim.setStartValue(QPoint(panel_x, overlay_h))
        self._panel_anim.setEndValue(QPoint(panel_x, panel_y_end))
        self._panel_anim.start()
        QTimer.singleShot(_ANIM_MS + 20, self._on_anim_done)

    def slide_out(self) -> None:
        self._visible = False
        self._animating = True
        overlay_h = self.height()
        panel_x = self._panel.x()

        self._panel_anim.stop()
        self._panel_anim.setEasingCurve(QEasingCurve.Type.InCubic)
        self._panel_anim.setStartValue(self._panel.pos())
        self._panel_anim.setEndValue(QPoint(panel_x, overlay_h))
        self._panel_anim.start()
        # Hide after animation finishes
        QTimer.singleShot(_ANIM_MS + 50, self._hide_if_closed)

    def navigate_left(self) -> None:
        if not self._visible or not self._cards:
            return
        self._current_idx = (self._current_idx - 1) % len(self._cards)
        self._update_selection()

    def navigate_right(self) -> None:
        if not self._visible or not self._cards:
            return
        self._current_idx = (self._current_idx + 1) % len(self._cards)
        self._update_selection()

    def confirm(self) -> None:
        if not self._visible or not self._presets:
            return
        preset = self._presets[self._current_idx]
        logger.info("Preset selected: %s", preset["name"])
        self.slide_out()
        self._on_select(preset)

    @property
    def is_visible(self) -> bool:
        return self._visible

    # ── Internals ────────────────────────────────────────────────────────

    def _load_presets(self) -> None:
        """Load user presets from DB, or guest defaults."""
        if self._username:
            user_presets = self._load_user_presets()
            if user_presets:
                self._presets = user_presets
                self._current_idx = 0
                return
        self._presets = list(GUEST_PRESETS)
        self._current_idx = 0

    def _load_user_presets(self) -> List[Dict[str, Any]]:
        """Load presets from the main database (shared with phone dashboard)."""
        try:
            import json
            import sqlite3
            from pathlib import Path
            db_path = (
                Path(__file__).resolve().parents[4]
                / "data" / "boxbunny_main.db"
            )
            if not db_path.exists():
                # Try absolute fallback
                db_path = Path(
                    "/home/boxbunny/Desktop/doomsday_integration/"
                    "boxing_robot_ws/data/boxbunny_main.db"
                )
            if not db_path.exists():
                return []
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
            # Get user_id from username
            user_row = conn.execute(
                "SELECT id FROM users WHERE username = ?",
                (self._username,),
            ).fetchone()
            if not user_row:
                conn.close()
                return []
            user_id = user_row["id"]
            rows = conn.execute(
                "SELECT * FROM presets WHERE user_id = ? "
                "AND (tags IS NULL OR tags != 'archived') "
                "ORDER BY is_favorite DESC, use_count DESC",
                (user_id,),
            ).fetchall()
            conn.close()
            presets = []
            for row in rows:
                presets.append(self._row_to_preset(dict(row)))
            logger.info("Loaded %d presets for %s", len(presets), self._username)
            return presets
        except Exception as exc:
            logger.debug("Could not load user presets: %s", exc)
            return []

    @staticmethod
    def _row_to_preset(row: dict) -> Dict[str, Any]:
        """Convert a main DB preset row to the overlay card format."""
        import json
        cfg = {}
        try:
            cfg = json.loads(row.get("config_json", "{}"))
        except (json.JSONDecodeError, TypeError):
            pass
        rounds = str(cfg.get("rounds", cfg.get("Rounds", "2")))
        work_sec = cfg.get("work_sec", 90)
        rest_sec = cfg.get("rest_sec", 30)
        speed = cfg.get("speed", cfg.get("Speed", "Medium (2s)"))
        combo_name = cfg.get("combo_name", cfg.get("combo", {}).get("name", ""))
        combo_seq = cfg.get("combo_seq", cfg.get("combo", {}).get("seq", ""))
        combo_id = cfg.get("combo_id", cfg.get("combo", {}).get("id"))
        difficulty = cfg.get("difficulty", "Beginner")
        ptype = row.get("preset_type", "training")
        route_map = {
            "training": "training_session", "sparring": "sparring_session",
            "performance": "power_test", "free": "training_session",
            "circuit": "training_session",
        }
        route = cfg.get("route", route_map.get(ptype, "training_session"))
        work_time = f"{int(work_sec)}s" if isinstance(work_sec, (int, float)) else str(work_sec)
        rest_time = f"{int(rest_sec)}s" if isinstance(rest_sec, (int, float)) else str(rest_sec)
        accent_map = {
            "training": Color.PRIMARY, "sparring": Color.DANGER,
            "performance": Color.WARNING, "free": Color.INFO,
            "circuit": Color.PURPLE,
        }
        return {
            "name": row.get("name", "Preset"),
            "tag": ptype.upper(),
            "desc": row.get("description", ""),
            "route": route,
            "combo": {"id": combo_id, "name": combo_name, "seq": combo_seq},
            "config": {
                "Rounds": rounds, "Work Time": work_time,
                "Rest Time": rest_time, "Speed": speed if isinstance(speed, str) else "Medium (2s)",
            },
            "difficulty": difficulty.title() if difficulty else "Beginner",
            "accent": accent_map.get(ptype, Color.PRIMARY),
        }

    def _build_cards(self) -> None:
        for card in self._cards:
            self._cards_layout.removeWidget(card)
            card.deleteLater()
        self._cards.clear()

        for preset in self._presets:
            card = _PresetCard(preset, self._panel)
            self._cards_layout.addWidget(card)
            self._cards.append(card)

        self._update_selection()

    def _update_selection(self) -> None:
        for i, card in enumerate(self._cards):
            card.set_selected(i == self._current_idx)

        # Move the highlight behind the selected card
        if not self._cards:
            self._highlight.hide()
            return

        selected = self._cards[self._current_idx]
        # Update highlight accent colour — just a clean border, no fill
        accent = selected.preset.get("accent", Color.PRIMARY)
        self._highlight.setStyleSheet(f"""
            background-color: transparent;
            border: 3px solid {accent};
            border-radius: {Size.RADIUS_LG + 3}px;
        """)
        self._highlight.show()

        # Calculate target position relative to the cards container
        # Need to defer slightly so layout has settled
        QTimer.singleShot(10, self._move_highlight)

    def _move_highlight(self) -> None:
        if not self._cards or self._current_idx >= len(self._cards):
            return
        selected = self._cards[self._current_idx]
        target = QPoint(
            selected.x() - 5,
            selected.y() - 5,
        )
        if self._highlight.isHidden():
            self._highlight.move(target)
            self._highlight.show()
        else:
            self._highlight_anim.stop()
            self._highlight_anim.setStartValue(self._highlight.pos())
            self._highlight_anim.setEndValue(target)
            self._highlight_anim.start()

    def _on_anim_done(self) -> None:
        self._animating = False

    def _hide_if_closed(self) -> None:
        """Hide the widget after slide-out, only if still closed."""
        self._animating = False
        if not self._visible:
            self.hide()

    def mousePressEvent(self, event) -> None:
        """Click on backdrop (outside panel) closes the overlay."""
        if self._visible:
            # Only close if click is outside the panel
            panel_rect = self._panel.geometry()
            if not panel_rect.contains(event.pos()):
                self.slide_out()
