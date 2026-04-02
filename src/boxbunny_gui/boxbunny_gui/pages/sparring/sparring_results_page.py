"""Post-sparring results — clean offense/defense breakdown with AI summary."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from boxbunny_gui.theme import Color, Icon, Size, font, PRIMARY_BTN, GHOST_BTN
from boxbunny_gui.widgets import BigButton

if TYPE_CHECKING:
    from boxbunny_gui.gui_bridge import GuiBridge
    from boxbunny_gui.nav.router import PageRouter

logger = logging.getLogger(__name__)

_PUNCH_COLORS: Dict[str, str] = {
    "Jab": Color.JAB, "Cross": Color.CROSS,
    "Hook": Color.L_HOOK, "Uppercut": Color.PURPLE,
}


def _stat_tile(title: str, value: str, accent: str) -> QWidget:
    """Clean stat tile with left accent and no child border artifacts."""
    w = QWidget()
    w.setObjectName("tile")
    w.setStyleSheet(f"""
        QWidget#tile {{
            background-color: #131920;
            border: 1px solid #1E2832;
            border-left: 3px solid {accent};
            border-radius: {Size.RADIUS}px;
        }}
        QWidget#tile QLabel {{
            background: transparent; border: none;
        }}
    """)
    lay = QVBoxLayout(w)
    lay.setContentsMargins(14, 10, 14, 10)
    lay.setSpacing(2)

    hdr = QLabel(title.upper())
    hdr.setStyleSheet(
        f"font-size: 10px; font-weight: 700; color: {Color.TEXT_DISABLED};"
        " letter-spacing: 0.8px;"
    )
    lay.addWidget(hdr)

    val = QLabel(value)
    val.setObjectName("val")
    val.setStyleSheet(
        f"font-size: 22px; font-weight: 700; color: {Color.TEXT};"
    )
    lay.addWidget(val)
    return w


class _DistBar(QWidget):
    """Horizontal punch distribution bar."""

    def __init__(self, name: str, value: int, max_val: int, color: str,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(32)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        dot = QLabel("\u25CF")
        dot.setFixedWidth(14)
        dot.setStyleSheet(f"color: {color}; font-size: 12px;")
        lay.addWidget(dot)

        lbl = QLabel(name)
        lbl.setFixedWidth(80)
        lbl.setStyleSheet(
            f"color: {Color.TEXT}; font-size: 13px; font-weight: 600;"
        )
        lay.addWidget(lbl)

        # Bar background
        bar_bg = QWidget()
        bar_bg.setFixedHeight(10)
        bar_bg.setStyleSheet(
            f"background-color: {Color.SURFACE_LIGHT}; border-radius: 5px;"
        )
        bar_lay = QHBoxLayout(bar_bg)
        bar_lay.setContentsMargins(0, 0, 0, 0)

        bar_fill = QWidget()
        width_pct = max(5, int(100 * value / max_val)) if max_val else 5
        bar_fill.setFixedHeight(10)
        bar_fill.setMaximumWidth(200)
        bar_fill.setMinimumWidth(int(200 * width_pct / 100))
        bar_fill.setStyleSheet(
            f"background-color: {color}; border-radius: 5px;"
        )
        bar_lay.addWidget(bar_fill)
        bar_lay.addStretch()
        lay.addWidget(bar_bg, stretch=1)

        cnt = QLabel(str(value))
        cnt.setFixedWidth(30)
        cnt.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        cnt.setStyleSheet(
            f"color: {Color.TEXT_SECONDARY}; font-size: 13px; font-weight: 700;"
        )
        lay.addWidget(cnt)


class SparringResultsPage(QWidget):
    """Offense + defense breakdown after a sparring session."""

    def __init__(
        self,
        router: PageRouter,
        bridge: Optional[GuiBridge] = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._router = router
        self._bridge = bridge
        self._config: Dict[str, Any] = {}
        self._username: str = ""
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(32, 14, 32, 18)
        root.setSpacing(0)

        root.addStretch(1)

        # ── Title bar ────────────────────────────────────────────────────
        title_row = QHBoxLayout()
        title = QLabel("Sparring Complete")
        title.setStyleSheet(
            f"font-size: 22px; font-weight: 700; color: {Color.DANGER};"
        )
        title_row.addWidget(title)
        title_row.addStretch()
        self._style_tag = QLabel("")
        self._style_tag.setStyleSheet(
            f"font-size: 12px; font-weight: 700; color: {Color.TEXT_SECONDARY};"
            f" background-color: {Color.SURFACE};"
            f" border: 1px solid {Color.BORDER};"
            " border-radius: 8px; padding: 4px 12px;"
        )
        title_row.addWidget(self._style_tag)
        root.addLayout(title_row)

        root.addSpacing(10)

        # ── Offense row ──────────────────────────────────────────────────
        self._off_total = _stat_tile("Total Punches", "--", Color.PRIMARY)
        root.addWidget(self._off_total)
        root.addSpacing(6)

        # Punch distribution bars
        self._dist_layout = QVBoxLayout()
        self._dist_layout.setSpacing(2)
        root.addLayout(self._dist_layout)

        root.addSpacing(10)

        # ── Defense grid ─────────────────────────────────────────────────
        def_grid = QGridLayout()
        def_grid.setSpacing(8)
        self._def_rate = _stat_tile("Defense Rate", "--%", Color.WARNING)
        self._def_blocks = _stat_tile("Blocks", "--", Color.PRIMARY)
        self._def_slips = _stat_tile("Slips", "--", Color.INFO)
        self._def_dodges = _stat_tile("Dodges", "--", Color.PURPLE)
        self._def_hits = _stat_tile("Hits Taken", "--", Color.DANGER)
        def_grid.addWidget(self._def_rate, 0, 0)
        def_grid.addWidget(self._def_blocks, 0, 1)
        def_grid.addWidget(self._def_slips, 0, 2)
        def_grid.addWidget(self._def_dodges, 1, 0)
        def_grid.addWidget(self._def_hits, 1, 1)
        root.addLayout(def_grid)

        root.addSpacing(10)

        # ── AI Coach Analysis ────────────────────────────────────────────
        ai_box = QWidget()
        ai_box.setObjectName("aibox")
        ai_box.setStyleSheet(f"""
            QWidget#aibox {{
                background-color: {Color.SURFACE};
                border: 1px solid {Color.BORDER};
                border-radius: {Size.RADIUS}px;
            }}
            QWidget#aibox QLabel {{
                background: transparent; border: none;
            }}
        """)
        ai_lay = QVBoxLayout(ai_box)
        ai_lay.setContentsMargins(16, 12, 16, 12)
        ai_lay.setSpacing(4)

        ai_title = QLabel("AI COACH ANALYSIS")
        ai_title.setStyleSheet(
            f"font-size: 10px; font-weight: 700; color: {Color.INFO};"
            " letter-spacing: 1px;"
        )
        ai_lay.addWidget(ai_title)

        self._ai_lbl = QLabel("AI analysis loading...")
        self._ai_lbl.setStyleSheet(
            f"font-size: 14px; color: {Color.TEXT};"
        )
        self._ai_lbl.setWordWrap(True)
        self._ai_lbl.setMinimumHeight(40)
        ai_lay.addWidget(self._ai_lbl)
        root.addWidget(ai_box)

        root.addStretch(1)

        # ── Action buttons ───────────────────────────────────────────────
        bottom = QHBoxLayout()
        bottom.setSpacing(12)

        btn_home = QPushButton(f"{Icon.BACK}  Home")
        btn_home.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_home.setFixedHeight(52)
        btn_home.setStyleSheet(f"""
            QPushButton {{
                font-size: 15px; font-weight: 600;
                background-color: {Color.SURFACE};
                color: {Color.TEXT};
                border: 1px solid {Color.BORDER_LIGHT};
                border-radius: {Size.RADIUS}px;
                padding: 0 24px;
            }}
            QPushButton:hover {{
                border-color: {Color.PRIMARY};
                background-color: {Color.SURFACE_HOVER};
            }}
        """)
        btn_home.clicked.connect(self._go_home)
        bottom.addWidget(btn_home)

        bottom.addStretch()

        btn_again = BigButton(
            f"{Icon.PLAY}  Spar Again", stylesheet=PRIMARY_BTN
        )
        btn_again.setFixedHeight(52)
        btn_again.clicked.connect(
            lambda: self._router.navigate(
                "sparring_select", username=self._username,
            )
        )
        bottom.addWidget(btn_again)
        root.addLayout(bottom)

    def _go_home(self) -> None:
        if self._username:
            self._router.navigate("home", username=self._username)
        else:
            self._router.navigate("home_guest")

    def _populate_bars(self, dist: Dict[str, int]) -> None:
        while self._dist_layout.count():
            item = self._dist_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        max_val = max(dist.values(), default=1)
        for name, count in dist.items():
            color = _PUNCH_COLORS.get(name, Color.TEXT_SECONDARY)
            self._dist_layout.addWidget(
                _DistBar(name, count, max_val, color, self)
            )

    def _request_llm(self) -> None:
        import json
        style = self._config.get("style", "Sparring")
        difficulty = self._config.get("difficulty", "medium")

        if self._bridge is None:
            self._ai_lbl.setText(
                f"Sparring session against {style} ({difficulty}) completed. "
                "Connect the AI Coach for detailed analysis."
            )
            return

        context = {
            "style": style,
            "difficulty": difficulty,
        }
        self._bridge.call_generate_llm(
            prompt=(
                f"Give a brief 2-sentence coaching analysis of this sparring session. "
                f"The user sparred against {style} style at {difficulty} difficulty."
            ),
            context_json=json.dumps(context),
            system_prompt_key="coach_summary",
            callback=self._on_llm,
        )

    def _on_llm(self, success: bool, response: str, _time: float) -> None:
        if success and response.strip():
            self._ai_lbl.setText(response.strip())
        else:
            style = self._config.get("style", "Sparring")
            self._ai_lbl.setText(
                f"Sparring session against {style} completed. "
                "Keep working on your defense and counter-punching!"
            )

    def on_enter(self, **kwargs: Any) -> None:
        self._config = kwargs.get("config", {})
        self._username = kwargs.get("username", "")

        style = self._config.get("style", "Sparring")
        self._style_tag.setText(style)

        self._populate_bars(
            {"Jab": 0, "Cross": 0, "Hook": 0, "Uppercut": 0}
        )
        self._request_llm()

        # Record session in history
        from boxbunny_gui.session_tracker import get_tracker
        style = self._config.get("style", "Sparring")
        get_tracker().add_session(
            mode="Sparring",
            duration=self._config.get("Duration", self._config.get("Work", "--")),
            punches="--",
            score=style,
        )
        logger.debug("SparringResultsPage entered")

    def on_leave(self) -> None:
        pass
