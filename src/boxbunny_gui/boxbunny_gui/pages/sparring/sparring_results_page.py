"""Post-sparring results — clean offense/defense breakdown with AI summary."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
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
        self._build_ui()

    def _build_ui(self) -> None:
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        wrapper = QWidget()
        wrapper.setStyleSheet("background: transparent;")
        root = QVBoxLayout(wrapper)
        root.setContentsMargins(32, 16, 32, 22)
        root.setSpacing(0)

        # ── Title ────────────────────────────────────────────────────────
        title = QLabel("Sparring Complete")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(
            f"font-size: 24px; font-weight: 700; color: {Color.TEXT};"
            " background-color: #1A1214;"
            " border: 1px solid #3D1A22;"
            f" border-radius: {Size.RADIUS}px;"
            " padding: 12px 24px;"
        )
        root.addWidget(title)

        root.addSpacing(16)

        # ── Offense section ──────────────────────────────────────────────
        off_lbl = QLabel("Offense")
        off_lbl.setStyleSheet(
            f"font-size: 13px; font-weight: 700; color: {Color.TEXT_SECONDARY};"
            " letter-spacing: 0.5px;"
        )
        root.addWidget(off_lbl)
        root.addSpacing(6)

        self._off_total = _stat_tile("Total Punches", "--", Color.PRIMARY)
        root.addWidget(self._off_total)
        root.addSpacing(6)

        # Punch distribution bars
        self._dist_layout = QVBoxLayout()
        self._dist_layout.setSpacing(2)
        root.addLayout(self._dist_layout)

        root.addSpacing(14)

        # ── Defense section ──────────────────────────────────────────────
        def_lbl = QLabel("Defense")
        def_lbl.setStyleSheet(
            f"font-size: 13px; font-weight: 700; color: {Color.TEXT_SECONDARY};"
            " letter-spacing: 0.5px;"
        )
        root.addWidget(def_lbl)
        root.addSpacing(6)

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

        root.addSpacing(14)

        # ── AI Coach Analysis ────────────────────────────────────────────
        ai_box = QWidget()
        ai_box.setObjectName("aibox")
        ai_box.setStyleSheet(f"""
            QWidget#aibox {{
                background-color: #101820;
                border: 1px solid #1A2E40;
                border-left: 3px solid {Color.INFO};
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
            f"font-size: 13px; color: {Color.TEXT};"
        )
        self._ai_lbl.setWordWrap(True)
        self._ai_lbl.setMinimumHeight(32)
        ai_lay.addWidget(self._ai_lbl)
        root.addWidget(ai_box)

        root.addSpacing(14)

        # ── Action buttons ───────────────────────────────────────────────
        bottom = QHBoxLayout()
        bottom.setSpacing(12)

        btn_home = QPushButton("Home")
        btn_home.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_home.setFixedSize(150, 48)
        btn_home.setStyleSheet(f"""
            QPushButton {{
                font-size: 15px; font-weight: 600;
                background-color: {Color.SURFACE};
                color: {Color.TEXT};
                border: 1px solid {Color.BORDER_LIGHT};
                border-radius: {Size.RADIUS}px;
            }}
            QPushButton:hover {{
                border-color: {Color.PRIMARY};
                background-color: {Color.SURFACE_HOVER};
            }}
        """)
        btn_home.clicked.connect(lambda: self._router.navigate("home"))
        bottom.addWidget(btn_home)

        bottom.addStretch()

        btn_again = BigButton(
            f"{Icon.PLAY}  Spar Again", stylesheet=PRIMARY_BTN
        )
        btn_again.setFixedHeight(48)
        btn_again.clicked.connect(
            lambda: self._router.navigate("sparring_select")
        )
        bottom.addWidget(btn_again)
        root.addLayout(bottom)

        scroll.setWidget(wrapper)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

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
        if self._bridge is None:
            self._ai_lbl.setText("AI Coach unavailable in offline mode.")
            return
        self._bridge.call_generate_llm(
            prompt="Summarize this sparring session in 1-2 sentences.",
            context_json="{}",
            system_prompt_key="coach_summary",
            callback=self._on_llm,
        )

    def _on_llm(self, success: bool, response: str, _time: float) -> None:
        self._ai_lbl.setText(
            response if success else "AI Coach analysis unavailable."
        )

    def on_enter(self, **kwargs: Any) -> None:
        self._config = kwargs.get("config", {})
        self._populate_bars(
            {"Jab": 0, "Cross": 0, "Hook": 0, "Uppercut": 0}
        )
        self._request_llm()
        logger.debug("SparringResultsPage entered")

    def on_leave(self) -> None:
        pass
