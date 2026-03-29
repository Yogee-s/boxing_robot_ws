"""Post-sparring results page.

Offense section with punch distribution bars, defense section with
defense rate / blocks / slips / dodges / hits taken, AI summary,
and action buttons.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from boxbunny_gui.theme import (
    Color, Size, font, GHOST_BTN, PRIMARY_BTN,
    section_title_style, back_link_style,
)
from boxbunny_gui.widgets import BigButton, StatCard

if TYPE_CHECKING:
    from boxbunny_gui.gui_bridge import GuiBridge
    from boxbunny_gui.nav.router import PageRouter

logger = logging.getLogger(__name__)

_PUNCH_COLORS: Dict[str, str] = {
    "Jab": Color.JAB, "Cross": Color.CROSS,
    "Hook": Color.L_HOOK, "Uppercut": Color.PURPLE,
}


class _DistBar(QFrame):
    """Horizontal coloured bar with label and count."""

    def __init__(
        self, name: str, value: int, max_val: int, color: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setFixedHeight(32)
        self.setStyleSheet(
            f"QFrame {{ background-color: {Color.SURFACE};"
            f" border-radius: {Size.RADIUS_SM}px; }}"
        )
        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 4, 12, 4)
        lay.setSpacing(Size.SPACING_SM)

        dot = QLabel("\u25CF")
        dot.setFixedWidth(14)
        dot.setStyleSheet(f"background: transparent; color: {color}; font-size: 12px;")
        lay.addWidget(dot)

        lbl = QLabel(name)
        lbl.setFixedWidth(80)
        lbl.setStyleSheet(
            "background: transparent;"
            f" color: {Color.TEXT}; font-size: 13px; font-weight: 600;"
        )
        lay.addWidget(lbl)

        bar = QFrame()
        width = max(6, int(180 * value / max_val)) if max_val else 6
        bar.setFixedSize(width, 10)
        bar.setStyleSheet(
            f"background-color: {color}; border-radius: 5px;"
        )
        lay.addWidget(bar)

        cnt = QLabel(str(value))
        cnt.setStyleSheet(
            "background: transparent;"
            f" color: {Color.TEXT_SECONDARY}; font-size: 13px; font-weight: 700;"
        )
        lay.addWidget(cnt)
        lay.addStretch()


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
        wrapper = QWidget()
        root = QVBoxLayout(wrapper)
        root.setContentsMargins(
            Size.SPACING_LG, Size.SPACING, Size.SPACING_LG, Size.SPACING
        )
        root.setSpacing(Size.SPACING)

        title = QLabel("Sparring Complete")
        title.setFont(font(Size.TEXT_SUBHEADER, bold=True))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(title)

        root.addSpacing(4)

        # Offense section
        off_lbl = QLabel("\u25CF  Offense")
        off_lbl.setStyleSheet(section_title_style())
        root.addWidget(off_lbl)

        off_grid = QHBoxLayout()
        off_grid.setSpacing(Size.SPACING_SM)
        self._off_punches = StatCard("Total Punches", "--", accent=Color.PRIMARY)
        off_grid.addWidget(self._off_punches)
        root.addLayout(off_grid)

        # Punch distribution bars
        self._dist_layout = QVBoxLayout()
        self._dist_layout.setSpacing(4)
        root.addLayout(self._dist_layout)

        # Defense section
        def_lbl = QLabel("\u25CF  Defense")
        def_lbl.setStyleSheet(
            f"color: {Color.WARNING}; font-size: 15px; font-weight: 600;"
        )
        root.addWidget(def_lbl)

        def_grid = QGridLayout()
        def_grid.setSpacing(Size.SPACING_SM)
        self._def_rate = StatCard("Defense Rate", "--%", accent=Color.WARNING)
        self._def_blocks = StatCard("Blocks", "--", accent=Color.PRIMARY)
        self._def_slips = StatCard("Slips", "--", accent=Color.INFO)
        self._def_dodges = StatCard("Dodges", "--", accent=Color.PURPLE)
        self._def_hits = StatCard("Hits Taken", "--", accent=Color.DANGER)
        def_grid.addWidget(self._def_rate, 0, 0)
        def_grid.addWidget(self._def_blocks, 0, 1)
        def_grid.addWidget(self._def_slips, 0, 2)
        def_grid.addWidget(self._def_dodges, 1, 0)
        def_grid.addWidget(self._def_hits, 1, 1)
        root.addLayout(def_grid)

        # AI summary
        ai_frame = QFrame()
        ai_frame.setStyleSheet(
            f"QFrame {{ background-color: {Color.SURFACE};"
            f" border: 1px solid {Color.BORDER};"
            f" border-left: 4px solid {Color.INFO};"
            f" border-radius: 14px; }}"
        )
        ai_inner = QVBoxLayout(ai_frame)
        ai_inner.setContentsMargins(16, 12, 16, 12)
        ai_inner.setSpacing(4)
        ai_title = QLabel("AI Coach Analysis")
        ai_title.setStyleSheet(
            "background: transparent;"
            f" color: {Color.INFO}; font-size: 12px; font-weight: 700;"
            " letter-spacing: 0.8px;"
        )
        ai_inner.addWidget(ai_title)
        self._ai_lbl = QLabel("AI analysis loading...")
        self._ai_lbl.setStyleSheet(
            "background: transparent;"
            f" color: {Color.TEXT_SECONDARY}; font-size: 14px;"
            " line-height: 1.4;"
        )
        self._ai_lbl.setWordWrap(True)
        self._ai_lbl.setMinimumHeight(36)
        ai_inner.addWidget(self._ai_lbl)
        root.addWidget(ai_frame)

        # Action buttons
        root.addSpacing(4)
        bottom = QHBoxLayout()
        bottom.setSpacing(Size.SPACING)

        btn_home = BigButton("Home", stylesheet=GHOST_BTN)
        btn_home.setFixedHeight(42)
        btn_home.clicked.connect(
            lambda: self._router.navigate("home")
        )
        bottom.addWidget(btn_home, stretch=1)

        bottom.addStretch()

        btn_again = BigButton("Spar Again", stylesheet=PRIMARY_BTN)
        btn_again.setFixedHeight(42)
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
        """Rebuild punch distribution bars."""
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

    # -- Lifecycle ----------------------------------------------------------
    def on_enter(self, **kwargs: Any) -> None:
        self._config = kwargs.get("config", {})
        self._populate_bars(
            {"Jab": 0, "Cross": 0, "Hook": 0, "Uppercut": 0}
        )
        self._request_llm()
        logger.debug("SparringResultsPage entered")

    def on_leave(self) -> None:
        pass
