"""Post-sparring results page.

Offense section with punch distribution bars, defense section with
defense rate / blocks / slips / dodges / hits taken, AI summary,
QR code area, and action buttons.
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

from boxbunny_gui.theme import Color, Size, font, GHOST_BTN, PRIMARY_BTN, SURFACE_BTN
from boxbunny_gui.widgets import BigButton, QRWidget, StatCard

if TYPE_CHECKING:
    from boxbunny_gui.gui_bridge import GuiBridge
    from boxbunny_gui.nav.router import PageRouter

logger = logging.getLogger(__name__)

_PUNCH_COLORS: Dict[str, str] = {
    "Jab": "#2196F3", "Cross": "#F44336",
    "Hook": "#FF9800", "Uppercut": "#9C27B0",
}


class _DistBar(QFrame):
    """Horizontal coloured bar with label and count."""

    def __init__(self, name: str, value: int, max_val: int, color: str,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(28)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(Size.SPACING_SM)
        lbl = QLabel(name)
        lbl.setFixedWidth(70)
        lbl.setStyleSheet(f"color: {Color.TEXT_SECONDARY}; font-size: 13px;")
        lay.addWidget(lbl)
        bar = QFrame()
        width = max(4, int(200 * value / max_val)) if max_val else 4
        bar.setFixedSize(width, 16)
        bar.setStyleSheet(f"background-color: {color}; border-radius: 4px;")
        lay.addWidget(bar)
        cnt = QLabel(str(value))
        cnt.setStyleSheet(f"color: {Color.TEXT}; font-size: 13px; font-weight: bold;")
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
        root.setContentsMargins(Size.SPACING, Size.SPACING_SM, Size.SPACING, Size.SPACING_SM)
        root.setSpacing(Size.SPACING_SM)

        title = QLabel("Sparring Complete")
        title.setFont(font(Size.TEXT_SUBHEADER, bold=True))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(title)

        # Offense section
        off_lbl = QLabel("Offense")
        off_lbl.setFont(font(16, bold=True))
        off_lbl.setStyleSheet(f"color: {Color.PRIMARY};")
        root.addWidget(off_lbl)

        off_grid = QHBoxLayout()
        off_grid.setSpacing(Size.SPACING_SM)
        self._off_punches = StatCard("Total Punches", "--")
        off_grid.addWidget(self._off_punches)
        root.addLayout(off_grid)

        # Punch distribution bars
        self._dist_layout = QVBoxLayout()
        self._dist_layout.setSpacing(2)
        root.addLayout(self._dist_layout)

        # Defense section
        def_lbl = QLabel("Defense")
        def_lbl.setFont(font(16, bold=True))
        def_lbl.setStyleSheet(f"color: {Color.WARNING};")
        root.addWidget(def_lbl)

        def_grid = QGridLayout()
        def_grid.setSpacing(Size.SPACING_SM)
        self._def_rate = StatCard("Defense Rate", "--%")
        self._def_blocks = StatCard("Blocks", "--")
        self._def_slips = StatCard("Slips", "--")
        self._def_dodges = StatCard("Dodges", "--")
        self._def_hits = StatCard("Hits Taken", "--")
        def_grid.addWidget(self._def_rate, 0, 0)
        def_grid.addWidget(self._def_blocks, 0, 1)
        def_grid.addWidget(self._def_slips, 0, 2)
        def_grid.addWidget(self._def_dodges, 1, 0)
        def_grid.addWidget(self._def_hits, 1, 1)
        root.addLayout(def_grid)

        # AI summary
        self._ai_lbl = QLabel("AI analysis loading...")
        self._ai_lbl.setStyleSheet(
            f"color: {Color.TEXT_SECONDARY}; font-size: 14px;"
            f" background-color: {Color.SURFACE}; border-radius: {Size.RADIUS_SM}px;"
            f" padding: {Size.SPACING_SM}px;"
        )
        self._ai_lbl.setWordWrap(True)
        root.addWidget(self._ai_lbl)

        # QR + buttons
        bottom = QHBoxLayout()
        self._qr = QRWidget(size=56)
        self._qr.set_text("https://boxbunny.local/session/latest")
        bottom.addWidget(self._qr)
        bottom.addStretch()
        btn_again = BigButton("Spar Again", stylesheet=PRIMARY_BTN)
        btn_again.setFixedWidth(150)
        btn_again.clicked.connect(lambda: self._router.navigate("sparring_config"))
        bottom.addWidget(btn_again)
        btn_home = BigButton("Home", stylesheet=GHOST_BTN)
        btn_home.setFixedWidth(100)
        btn_home.clicked.connect(lambda: self._router.navigate("home_individual"))
        bottom.addWidget(btn_home)
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
            self._dist_layout.addWidget(_DistBar(name, count, max_val, color, self))

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
        self._ai_lbl.setText(response if success else "AI Coach analysis unavailable.")

    # -- Lifecycle ----------------------------------------------------------
    def on_enter(self, **kwargs: Any) -> None:
        self._config = kwargs.get("config", {})
        self._populate_bars({"Jab": 0, "Cross": 0, "Hook": 0, "Uppercut": 0})
        self._request_llm()
        logger.debug("SparringResultsPage entered")

    def on_leave(self) -> None:
        pass
