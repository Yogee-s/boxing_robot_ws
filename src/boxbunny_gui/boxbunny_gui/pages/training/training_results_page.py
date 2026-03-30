"""Post-training results with mastery progress and level-up detection.

Shows session stats, combo mastery progress (Anki-style), AI coach
summary, and action buttons. Enhanced with accent-colored stat cards.
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
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

from boxbunny_gui.theme import (
    Color, Icon, Size, font, GHOST_BTN, PRIMARY_BTN, SURFACE_BTN,
    section_title_style, accent_frame_style,
)
from boxbunny_gui.widgets import BigButton, StatCard

if TYPE_CHECKING:
    from boxbunny_gui.curriculum import ComboCurriculum
    from boxbunny_gui.gui_bridge import GuiBridge
    from boxbunny_gui.nav.router import PageRouter

logger = logging.getLogger(__name__)


def _progress_bar_style(accent: str = Color.PRIMARY) -> str:
    return f"""
        QProgressBar {{
            background-color: {Color.SURFACE_LIGHT};
            border: none; border-radius: 5px;
            height: 10px; text-align: center; font-size: 0px;
        }}
        QProgressBar::chunk {{
            background-color: {accent};
            border-radius: 5px;
        }}
    """


class TrainingResultsPage(QWidget):
    """Post-session results with mastery tracking and AI coach feedback."""

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
        self._curriculum: Optional[ComboCurriculum] = None
        self._combo_id: Optional[str] = None
        self._difficulty: Optional[str] = None
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 12, 24, 12)
        root.setSpacing(8)

        # Title
        title = QLabel(f"{Icon.CHECK}  Session Complete")
        title.setFont(font(Size.TEXT_SUBHEADER, bold=True))
        title.setStyleSheet(f"color: {Color.PRIMARY};")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(title)

        # Stat cards (2x2) with distinct accent colours
        stats = QGridLayout()
        stats.setSpacing(10)
        stats.setContentsMargins(8, 0, 8, 0)
        self._stat_punches = StatCard("Total Punches", "0", accent=Color.PRIMARY)
        self._stat_accuracy = StatCard("Accuracy", "0%", accent=Color.INFO)
        self._stat_best = StatCard("Best Round", "--", accent=Color.SUCCESS)
        self._stat_fatigue = StatCard("Fatigue Index", "--", accent=Color.WARNING)
        stats.addWidget(self._stat_punches, 0, 0)
        stats.addWidget(self._stat_accuracy, 0, 1)
        stats.addWidget(self._stat_best, 1, 0)
        stats.addWidget(self._stat_fatigue, 1, 1)
        root.addLayout(stats)

        # ── Mastery progress card ────────────────────────────────────
        self._mastery_card = QFrame()
        self._mastery_card.setStyleSheet(accent_frame_style(Color.PRIMARY))
        mc_lay = QVBoxLayout(self._mastery_card)
        mc_lay.setContentsMargins(16, 10, 16, 10)
        mc_lay.setSpacing(6)

        mc_header = QLabel("COMBO MASTERY")
        mc_header.setStyleSheet(
            f"color: {Color.TEXT_DISABLED}; font-size: 10px;"
            " font-weight: 700; letter-spacing: 0.8px; border: none;"
        )
        mc_lay.addWidget(mc_header)

        mc_top = QHBoxLayout()
        self._mastery_name_lbl = QLabel("")
        self._mastery_name_lbl.setStyleSheet(
            f"color: {Color.TEXT}; font-size: 15px; font-weight: 600;"
            " border: none;"
        )
        mc_top.addWidget(self._mastery_name_lbl)
        mc_top.addStretch()
        self._mastery_pct_lbl = QLabel("")
        self._mastery_pct_lbl.setStyleSheet(
            f"color: {Color.PRIMARY}; font-size: 15px; font-weight: 700;"
            " border: none;"
        )
        mc_top.addWidget(self._mastery_pct_lbl)
        mc_lay.addLayout(mc_top)

        self._mastery_bar = QProgressBar()
        self._mastery_bar.setFixedHeight(10)
        self._mastery_bar.setRange(0, 100)
        self._mastery_bar.setTextVisible(False)
        self._mastery_bar.setStyleSheet(_progress_bar_style())
        mc_lay.addWidget(self._mastery_bar)

        self._mastery_detail_lbl = QLabel("")
        self._mastery_detail_lbl.setStyleSheet(
            f"color: {Color.TEXT_SECONDARY}; font-size: 12px; border: none;"
        )
        mc_lay.addWidget(self._mastery_detail_lbl)

        self._levelup_lbl = QLabel("")
        self._levelup_lbl.setAlignment(Qt.AlignCenter)
        self._levelup_lbl.setStyleSheet(
            f"color: {Color.WARNING}; font-size: 14px; font-weight: 700;"
            " border: none;"
        )
        self._levelup_lbl.hide()
        mc_lay.addWidget(self._levelup_lbl)

        root.addWidget(self._mastery_card)

        # ── AI Coach summary ──────────────────────────────────────────
        coach_section = QVBoxLayout()
        coach_section.setSpacing(4)
        coach_header = QLabel("AI COACH")
        coach_header.setStyleSheet(section_title_style(Color.INFO))
        coach_section.addWidget(coach_header)

        self._coach_lbl = QLabel("AI Coach analysis loading...")
        self._coach_lbl.setStyleSheet(
            f"color: {Color.TEXT_SECONDARY}; font-size: 13px;"
            f" background-color: {Color.SURFACE};"
            f" border: 1px solid {Color.BORDER};"
            f" border-left: {Size.ACCENT_BAR_W}px solid {Color.INFO};"
            f" border-radius: {Size.RADIUS}px;"
            " padding: 10px 14px;"
        )
        self._coach_lbl.setWordWrap(True)
        self._coach_lbl.setMinimumHeight(40)
        self._coach_lbl.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        coach_section.addWidget(self._coach_lbl)
        root.addLayout(coach_section)

        root.addStretch()

        # ── Action buttons ────────────────────────────────────────────
        bottom = QHBoxLayout()
        bottom.setSpacing(8)

        self._btn_home = BigButton("Home", stylesheet=GHOST_BTN)
        self._btn_home.setFixedHeight(42)
        self._btn_home.clicked.connect(
            lambda: self._router.navigate("home_guest")
        )
        bottom.addWidget(self._btn_home, stretch=1)

        self._btn_combos = BigButton("Combos", stylesheet=SURFACE_BTN)
        self._btn_combos.setFixedHeight(42)
        self._btn_combos.clicked.connect(self._on_back_to_combos)
        bottom.addWidget(self._btn_combos, stretch=1)

        self._btn_again = BigButton(f"{Icon.PLAY} Train Again", stylesheet=PRIMARY_BTN)
        self._btn_again.setFixedHeight(42)
        self._btn_again.clicked.connect(self._on_train_again)
        bottom.addWidget(self._btn_again, stretch=1)
        root.addLayout(bottom)

    # ── Mastery update ────────────────────────────────────────────────

    def _update_mastery(self) -> None:
        """Refresh mastery card from curriculum data."""
        if not self._curriculum or not self._combo_id:
            self._mastery_card.hide()
            return

        self._mastery_card.show()
        stats = self._curriculum.get_combo_stats(self._combo_id)
        if not stats:
            self._mastery_card.hide()
            return

        threshold = stats["threshold"]
        avg = stats["average_score"]
        pct = min(int((avg / threshold) * 100), 100) if threshold > 0 else 0
        attempts = stats["total_attempts"]

        self._mastery_name_lbl.setText(stats["combo_name"])
        self._mastery_pct_lbl.setText(
            f"{Icon.CHECK} Mastered" if stats["is_mastered"] else f"{pct}%"
        )
        self._mastery_bar.setValue(pct)

        scores_str = ", ".join(f"{s:.1f}" for s in stats["last_5_scores"])
        self._mastery_detail_lbl.setText(
            f"Avg: {avg:.1f}/{threshold:.0f}  |  "
            f"Attempts: {attempts}  |  "
            f"Recent: [{scores_str}]"
        )

        # Check level-up
        self._levelup_lbl.hide()
        if self._difficulty:
            progress = self._curriculum.get_level_progress(self._difficulty)
            if progress["can_level_up"]:
                nxt = self._curriculum.get_next_difficulty(self._difficulty)
                if nxt:
                    self._levelup_lbl.setText(
                        f"Ready to advance to {nxt}!"
                    )
                    self._levelup_lbl.show()

    # ── Actions ───────────────────────────────────────────────────────

    def _on_back_to_combos(self) -> None:
        self._router.navigate(
            "training_select",
            level=self._difficulty or "Beginner",
        )

    def _on_train_again(self) -> None:
        self._router.navigate(
            "training_config",
            combo=self._config.get("combo", {}),
            difficulty=self._difficulty,
            curriculum=self._curriculum,
        )

    def _request_llm_summary(self) -> None:
        if self._bridge is None:
            self._coach_lbl.setText("AI Coach unavailable in offline mode.")
            return
        self._bridge.call_generate_llm(
            prompt="Summarize this boxing training session in 1-2 sentences.",
            context_json="{}",
            system_prompt_key="coach_summary",
            callback=self._on_llm_response,
        )

    def _on_llm_response(
        self, success: bool, response: str, gen_time: float,
    ) -> None:
        if success:
            self._coach_lbl.setText(response)
        else:
            self._coach_lbl.setText("AI Coach analysis unavailable.")

    # ── Lifecycle ─────────────────────────────────────────────────────
    def on_enter(self, **kwargs: Any) -> None:
        self._config = kwargs.get("config", {})
        self._curriculum = kwargs.get("curriculum")
        self._combo_id = kwargs.get("combo_id")
        self._difficulty = kwargs.get("difficulty")

        self._stat_punches.set_value("--")
        self._stat_accuracy.set_value("--%")
        self._stat_best.set_value("--")
        self._stat_fatigue.set_value("--")

        self._update_mastery()
        self._request_llm_summary()
        logger.info("TrainingResultsPage entered (combo=%s)", self._combo_id)

    def on_leave(self) -> None:
        pass
