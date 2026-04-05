"""Post-training results — matches sparring results design.

Clean stat tiles, mastery progress, AI coach summary, action buttons.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from boxbunny_gui.theme import Color, Icon, Size, font, PRIMARY_BTN

if TYPE_CHECKING:
    from boxbunny_gui.curriculum import ComboCurriculum
    from boxbunny_gui.gui_bridge import GuiBridge
    from boxbunny_gui.nav.router import PageRouter

logger = logging.getLogger(__name__)


def _stat_tile(title: str, value: str, accent: str) -> QWidget:
    """Clean stat tile — no child border artifacts."""
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


def _progress_bar_style(accent: str = Color.PRIMARY) -> str:
    return f"""
        QProgressBar {{
            background-color: {Color.SURFACE_LIGHT};
            border: none; border-radius: 5px;
            height: 10px; text-align: center; font-size: 1px;
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
        self._username: str = ""
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(32, 14, 32, 18)
        root.setSpacing(0)

        root.addStretch(1)

        # ── Title bar ────────────────────────────────────────────────────
        title_row = QHBoxLayout()
        title = QLabel("Session Complete")
        title.setStyleSheet(
            f"font-size: 22px; font-weight: 700; color: {Color.PRIMARY};"
        )
        title_row.addWidget(title)
        title_row.addStretch()
        self._combo_tag = QLabel("")
        self._combo_tag.setStyleSheet(
            f"font-size: 12px; font-weight: 700; color: {Color.TEXT_SECONDARY};"
            f" background-color: {Color.SURFACE};"
            f" border: 1px solid {Color.BORDER};"
            " border-radius: 8px; padding: 4px 12px;"
        )
        title_row.addWidget(self._combo_tag)
        root.addLayout(title_row)

        root.addSpacing(12)

        # ── Stats (2x2) ─────────────────────────────────────────────────
        stats = QGridLayout()
        stats.setSpacing(10)
        self._stat_punches = _stat_tile("Total Punches", "0", Color.PRIMARY)
        self._stat_accuracy = _stat_tile("Accuracy", "0%", Color.INFO)
        self._stat_best = _stat_tile("Best Round", "--", Color.SUCCESS)
        self._stat_fatigue = _stat_tile("Fatigue Index", "--", Color.WARNING)
        stats.addWidget(self._stat_punches, 0, 0)
        stats.addWidget(self._stat_accuracy, 0, 1)
        stats.addWidget(self._stat_best, 1, 0)
        stats.addWidget(self._stat_fatigue, 1, 1)
        root.addLayout(stats)

        root.addSpacing(10)

        # ── Mastery progress ─────────────────────────────────────────────
        self._mastery_card = QWidget()
        self._mastery_card.setObjectName("mastery")
        self._mastery_card.setStyleSheet(f"""
            QWidget#mastery {{
                background-color: {Color.SURFACE};
                border: 1px solid {Color.BORDER};
                border-radius: {Size.RADIUS}px;
            }}
            QWidget#mastery QLabel {{
                background: transparent; border: none;
            }}
            QWidget#mastery QProgressBar {{
                border: none;
            }}
        """)
        mc_lay = QVBoxLayout(self._mastery_card)
        mc_lay.setContentsMargins(16, 12, 16, 12)
        mc_lay.setSpacing(6)

        mc_top = QHBoxLayout()
        mc_header = QLabel("COMBO MASTERY")
        mc_header.setStyleSheet(
            f"font-size: 10px; font-weight: 700; color: {Color.TEXT_DISABLED};"
            " letter-spacing: 0.8px;"
        )
        mc_top.addWidget(mc_header)
        mc_top.addStretch()
        self._mastery_name_lbl = QLabel("")
        self._mastery_name_lbl.setStyleSheet(
            f"font-size: 14px; font-weight: 600; color: {Color.TEXT};"
        )
        mc_top.addWidget(self._mastery_name_lbl)
        mc_top.addStretch()
        self._mastery_pct_lbl = QLabel("")
        self._mastery_pct_lbl.setStyleSheet(
            f"font-size: 14px; font-weight: 700; color: {Color.PRIMARY};"
        )
        mc_top.addWidget(self._mastery_pct_lbl)
        mc_lay.addLayout(mc_top)

        self._mastery_bar = QProgressBar()
        self._mastery_bar.setFixedHeight(8)
        self._mastery_bar.setRange(0, 100)
        self._mastery_bar.setTextVisible(False)
        self._mastery_bar.setStyleSheet(_progress_bar_style())
        mc_lay.addWidget(self._mastery_bar)

        self._mastery_detail_lbl = QLabel("")
        self._mastery_detail_lbl.setStyleSheet(
            f"font-size: 11px; color: {Color.TEXT_SECONDARY};"
        )
        mc_lay.addWidget(self._mastery_detail_lbl)

        self._levelup_lbl = QLabel("")
        self._levelup_lbl.setAlignment(Qt.AlignCenter)
        self._levelup_lbl.setStyleSheet(
            f"font-size: 14px; font-weight: 700; color: {Color.WARNING};"
        )
        self._levelup_lbl.hide()
        mc_lay.addWidget(self._levelup_lbl)

        root.addWidget(self._mastery_card)

        root.addSpacing(10)

        # ── AI Coach ─────────────────────────────────────────────────────
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

        self._coach_lbl = QLabel("AI analysis loading...")
        self._coach_lbl.setStyleSheet(
            f"font-size: 14px; color: {Color.TEXT};"
        )
        self._coach_lbl.setWordWrap(True)
        self._coach_lbl.setMinimumHeight(40)
        ai_lay.addWidget(self._coach_lbl)
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

        self._btn_combos = QPushButton("Combos")
        self._btn_combos.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_combos.setFixedHeight(52)
        self._btn_combos.setStyleSheet(f"""
            QPushButton {{
                font-size: 15px; font-weight: 600;
                background-color: {Color.SURFACE};
                color: {Color.TEXT};
                border: 1px solid {Color.BORDER_LIGHT};
                border-radius: {Size.RADIUS}px;
                padding: 0 24px;
            }}
            QPushButton:hover {{
                border-color: {Color.INFO};
                background-color: {Color.SURFACE_HOVER};
            }}
        """)
        self._btn_combos.clicked.connect(self._on_back_to_combos)
        bottom.addWidget(self._btn_combos)

        self._btn_save = QPushButton("Save Preset")
        self._btn_save.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_save.setFixedHeight(52)
        self._btn_save.setStyleSheet(f"""
            QPushButton {{
                font-size: 14px; font-weight: 600;
                background-color: {Color.SURFACE};
                color: {Color.TEXT_SECONDARY};
                border: 1px solid {Color.BORDER_LIGHT};
                border-radius: {Size.RADIUS}px;
                padding: 0 16px;
            }}
            QPushButton:hover {{
                color: {Color.PRIMARY}; border-color: {Color.PRIMARY};
            }}
        """)
        self._btn_save.clicked.connect(self._save_as_preset)
        bottom.addWidget(self._btn_save)

        bottom.addStretch()

        from boxbunny_gui.widgets import BigButton
        self._btn_again = BigButton(
            f"{Icon.PLAY} Train Again", stylesheet=PRIMARY_BTN
        )
        self._btn_again.setFixedHeight(52)
        self._btn_again.clicked.connect(self._on_train_again)
        bottom.addWidget(self._btn_again)
        root.addLayout(bottom)

    def _go_home(self) -> None:
        if self._username:
            self._router.navigate("home", username=self._username)
        else:
            self._router.navigate("home_guest")

    def _update_mastery(self) -> None:
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

    def _save_as_preset(self) -> None:
        """Save this training session config as a preset."""
        if not self._username:
            return
        import json, sqlite3
        from pathlib import Path
        combo = self._config.get("combo", {})
        cfg_json = json.dumps({
            "rounds": int(self._config.get("Rounds", "2")),
            "work_sec": int(self._config.get("Work Time", "90s").rstrip("s") if isinstance(self._config.get("Work Time", "90s"), str) else 90),
            "rest_sec": int(self._config.get("Rest Time", "30s").rstrip("s") if isinstance(self._config.get("Rest Time", "30s"), str) else 30),
            "speed": self._config.get("Speed", "Medium (2s)"),
            "combo_seq": combo.get("seq", ""),
            "combo_name": combo.get("name", ""),
            "combo_id": combo.get("id"),
            "difficulty": self._difficulty or "beginner",
        })
        name = combo.get("name", "Custom Training")
        try:
            db_path = Path("/home/boxbunny/Desktop/doomsday_integration/boxing_robot_ws/data/boxbunny_main.db")
            conn = sqlite3.connect(str(db_path))
            user_row = conn.execute("SELECT id FROM users WHERE username = ?", (self._username,)).fetchone()
            if user_row:
                conn.execute(
                    "INSERT INTO presets (user_id, name, description, preset_type, config_json) VALUES (?, ?, ?, 'training', ?)",
                    (user_row[0], name, f"{self._difficulty or 'Beginner'} drill", cfg_json),
                )
                conn.commit()
            conn.close()
            self._btn_save.setText(f"{Icon.CHECK} Saved!")
            self._btn_save.setEnabled(False)
        except Exception as exc:
            logger.warning("Failed to save preset: %s", exc)

    def _on_back_to_combos(self) -> None:
        self._router.navigate(
            "training_select",
            level=self._difficulty or "Beginner",
            username=self._username,
        )

    def _on_train_again(self) -> None:
        combo = self._config.get("combo", {})
        is_free = not combo.get("seq", "")
        if is_free:
            # Free training — go straight back to session page
            self._router.navigate(
                "training_session",
                config=self._config,
                combo_id=None,
                difficulty=self._difficulty,
                username=self._username,
            )
        else:
            self._router.navigate(
                "training_config",
                combo=combo,
                difficulty=self._difficulty,
                curriculum=self._curriculum,
                username=self._username,
            )

    def _request_llm_summary(self) -> None:
        import json
        combo_name = self._config.get("combo", {}).get("name", "Free Training")
        punches = self._total_punches
        combos = self._combos_done
        rounds_cfg = self._config.get("Rounds", "1")
        work_time = self._config.get("Work Time", "90s")
        difficulty = self._difficulty or "beginner"

        # Build an accurate description of what happened
        if punches == 0 and combos == 0:
            session_desc = (
                f"The user started a {difficulty} training session "
                f"with combo '{combo_name}' but ended it early without "
                f"throwing any punches or completing any combos."
            )
        elif punches > 0 and combos == 0:
            session_desc = (
                f"The user did a {difficulty} session with combo '{combo_name}'. "
                f"They threw {punches} punches but didn't complete any full combo cycles. "
                f"Config: {rounds_cfg} rounds, {work_time} work time."
            )
        else:
            session_desc = (
                f"The user completed a {difficulty} session with combo '{combo_name}'. "
                f"They threw {punches} punches and completed {combos} full combo cycles. "
                f"Config: {rounds_cfg} rounds, {work_time} work time."
            )

        # If no bridge or LLM unavailable, generate a local summary
        if self._bridge is None:
            self._coach_lbl.setText(self._local_summary(punches, combos, combo_name))
            return

        context = {
            "combo": combo_name,
            "difficulty": difficulty,
            "total_punches": punches,
            "combos_completed": combos,
            "rounds": rounds_cfg,
            "work_time": work_time,
        }
        self._bridge.call_generate_llm(
            prompt=(
                f"Give a brief 2-sentence coaching analysis of this session. "
                f"Be specific about what happened and give one tip. "
                f"{session_desc}"
            ),
            context_json=json.dumps(context),
            system_prompt_key="coach_summary",
            callback=self._on_llm_response,
        )

    @staticmethod
    def _local_summary(punches: int, combos: int, combo_name: str) -> str:
        """Fallback summary when LLM is unavailable."""
        if punches == 0:
            return (
                "Session ended early with no punches thrown. "
                "No worries — when you're ready, try starting with "
                "a simple jab-cross drill to warm up!"
            )
        if combos == 0:
            return (
                f"You threw {punches} punches but didn't complete any full "
                f"combo cycles. Focus on following the sequence cues "
                f"and matching each punch in order."
            )
        return (
            f"Nice work! You threw {punches} punches and completed "
            f"{combos} full cycles of {combo_name}. "
            f"Keep practicing to build muscle memory and speed."
        )

    def _on_llm_response(
        self, success: bool, response: str, gen_time: float,
    ) -> None:
        if success and response.strip():
            self._coach_lbl.setText(response.strip())
        else:
            # LLM failed — use local fallback
            combo_name = self._config.get("combo", {}).get("name", "Free Training")
            self._coach_lbl.setText(
                self._local_summary(self._total_punches, self._combos_done, combo_name)
            )
    def on_enter(self, **kwargs: Any) -> None:
        self._config = kwargs.get("config", {})
        self._curriculum = kwargs.get("curriculum")
        self._combo_id = kwargs.get("combo_id")
        self._difficulty = kwargs.get("difficulty")
        self._username = kwargs.get("username", "")
        self._total_punches = kwargs.get("total_punches", 0)
        self._combos_done = kwargs.get("combos_completed", 0)
        total_punches = self._total_punches
        combos_done = self._combos_done

        # Detect free training mode (no combo sequence)
        combo = self._config.get("combo", {})
        combo_name = combo.get("name", "")
        is_free = not combo.get("seq", "")

        # Populate stat tiles
        self._stat_punches.findChild(QLabel, "val").setText(str(total_punches))

        if is_free:
            # Free training: hide irrelevant tiles, show only punches + speed
            self._stat_accuracy.findChild(QLabel, "val").setText(
                self._config.get("Speed", "Medium")
            )
            self._stat_best.setVisible(False)
            self._stat_fatigue.setVisible(False)
            self._combo_tag.setText("Free Training")
            self._btn_combos.setVisible(False)
            self._btn_save.setVisible(False)
        else:
            self._stat_accuracy.findChild(QLabel, "val").setText(
                f"{combos_done} combos"
            )
            rounds = self._config.get("Rounds", "1")
            self._stat_best.findChild(QLabel, "val").setText(
                f"{rounds} rounds"
            )
            work = self._config.get("Work Time", "--")
            self._stat_fatigue.findChild(QLabel, "val").setText(work)
            self._stat_best.setVisible(True)
            self._stat_fatigue.setVisible(True)
            self._combo_tag.setText(combo_name if combo_name else "Training")
            self._btn_combos.setVisible(True)
            self._btn_save.setVisible(True)

        self._update_mastery()
        self._request_llm_summary()

        # Record session in history
        from boxbunny_gui.session_tracker import get_tracker
        combo_name = combo_name or "Free Training"
        get_tracker().add_session(
            mode="Free Training" if is_free else "Training",
            duration=self._config.get("Work Time", "--"),
            punches=str(total_punches),
            score=f"{combos_done} combos" if not is_free else f"{total_punches} punches",
        )
        logger.info("TrainingResultsPage entered (combo=%s)", self._combo_id)

    def on_leave(self) -> None:
        pass
