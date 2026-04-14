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
    "L Hook": Color.L_HOOK, "R Hook": Color.R_HOOK,
    "L Upper": Color.L_UPPERCUT, "R Upper": Color.R_UPPERCUT,
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

        # ── Robot stats ──────────────────────────────────────────────────
        robot_row = QHBoxLayout()
        robot_row.setSpacing(8)
        self._robot_attacks = _stat_tile("Robot Attacks", "--", Color.DANGER)
        self._robot_blocks = _stat_tile("Blocks Detected", "--", Color.INFO)
        self._defense_rate = _stat_tile("Defense Rate", "--", Color.SUCCESS)
        robot_row.addWidget(self._robot_attacks)
        robot_row.addWidget(self._robot_blocks)
        robot_row.addWidget(self._defense_rate)
        root.addLayout(robot_row)

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
        total = self._total_punches
        attacks = self._robot_attacks
        blocks = self._blocks_detected
        defense = self._defense_rate

        if total == 0 and attacks == 0:
            self._ai_lbl.setText(
                "Session ended early — no punches were thrown and no robot "
                "attacks occurred. No data to analyse."
            )
            return

        # Build an accurate description of what happened
        dist_str = ", ".join(
            f"{k}: {v}" for k, v in self._punch_dist.items() if v > 0
        ) or "none recorded"

        if total == 0:
            session_desc = (
                f"The user started a sparring session against {style} style "
                f"at {difficulty} difficulty but threw no punches. "
                f"The robot attacked {attacks} times and the user blocked "
                f"{blocks} ({defense:.0%} defense rate)."
            )
        else:
            session_desc = (
                f"The user sparred against {style} style at {difficulty} difficulty. "
                f"They threw {total} punches ({dist_str}). "
                f"The robot attacked {attacks} times, the user blocked {blocks} "
                f"({defense:.0%} defense rate)."
            )

        if self._bridge is None:
            self._ai_lbl.setText(self._local_summary(
                total, attacks, blocks, defense, style,
            ))
            return

        context = {
            "style": style,
            "difficulty": difficulty,
            "total_punches": total,
            "punch_distribution": {k: v for k, v in self._punch_dist.items() if v > 0},
            "robot_attacks": attacks,
            "blocks_detected": blocks,
            "defense_rate": round(defense, 3),
        }
        self._ai_lbl.setText("AI analysis loading...")
        self._bridge.call_generate_llm(
            prompt=(
                f"Give a brief 2-sentence coaching analysis of this sparring session. "
                f"Be specific about the actual numbers and give one actionable tip. "
                f"{session_desc}"
            ),
            context_json=json.dumps(context),
            system_prompt_key="coach_summary",
            callback=self._on_llm,
        )

    @staticmethod
    def _local_summary(
        total: int, attacks: int, blocks: int, defense: float, style: str,
    ) -> str:
        """Fallback summary when LLM is unavailable."""
        if total == 0:
            return (
                f"Sparring session against {style} ended with no punches thrown. "
                "Try to be more aggressive next round!"
            )
        parts = [f"You threw {total} punches against {style} style"]
        if attacks > 0:
            parts.append(f" and blocked {blocks}/{attacks} robot attacks ({defense:.0%})")
        parts.append(".")
        if defense < 0.4 and attacks > 0:
            parts.append(" Focus on keeping your guard up between combos.")
        elif defense > 0.7 and attacks > 0:
            parts.append(" Great defensive awareness — keep it up!")
        else:
            parts.append(" Keep working on mixing offense and defense.")
        return "".join(parts)

    def _on_llm(self, success: bool, response: str, _time: float) -> None:
        if success and response.strip():
            self._ai_lbl.setText(response.strip())
        else:
            self._ai_lbl.setText(self._local_summary(
                self._total_punches, self._robot_attacks,
                self._blocks_detected, self._defense_rate,
                self._config.get("style", "Sparring"),
            ))

    def on_enter(self, **kwargs: Any) -> None:
        self._config = kwargs.get("config", {})
        self._username = kwargs.get("username", "")
        total_punches = kwargs.get("total_punches", 0)
        robot_attacks = kwargs.get("robot_attacks", 0)
        blocks_detected = kwargs.get("blocks_detected", 0)
        punch_dist = kwargs.get("punch_dist", {})
        defense_rate = kwargs.get("defense_rate", 0.0)

        # Store for LLM access
        self._total_punches = total_punches
        self._robot_attacks = robot_attacks
        self._blocks_detected = blocks_detected
        self._punch_dist = punch_dist
        self._defense_rate = defense_rate

        style = self._config.get("style", "Sparring")
        self._style_tag.setText(style)

        # Total punches (fusion-filtered IMU+CV confirmed hits)
        self._off_total.findChild(QLabel, "val").setText(str(total_punches))

        # Punch distribution with separate L/R (fusion-filtered)
        dist = {"Jab": 0, "Cross": 0, "L Hook": 0, "R Hook": 0,
                "L Upper": 0, "R Upper": 0}
        dist.update(punch_dist)
        self._populate_bars(dist)

        # Robot stats
        self._robot_attacks.findChild(QLabel, "val").setText(str(robot_attacks))
        self._robot_blocks.findChild(QLabel, "val").setText(str(blocks_detected))
        self._defense_rate.findChild(QLabel, "val").setText(
            f"{int(defense_rate * 100)}%"
        )

        self._request_llm()

        # Record session in history
        from boxbunny_gui.session_tracker import get_tracker
        get_tracker().add_session(
            mode="Sparring",
            duration=self._config.get("Duration", self._config.get("Work", "--")),
            punches=str(total_punches),
            score=style,
        )
        logger.debug("SparringResultsPage entered")

    def on_leave(self) -> None:
        pass
