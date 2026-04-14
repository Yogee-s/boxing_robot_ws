"""Coach home page with station management.

Idle state: Welcome banner, Start Station, preset cards, recent class history.
Active state: current config, participant count, session timer, controls.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QScroller,
    QVBoxLayout,
    QWidget,
)

from boxbunny_gui.theme import Color, Icon, Size, font
from boxbunny_gui.widgets import BigButton, TimerDisplay

if TYPE_CHECKING:
    from boxbunny_gui.nav.router import PageRouter

logger = logging.getLogger(__name__)

_DB_PATH = Path(__file__).resolve().parents[5] / "data" / "boxbunny_main.db"


# ── Helper widgets ───────────────────────────────────────────────────────

def _preset_btn(name: str, ptype: str, is_fav: bool) -> QPushButton:
    """Styled preset button for the coach grid."""
    fav_star = " \u2605" if is_fav else ""
    label = f"{name}{fav_star}".replace("&", "&&")  # escape Qt shortcut prefix
    btn = QPushButton(label)
    btn.setCursor(Qt.CursorShape.PointingHandCursor)
    btn.setFixedHeight(56)

    type_colors = {
        "circuit": "#00BCD4", "training": "#2196F3",
        "sparring": "#FF5722", "free": "#9C27B0",
    }
    accent = type_colors.get(ptype, Color.PRIMARY)

    btn.setStyleSheet(f"""
        QPushButton {{
            font-size: 14px; font-weight: 600;
            background-color: {Color.SURFACE};
            color: {Color.TEXT};
            border: 1px solid {Color.BORDER};
            border-left: 3px solid {accent};
            border-radius: {Size.RADIUS}px;
            padding: 0 16px;
            text-align: left;
        }}
        QPushButton:hover {{
            background-color: {Color.SURFACE_LIGHT};
            border-color: {accent};
        }}
    """)
    return btn


class _ClassCard(QWidget):
    """Card showing a past coaching session summary."""

    def __init__(
        self, name: str, date: str, total: int,
        avg_punches: int, avg_reaction: int,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {Color.SURFACE};
                border: 1px solid {Color.BORDER};
                border-radius: {Size.RADIUS}px;
            }}
            QLabel {{ background: transparent; border: none; }}
        """)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(6)

        # Top: name + date
        top = QHBoxLayout()
        top.setSpacing(8)
        title = QLabel(name)
        title.setStyleSheet(
            f"font-size: 14px; font-weight: 700; color: {Color.TEXT};"
        )
        top.addWidget(title)
        top.addStretch()
        dt = QLabel(date)
        dt.setStyleSheet(
            f"font-size: 11px; color: {Color.TEXT_DISABLED};"
        )
        top.addWidget(dt)
        lay.addLayout(top)

        # Stats row
        stats = QHBoxLayout()
        stats.setSpacing(16)
        for label, value, color in [
            ("Students", str(total), Color.PRIMARY),
            ("Avg Punches", str(avg_punches), Color.INFO),
            ("Avg Reaction", f"{avg_reaction}ms", Color.WARNING),
        ]:
            col = QVBoxLayout()
            col.setSpacing(0)
            v = QLabel(value)
            v.setStyleSheet(
                f"font-size: 16px; font-weight: 700; color: {color};"
            )
            col.addWidget(v)
            l = QLabel(label)
            l.setStyleSheet(
                f"font-size: 10px; color: {Color.TEXT_DISABLED};"
                " letter-spacing: 0.5px;"
            )
            col.addWidget(l)
            stats.addLayout(col)
        stats.addStretch()
        lay.addLayout(stats)


class HomeCoachPage(QWidget):
    """Coach dashboard with session management."""

    def __init__(self, router: PageRouter, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._router = router
        self._session_active: bool = False
        self._participant_num: int = 0
        self._username: str = ""
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(32, 18, 32, 18)
        root.setSpacing(0)

        # ── Top bar ──────────────────────────────────────────────────────
        top = QHBoxLayout()
        top.setSpacing(10)

        self._name_lbl = QLabel("Welcome, Coach!")
        self._name_lbl.setStyleSheet(
            f"font-size: 28px; font-weight: 700; color: {Color.PRIMARY};"
        )
        top.addWidget(self._name_lbl)
        top.addStretch()

        self._status_badge = QLabel("")
        self._status_badge.setStyleSheet(f"""
            font-size: 12px; font-weight: 700;
            color: {Color.TEXT_SECONDARY};
            background-color: {Color.SURFACE};
            border: 1px solid {Color.BORDER};
            border-radius: 8px; padding: 4px 14px;
        """)
        self._status_badge.setVisible(False)
        top.addWidget(self._status_badge)

        presets_btn = QPushButton("Edit Presets")
        presets_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        presets_btn.setFixedSize(120, 40)
        presets_btn.setStyleSheet(f"""
            QPushButton {{
                font-size: 13px; font-weight: 600;
                background-color: {Color.SURFACE}; color: {Color.TEXT_SECONDARY};
                border: 1px solid {Color.BORDER_LIGHT}; border-radius: 8px;
            }}
            QPushButton:hover {{
                color: {Color.PRIMARY}; border-color: {Color.PRIMARY};
                background-color: {Color.SURFACE_LIGHT};
            }}
        """)
        presets_btn.clicked.connect(
            lambda: self._router.navigate("presets", username=self._username)
        )
        top.addWidget(presets_btn)

        logout_btn = QPushButton("Log Out")
        logout_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        logout_btn.setFixedSize(100, 40)
        logout_btn.setStyleSheet(f"""
            QPushButton {{
                font-size: 13px; font-weight: 600;
                background-color: {Color.SURFACE}; color: {Color.TEXT_SECONDARY};
                border: 1px solid {Color.BORDER_LIGHT}; border-radius: 8px;
            }}
            QPushButton:hover {{
                color: {Color.DANGER}; border-color: {Color.DANGER};
                background-color: {Color.SURFACE_LIGHT};
            }}
        """)
        logout_btn.clicked.connect(lambda: self._router.navigate("auth"))
        top.addWidget(logout_btn)
        root.addLayout(top)

        # ════════════════════════════════════════════════════════════════
        # SCROLLABLE CONTENT
        # ════════════════════════════════════════════════════════════════
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(
            "QScrollArea { border: none; background: transparent; }"
            "QScrollArea > QWidget > QWidget { background: transparent; }"
        )
        scroll_content = QWidget()
        self._content_lay = QVBoxLayout(scroll_content)
        self._content_lay.setContentsMargins(0, 8, 0, 0)
        self._content_lay.setSpacing(0)
        scroll.setWidget(scroll_content)
        QScroller.grabGesture(
            scroll.viewport(),
            QScroller.ScrollerGestureType.LeftMouseButtonGesture,
        )
        root.addWidget(scroll, stretch=1)

        # ── IDLE STATE ───────────────────────────────────────────────────
        self._idle_widget = QWidget()
        idle_lay = QVBoxLayout(self._idle_widget)
        idle_lay.setContentsMargins(0, 0, 0, 0)
        idle_lay.setSpacing(0)

        # Student count + Start Station row
        idle_lay.addSpacing(8)
        start_row = QHBoxLayout()
        start_row.setSpacing(12)

        # Student counter
        students_lbl = QLabel("Students:")
        students_lbl.setStyleSheet(
            f"font-size: 14px; font-weight: 600; color: {Color.TEXT_SECONDARY};"
        )
        start_row.addWidget(students_lbl)

        self._student_count = 6
        self._student_lbl = QLabel(str(self._student_count))
        self._student_lbl.setFixedWidth(40)
        self._student_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._student_lbl.setStyleSheet(
            f"font-size: 22px; font-weight: 700; color: {Color.TEXT};"
        )

        btn_minus = QPushButton("-")
        btn_minus.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_minus.setFixedSize(40, 40)
        btn_minus.setStyleSheet(f"""
            QPushButton {{
                font-size: 22px; font-weight: 700;
                background-color: {Color.SURFACE_LIGHT}; color: {Color.TEXT};
                border: 1px solid {Color.BORDER}; border-radius: 8px;
            }}
            QPushButton:hover {{ background-color: {Color.PRIMARY}; color: #FFF; }}
        """)
        btn_minus.clicked.connect(lambda: self._adj_students(-1))

        btn_plus = QPushButton("+")
        btn_plus.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_plus.setFixedSize(40, 40)
        btn_plus.setStyleSheet(f"""
            QPushButton {{
                font-size: 22px; font-weight: 700;
                background-color: {Color.SURFACE_LIGHT}; color: {Color.TEXT};
                border: 1px solid {Color.BORDER}; border-radius: 8px;
            }}
            QPushButton:hover {{ background-color: {Color.PRIMARY}; color: #FFF; }}
        """)
        btn_plus.clicked.connect(lambda: self._adj_students(1))

        start_row.addWidget(btn_minus)
        start_row.addWidget(self._student_lbl)
        start_row.addWidget(btn_plus)

        start_row.addSpacing(8)

        # Start Station button
        self._btn_start_station = QPushButton("Start Station")
        self._btn_start_station.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_start_station.setFixedHeight(50)
        self._btn_start_station.setStyleSheet(f"""
            QPushButton {{
                font-size: 18px; font-weight: 700;
                background-color: {Color.PRIMARY}; color: #FFFFFF;
                border: none; border-radius: {Size.RADIUS}px;
            }}
            QPushButton:hover {{
                background-color: {Color.PRIMARY_DARK};
            }}
        """)
        self._btn_start_station.clicked.connect(self._start_session)
        start_row.addWidget(self._btn_start_station, stretch=1)

        idle_lay.addLayout(start_row)

        # ── Presets section ──────────────────────────────────────────────
        idle_lay.addSpacing(14)
        self._add_section_header(idle_lay, "STATION PRESETS")
        idle_lay.addSpacing(6)

        self._preset_grid = QGridLayout()
        self._preset_grid.setSpacing(8)
        self._preset_grid.setColumnStretch(0, 1)
        self._preset_grid.setColumnStretch(1, 1)
        idle_lay.addLayout(self._preset_grid)

        # ── Recent classes section ───────────────────────────────────────
        idle_lay.addSpacing(18)
        self._add_section_header(idle_lay, "RECENT CLASSES")
        idle_lay.addSpacing(6)

        self._history_layout = QVBoxLayout()
        self._history_layout.setSpacing(8)
        idle_lay.addLayout(self._history_layout)

        idle_lay.addStretch()
        self._content_lay.addWidget(self._idle_widget)

        # ── ACTIVE STATE (setup → ready → running sub-states) ─────────
        self._active_widget = QWidget()
        active_lay = QVBoxLayout(self._active_widget)
        active_lay.setContentsMargins(0, 8, 0, 0)
        active_lay.setSpacing(12)

        # Config + participant header (always visible in active)
        info_row = QHBoxLayout()
        self._config_lbl = QLabel("Free Station")
        self._config_lbl.setStyleSheet(
            f"font-size: 20px; font-weight: 700; color: {Color.TEXT};"
        )
        self._config_lbl.setWordWrap(True)
        info_row.addWidget(self._config_lbl)
        info_row.addStretch()
        self._participant_lbl = QLabel("")
        self._participant_lbl.setStyleSheet(
            f"font-size: 18px; font-weight: 700; color: {Color.PRIMARY};"
        )
        info_row.addWidget(self._participant_lbl)
        active_lay.addLayout(info_row)

        # ── Ready sub-state: big GO button ───────────────────────────
        self._ready_widget = QWidget()
        ready_lay = QVBoxLayout(self._ready_widget)
        ready_lay.setContentsMargins(0, 0, 0, 0)
        ready_lay.setAlignment(Qt.AlignmentFlag.AlignCenter)

        ready_lay.addStretch()
        self._btn_go = QPushButton("GO")
        self._btn_go.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_go.setFixedSize(220, 220)
        self._btn_go.setStyleSheet(f"""
            QPushButton {{
                font-size: 48px; font-weight: 800;
                background-color: {Color.PRIMARY}; color: #FFFFFF;
                border: none; border-radius: 110px;
            }}
            QPushButton:hover {{
                background-color: {Color.PRIMARY_DARK};
            }}
        """)
        self._btn_go.clicked.connect(self._begin_round)
        ready_lay.addWidget(self._btn_go, alignment=Qt.AlignmentFlag.AlignCenter)

        ready_lay.addSpacing(8)
        self._ready_hint = QLabel("Hit the centre pad or tap GO")
        self._ready_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._ready_hint.setStyleSheet(
            f"font-size: 14px; color: {Color.TEXT_DISABLED};"
        )
        ready_lay.addWidget(self._ready_hint)
        ready_lay.addStretch()

        # End session from ready screen
        ready_end = QPushButton("End Session")
        ready_end.setCursor(Qt.CursorShape.PointingHandCursor)
        ready_end.setFixedHeight(40)
        ready_end.setStyleSheet(f"""
            QPushButton {{
                font-size: 13px; font-weight: 600;
                background: transparent; color: {Color.DANGER};
                border: none;
            }}
            QPushButton:hover {{ color: {Color.TEXT}; }}
        """)
        ready_end.clicked.connect(self._end_session)
        ready_lay.addWidget(ready_end, alignment=Qt.AlignmentFlag.AlignCenter)

        self._ready_widget.setVisible(False)
        active_lay.addWidget(self._ready_widget)

        # ── Running sub-state: timer + controls ──────────────────────
        self._running_widget = QWidget()
        run_lay = QVBoxLayout(self._running_widget)
        run_lay.setContentsMargins(0, 0, 0, 0)
        run_lay.setSpacing(12)

        self._timer = TimerDisplay(font_size=Size.TEXT_TIMER, show_ring=True)
        self._timer.finished.connect(self._on_timer_done)
        run_lay.addWidget(self._timer, stretch=1)

        self._punch_count = 0
        self._punch_lbl = QLabel("0 punches")
        self._punch_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._punch_lbl.setStyleSheet(
            f"font-size: 16px; font-weight: 600; color: {Color.TEXT_SECONDARY};"
        )
        run_lay.addWidget(self._punch_lbl)

        run_btn_row = QHBoxLayout()
        run_btn_row.setSpacing(12)

        self._btn_pause = QPushButton("Pause")
        self._btn_pause.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_pause.setFixedHeight(52)
        self._btn_pause.setStyleSheet(f"""
            QPushButton {{
                font-size: 15px; font-weight: 600;
                background-color: {Color.SURFACE}; color: {Color.TEXT};
                border: 1px solid {Color.BORDER_LIGHT};
                border-radius: {Size.RADIUS}px; padding: 0 24px;
            }}
            QPushButton:hover {{
                border-color: {Color.PRIMARY};
                background-color: {Color.SURFACE_HOVER};
            }}
        """)
        self._btn_pause.clicked.connect(self._toggle_pause)
        run_btn_row.addWidget(self._btn_pause)

        run_btn_row.addStretch()

        self._btn_end_run = QPushButton("End Session")
        self._btn_end_run.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_end_run.setFixedHeight(52)
        self._btn_end_run.setStyleSheet(f"""
            QPushButton {{
                font-size: 15px; font-weight: 600;
                background-color: {Color.SURFACE}; color: {Color.DANGER};
                border: 1px solid {Color.DANGER};
                border-radius: {Size.RADIUS}px; padding: 0 24px;
            }}
            QPushButton:hover {{
                background-color: {Color.DANGER}; color: #FFFFFF;
            }}
        """)
        self._btn_end_run.clicked.connect(self._end_session)
        run_btn_row.addWidget(self._btn_end_run)
        run_lay.addLayout(run_btn_row)

        self._running_widget.setVisible(False)
        active_lay.addWidget(self._running_widget)

        self._active_widget.setVisible(False)
        self._content_lay.addWidget(self._active_widget)

    @staticmethod
    def _add_section_header(layout: QVBoxLayout, text: str) -> None:
        row = QHBoxLayout()
        lbl = QLabel(text)
        lbl.setStyleSheet(
            f"font-size: 11px; font-weight: 700; color: {Color.TEXT_DISABLED};"
            " letter-spacing: 2px;"
        )
        row.addWidget(lbl)
        div = QWidget()
        div.setFixedHeight(1)
        div.setStyleSheet(f"background-color: {Color.BORDER};")
        row.addWidget(div, stretch=1)
        layout.addLayout(row)

    # ── Data loading ─────────────────────────────────────────────────────

    def _load_presets(self) -> List[Dict[str, Any]]:
        if not self._username or not _DB_PATH.exists():
            return []
        try:
            conn = sqlite3.connect(str(_DB_PATH))
            conn.row_factory = sqlite3.Row
            uid_row = conn.execute(
                "SELECT id FROM users WHERE username = ?", (self._username,)
            ).fetchone()
            if not uid_row:
                conn.close()
                return []
            rows = conn.execute(
                "SELECT * FROM presets WHERE user_id = ? "
                "ORDER BY is_favorite DESC, use_count DESC",
                (uid_row["id"],),
            ).fetchall()
            conn.close()
            return [dict(r) for r in rows]
        except Exception as exc:
            logger.warning("Failed to load coach presets: %s", exc)
            return []

    def _load_coaching_history(self) -> List[Dict[str, Any]]:
        if not self._username or not _DB_PATH.exists():
            return []
        try:
            conn = sqlite3.connect(str(_DB_PATH))
            conn.row_factory = sqlite3.Row
            uid_row = conn.execute(
                "SELECT id FROM users WHERE username = ?", (self._username,)
            ).fetchone()
            if not uid_row:
                conn.close()
                return []
            sessions = conn.execute(
                "SELECT * FROM coaching_sessions WHERE coach_user_id = ? "
                "ORDER BY started_at DESC LIMIT 5",
                (uid_row["id"],),
            ).fetchall()
            result = []
            for s in sessions:
                parts = conn.execute(
                    "SELECT session_data_json FROM coaching_participants "
                    "WHERE coaching_session_id = ?",
                    (s["id"],),
                ).fetchall()
                punch_counts = []
                reaction_times = []
                for p in parts:
                    data = json.loads(p["session_data_json"])
                    punch_counts.append(data.get("punch_count", 0))
                    rt = data.get("reaction_time_ms", 0)
                    if rt > 0:
                        reaction_times.append(rt)
                avg_p = int(sum(punch_counts) / max(len(punch_counts), 1))
                avg_r = int(sum(reaction_times) / max(len(reaction_times), 1))
                date_str = s["started_at"][:10] if s["started_at"] else ""
                result.append({
                    "name": s["notes"] or "Coaching Session",
                    "date": date_str,
                    "total": s["total_participants"],
                    "avg_punches": avg_p,
                    "avg_reaction": avg_r,
                })
            conn.close()
            return result
        except Exception as exc:
            logger.warning("Failed to load coaching history: %s", exc)
            return []

    # ── Populate UI ──────────────────────────────────────────────────────

    def _populate(self) -> None:
        # Clear preset grid
        while self._preset_grid.count():
            item = self._preset_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        presets = self._load_presets()
        for i, preset in enumerate(presets):
            btn = _preset_btn(
                preset["name"],
                preset.get("preset_type", "training"),
                bool(preset.get("is_favorite", False)),
            )
            btn.clicked.connect(
                lambda _c=False, p=preset: self._select_preset(p)
            )
            self._preset_grid.addWidget(btn, i // 2, i % 2)

        if not presets:
            empty = QLabel("No presets yet")
            empty.setStyleSheet(
                f"color: {Color.TEXT_DISABLED}; font-size: 13px;"
            )
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._preset_grid.addWidget(empty, 0, 0, 1, 2)

        # Clear history
        while self._history_layout.count():
            item = self._history_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        history = self._load_coaching_history()
        for entry in history:
            card = _ClassCard(
                name=entry["name"],
                date=entry["date"],
                total=entry["total"],
                avg_punches=entry["avg_punches"],
                avg_reaction=entry["avg_reaction"],
                parent=self,
            )
            self._history_layout.addWidget(card)

        if not history:
            empty = QLabel("No coaching sessions yet")
            empty.setStyleSheet(
                f"color: {Color.TEXT_DISABLED}; font-size: 13px;"
            )
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._history_layout.addWidget(empty)

    def _select_preset(self, preset: Dict[str, Any]) -> None:
        try:
            config = json.loads(preset.get("config_json", "{}"))
        except (json.JSONDecodeError, TypeError):
            config = {}
        self._config_lbl.setText(preset.get("name", "Station"))
        self._work_time = int(config.get("work_sec", 180))
        self._session_active = True
        self._total_students = self._student_count
        self._participant_num = 1
        self._punch_count = 0
        self._participant_lbl.setText(f"1 of {self._total_students}")
        self._update_state()
        self._show_sub("ready")
        logger.info("Coach preset loaded: %s (%d students)", preset["name"], self._total_students)

    # ── Session control ──────────────────────────────────────────────────

    def _adj_students(self, delta: int) -> None:
        self._student_count = max(1, min(30, self._student_count + delta))
        self._student_lbl.setText(str(self._student_count))

    def _start_session(self) -> None:
        """Go straight to ready screen with student count."""
        self._session_active = True
        self._total_students = self._student_count
        self._participant_num = 1
        self._work_time = 180
        self._punch_count = 0
        self._config_lbl.setText("Free Station")
        self._participant_lbl.setText(f"1 of {self._total_students}")
        self._update_state()
        self._show_sub("ready")
        logger.info("Coach station started: %d students", self._total_students)

    def _begin_round(self) -> None:
        """Start the timer (GO pressed or pad hit)."""
        self._punch_count = 0
        self._punch_lbl.setText("0 punches")
        self._btn_pause.setText("Pause")
        self._btn_pause.setEnabled(True)
        self._show_sub("running")
        self._timer.start(self._work_time)
        logger.info("Round started for participant #%d", self._participant_num)

    def _on_timer_done(self) -> None:
        """Timer finished — auto-advance after 3s."""
        self._btn_pause.setText("Done!")
        self._btn_pause.setEnabled(False)
        logger.info(
            "Participant #%d finished — %d punches",
            self._participant_num, self._punch_count,
        )
        QTimer.singleShot(3000, self._auto_next)

    def _auto_next(self) -> None:
        """Auto-advance to next participant or end session."""
        if not self._session_active:
            return
        if self._participant_num >= self._total_students:
            self._end_session()
            return
        self._participant_num += 1
        self._participant_lbl.setText(
            f"Participant {self._participant_num} of {self._total_students}"
        )
        self._timer.reset()
        self._show_sub("ready")
        logger.info("Ready for participant #%d", self._participant_num)

    def _show_sub(self, sub: str) -> None:
        """Show one sub-state widget, hide others."""
        self._ready_widget.setVisible(sub == "ready")
        self._running_widget.setVisible(sub == "running")

    def _end_session(self) -> None:
        self._session_active = False
        self._timer.reset()
        self._btn_pause.setEnabled(True)
        self._update_state()
        logger.info(
            "Coach station ended after %d participants",
            self._participant_num,
        )

    def on_pad_hit(self, pad: str) -> None:
        """Called by bridge when centre pad is hit — starts round if ready."""
        if (self._session_active
                and self._ready_widget.isVisible()
                and pad == "centre"):
            self._begin_round()

    def _toggle_pause(self) -> None:
        if self._timer._running:
            self._timer.pause()
            self._btn_pause.setText("Resume")
        else:
            self._timer.resume()
            self._btn_pause.setText("Pause")

    def _update_state(self) -> None:
        self._idle_widget.setVisible(not self._session_active)
        self._active_widget.setVisible(self._session_active)
        self._status_badge.setVisible(self._session_active)
        if self._session_active:
            self._status_badge.setText("In Session")
            self._status_badge.setStyleSheet(f"""
                font-size: 12px; font-weight: 700;
                color: #FFFFFF;
                background-color: {Color.PRIMARY};
                border: 1px solid {Color.PRIMARY};
                border-radius: 8px; padding: 4px 14px;
            """)
        self._btn_pause.setText("Pause")

    # ── Lifecycle ──────────────────────────────────────────────────────
    def on_enter(self, **kwargs: Any) -> None:
        self._username = kwargs.get("username", "")
        # Look up display name from DB
        display = self._username or "Coach"
        if self._username and _DB_PATH.exists():
            try:
                conn = sqlite3.connect(str(_DB_PATH))
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    "SELECT display_name FROM users WHERE username = ?",
                    (self._username,),
                ).fetchone()
                conn.close()
                if row and row["display_name"]:
                    display = row["display_name"]
            except Exception:
                pass
        self._name_lbl.setText(f"Welcome, {display}!")
        self._populate()
        self._update_state()
        logger.debug("HomeCoachPage entered (user=%s)", self._username)

    def on_leave(self) -> None:
        pass
