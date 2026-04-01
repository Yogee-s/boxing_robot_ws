"""Simple in-memory session tracker for training history.

Guest sessions are stored in memory only (lost on close).
Logged-in users load/save from the database.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class SessionTracker:
    """Tracks training/sparring/performance sessions."""

    def __init__(self) -> None:
        self._sessions: List[Dict[str, str]] = []

    def add_session(
        self,
        mode: str,
        duration: str = "--",
        punches: str = "0",
        score: str = "--",
    ) -> None:
        """Record a completed session."""
        session = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "time": datetime.now().strftime("%H:%M"),
            "mode": mode,
            "duration": duration,
            "punches": punches,
            "score": score,
        }
        self._sessions.insert(0, session)  # newest first
        logger.info("Session recorded: %s", session)

    @property
    def sessions(self) -> List[Dict[str, str]]:
        return list(self._sessions)

    def clear(self) -> None:
        self._sessions.clear()

    def load_for_user(self, username: str) -> None:
        """Load sessions from database for a logged-in user."""
        try:
            import json
            import sqlite3
            from pathlib import Path

            db_path = (
                Path(__file__).resolve().parents[3]
                / "data" / "users" / username / "boxbunny.db"
            )
            if not db_path.exists():
                logger.debug("No DB for user %s at %s", username, db_path)
                return

            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM training_sessions ORDER BY started_at DESC LIMIT 50"
            ).fetchall()
            conn.close()

            self._sessions = []
            for row in rows:
                # Parse summary_json for punch count
                summary = {}
                try:
                    summary = json.loads(row["summary_json"] or "{}")
                except Exception:
                    pass

                total_punches = summary.get("total_punches", 0)
                duration_min = summary.get(
                    "duration_minutes",
                    (row["work_time_sec"] * row["rounds_completed"]) / 60
                    if row["rounds_completed"] else 0,
                )
                mode_raw = row["mode"] or "training"
                mode_display = mode_raw.replace("_", " ").title()

                self._sessions.append({
                    "date": row["started_at"][:10] if row["started_at"] else "--",
                    "time": (
                        row["started_at"][11:16]
                        if row["started_at"] and len(row["started_at"]) > 11
                        else ""
                    ),
                    "mode": mode_display,
                    "duration": f"{int(duration_min)}m",
                    "punches": str(total_punches),
                    "score": f"{row['rounds_completed']}/{row['rounds_total']} rounds",
                })
            logger.info(
                "Loaded %d sessions for user %s", len(self._sessions), username,
            )
        except Exception as exc:
            logger.warning("Could not load user sessions: %s", exc)
            self._sessions = []


# Global singleton — shared across all pages
_tracker: Optional[SessionTracker] = None


def get_tracker() -> SessionTracker:
    """Get the global session tracker."""
    global _tracker  # noqa: PLW0603
    if _tracker is None:
        _tracker = SessionTracker()
    return _tracker


def reset_tracker() -> None:
    """Reset the tracker (e.g. on logout)."""
    global _tracker  # noqa: PLW0603
    _tracker = SessionTracker()
