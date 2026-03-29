"""Centralized database access for BoxBunny.

Manages the shared main database and per-user databases.
Uses bcrypt for password and pattern hashing.
"""

import json
import logging
import os
import secrets
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import bcrypt

logger = logging.getLogger("boxbunny.db")


class DatabaseManager:
    """Unified database manager for BoxBunny."""

    def __init__(self, data_dir: str) -> None:
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._main_db_path = self._data_dir / "boxbunny_main.db"
        self._users_dir = self._data_dir / "users"
        self._users_dir.mkdir(exist_ok=True)
        self._schema_dir = self._data_dir / "schema"
        self._init_main_db()

    def _get_main_conn(self) -> sqlite3.Connection:
        """Get connection to the main shared database."""
        conn = sqlite3.connect(str(self._main_db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _get_user_conn(self, username: str) -> sqlite3.Connection:
        """Get connection to a user's personal database."""
        user_dir = self._users_dir / username
        user_dir.mkdir(exist_ok=True)
        db_path = user_dir / "boxbunny.db"
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_main_db(self) -> None:
        """Initialize main database with schema."""
        schema_path = self._schema_dir / "main_schema.sql"
        if not schema_path.exists():
            logger.warning("Main schema file not found at %s", schema_path)
            return
        with self._get_main_conn() as conn:
            conn.executescript(schema_path.read_text())
        logger.info("Main database initialized at %s", self._main_db_path)

    def _init_user_db(self, username: str) -> None:
        """Initialize a user's personal database with schema."""
        schema_path = self._schema_dir / "user_schema.sql"
        if not schema_path.exists():
            logger.warning("User schema file not found at %s", schema_path)
            return
        with self._get_user_conn(username) as conn:
            conn.executescript(schema_path.read_text())
        logger.info("User database initialized for %s", username)

    # --- User Management ---

    def create_user(
        self,
        username: str,
        password: str,
        display_name: str,
        user_type: str = "individual",
        level: str = "beginner",
    ) -> Optional[int]:
        """Create a new user account. Returns user ID or None on failure."""
        password_hash = bcrypt.hashpw(
            password.encode("utf-8"), bcrypt.gensalt()
        ).decode("utf-8")
        try:
            with self._get_main_conn() as conn:
                cursor = conn.execute(
                    """INSERT INTO users (username, password_hash, display_name,
                       user_type, level) VALUES (?, ?, ?, ?, ?)""",
                    (username, password_hash, display_name, user_type, level),
                )
                user_id = cursor.lastrowid
            self._init_user_db(username)
            logger.info("Created user: %s (id=%d, type=%s)", username, user_id, user_type)
            return user_id
        except sqlite3.IntegrityError:
            logger.warning("Username already exists: %s", username)
            return None

    def verify_password(self, username: str, password: str) -> Optional[Dict]:
        """Verify password and return user dict, or None if invalid."""
        with self._get_main_conn() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE username = ?", (username,)
            ).fetchone()
        if row is None:
            return None
        if bcrypt.checkpw(password.encode("utf-8"), row["password_hash"].encode("utf-8")):
            self._update_last_login(username)
            return dict(row)
        return None

    def set_pattern(self, user_id: int, pattern_sequence: List[int]) -> bool:
        """Set pattern lock for a user. Pattern is a list of dot indices."""
        pattern_str = "-".join(str(s) for s in pattern_sequence)
        pattern_hash = bcrypt.hashpw(
            pattern_str.encode("utf-8"), bcrypt.gensalt()
        ).decode("utf-8")
        with self._get_main_conn() as conn:
            conn.execute(
                "UPDATE users SET pattern_hash = ? WHERE id = ?",
                (pattern_hash, user_id),
            )
        return True

    def verify_pattern(self, user_id: int, pattern_sequence: List[int]) -> bool:
        """Verify a user's pattern lock."""
        with self._get_main_conn() as conn:
            row = conn.execute(
                "SELECT pattern_hash, username FROM users WHERE id = ?", (user_id,)
            ).fetchone()
        if row is None or row["pattern_hash"] is None:
            return False
        pattern_str = "-".join(str(s) for s in pattern_sequence)
        if bcrypt.checkpw(pattern_str.encode("utf-8"), row["pattern_hash"].encode("utf-8")):
            self._update_last_login(row["username"])
            return True
        return False

    def _update_last_login(self, username: str) -> None:
        """Update last login timestamp."""
        with self._get_main_conn() as conn:
            conn.execute(
                "UPDATE users SET last_login = datetime('now') WHERE username = ?",
                (username,),
            )

    def get_user(self, user_id: int) -> Optional[Dict]:
        """Get user by ID."""
        with self._get_main_conn() as conn:
            row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return dict(row) if row else None

    def get_user_by_username(self, username: str) -> Optional[Dict]:
        """Get user by username."""
        with self._get_main_conn() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE username = ?", (username,)
            ).fetchone()
        return dict(row) if row else None

    def list_users(self, user_type: Optional[str] = None) -> List[Dict]:
        """List all users, optionally filtered by type."""
        with self._get_main_conn() as conn:
            if user_type:
                rows = conn.execute(
                    "SELECT id, username, display_name, user_type, level, last_login "
                    "FROM users WHERE user_type = ? ORDER BY display_name",
                    (user_type,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, username, display_name, user_type, level, last_login "
                    "FROM users ORDER BY display_name"
                ).fetchall()
        return [dict(r) for r in rows]

    # --- Auth Sessions ---

    def create_auth_session(
        self, user_id: Optional[int], device_type: str, hours: int = 168
    ) -> str:
        """Create an auth session token. Returns the token."""
        token = secrets.token_urlsafe(32)
        expires_at = (datetime.utcnow() + timedelta(hours=hours)).isoformat()
        with self._get_main_conn() as conn:
            conn.execute(
                """INSERT INTO auth_sessions (user_id, session_token, device_type, expires_at)
                   VALUES (?, ?, ?, ?)""",
                (user_id, token, device_type, expires_at),
            )
        return token

    def validate_session_token(self, token: str) -> Optional[Dict]:
        """Validate a session token. Returns user dict or None."""
        with self._get_main_conn() as conn:
            row = conn.execute(
                """SELECT a.*, u.username, u.display_name, u.user_type, u.level
                   FROM auth_sessions a
                   LEFT JOIN users u ON a.user_id = u.id
                   WHERE a.session_token = ? AND a.is_active = 1
                   AND a.expires_at > datetime('now')""",
                (token,),
            ).fetchone()
        return dict(row) if row else None

    def invalidate_session(self, token: str) -> None:
        """Invalidate a session token."""
        with self._get_main_conn() as conn:
            conn.execute(
                "UPDATE auth_sessions SET is_active = 0 WHERE session_token = ?",
                (token,),
            )

    # --- Guest Sessions ---

    def create_guest_session(self, ttl_days: int = 7) -> str:
        """Create a guest session. Returns the guest token."""
        token = secrets.token_urlsafe(16)
        expires_at = (datetime.utcnow() + timedelta(days=ttl_days)).isoformat()
        with self._get_main_conn() as conn:
            conn.execute(
                "INSERT INTO guest_sessions (guest_session_token, expires_at) VALUES (?, ?)",
                (token, expires_at),
            )
        return token

    def claim_guest_session(self, guest_token: str, user_id: int) -> bool:
        """Link a guest session to a user account."""
        with self._get_main_conn() as conn:
            cursor = conn.execute(
                """UPDATE guest_sessions SET claimed_by_user_id = ?, claimed_at = datetime('now')
                   WHERE guest_session_token = ? AND claimed_by_user_id IS NULL""",
                (user_id, guest_token),
            )
        return cursor.rowcount > 0

    def cleanup_expired_guests(self) -> int:
        """Remove expired unclaimed guest sessions. Returns count removed."""
        with self._get_main_conn() as conn:
            cursor = conn.execute(
                """DELETE FROM guest_sessions
                   WHERE expires_at < datetime('now') AND claimed_by_user_id IS NULL"""
            )
        count = cursor.rowcount
        if count > 0:
            logger.info("Cleaned up %d expired guest sessions", count)
        return count

    # --- Presets ---

    def get_presets(self, user_id: int) -> List[Dict]:
        """Get all presets for a user."""
        with self._get_main_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM presets WHERE user_id = ? ORDER BY is_favorite DESC, use_count DESC",
                (user_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def create_preset(
        self,
        user_id: int,
        name: str,
        preset_type: str,
        config_json: str,
        description: str = "",
        tags: str = "",
    ) -> int:
        """Create a preset. Returns preset ID."""
        with self._get_main_conn() as conn:
            cursor = conn.execute(
                """INSERT INTO presets (user_id, name, description, preset_type,
                   config_json, tags) VALUES (?, ?, ?, ?, ?, ?)""",
                (user_id, name, description, preset_type, config_json, tags),
            )
        return cursor.lastrowid

    def update_preset(self, preset_id: int, **kwargs: Any) -> bool:
        """Update preset fields."""
        allowed = {"name", "description", "config_json", "is_favorite", "tags"}
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return False
        updates["updated_at"] = datetime.utcnow().isoformat()
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [preset_id]
        with self._get_main_conn() as conn:
            conn.execute(f"UPDATE presets SET {set_clause} WHERE id = ?", values)
        return True

    def increment_preset_use(self, preset_id: int) -> None:
        """Increment preset use count."""
        with self._get_main_conn() as conn:
            conn.execute(
                "UPDATE presets SET use_count = use_count + 1 WHERE id = ?",
                (preset_id,),
            )

    # --- Training Sessions (per-user DB) ---

    def save_training_session(self, username: str, session_data: Dict) -> str:
        """Save a training session record. Returns session_id."""
        session_id = session_data.get("session_id", secrets.token_urlsafe(12))
        with self._get_user_conn(username) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO training_sessions
                   (session_id, mode, difficulty, started_at, ended_at, is_complete,
                    rounds_completed, rounds_total, work_time_sec, rest_time_sec,
                    config_json, summary_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    session_data.get("mode", "training"),
                    session_data.get("difficulty", "beginner"),
                    session_data.get("started_at", datetime.utcnow().isoformat()),
                    session_data.get("ended_at"),
                    int(session_data.get("is_complete", False)),
                    session_data.get("rounds_completed", 0),
                    session_data.get("rounds_total", 0),
                    session_data.get("work_time_sec", 180),
                    session_data.get("rest_time_sec", 60),
                    json.dumps(session_data.get("config", {})),
                    json.dumps(session_data.get("summary", {})),
                ),
            )
        return session_id

    def save_session_event(
        self, username: str, session_id: str, timestamp: float,
        event_type: str, data: Dict,
    ) -> None:
        """Save a timestamped event within a session."""
        with self._get_user_conn(username) as conn:
            conn.execute(
                """INSERT INTO session_events (session_id, timestamp, event_type, data_json)
                   VALUES (?, ?, ?, ?)""",
                (session_id, timestamp, event_type, json.dumps(data)),
            )

    def get_session_history(
        self, username: str, limit: int = 50, mode: Optional[str] = None,
    ) -> List[Dict]:
        """Get training session history for a user."""
        with self._get_user_conn(username) as conn:
            if mode:
                rows = conn.execute(
                    """SELECT * FROM training_sessions WHERE mode = ?
                       ORDER BY started_at DESC LIMIT ?""",
                    (mode, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM training_sessions ORDER BY started_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [dict(r) for r in rows]

    def get_session_detail(self, username: str, session_id: str) -> Optional[Dict]:
        """Get full session detail including events."""
        with self._get_user_conn(username) as conn:
            session = conn.execute(
                "SELECT * FROM training_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if session is None:
                return None
            events = conn.execute(
                "SELECT * FROM session_events WHERE session_id = ? ORDER BY timestamp",
                (session_id,),
            ).fetchall()
        result = dict(session)
        result["events"] = [dict(e) for e in events]
        return result

    # --- Performance Tests (per-user DB) ---

    def save_power_test(self, username: str, data: Dict) -> int:
        """Save a power test result."""
        with self._get_user_conn(username) as conn:
            cursor = conn.execute(
                """INSERT INTO power_tests (peak_force, avg_force, punch_count, results_json)
                   VALUES (?, ?, ?, ?)""",
                (data["peak_force"], data["avg_force"], data["punch_count"],
                 json.dumps(data.get("results", []))),
            )
        return cursor.lastrowid

    def save_stamina_test(self, username: str, data: Dict) -> int:
        """Save a stamina test result."""
        with self._get_user_conn(username) as conn:
            cursor = conn.execute(
                """INSERT INTO stamina_tests
                   (duration_sec, total_punches, punches_per_minute, fatigue_index, results_json)
                   VALUES (?, ?, ?, ?, ?)""",
                (data["duration_sec"], data["total_punches"],
                 data["punches_per_minute"], data["fatigue_index"],
                 json.dumps(data.get("results", []))),
            )
        return cursor.lastrowid

    def save_reaction_test(self, username: str, data: Dict) -> int:
        """Save a reaction time test result."""
        with self._get_user_conn(username) as conn:
            cursor = conn.execute(
                """INSERT INTO reaction_tests
                   (num_trials, avg_reaction_ms, best_reaction_ms, worst_reaction_ms,
                    tier, results_json)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (data["num_trials"], data["avg_reaction_ms"],
                 data["best_reaction_ms"], data["worst_reaction_ms"],
                 data["tier"], json.dumps(data.get("results", []))),
            )
        return cursor.lastrowid

    # --- Gamification (per-user DB) ---

    def get_user_xp(self, username: str) -> Dict:
        """Get user's XP and rank data."""
        with self._get_user_conn(username) as conn:
            row = conn.execute("SELECT * FROM user_xp WHERE id = 1").fetchone()
        return dict(row) if row else {"total_xp": 0, "current_rank": "Novice"}

    def add_xp(self, username: str, xp_amount: int) -> Dict:
        """Add XP and update rank if needed. Returns updated XP data."""
        ranks = [
            ("Novice", 0), ("Contender", 500), ("Fighter", 1500),
            ("Warrior", 4000), ("Champion", 10000), ("Elite", 25000),
        ]
        with self._get_user_conn(username) as conn:
            current = conn.execute("SELECT * FROM user_xp WHERE id = 1").fetchone()
            new_xp = (current["total_xp"] if current else 0) + xp_amount
            new_rank = "Novice"
            for rank_name, threshold in reversed(ranks):
                if new_xp >= threshold:
                    new_rank = rank_name
                    break
            old_rank = current["current_rank"] if current else "Novice"
            rank_history = json.loads(current["rank_history_json"]) if current else []
            if new_rank != old_rank:
                rank_history.append({
                    "rank": new_rank, "achieved_at": datetime.utcnow().isoformat(),
                    "xp": new_xp,
                })
            conn.execute(
                """UPDATE user_xp SET total_xp = ?, current_rank = ?,
                   rank_history_json = ? WHERE id = 1""",
                (new_xp, new_rank, json.dumps(rank_history)),
            )
        return {"total_xp": new_xp, "current_rank": new_rank, "ranked_up": new_rank != old_rank}

    def check_personal_record(
        self, username: str, record_type: str, value: float,
    ) -> Optional[Dict]:
        """Check and update a personal record. Returns PR data if broken."""
        with self._get_user_conn(username) as conn:
            existing = conn.execute(
                "SELECT * FROM personal_records WHERE record_type = ?",
                (record_type,),
            ).fetchone()
            if existing is None or value > existing["value"]:
                previous = existing["value"] if existing else 0.0
                conn.execute(
                    """INSERT OR REPLACE INTO personal_records
                       (record_type, value, achieved_at, previous_value)
                       VALUES (?, ?, datetime('now'), ?)""",
                    (record_type, value, previous),
                )
                return {
                    "record_type": record_type, "value": value,
                    "previous_value": previous, "is_new": existing is None,
                }
        return None

    def update_streak(self, username: str) -> Dict:
        """Update training streak. Call after each session."""
        today = datetime.utcnow().date().isoformat()
        with self._get_user_conn(username) as conn:
            row = conn.execute("SELECT * FROM streaks WHERE id = 1").fetchone()
            streak_data = dict(row) if row else {
                "current_streak": 0, "longest_streak": 0,
                "last_training_date": None, "weekly_goal": 3,
                "weekly_progress": 0, "week_start_date": None,
            }
            last_date = streak_data["last_training_date"]
            if last_date == today:
                return streak_data
            yesterday = (datetime.utcnow().date() - timedelta(days=1)).isoformat()
            if last_date == yesterday:
                streak_data["current_streak"] += 1
            elif last_date is None:
                streak_data["current_streak"] = 1
            else:
                streak_data["current_streak"] = 1
            streak_data["longest_streak"] = max(
                streak_data["longest_streak"], streak_data["current_streak"]
            )
            streak_data["last_training_date"] = today
            streak_data["weekly_progress"] += 1
            conn.execute(
                """UPDATE streaks SET current_streak = ?, longest_streak = ?,
                   last_training_date = ?, weekly_progress = ? WHERE id = 1""",
                (streak_data["current_streak"], streak_data["longest_streak"],
                 today, streak_data["weekly_progress"]),
            )
        return streak_data

    def unlock_achievement(self, username: str, achievement_id: str) -> bool:
        """Unlock an achievement. Returns True if newly unlocked."""
        try:
            with self._get_user_conn(username) as conn:
                conn.execute(
                    "INSERT INTO achievements (achievement_id) VALUES (?)",
                    (achievement_id,),
                )
            return True
        except sqlite3.IntegrityError:
            return False

    def get_achievements(self, username: str) -> List[Dict]:
        """Get all unlocked achievements."""
        with self._get_user_conn(username) as conn:
            rows = conn.execute(
                "SELECT * FROM achievements ORDER BY unlocked_at"
            ).fetchall()
        return [dict(r) for r in rows]
