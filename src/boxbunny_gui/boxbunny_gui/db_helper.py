"""Lightweight DB helper using direct sqlite3 + hashlib.

Avoids importing DatabaseManager which pulls in native libs that
conflict with the conda environment on aarch64.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_WS_ROOT = Path(__file__).resolve().parents[3]  # boxing_robot_ws/
_DB_PATH = _WS_ROOT / "data" / "boxbunny_main.db"


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(str(_DB_PATH))
    c.row_factory = sqlite3.Row
    return c


def _hash_pw(password: str) -> str:
    """Simple salted SHA-256 hash. Format: sha256:<salt>:<hash>"""
    import os
    salt = os.urandom(16).hex()
    h = hashlib.sha256(f"{salt}:{password}".encode()).hexdigest()
    return f"sha256:{salt}:{h}"


def _verify_hash(password: str, stored: str) -> bool:
    """Verify against sha256:<salt>:<hash> or bcrypt format."""
    if stored.startswith("sha256:"):
        _, salt, expected = stored.split(":", 2)
        h = hashlib.sha256(f"{salt}:{password}".encode()).hexdigest()
        return hmac.compare_digest(h, expected)
    # Try bcrypt fallback
    try:
        import bcrypt
        return bcrypt.checkpw(password.encode("utf-8"),
                              stored.encode("utf-8"))
    except Exception:
        return False


def list_users() -> List[Dict]:
    """List all users."""
    try:
        with _conn() as c:
            rows = c.execute(
                "SELECT id, username, display_name, user_type, level "
                "FROM users ORDER BY display_name"
            ).fetchall()
        return [dict(r) for r in rows]
    except Exception as exc:
        logger.warning("list_users failed: %s", exc)
        return []


def get_user(user_id: int) -> Optional[Dict]:
    try:
        with _conn() as c:
            row = c.execute(
                "SELECT * FROM users WHERE id = ?", (user_id,)
            ).fetchone()
        return dict(row) if row else None
    except Exception as exc:
        logger.warning("get_user failed: %s", exc)
        return None


def get_user_by_username(username: str) -> Optional[Dict]:
    try:
        with _conn() as c:
            row = c.execute(
                "SELECT * FROM users WHERE username = ?", (username,)
            ).fetchone()
        return dict(row) if row else None
    except Exception as exc:
        logger.warning("get_user_by_username failed: %s", exc)
        return None


def verify_password(username: str, password: str) -> Optional[Dict]:
    """Verify password, return user dict or None."""
    user = get_user_by_username(username)
    if not user:
        return None
    if _verify_hash(password, user["password_hash"]):
        return user
    return None


def verify_pattern(user_id: int, pattern: List[int]) -> bool:
    """Verify pattern lock."""
    user = get_user(user_id)
    if not user or not user.get("pattern_hash"):
        return False
    pattern_str = "-".join(str(s) for s in pattern)
    return _verify_hash(pattern_str, user["pattern_hash"])


def update_password(username: str, new_password: str) -> bool:
    """Update a user's password."""
    try:
        pw_hash = _hash_pw(new_password)
        with _conn() as c:
            c.execute(
                "UPDATE users SET password_hash = ? WHERE username = ?",
                (pw_hash, username),
            )
        return True
    except Exception as exc:
        logger.warning("update_password failed: %s", exc)
        return False


def update_pattern(username: str, pattern: List[int]) -> bool:
    """Update a user's pattern lock."""
    try:
        pattern_str = "-".join(str(s) for s in pattern)
        pat_hash = _hash_pw(pattern_str)
        with _conn() as c:
            c.execute(
                "UPDATE users SET pattern_hash = ? WHERE username = ?",
                (pat_hash, username),
            )
        return True
    except Exception as exc:
        logger.warning("update_pattern failed: %s", exc)
        return False
