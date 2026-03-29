"""Gamification endpoints and engine for BoxBunny Dashboard.

Provides XP, rank progression, achievements, and coaching leaderboards.
Ranks: Novice(0) -> Contender(500) -> Fighter(1500) -> Warrior(4000) ->
       Champion(10000) -> Elite(25000)
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel

from boxbunny_dashboard.api.auth import get_current_user
from boxbunny_dashboard.db.manager import DatabaseManager

logger = logging.getLogger("boxbunny.dashboard.gamification")
router = APIRouter()

# ---- Constants ----

RANKS = [
    ("Novice", 0),
    ("Contender", 500),
    ("Fighter", 1500),
    ("Warrior", 4000),
    ("Champion", 10000),
    ("Elite", 25000),
]


# ---- Pydantic models ----

class GamificationProfile(BaseModel):
    total_xp: int
    current_rank: str
    next_rank: Optional[str] = None
    xp_to_next_rank: int = 0
    current_streak: int = 0
    longest_streak: int = 0
    weekly_goal: int = 3
    weekly_progress: int = 0
    personal_records: List[Dict[str, Any]] = []


class Achievement(BaseModel):
    achievement_id: str
    unlocked_at: str


class LeaderboardEntry(BaseModel):
    username: str
    display_name: str
    score: int
    rank: str


# ---- GamificationEngine ----

class GamificationEngine:
    """Calculates XP, scores, and achievement checks for training sessions."""

    # XP multipliers by mode
    _MODE_XP = {
        "reaction": 1.0,
        "shadow": 1.5,
        "defence": 2.0,
        "power_test": 0.8,
        "stamina_test": 1.2,
    }

    @staticmethod
    def calculate_session_xp(session_data: Dict[str, Any]) -> int:
        """Calculate XP earned from a training session."""
        base_xp = 50  # base participation XP
        mode = session_data.get("mode", "training")
        multiplier = GamificationEngine._MODE_XP.get(mode, 1.0)

        # Bonus for rounds completed
        rounds = session_data.get("rounds_completed", 0)
        round_xp = rounds * 15

        # Bonus for session completion
        completion_bonus = 25 if session_data.get("is_complete", False) else 0

        # Bonus for difficulty
        difficulty_mult = {
            "beginner": 1.0, "intermediate": 1.3, "advanced": 1.6, "elite": 2.0,
        }.get(session_data.get("difficulty", "beginner"), 1.0)

        total = int((base_xp + round_xp + completion_bonus) * multiplier * difficulty_mult)
        return max(total, 10)  # minimum 10 XP per session

    @staticmethod
    def calculate_session_score(session_data: Dict[str, Any]) -> int:
        """Calculate a performance score (0-100) for a session."""
        score = 50  # baseline
        summary = session_data.get("summary", {})

        # Accuracy component (if available)
        accuracy = summary.get("accuracy", 0.0)
        score += int(accuracy * 30)

        # Speed component
        avg_reaction = summary.get("avg_reaction_ms", 500)
        if avg_reaction < 200:
            score += 20
        elif avg_reaction < 350:
            score += 10
        elif avg_reaction < 500:
            score += 5

        # Completion bonus
        if session_data.get("is_complete", False):
            score += 10

        return min(max(score, 0), 100)

    @staticmethod
    def check_achievements(
        username: str, session_data: Dict[str, Any],
    ) -> List[str]:
        """Check which achievements should be unlocked based on session data.

        Returns a list of achievement_id strings to unlock.
        """
        unlockable: List[str] = []
        summary = session_data.get("summary", {})

        # First session
        if session_data.get("total_sessions", 0) == 1:
            unlockable.append("first_blood")

        # Punch count milestones
        punch_count = summary.get("total_punches", 0)
        if punch_count >= 100:
            unlockable.append("century")
        if punch_count >= 500:
            unlockable.append("fury")
        if punch_count >= 1000:
            unlockable.append("thousand_fists")

        # Reaction time tier
        tier = summary.get("reaction_tier", "")
        if tier == "lightning":
            unlockable.append("speed_demon")

        # Streak-based
        streak = session_data.get("current_streak", 0)
        if streak >= 7:
            unlockable.append("weekly_warrior")
        if streak >= 30:
            unlockable.append("consistent")

        # Total sessions
        total = session_data.get("total_sessions", 0)
        if total >= 10:
            unlockable.append("iron_chin")
        if total >= 50:
            unlockable.append("marathon")
        if total >= 100:
            unlockable.append("centurion")

        # Mode diversity
        modes_played = session_data.get("modes_played", [])
        if len(modes_played) >= 3:
            unlockable.append("well_rounded")

        # Perfect session
        if summary.get("accuracy", 0.0) >= 1.0 and session_data.get("is_complete"):
            unlockable.append("perfect_round")

        return unlockable


# ---- Helpers ----

def _get_db(request: Request) -> DatabaseManager:
    return request.app.state.db


def _next_rank(current_xp: int) -> tuple:
    """Return (next_rank_name, xp_needed) or (None, 0) if max rank."""
    for rank_name, threshold in RANKS:
        if current_xp < threshold:
            return rank_name, threshold - current_xp
    return None, 0


# ---- Endpoints ----

@router.get("/profile", response_model=GamificationProfile)
async def get_profile(
    user: dict = Depends(get_current_user),
    db: DatabaseManager = Depends(_get_db),
) -> GamificationProfile:
    """Return the user's gamification profile: rank, XP, streak, PRs."""
    username = user["username"]
    xp_data = db.get_user_xp(username)
    streak_data = db.update_streak(username)
    next_rank_name, xp_to_next = _next_rank(xp_data.get("total_xp", 0))

    return GamificationProfile(
        total_xp=xp_data.get("total_xp", 0),
        current_rank=xp_data.get("current_rank", "Novice"),
        next_rank=next_rank_name,
        xp_to_next_rank=xp_to_next,
        current_streak=streak_data.get("current_streak", 0),
        longest_streak=streak_data.get("longest_streak", 0),
        weekly_goal=streak_data.get("weekly_goal", 3),
        weekly_progress=streak_data.get("weekly_progress", 0),
    )


@router.get("/achievements", response_model=List[Achievement])
async def get_achievements(
    user: dict = Depends(get_current_user),
    db: DatabaseManager = Depends(_get_db),
) -> List[Achievement]:
    """Return all unlocked achievements for the user."""
    username = user["username"]
    rows = db.get_achievements(username)
    return [
        Achievement(
            achievement_id=r["achievement_id"],
            unlocked_at=r.get("unlocked_at", ""),
        )
        for r in rows
    ]


@router.get("/leaderboard/{coaching_session_id}", response_model=List[LeaderboardEntry])
async def get_leaderboard(
    coaching_session_id: str,
    user: dict = Depends(get_current_user),
    db: DatabaseManager = Depends(_get_db),
) -> List[LeaderboardEntry]:
    """Return the leaderboard for a coaching session.

    Aggregates scores from all participants in the coaching session.
    """
    # For coaching sessions, query participants from the main DB
    participants = db.list_users(user_type="individual")
    entries: List[LeaderboardEntry] = []
    for p in participants:
        history = db.get_session_history(p["username"], limit=5)
        total_score = sum(
            GamificationEngine.calculate_session_score(s) for s in history
        )
        xp_data = db.get_user_xp(p["username"])
        entries.append(LeaderboardEntry(
            username=p["username"],
            display_name=p.get("display_name", p["username"]),
            score=total_score,
            rank=xp_data.get("current_rank", "Novice"),
        ))

    entries.sort(key=lambda e: e.score, reverse=True)
    return entries


# ---- Benchmarks / Percentile Comparison ----

@router.get("/benchmarks")
async def get_benchmarks(
    request: Request,
    user: dict = Depends(get_current_user),
) -> dict:
    """Get percentile rankings against population norms for the current user.

    Compares the user's best performance metrics against age/gender norms
    from sports science research.
    """
    db: DatabaseManager = request.app.state.db
    username = user["username"]

    # Gather user's best stats from session history
    sessions = db.get_session_history(username, limit=50)
    if not sessions:
        return {"benchmarks": {}, "message": "No sessions yet — train first!"}

    # Aggregate best metrics across all sessions
    import json as _json
    best_stats: dict = {}
    for session in sessions:
        summary = session.get("summary_json", "{}")
        if isinstance(summary, str):
            try:
                summary = _json.loads(summary)
            except _json.JSONDecodeError:
                summary = {}
        if summary.get("total_punches", 0) > best_stats.get("total_punches", 0):
            best_stats["total_punches"] = summary["total_punches"]

    # Get performance test bests
    reaction_tests = db.get_session_history(username, mode="reaction")
    for rt in reaction_tests:
        s = rt.get("summary_json", "{}")
        if isinstance(s, str):
            try:
                s = _json.loads(s)
            except _json.JSONDecodeError:
                s = {}
        avg = s.get("avg_reaction_ms", 9999)
        if avg < best_stats.get("avg_reaction_ms", 9999):
            best_stats["avg_reaction_ms"] = avg

    # Compute percentiles using the benchmark engine
    try:
        from boxbunny_dashboard.benchmarks import BenchmarkEngine
        engine = BenchmarkEngine()
        results = engine.get_all_percentiles(
            best_stats,
            age=user.get("age"),
            gender=user.get("gender"),
            level=user.get("level"),
        )
    except Exception as e:
        logger.error("Benchmark computation failed: %s", e)
        results = {}

    return {
        "benchmarks": results,
        "user_stats": best_stats,
        "demographics": {
            "age": user.get("age"),
            "gender": user.get("gender"),
            "height_cm": user.get("height_cm"),
            "weight_kg": user.get("weight_kg"),
        },
    }
