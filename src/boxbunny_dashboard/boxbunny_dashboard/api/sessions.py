"""Training session data endpoints for BoxBunny Dashboard.

Provides access to current live session data, detailed session summaries,
paginated session history, and trend analytics. All data is per-user.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel

from boxbunny_dashboard.api.auth import get_current_user
from boxbunny_dashboard.db.manager import DatabaseManager

logger = logging.getLogger("boxbunny.dashboard.sessions")
router = APIRouter()


# ---- Pydantic models ----

class SessionSummary(BaseModel):
    session_id: str
    mode: str
    difficulty: str
    started_at: str
    ended_at: Optional[str] = None
    is_complete: bool = False
    rounds_completed: int = 0
    rounds_total: int = 0
    work_time_sec: int = 0
    rest_time_sec: int = 0


class SessionDetail(SessionSummary):
    config: Dict[str, Any] = {}
    summary: Dict[str, Any] = {}
    events: List[Dict[str, Any]] = []


class SessionHistoryResponse(BaseModel):
    sessions: List[SessionSummary]
    total: int
    page: int
    page_size: int


# ---- Helpers ----

def _get_db(request: Request) -> DatabaseManager:
    return request.app.state.db


def _parse_json_field(value: Optional[str]) -> Dict[str, Any]:
    """Safely parse a JSON string field."""
    if not value:
        return {}
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return {}


def _row_to_summary(row: Dict[str, Any]) -> SessionSummary:
    return SessionSummary(
        session_id=row["session_id"],
        mode=row.get("mode", "training"),
        difficulty=row.get("difficulty", "beginner"),
        started_at=row.get("started_at", ""),
        ended_at=row.get("ended_at"),
        is_complete=bool(row.get("is_complete", False)),
        rounds_completed=row.get("rounds_completed", 0),
        rounds_total=row.get("rounds_total", 0),
        work_time_sec=row.get("work_time_sec", 0),
        rest_time_sec=row.get("rest_time_sec", 0),
    )


# ---- Endpoints ----

@router.get("/current")
async def get_current_session(
    request: Request,
    user: dict = Depends(get_current_user),
    db: DatabaseManager = Depends(_get_db),
) -> Dict[str, Any]:
    """Return the latest live session data (most recent incomplete session)."""
    username = user["username"]
    sessions = db.get_session_history(username, limit=1)
    if not sessions:
        return {"active": False, "session": None}

    latest = sessions[0]
    is_active = not bool(latest.get("is_complete", True))
    response: Dict[str, Any] = {
        "active": is_active,
        "session": _row_to_summary(latest).model_dump(),
    }

    # Include live state from WebSocket manager if available
    ws_manager = request.app.state.ws_manager
    state = ws_manager._state_buffer.get(username)
    if state and is_active:
        response["live_state"] = state

    return response


@router.get("/history", response_model=SessionHistoryResponse)
async def get_session_history(
    user: dict = Depends(get_current_user),
    db: DatabaseManager = Depends(_get_db),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    mode: Optional[str] = Query(default=None),
) -> SessionHistoryResponse:
    """Return paginated training session history."""
    username = user["username"]
    # Fetch extra to determine total (SQLite doesn't have cheap COUNT with LIMIT)
    all_sessions = db.get_session_history(username, limit=1000, mode=mode)
    total = len(all_sessions)
    start = (page - 1) * page_size
    page_slice = all_sessions[start : start + page_size]

    return SessionHistoryResponse(
        sessions=[_row_to_summary(s) for s in page_slice],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/trends")
async def get_session_trends(
    request: Request,
    user: dict = Depends(get_current_user),
    db: DatabaseManager = Depends(_get_db),
    range: str = Query(default="30d", pattern="^(7d|30d|90d|all)$"),
) -> Dict[str, Any]:
    """Return time-series trend data for analytics dashboards.

    Aggregates punch volume, reaction time, defense rate, personal bests,
    weekly summaries, and period-over-period comparisons.
    """
    username = user["username"]

    # Determine cutoff date from range
    range_days = {"7d": 7, "30d": 30, "90d": 90, "all": 3650}
    days = range_days.get(range, 30)

    all_sessions = db.get_session_history(username, limit=1000)
    if not all_sessions:
        return {
            "punch_volume": [],
            "reaction_time": [],
            "defense_rate": [],
            "power": [],
            "stamina": [],
            "personal_bests": {},
            "weekly_summary": {"sessions": 0, "total_punches": 0, "avg_score": 0},
            "period_comparison": {"vs_last_week": "0%", "vs_last_month": "0%"},
        }

    cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()

    # Filter sessions within range
    sessions_in_range = []
    for s in all_sessions:
        started = s.get("started_at", "")
        if started >= cutoff or range == "all":
            sessions_in_range.append(s)

    # Build trend arrays
    punch_volume = []
    reaction_time = []
    defense_rate = []
    power_data = []
    stamina_data = []
    total_punches_all = 0
    scores = []

    # Track personal bests
    best_reaction = 9999
    best_punches = 0
    best_defense = 0.0
    best_power = 0
    best_stamina = 0

    for s in reversed(sessions_in_range):
        date_str = s.get("started_at", "")[:10]
        summary = _parse_json_field(s.get("summary_json"))

        # Punch volume
        punches = summary.get("total_punches", (s.get("rounds_completed", 0) * 20))
        punch_volume.append({"date": date_str, "value": punches})
        total_punches_all += punches

        if punches > best_punches:
            best_punches = punches

        # Reaction time
        react_ms = summary.get("avg_reaction_ms")
        if react_ms:
            reaction_time.append({"date": date_str, "value": react_ms})
            if react_ms < best_reaction:
                best_reaction = react_ms

        # Defense rate
        defense = summary.get("defense_rate")
        if defense is not None:
            defense_rate.append({"date": date_str, "value": round(defense, 3)})
            if defense > best_defense:
                best_defense = defense

        # Power
        pw = summary.get("max_power", 0)
        if pw:
            power_data.append({"date": date_str, "value": pw})
            if pw > best_power:
                best_power = pw

        # Stamina (punches per minute)
        ppm = summary.get("punches_per_minute", 0)
        if ppm:
            stamina_data.append({"date": date_str, "value": round(ppm, 1)})
            if ppm > best_stamina:
                best_stamina = ppm

        # Score tracking
        accuracy = summary.get("accuracy", 0.5)
        scores.append(int(accuracy * 100))

    # Weekly summary (last 7 days)
    week_cutoff = (datetime.utcnow() - timedelta(days=7)).isoformat()
    week_sessions = [s for s in all_sessions if s.get("started_at", "") >= week_cutoff]
    week_punches = 0
    for s in week_sessions:
        sm = _parse_json_field(s.get("summary_json"))
        week_punches += sm.get("total_punches", (s.get("rounds_completed", 0) * 20))

    avg_score = round(sum(scores) / len(scores)) if scores else 0

    # Period comparison
    prev_week_cutoff = (datetime.utcnow() - timedelta(days=14)).isoformat()
    prev_week_sessions = [
        s for s in all_sessions
        if prev_week_cutoff <= s.get("started_at", "") < week_cutoff
    ]
    prev_month_cutoff = (datetime.utcnow() - timedelta(days=60)).isoformat()
    month_cutoff = (datetime.utcnow() - timedelta(days=30)).isoformat()
    prev_month_sessions = [
        s for s in all_sessions
        if prev_month_cutoff <= s.get("started_at", "") < month_cutoff
    ]

    def _pct_change(current: int, previous: int) -> str:
        if previous == 0:
            return "+100%" if current > 0 else "0%"
        change = round(((current - previous) / previous) * 100)
        return f"+{change}%" if change >= 0 else f"{change}%"

    vs_last_week = _pct_change(len(week_sessions), len(prev_week_sessions))
    month_sessions = [
        s for s in all_sessions if s.get("started_at", "") >= month_cutoff
    ]
    vs_last_month = _pct_change(len(month_sessions), len(prev_month_sessions))

    # Training days for heat map (current week, Mon=0 to Sun=6)
    today = datetime.utcnow().date()
    week_start = today - timedelta(days=today.weekday())
    training_days = set()
    for s in week_sessions:
        try:
            d = datetime.fromisoformat(s["started_at"].replace("Z", "+00:00")).date()
            day_idx = (d - week_start).days
            if 0 <= day_idx <= 6:
                training_days.add(day_idx)
        except (ValueError, KeyError):
            pass

    return {
        "punch_volume": punch_volume,
        "reaction_time": reaction_time,
        "defense_rate": defense_rate,
        "power": power_data,
        "stamina": stamina_data,
        "personal_bests": {
            "fastest_reaction_ms": best_reaction if best_reaction < 9999 else None,
            "most_punches": best_punches if best_punches > 0 else None,
            "best_defense_rate": round(best_defense, 3) if best_defense > 0 else None,
            "max_power": best_power if best_power > 0 else None,
            "best_stamina_ppm": best_stamina if best_stamina > 0 else None,
        },
        "weekly_summary": {
            "sessions": len(week_sessions),
            "total_punches": week_punches,
            "avg_score": avg_score,
        },
        "period_comparison": {
            "vs_last_week": vs_last_week,
            "vs_last_month": vs_last_month,
        },
        "training_days": sorted(training_days),
    }


@router.get("/{session_id}/raw")
async def get_session_raw_data(
    session_id: str,
    user: dict = Depends(get_current_user),
    db: DatabaseManager = Depends(_get_db),
) -> Dict[str, Any]:
    """Return raw CV predictions, IMU strikes, and direction timeline for a session."""
    username = user["username"]

    # Get session events from database
    events = db.get_session_events(username, session_id) or []

    raw_cv: List[Dict[str, Any]] = []
    raw_imu: List[Dict[str, Any]] = []
    direction: List[Dict[str, Any]] = []

    for evt in events:
        if evt.get("event_type") == "raw_cv_predictions":
            raw_cv = evt.get("data", [])
        elif evt.get("event_type") == "raw_imu_strikes":
            raw_imu = evt.get("data", [])
        elif evt.get("event_type") == "direction_timeline":
            direction = evt.get("data", [])

    # Also get experimental data from summary
    detail = db.get_session_detail(username, session_id)
    summary: Dict[str, Any] = {}
    if detail and detail.get("summary_json"):
        try:
            summary = (
                json.loads(detail["summary_json"])
                if isinstance(detail["summary_json"], str)
                else detail["summary_json"]
            )
        except (json.JSONDecodeError, TypeError):
            pass

    return {
        "cv_predictions": raw_cv,
        "imu_strikes": raw_imu,
        "direction_timeline": direction,
        "cv_prediction_summary": summary.get("cv_prediction_summary", {}),
        "imu_strike_summary": summary.get("imu_strike_summary", {}),
        "direction_summary": summary.get("direction_summary", {}),
        "experimental": summary.get("experimental", {}),
    }


@router.get("/{session_id}", response_model=SessionDetail)
async def get_session_detail(
    session_id: str,
    user: dict = Depends(get_current_user),
    db: DatabaseManager = Depends(_get_db),
) -> SessionDetail:
    """Return detailed data for a specific training session."""
    username = user["username"]
    detail = db.get_session_detail(username, session_id)
    if detail is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )
    return SessionDetail(
        session_id=detail["session_id"],
        mode=detail.get("mode", "training"),
        difficulty=detail.get("difficulty", "beginner"),
        started_at=detail.get("started_at", ""),
        ended_at=detail.get("ended_at"),
        is_complete=bool(detail.get("is_complete", False)),
        rounds_completed=detail.get("rounds_completed", 0),
        rounds_total=detail.get("rounds_total", 0),
        work_time_sec=detail.get("work_time_sec", 0),
        rest_time_sec=detail.get("rest_time_sec", 0),
        config=_parse_json_field(detail.get("config_json")),
        summary=_parse_json_field(detail.get("summary_json")),
        events=detail.get("events", []),
    )
