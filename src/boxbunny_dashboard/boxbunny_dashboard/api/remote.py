"""Remote GUI control endpoints.

Allows the phone dashboard to send commands to the GUI (start training,
open presets, etc.) via a shared command file that the GUI polls.
"""
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from boxbunny_dashboard.api.auth import get_current_user
from boxbunny_dashboard.db.manager import DatabaseManager

logger = logging.getLogger("boxbunny.dashboard.remote")
router = APIRouter()

_CMD_FILE = Path("/tmp/boxbunny_gui_command.json")


class RemoteCommand(BaseModel):
    action: str = Field(..., description="start_training | start_preset | open_presets | stop_session | navigate")
    config: Dict[str, Any] = Field(default_factory=dict)


class RemoteStatus(BaseModel):
    success: bool
    message: str


def _get_db(request: Request) -> DatabaseManager:
    return request.app.state.db


def _db_preset_to_gui(row: dict) -> dict:
    """Convert main DB preset row to GUI preset format."""
    cfg = {}
    try:
        cfg = json.loads(row.get("config_json", "{}"))
    except (json.JSONDecodeError, TypeError):
        pass

    # Convert DB format to GUI format
    rounds = str(cfg.get("rounds", cfg.get("Rounds", "2")))
    work_sec = cfg.get("work_sec", 90)
    rest_sec = cfg.get("rest_sec", 30)
    speed = cfg.get("speed", cfg.get("Speed", "Medium (2s)"))
    combo_name = cfg.get("combo_name", cfg.get("combo", {}).get("name", ""))
    combo_seq = cfg.get("combo_seq", cfg.get("combo", {}).get("seq", ""))
    combo_id = cfg.get("combo_id", cfg.get("combo", {}).get("id"))
    difficulty = cfg.get("difficulty", row.get("preset_type", "Beginner")).title()

    # Map work_sec to string if needed
    if isinstance(work_sec, (int, float)):
        work_time = f"{int(work_sec)}s"
    else:
        work_time = str(work_sec)
    if isinstance(rest_sec, (int, float)):
        rest_time = f"{int(rest_sec)}s"
    else:
        rest_time = str(rest_sec)

    # Determine route
    ptype = row.get("preset_type", "training")
    route_map = {
        "training": "training_session",
        "sparring": "sparring_session",
        "performance": "power_test",
        "free": "training_session",
        "circuit": "training_session",
    }
    route = cfg.get("route", route_map.get(ptype, "training_session"))

    return {
        "name": row.get("name", "Preset"),
        "tag": ptype.upper(),
        "desc": row.get("description", ""),
        "route": route,
        "combo": {
            "id": combo_id,
            "name": combo_name or row.get("name", ""),
            "seq": combo_seq,
        },
        "config": {
            "Rounds": rounds,
            "Work Time": work_time,
            "Rest Time": rest_time,
            "Speed": speed if isinstance(speed, str) else "Medium (2s)",
        },
        "difficulty": difficulty,
        "accent": "#FF6B35",
        "id": row.get("id"),
    }


@router.post("/command", response_model=RemoteStatus)
async def send_command(
    body: RemoteCommand,
    user: dict = Depends(get_current_user),
) -> RemoteStatus:
    """Send a remote command to the GUI."""
    cmd = {
        "action": body.action,
        "config": body.config,
        "username": user["username"],
        "timestamp": time.time(),
    }
    try:
        _CMD_FILE.write_text(json.dumps(cmd))
        logger.info("Remote command: %s from %s", body.action, user["username"])
        return RemoteStatus(success=True, message=f"Command '{body.action}' sent")
    except Exception as exc:
        logger.warning("Failed to write command: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to send command to GUI",
        )


class HeightAction(BaseModel):
    action: str = Field(..., description="up | down | stop")


# Height control state
_height_active: bool = False
_height_direction: str = "stop"
_height_last_cmd: float = 0.0


@router.post("/height", response_model=RemoteStatus)
async def height_control(
    body: HeightAction,
    user: dict = Depends(get_current_user),
) -> RemoteStatus:
    """Control robot height via press-and-hold.

    Send "up"/"down" to start moving, "stop" to halt.
    Auto-stops after 5 seconds if no new command (dead-man switch).
    """
    global _height_active, _height_direction, _height_last_cmd

    action_map = {"up": "manual_up", "down": "manual_down", "stop": "stop"}
    mapped = action_map.get(body.action)
    if mapped is None:
        raise HTTPException(400, f"Invalid action: {body.action}")

    _height_direction = mapped
    _height_last_cmd = time.time()
    _height_active = mapped != "stop"

    # Write height command for GUI to pick up
    cmd = {
        "action": "height_adjust",
        "config": {"height_action": mapped},
        "timestamp": time.time(),
    }
    try:
        _CMD_FILE.write_text(json.dumps(cmd))
        logger.info("Height: %s from %s", mapped, user["username"])
        return RemoteStatus(success=True, message=f"Height: {body.action}")
    except Exception as exc:
        logger.warning("Failed to write height command: %s", exc)
        raise HTTPException(500, "Failed to send height command")


@router.get("/presets")
async def get_user_presets(
    user: dict = Depends(get_current_user),
    db: DatabaseManager = Depends(_get_db),
) -> list:
    """Get user's presets from the main DB (same source as /presets/ API)."""
    try:
        with db._get_main_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM presets WHERE user_id = ? AND (tags IS NULL OR tags != 'archived') ORDER BY is_favorite DESC, use_count DESC",
                (user["user_id"],),
            ).fetchall()
        return [_db_preset_to_gui(dict(r)) for r in rows]
    except Exception as exc:
        logger.warning("Failed to load presets: %s", exc)
        return _default_presets()


def _default_presets() -> list:
    return [
        {
            "name": "Free Training",
            "tag": "OPEN SESSION",
            "desc": "Punch freely with no combos",
            "route": "training_session",
            "combo": {"id": None, "name": "Free Training", "seq": ""},
            "config": {"Rounds": "1", "Work Time": "120s", "Rest Time": "30s", "Speed": "Medium (2s)"},
            "difficulty": "Beginner",
            "accent": "#58A6FF",
        },
        {
            "name": "Jab-Cross Drill",
            "tag": "TECHNIQUE",
            "desc": "Classic 1-2 combo drill",
            "route": "training_session",
            "combo": {"id": "beginner_007", "name": "Jab-Cross", "seq": "1-2"},
            "config": {"Rounds": "2", "Work Time": "60s", "Rest Time": "30s", "Speed": "Medium (2s)"},
            "difficulty": "Beginner",
            "accent": "#FF6B35",
        },
        {
            "name": "Power Test",
            "tag": "PERFORMANCE",
            "desc": "Test your max punch force",
            "route": "power_test",
            "combo": {},
            "config": {},
            "difficulty": "",
            "accent": "#FF5C5C",
        },
    ]
