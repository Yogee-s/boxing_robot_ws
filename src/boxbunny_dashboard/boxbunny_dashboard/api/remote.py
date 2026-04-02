"""Remote GUI control endpoints.

Allows the phone dashboard to send commands to the GUI (start training,
open presets, etc.) via a shared command file that the GUI polls.
"""
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from boxbunny_dashboard.api.auth import get_current_user
from boxbunny_dashboard.db.manager import DatabaseManager

logger = logging.getLogger("boxbunny.dashboard.remote")
router = APIRouter()

_CMD_FILE = Path("/tmp/boxbunny_gui_command.json")


class RemoteCommand(BaseModel):
    """A command to send to the GUI."""
    action: str = Field(..., description="start_training | start_preset | open_presets | stop_session")
    config: Dict[str, Any] = Field(default_factory=dict)


class RemoteStatus(BaseModel):
    success: bool
    message: str


def _get_db(request: Request) -> DatabaseManager:
    return request.app.state.db


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


@router.get("/presets")
async def get_user_presets(
    user: dict = Depends(get_current_user),
    db: DatabaseManager = Depends(_get_db),
) -> list:
    """Get the user's saved presets for the remote preset picker."""
    try:
        import sqlite3
        username = user["username"]
        db_path = Path(db._data_dir) / "users" / username / "boxbunny.db"
        if not db_path.exists():
            return _default_presets()
        conn = sqlite3.connect(str(db_path))
        rows = conn.execute("SELECT preset_json FROM presets ORDER BY id").fetchall()
        conn.close()
        return [json.loads(r[0]) for r in rows] if rows else _default_presets()
    except Exception:
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
        },
        {
            "name": "Jab-Cross Drill",
            "tag": "TECHNIQUE",
            "desc": "Classic 1-2 combo drill",
            "route": "training_session",
            "combo": {"id": "beginner_007", "name": "Jab-Cross", "seq": "1-2"},
            "config": {"Rounds": "2", "Work Time": "60s", "Rest Time": "30s", "Speed": "Medium (2s)"},
            "difficulty": "Beginner",
        },
        {
            "name": "Power Test",
            "tag": "PERFORMANCE",
            "desc": "Test your max punch force",
            "route": "power_test",
            "combo": {},
            "config": {},
            "difficulty": "",
        },
    ]
