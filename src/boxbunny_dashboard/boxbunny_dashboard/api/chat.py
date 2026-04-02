"""AI Coach chat endpoints for BoxBunny Dashboard.

Proxies messages to the ROS GenerateLlm service (or a direct LLM call)
and returns AI coaching responses. Stores chat history per user.
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field

from boxbunny_dashboard.api.auth import get_current_user
from boxbunny_dashboard.db.manager import DatabaseManager

logger = logging.getLogger("boxbunny.dashboard.chat")
router = APIRouter()


# ---- Pydantic models ----

class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: Optional[str] = None


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    context: Dict[str, Any] = Field(default_factory=dict)


class TrainingAction(BaseModel):
    """An actionable training suggestion embedded in a chat response."""
    label: str  # Button text
    type: str  # "training", "power_test", "reaction_test", "preset"
    config: Dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    reply: str
    timestamp: str
    actions: Optional[List[TrainingAction]] = None


# ---- Helpers ----

def _get_db(request: Request) -> DatabaseManager:
    return request.app.state.db


def _get_user_history(db: DatabaseManager, user: dict) -> str:
    """Fetch recent training sessions for LLM context."""
    try:
        import sqlite3
        from pathlib import Path
        username = user.get("username", "")
        db_path = Path(db._data_dir) / "users" / username / "boxbunny.db"
        if not db_path.exists():
            return "No training history yet."
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT mode, difficulty, started_at, rounds_completed, rounds_total, "
            "work_time_sec, summary_json FROM training_sessions "
            "ORDER BY started_at DESC LIMIT 5"
        ).fetchall()
        conn.close()
        if not rows:
            return "No training sessions recorded yet."
        lines = []
        for r in rows:
            summary = {}
            try:
                summary = json.loads(r["summary_json"] or "{}")
            except Exception:
                pass
            punches = summary.get("total_punches", 0)
            date = r["started_at"][:10] if r["started_at"] else "?"
            lines.append(
                f"- {date}: {r['mode']} ({r['difficulty']}), "
                f"{r['rounds_completed']}/{r['rounds_total']} rounds, "
                f"{punches} punches"
            )
        return "\n".join(lines)
    except Exception:
        return "Could not load training history."


def _build_system_prompt(user: dict, context: Dict[str, Any]) -> str:
    """Build a system prompt for the AI boxing coach."""
    level = user.get('level', 'beginner')
    name = user.get('display_name', 'Boxer')
    return (
        f"You are BoxBunny AI Coach, an expert boxing trainer assistant based on AIBA coaching methodology. "
        f"The user's name is {name} (level: {level}). "
        "You have deep knowledge of boxing techniques, training methods, stance, footwork, "
        "straight punches (jab, cross), hooks, uppercuts, defensive moves (slips, blocks), "
        "combinations, physical conditioning, and fight strategy from multiple schools "
        "(European, Russian, American, Cuban styles). "
        "Provide concise, actionable advice. Be encouraging but technically precise. "
        "\n\n"
        "IMPORTANT: When you suggest a specific training drill, include it as a tag like this:\n"
        "[DRILL:Jab-Cross Drill|combo=1-2|rounds=2|work=60s|speed=Medium (2s)]\n"
        "[DRILL:Power Test|type=power_test]\n"
        "[DRILL:Reaction Time|type=reaction_test]\n"
        "[DRILL:Hook Combo|combo=1-2-3|rounds=3|work=90s|speed=Medium (2s)]\n"
        "ONLY include drill tags when the user specifically asks for a drill, training suggestion, "
        "or says something like 'what should I practice', 'suggest a drill', 'give me a workout', "
        "'what combo should I try', etc. Do NOT suggest drills unprompted — just give advice. "
        "When you do include a drill, explain WHY before the tag."
        "\n\n"
        f"USER'S RECENT TRAINING HISTORY:\n{context.get('history', 'No history available.')}"
    )


def _parse_actions(text: str) -> tuple:
    """Extract [DRILL:...] tags from LLM output and return (clean_text, actions)."""
    import re
    actions = []
    pattern = r'\[DRILL:([^]]+)\]'

    for match in re.finditer(pattern, text):
        tag = match.group(1)
        parts = tag.split('|')
        label = parts[0].strip()
        config = {}
        drill_type = "training"

        for part in parts[1:]:
            if '=' in part:
                k, v = part.split('=', 1)
                k, v = k.strip(), v.strip()
                if k == 'type':
                    drill_type = v
                elif k == 'combo':
                    config['combo_seq'] = v
                elif k == 'rounds':
                    config['Rounds'] = v
                elif k == 'work':
                    config['Work Time'] = v
                elif k == 'speed':
                    config['Speed'] = v

        route_map = {
            "training": "training_session",
            "power_test": "power_test",
            "reaction_test": "reaction_test",
            "stamina_test": "stamina_test",
        }
        config['route'] = route_map.get(drill_type, "training_session")
        config['name'] = label

        actions.append(TrainingAction(
            label=f"Start: {label}",
            type=drill_type,
            config=config,
        ))

    # Remove the tags from the text
    clean_text = re.sub(pattern, '', text).strip()
    # Clean up any double newlines left by tag removal
    clean_text = re.sub(r'\n{3,}', '\n\n', clean_text)

    return clean_text, actions if actions else None


# Singleton ROS node for dashboard LLM calls — avoids leaking nodes
_ros_node = None
_ros_client = None


def _get_ros_llm_client():
    """Get or create a persistent ROS node + LLM service client."""
    global _ros_node, _ros_client  # noqa: PLW0603
    try:
        import rclpy
        from boxbunny_msgs.srv import GenerateLlm

        if not rclpy.ok():
            rclpy.init()
        if _ros_node is None:
            _ros_node = rclpy.create_node("dashboard_llm_client")
            _ros_client = _ros_node.create_client(
                GenerateLlm, "/boxbunny/llm/generate",
            )
            logger.info("Dashboard LLM ROS client created")
        return _ros_node, _ros_client
    except Exception:
        return None, None


def _call_llm_sync(prompt: str, system_prompt: str) -> str:
    """Blocking LLM call via ROS service — run in a thread pool."""
    try:
        import rclpy
        from boxbunny_msgs.srv import GenerateLlm

        node, client = _get_ros_llm_client()
        if node is None or client is None:
            raise RuntimeError("ROS not available")

        if not client.wait_for_service(timeout_sec=5.0):
            logger.warning("LLM service not available within timeout")
            raise RuntimeError("LLM service not available")

        req = GenerateLlm.Request()
        req.prompt = prompt
        req.context_json = json.dumps({"system_prompt": system_prompt})
        req.system_prompt_key = "coach_chat"
        future = client.call_async(req)
        rclpy.spin_until_future_complete(node, future, timeout_sec=30.0)
        if future.result() is not None and future.result().success:
            return future.result().response
        logger.warning("LLM service returned failure or timed out")
    except Exception as exc:
        logger.warning("ROS LLM call failed: %s", exc, exc_info=True)

    return ""


_direct_model = None
_direct_model_loading = False


def _preload_direct_model() -> None:
    """Pre-load the GGUF model only if the ROS LLM service is NOT available."""
    global _direct_model, _direct_model_loading  # noqa: PLW0603
    if _direct_model is not None or _direct_model_loading:
        return
    _direct_model_loading = True
    try:
        # First check if ROS LLM service is available — if so, skip direct load
        import time
        time.sleep(5)  # Wait for ROS nodes to start
        node, client = _get_ros_llm_client()
        if node and client and client.wait_for_service(timeout_sec=3.0):
            logger.info("ROS LLM service available — skipping direct model load")
            _direct_model_loading = False
            return
    except Exception:
        pass

    # ROS not available — load model directly as fallback
    try:
        from llama_cpp import Llama
        from pathlib import Path
        model_path = (
            Path(__file__).resolve().parents[4]
            / "models" / "llm" / "qwen2.5-3b-instruct-q4_k_m.gguf"
        )
        if model_path.exists():
            logger.info("Pre-loading LLM model directly: %s", model_path)
            _direct_model = Llama(
                model_path=str(model_path),
                n_ctx=2048, n_gpu_layers=-1, verbose=False,
            )
            logger.info("LLM model pre-loaded and ready")
    except Exception as exc:
        logger.warning("Failed to pre-load LLM: %s", exc)
    finally:
        _direct_model_loading = False


# Start pre-loading in a background thread on module import
# Only if not already loaded by the ROS LLM node (avoid double-loading)
import threading
try:
    threading.Thread(target=_preload_direct_model, daemon=True).start()
except Exception:
    pass


def _call_llm_direct(prompt: str, system_prompt: str) -> str:
    """Direct LLM call as fallback when ROS is unavailable."""
    global _direct_model  # noqa: PLW0603
    try:
        from llama_cpp import Llama
        from pathlib import Path
        model_path = (
            Path(__file__).resolve().parents[4]
            / "models" / "llm" / "qwen2.5-3b-instruct-q4_k_m.gguf"
        )
        if _direct_model is None:
            if not model_path.exists():
                return ""
            logger.info("Loading LLM model directly: %s", model_path)
            _direct_model = Llama(
                model_path=str(model_path),
                n_ctx=2048, n_gpu_layers=-1, verbose=False,
            )
        resp = _direct_model.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_tokens=128, temperature=0.7,
        )
        return resp["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        logger.warning("Direct LLM call failed: %s", exc)
        return ""


async def _call_llm(prompt: str, system_prompt: str) -> str:
    """Call LLM — tries ROS service first, then direct model, then fallback."""
    loop = asyncio.get_event_loop()
    # Try ROS service
    result = await loop.run_in_executor(
        None, _call_llm_sync, prompt, system_prompt,
    )
    if result:
        return result
    # Try direct model call
    result = await loop.run_in_executor(
        None, _call_llm_direct, prompt, system_prompt,
    )
    if result:
        return result
    return (
        "I'm currently running in offline mode. Connect the LLM service "
        "for personalized coaching feedback. In the meantime, remember: "
        "keep your guard up, rotate your hips on crosses, and breathe!"
    )


# ---- Endpoints ----

@router.get("/status")
async def get_llm_status() -> dict:
    """Check if the LLM is ready to accept messages."""
    # Check ROS service
    try:
        node, client = _get_ros_llm_client()
        if node and client and client.service_is_ready():
            return {"ready": True, "source": "ros"}
    except Exception:
        pass
    # Check direct model
    try:
        from pathlib import Path
        model_path = (
            Path(__file__).resolve().parents[4]
            / "models" / "llm" / "qwen2.5-3b-instruct-q4_k_m.gguf"
        )
        if model_path.exists():
            return {"ready": True, "source": "direct"}
    except Exception:
        pass
    return {"ready": False, "source": "none"}


@router.post("/message", response_model=ChatResponse)
async def send_message(
    body: ChatRequest,
    user: dict = Depends(get_current_user),
    db: DatabaseManager = Depends(_get_db),
) -> ChatResponse:
    """Send a message to the AI coach and get a response."""
    # Fetch user's recent training history for context
    history_context = _get_user_history(db, user)
    context_with_history = {**body.context, "history": history_context}
    system_prompt = _build_system_prompt(user, context_with_history)
    reply = await _call_llm(body.message, system_prompt)
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # Persist both user message and reply in the user's session events
    username = user["username"]
    try:
        db.save_session_event(
            username=username,
            session_id="chat",
            timestamp=time.time(),
            event_type="chat_message",
            data={"role": "user", "content": body.message},
        )
        db.save_session_event(
            username=username,
            session_id="chat",
            timestamp=time.time(),
            event_type="chat_message",
            data={"role": "assistant", "content": reply},
        )
    except Exception:
        logger.warning("Failed to persist chat message for %s", username)

    # Parse action cards from the LLM output
    clean_reply, actions = _parse_actions(reply)

    return ChatResponse(reply=clean_reply, timestamp=timestamp, actions=actions)


@router.get("/history", response_model=List[ChatMessage])
async def get_chat_history(
    user: dict = Depends(get_current_user),
    db: DatabaseManager = Depends(_get_db),
    limit: int = Query(default=50, ge=1, le=200),
) -> List[ChatMessage]:
    """Return recent chat history for the authenticated user."""
    username = user["username"]
    detail = db.get_session_detail(username, "chat")
    if detail is None:
        return []

    events = detail.get("events", [])
    chat_events = [
        e for e in events
        if e.get("event_type") == "chat_message"
    ]
    messages: List[ChatMessage] = []
    for e in chat_events[-limit:]:
        data = e.get("data_json", "{}")
        if isinstance(data, str):
            data = json.loads(data)
        messages.append(ChatMessage(
            role=data.get("role", "user"),
            content=data.get("content", ""),
            timestamp=str(e.get("timestamp", "")),
        ))
    return messages
