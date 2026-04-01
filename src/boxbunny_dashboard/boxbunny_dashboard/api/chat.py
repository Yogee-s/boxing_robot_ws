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


class ChatResponse(BaseModel):
    reply: str
    timestamp: str


# ---- Helpers ----

def _get_db(request: Request) -> DatabaseManager:
    return request.app.state.db


def _build_system_prompt(user: dict, context: Dict[str, Any]) -> str:
    """Build a system prompt for the AI boxing coach."""
    return (
        "You are BoxBunny AI Coach, an expert boxing trainer assistant. "
        f"The user's name is {user.get('display_name', 'Boxer')} "
        f"(level: {user.get('level', 'beginner')}). "
        "Provide concise, actionable boxing advice. "
        "Reference their recent training data when available."
    )


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
        logger.debug("ROS LLM service unavailable: %s", exc)

    return ""


async def _call_llm(prompt: str, system_prompt: str) -> str:
    """Call LLM in a thread pool to avoid blocking the async event loop."""
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, _call_llm_sync, prompt, system_prompt,
    )
    if result:
        return result
    return (
        "I'm currently running in offline mode. Connect the LLM service "
        "for personalized coaching feedback. In the meantime, remember: "
        "keep your guard up, rotate your hips on crosses, and breathe!"
    )


# ---- Endpoints ----

@router.post("/message", response_model=ChatResponse)
async def send_message(
    body: ChatRequest,
    user: dict = Depends(get_current_user),
    db: DatabaseManager = Depends(_get_db),
) -> ChatResponse:
    """Send a message to the AI coach and get a response."""
    system_prompt = _build_system_prompt(user, body.context)
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

    return ChatResponse(reply=reply, timestamp=timestamp)


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
