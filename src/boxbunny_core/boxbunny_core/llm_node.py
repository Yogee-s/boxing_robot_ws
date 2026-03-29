"""Local LLM AI Coach node for BoxBunny.

Hosts a local LLM on the Jetson GPU for real-time coaching tips,
post-session analysis, and chat. Uses llama-cpp-python for inference.
Degrades gracefully if model unavailable — serves pre-written fallback tips.
"""

import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

import rclpy
from rclpy.node import Node

from boxbunny_msgs.msg import (
    CoachTip,
    ConfirmedPunch,
    DrillEvent,
    SessionPunchSummary,
    SessionState,
)
from boxbunny_msgs.srv import GenerateLlm

logger = logging.getLogger("boxbunny.llm_node")


SYSTEM_PROMPT = """You are BoxBunny AI Coach, an expert boxing trainer built into a boxing training robot. You provide concise, actionable coaching feedback.

Key traits:
- Expert knowledge of boxing technique, combinations, footwork, and defense
- Adjusts language complexity to the user's skill level
- Safety-focused: always prioritize proper form to prevent injury
- Encouraging but honest about areas needing improvement
- Keep tips SHORT (1-2 sentences max for real-time tips, 2-3 paragraphs for analysis)
- Reference specific punch types and stats when available
- Use boxing terminology naturally but explain jargon for beginners
"""

TIP_INTERVAL_S = 18.0  # Seconds between coaching tips


class LlmNode(Node):
    """ROS 2 node for local LLM AI coaching."""

    def __init__(self) -> None:
        super().__init__("llm_node")

        # Parameters
        self.declare_parameter("model_path", "")
        self.declare_parameter("n_gpu_layers", -1)
        self.declare_parameter("n_ctx", 2048)
        self.declare_parameter("max_tokens", 128)
        self.declare_parameter("temperature", 0.7)
        self.declare_parameter("fallback_tips_path", "")

        self._model_path = self.get_parameter("model_path").value
        self._n_gpu_layers = self.get_parameter("n_gpu_layers").value
        self._n_ctx = self.get_parameter("n_ctx").value
        self._max_tokens = self.get_parameter("max_tokens").value
        self._temperature = self.get_parameter("temperature").value
        fallback_path = self.get_parameter("fallback_tips_path").value

        # State
        self._llm = None
        self._available = False
        self._session_active = False
        self._session_punches = 0
        self._session_mode = ""
        self._last_tip_time = 0.0
        self._recent_events: List[str] = []

        # Load fallback tips
        self._fallback_tips = self._load_fallback_tips(fallback_path)

        # Subscribers
        self.create_subscription(
            SessionState, "/boxbunny/session/state", self._on_session_state, 10
        )
        self.create_subscription(
            ConfirmedPunch, "/boxbunny/punch/confirmed", self._on_punch, 10
        )
        self.create_subscription(
            DrillEvent, "/boxbunny/drill/event", self._on_drill_event, 10
        )
        self.create_subscription(
            SessionPunchSummary, "/boxbunny/punch/session_summary",
            self._on_session_summary, 10
        )

        # Publisher
        self._pub_tip = self.create_publisher(CoachTip, "/boxbunny/coach/tip", 10)

        # Service
        self.create_service(GenerateLlm, "/boxbunny/llm/generate", self._handle_generate)

        # Tip timer
        self.create_timer(3.0, self._tip_tick)

        logger.info("LLM node initialized (model=%s)", self._model_path or "none")

    def _load_fallback_tips(self, path: str) -> Dict[str, List[str]]:
        """Load pre-written fallback tips from JSON."""
        if not path:
            ws_root = Path(__file__).resolve().parents[3]
            path = str(ws_root / "config" / "fallback_tips.json")
        try:
            with open(path, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning("Fallback tips not loaded: %s", e)
            return {
                "technique": ["Keep your guard up between punches."],
                "encouragement": ["Great work! Keep pushing."],
                "correction": ["Watch your form on hooks."],
                "suggestion": ["Try mixing up your combinations."],
            }

    def _lazy_load_model(self) -> bool:
        """Lazy-load the LLM model on first use."""
        if self._llm is not None:
            return True
        if not self._model_path or not os.path.exists(self._model_path):
            logger.info("LLM model not found at %s — using fallback tips", self._model_path)
            return False
        try:
            from llama_cpp import Llama
            self._llm = Llama(
                model_path=self._model_path,
                n_gpu_layers=self._n_gpu_layers,
                n_ctx=self._n_ctx,
                verbose=False,
            )
            self._available = True
            logger.info("LLM model loaded: %s", self._model_path)
            return True
        except Exception as e:
            logger.error("Failed to load LLM: %s", e)
            return False

    def _generate(self, prompt: str, system: str = "", max_tokens: int = 0) -> str:
        """Generate text from the LLM."""
        if not self._lazy_load_model():
            return ""
        if max_tokens <= 0:
            max_tokens = self._max_tokens
        try:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            result = self._llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=self._temperature,
            )
            return result["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.error("LLM generation failed: %s", e)
            return ""

    def _get_fallback_tip(self, tip_type: str = "technique") -> str:
        """Get a random fallback tip when LLM is unavailable."""
        tips = self._fallback_tips.get(tip_type, self._fallback_tips.get("technique", []))
        return random.choice(tips) if tips else "Keep training!"

    def _on_session_state(self, msg: SessionState) -> None:
        """Track session state for tip timing."""
        self._session_active = msg.state in ("active", "rest")
        self._session_mode = msg.mode
        if msg.state == "active":
            self._session_punches = 0
            self._recent_events.clear()

    def _on_punch(self, msg: ConfirmedPunch) -> None:
        """Track punch events for context."""
        if self._session_active:
            self._session_punches += 1
            self._recent_events.append(f"punch:{msg.punch_type}")
            if len(self._recent_events) > 20:
                self._recent_events.pop(0)

    def _on_drill_event(self, msg: DrillEvent) -> None:
        """Track drill events for tips."""
        if msg.event_type == "combo_missed":
            self._recent_events.append("combo_missed")
        elif msg.event_type == "combo_completed":
            self._recent_events.append(f"combo_ok:acc={msg.accuracy:.0%}")

    def _on_session_summary(self, msg: SessionPunchSummary) -> None:
        """Generate post-session analysis when summary is received."""
        if msg.total_punches == 0:
            return
        tip = self._generate_session_analysis(msg)
        if tip:
            self._publish_tip(tip, "suggestion", "session_end", priority=2)

    def _tip_tick(self) -> None:
        """Periodically generate coaching tips during active sessions."""
        if not self._session_active:
            return
        now = time.time()
        if now - self._last_tip_time < TIP_INTERVAL_S:
            return
        self._last_tip_time = now

        # Determine tip type based on recent events
        tip_type = "technique"
        trigger = "periodic"
        missed_count = sum(1 for e in self._recent_events if e == "combo_missed")
        if missed_count >= 2:
            tip_type = "correction"
            trigger = "low_accuracy"
        elif self._session_punches > 50:
            tip_type = "encouragement"
            trigger = "milestone"

        # Try LLM first, fall back to pre-written tips
        if self._available:
            context = f"Mode: {self._session_mode}, Punches: {self._session_punches}"
            prompt = f"Give a brief 1-sentence coaching tip. {context}"
            tip_text = self._generate(prompt, SYSTEM_PROMPT, max_tokens=50)
        else:
            tip_text = ""

        if not tip_text:
            tip_text = self._get_fallback_tip(tip_type)

        self._publish_tip(tip_text, tip_type, trigger)

    def _generate_session_analysis(self, summary: SessionPunchSummary) -> str:
        """Generate a brief post-session analysis."""
        stats = f"Punches: {summary.total_punches}, Defense rate: {summary.defense_rate:.0%}"
        if self._available:
            prompt = (
                f"Analyze this boxing session briefly (1-2 sentences for robot screen).\n"
                f"{stats}\nDistribution: {summary.punch_distribution_json}"
            )
            return self._generate(prompt, SYSTEM_PROMPT, max_tokens=80)
        return f"Session complete: {summary.total_punches} punches thrown."

    def _publish_tip(
        self, text: str, tip_type: str, trigger: str, priority: int = 1
    ) -> None:
        """Publish a coaching tip."""
        msg = CoachTip()
        msg.timestamp = time.time()
        msg.tip_text = text
        msg.tip_type = tip_type
        msg.trigger = trigger
        msg.priority = priority
        self._pub_tip.publish(msg)
        logger.debug("Coach tip [%s/%s]: %s", tip_type, trigger, text[:60])

    def _handle_generate(
        self, request: GenerateLlm.Request, response: GenerateLlm.Response
    ) -> GenerateLlm.Response:
        """Handle LLM generation service request."""
        start = time.time()
        system = SYSTEM_PROMPT
        if request.system_prompt_key == "general":
            system = SYSTEM_PROMPT
        elif request.system_prompt_key:
            system = SYSTEM_PROMPT + f"\nContext: {request.system_prompt_key}"

        context = request.context_json or "{}"
        prompt = f"{request.prompt}\n\nUser data: {context}"

        text = self._generate(prompt, system)
        if text:
            response.success = True
            response.response = text
        else:
            response.success = False
            response.response = "AI Coach is currently unavailable."
        response.generation_time_sec = time.time() - start
        return response


def main(args=None) -> None:
    """Entry point for the LLM node."""
    rclpy.init(args=args)
    node = LlmNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
