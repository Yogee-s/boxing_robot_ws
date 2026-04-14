"""Local LLM AI Coach node for BoxBunny.

Hosts a local LLM on the Jetson GPU for real-time coaching tips,
post-session analysis, and chat. Uses llama-cpp-python for inference.
Degrades gracefully if model unavailable — serves pre-written fallback tips.
"""

import base64
import json
import logging
import os
import random
import tempfile
import threading
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

from boxbunny_core.constants import Services, Topics

logger = logging.getLogger("boxbunny.llm_node")


SYSTEM_PROMPT = """You are BoxBunny AI Coach, an expert boxing trainer built into a boxing training robot. Your knowledge is based on the AIBA Coaches Manual and professional boxing coaching methodology.

Key traits:
- Deep knowledge of boxing technique: stance, footwork, all 6 basic punches (jab=1, cross=2, L hook=3, R hook=4, L uppercut=5, R uppercut=6), combinations, defenses (slip, block, bob-and-weave)
- Expert in training methodology: initiation, basic, specialization, and high-performance stages
- Knows European, Russian, American, and Cuban boxing styles
- Adjusts advice to the user's skill level (beginner/intermediate/advanced)
- Safety-focused: always prioritize proper form to prevent injury
- Encouraging but honest about areas needing improvement
- Keep tips SHORT (1-2 sentences max for real-time tips, 2-3 paragraphs for analysis)
- Always finish your sentences completely. Never stop mid-thought.
- Reference specific punch types and stats when available
- Do NOT suggest drills or training sessions unless the user explicitly asks for one
- When movement data is provided (avg_depth, lateral_movement, max_lateral_displacement, direction_summary):
  Analyze the user's positioning and footwork. In boxing:
  * Consistent lateral movement avoids being a stationary target
  * Favouring one side (e.g. spending 70%+ of time on the right) exposes you to hooks and crosses from that direction — the opponent can herd you into a corner
  * Good depth management (varied in/out movement, depth_range > 0.3m) shows ring awareness
  * Staying too far back (avg_depth > 2.0m) limits power punch effectiveness
  * Crowding in too close (avg_depth < 0.8m) makes you vulnerable to uppercuts and clinches
  * A narrow lateral range suggests the user is planted — encourage movement drills
  Give specific, actionable footwork and positioning advice based on the numbers.
- NEVER use markdown formatting like ** or * or # in your replies. Write in plain text only.
"""

TIP_INTERVAL_S = 18.0  # Seconds between coaching tips
IMAGE_INFERENCE_TIMEOUT_S = 40.0  # Longer timeout for vision inference


class LlmNode(Node):
    """ROS 2 node for local LLM AI coaching."""

    def __init__(self) -> None:
        super().__init__("llm_node")

        # Parameters
        self.declare_parameter("model_path", "")
        self.declare_parameter("mmproj_path", "")
        self.declare_parameter("n_gpu_layers", -1)
        self.declare_parameter("n_ctx", 2048)
        self.declare_parameter("max_tokens", 200)
        self.declare_parameter("temperature", 0.7)
        self.declare_parameter("fallback_tips_path", "")

        self._model_path = self.get_parameter("model_path").value
        self._mmproj_path = self.get_parameter("mmproj_path").value
        self._n_gpu_layers = self.get_parameter("n_gpu_layers").value
        self._n_ctx = self.get_parameter("n_ctx").value
        self._max_tokens = self.get_parameter("max_tokens").value
        self._temperature = self.get_parameter("temperature").value
        fallback_path = self.get_parameter("fallback_tips_path").value

        # State
        self._llm = None
        self._chat_handler = None  # Vision handler, lazy-loaded
        self._vision_available = False
        self._available = False
        self._session_active = False
        self._session_punches = 0
        self._session_mode = ""
        self._last_tip_time = 0.0
        self._recent_events: List[str] = []
        self._consecutive_failures: int = 0
        self._last_success_time: float = time.time()
        self._retry_timer = None

        # Load fallback tips
        self._fallback_tips = self._load_fallback_tips(fallback_path)

        # Subscribers
        self.create_subscription(
            SessionState, Topics.SESSION_STATE, self._on_session_state, 10
        )
        self.create_subscription(
            ConfirmedPunch, Topics.PUNCH_CONFIRMED, self._on_punch, 10
        )
        self.create_subscription(
            DrillEvent, Topics.DRILL_EVENT, self._on_drill_event, 10
        )
        self.create_subscription(
            SessionPunchSummary, Topics.PUNCH_SESSION_SUMMARY,
            self._on_session_summary, 10
        )

        # Publisher
        self._pub_tip = self.create_publisher(CoachTip, Topics.COACH_TIP, 10)

        # Service
        self.create_service(GenerateLlm, Services.GENERATE_LLM, self._handle_generate)

        # Tip timer
        self.create_timer(3.0, self._tip_tick)

        # Pre-load the model on a timer so it's ready before the first request
        self._preload_timer = self.create_timer(2.0, self._preload_model)

        logger.info("LLM node initialized (model=%s)", self._model_path or "none")

    def _preload_model(self) -> None:
        """Pre-load the LLM model at startup so it's ready for requests."""
        self._preload_timer.cancel()  # Only run once
        logger.info("Pre-loading LLM model...")
        if self._lazy_load_model():
            logger.info("LLM model pre-loaded and ready")
        else:
            logger.warning("LLM model pre-load failed")

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
                n_threads=6,
                flash_attn=True,
                verbose=False,
            )
            self._available = True
            logger.info("LLM model loaded: %s", self._model_path)
            return True
        except Exception as e:
            logger.error("Failed to load LLM: %s", e)
            self._available = False
            self._schedule_retry()
            return False

    def _schedule_retry(self) -> None:
        """Schedule a single retry attempt in 30 seconds if not already scheduled."""
        if self._retry_timer is not None:
            return
        logger.info("Scheduling LLM model reload retry in 30s")
        self._retry_timer = self.create_timer(30.0, self._retry_load)

    def _retry_load(self) -> None:
        """Attempt to reload the LLM model once."""
        if self._retry_timer is not None:
            self._retry_timer.cancel()
            self._retry_timer = None
        if self._available and self._llm is not None:
            return
        logger.info("Retrying LLM model load...")
        self._llm = None
        if self._lazy_load_model():
            logger.info("LLM model reload succeeded")
        else:
            logger.warning("LLM model reload failed — will retry again in 30s")

    def _check_reload(self) -> None:
        """Attempt model reload after 3 consecutive failures."""
        if self._consecutive_failures >= 3:
            logger.warning("3 consecutive LLM failures — attempting model reload")
            self._consecutive_failures = 0
            self._available = False
            self._llm = None
            try:
                if not self._lazy_load_model():
                    self._schedule_retry()
            except Exception as e:
                logger.error("Model reload failed: %s", e)
                self._schedule_retry()

    def _lazy_load_vision(self) -> bool:
        """Lazy-load the multimodal vision projector on first image request.

        This is separate from _lazy_load_model() to avoid loading the ~1GB
        mmproj into VRAM until a user actually sends an image via chat.
        Real-time coaching tips never call this method.
        """
        if self._chat_handler is not None:
            return True
        if not self._mmproj_path or not os.path.exists(self._mmproj_path):
            logger.info("Vision projector not found at %s", self._mmproj_path)
            return False
        try:
            from llama_cpp.llama_chat_format import Llava15ChatHandler
            self._chat_handler = Llava15ChatHandler(
                clip_model_path=self._mmproj_path, verbose=False,
            )
            self._vision_available = True
            logger.info("Vision projector loaded: %s", self._mmproj_path)
            return True
        except Exception as e:
            logger.error("Failed to load vision projector: %s", e)
            self._vision_available = False
            return False

    def _generate_with_image(
        self, prompt: str, image_b64: str, system: str = "", max_tokens: int = 0,
    ) -> str:
        """Generate text from an image + text prompt. Only used by dashboard chat.

        Loads the vision projector lazily on first call. Uses a longer timeout
        (40s) than text-only generation (20s) since image encoding is slower.
        """
        if not self._lazy_load_model():
            return ""
        if not self._lazy_load_vision():
            return self._generate(prompt + "\n(Image was attached but vision is unavailable)", system, max_tokens)
        if max_tokens <= 0:
            max_tokens = self._max_tokens

        try:
            result = [None]
            error = [None]

            def _infer() -> None:
                try:
                    data_uri = f"data:image/jpeg;base64,{image_b64}"
                    messages = []
                    if system:
                        messages.append({"role": "system", "content": system})
                    messages.append({
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": data_uri}},
                            {"type": "text", "text": prompt},
                        ],
                    })
                    # Set vision handler on model for this call, then restore
                    self._llm.chat_handler = self._chat_handler
                    try:
                        result[0] = self._llm.create_chat_completion(
                            messages=messages,
                            max_tokens=max_tokens,
                            temperature=self._temperature,
                        )
                    finally:
                        self._llm.chat_handler = None
                except Exception as e:
                    error[0] = e
                    logger.warning("Vision inference error: %s", e)

            t = threading.Thread(target=_infer, daemon=True)
            t.start()
            t.join(timeout=IMAGE_INFERENCE_TIMEOUT_S)

            if t.is_alive():
                logger.warning("Vision inference timed out (%.0fs)", IMAGE_INFERENCE_TIMEOUT_S)
                return ""

            if error[0] is not None or result[0] is None:
                return ""

            text = result[0]["choices"][0]["message"]["content"].strip()
            return self._clean_markdown(text)
        except Exception as e:
            logger.warning("Vision generation failed: %s", e)
            return ""

    def _generate(self, prompt: str, system: str = "", max_tokens: int = 0) -> str:
        """Generate text from the LLM with a 20-second timeout."""
        if not self._lazy_load_model():
            return ""
        if max_tokens <= 0:
            max_tokens = self._max_tokens
        try:
            result = [None]
            error = [None]

            def _infer() -> None:
                try:
                    messages = []
                    if system:
                        messages.append({"role": "system", "content": system})
                    messages.append({"role": "user", "content": prompt})
                    result[0] = self._llm.create_chat_completion(
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=self._temperature,
                    )
                except Exception as e:
                    error[0] = e
                    logger.warning("LLM inference error: %s", e)

            t = threading.Thread(target=_infer, daemon=True)
            t.start()
            t.join(timeout=20.0)

            if t.is_alive():
                logger.warning("LLM inference timed out (20s)")
                self._consecutive_failures += 1
                self._check_reload()
                return ""

            if error[0] is not None or result[0] is None:
                self._consecutive_failures += 1
                self._check_reload()
                return ""

            self._consecutive_failures = 0
            self._last_success_time = time.time()
            text = result[0]["choices"][0]["message"]["content"].strip()
            return self._clean_markdown(text)
        except Exception as e:
            logger.warning("LLM generation failed: %s", e)
            self._consecutive_failures += 1
            self._check_reload()
            return ""

    @staticmethod
    def _clean_markdown(text: str) -> str:
        """Strip markdown formatting that small LLMs produce despite instructions."""
        import re
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)  # **bold** → bold
        text = re.sub(r'\*(.+?)\*', r'\1', text)       # *italic* → italic
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)  # # headers
        text = re.sub(r'^\s*[-*]\s+', '- ', text, flags=re.MULTILINE)  # normalize bullets
        return text.strip()

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
            # Build context from actual recent events
            recent = self._recent_events[-10:] if self._recent_events else []
            punch_types = [e.split(":")[1] for e in recent if e.startswith("punch:")]
            misses = sum(1 for e in recent if e == "combo_missed")
            hits = [e for e in recent if e.startswith("combo_ok")]

            parts = [f"Mode: {self._session_mode}, Total punches so far: {self._session_punches}."]
            if punch_types:
                parts.append(f"Recent punches: {', '.join(punch_types[-5:])}.")
            if misses:
                parts.append(f"Missed {misses} combos recently.")
            if hits:
                parts.append(f"Completed combos recently: {', '.join(hits[-3:])}.")
            if not punch_types and self._session_punches == 0:
                parts.append("No punches thrown yet.")

            context = " ".join(parts)
            prompt = (
                f"Give a brief 1-sentence coaching tip based on this live data. "
                f"Only reference what the data shows. {context}"
            )
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
        """Handle LLM generation service request.

        Routes to vision inference if context_json contains image_base64,
        otherwise uses fast text-only generation. Chat requests from the
        dashboard get a higher max_tokens (256) so responses finish
        naturally without mid-sentence cutoffs.
        """
        start = time.time()
        system = SYSTEM_PROMPT
        if request.system_prompt_key == "general":
            system = SYSTEM_PROMPT
        elif request.system_prompt_key:
            system = SYSTEM_PROMPT + f"\nContext: {request.system_prompt_key}"

        context_raw = request.context_json or "{}"
        try:
            context = json.loads(context_raw)
        except json.JSONDecodeError:
            context = {}

        # Use the dashboard's custom system prompt if provided
        if "system_prompt" in context:
            system = context.pop("system_prompt")

        image_b64 = context.pop("image_base64", None)
        prompt = f"{request.prompt}\n\nUser data: {json.dumps(context)}"

        # Chat requests get more tokens so replies finish naturally.
        # Adjust based on reply_depth preference from the dashboard.
        is_chat = request.system_prompt_key == "coach_chat"
        if is_chat:
            reply_depth = context.pop("reply_depth", "normal")
            depth_tokens = {"short": 100, "normal": 256, "detailed": 512}
            chat_max_tokens = depth_tokens.get(reply_depth, 256)
        else:
            chat_max_tokens = self._max_tokens

        if image_b64:
            text = self._generate_with_image(
                prompt, image_b64, system, max_tokens=chat_max_tokens,
            )
        else:
            text = self._generate(prompt, system, max_tokens=chat_max_tokens)

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
