import os
import random
from typing import Dict, List

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from boxbunny_msgs.msg import PunchEvent, DrillEvent, TrashTalk
from boxbunny_msgs.srv import GenerateLLM

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


class TrashTalkNode(Node):
    def __init__(self) -> None:
        super().__init__("trash_talk_node")

        self.declare_parameter("use_llm_if_available", True)
        self.declare_parameter("model_path", "")
        self.declare_parameter("max_tokens", 32)
        self.declare_parameter("temperature", 0.7)
        self.declare_parameter("mode", "coach")
        self.declare_parameter("persona_examples_path", "")
        self.declare_parameter("dataset_path", "")
        self.declare_parameter("use_stats_context", True)
        self.declare_parameter("n_ctx", 512)
        self.declare_parameter("n_threads", 4)

        self.pub = self.create_publisher(TrashTalk, "trash_talk", 10)
        self.punch_sub = self.create_subscription(PunchEvent, "punch_events", self._on_punch, 10)
        self.drill_sub = self.create_subscription(DrillEvent, "drill_events", self._on_drill_event, 10)
        self.summary_sub = self.create_subscription(String, "drill_summary", self._on_summary, 10)
        self.stats_sub = self.create_subscription(String, "punch_stats", self._on_stats, 10)
        self.srv = self.create_service(GenerateLLM, "llm/generate", self._on_generate)

        self._llm = None
        self._persona_examples = self._load_persona_examples()
        self._dataset_examples = self._load_dataset_examples()
        self._stats_context = ""
        self._init_llm()

        self.templates = {
            "coach": [
                "Nice jab. Keep your guard up.",
                "Good timing. Reset and breathe.",
                "That was sharp—stay balanced.",
                "Quick reaction. Now do it again.",
            ],
            "encourage": [
                "You got this—stay loose.",
                "Great effort. Keep pushing.",
                "That was fast! Keep the rhythm.",
                "Strong work. Next one’s even better.",
            ],
            "trash": [
                "Fast punch! For a human.",
                "Nice swing. Did you plan that?",
                "I’ve seen snails dodge faster.",
                "That one had some spice. Not bad.",
            ],
            "analysis": [
                "Your straights are clean—mix in hooks to balance.",
                "Good pace. Watch your guard on the return.",
                "Solid volume, but work on consistent accuracy.",
                "Your velocity is good; keep your stance stable.",
            ],
        }

        self.get_logger().info("LLM coach node ready")

    def _load_persona_examples(self) -> Dict[str, List[Dict[str, str]]]:
        path = self.get_parameter("persona_examples_path").value
        if not path or not yaml or not os.path.exists(path):
            return {}
        try:
            with open(path, "r") as f:
                data = yaml.safe_load(f) or {}
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _load_dataset_examples(self) -> Dict[str, List[Dict[str, str]]]:
        path = self.get_parameter("dataset_path").value
        if not path or not yaml or not os.path.exists(path):
            return {}
        try:
            with open(path, "r") as f:
                data = yaml.safe_load(f) or {}
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _init_llm(self) -> None:
        if not self.get_parameter("use_llm_if_available").value:
            return

        model_path = self.get_parameter("model_path").value
        if not model_path or not os.path.exists(model_path):
            self.get_logger().warn("LLM model not found. Using fallback templates.")
            return
        try:
            from llama_cpp import Llama  # type: ignore

            self._llm = Llama(
                model_path=model_path,
                n_ctx=int(self.get_parameter("n_ctx").value),
                n_threads=int(self.get_parameter("n_threads").value),
            )
            self.get_logger().info("LLM loaded")
        except Exception as exc:  # pragma: no cover
            self.get_logger().warn(f"LLM load failed: {exc}")

    def _on_punch(self, msg: PunchEvent) -> None:
        line = self._generate_line("coach", f"punch:{msg.glove}:{msg.punch_type or 'unknown'}")
        self._publish(line)

    def _on_drill_event(self, msg: DrillEvent) -> None:
        if msg.event_type == "punch_detected":
            line = self._generate_line("coach", f"reaction:{msg.value:.2f}")
            self._publish(line)

    def _on_summary(self, msg: String) -> None:
        if random.random() < 0.15:
            line = self._generate_line("coach", "summary")
            self._publish(line)

    def _on_stats(self, msg: String) -> None:
        self._stats_context = msg.data

    def _on_generate(self, request, response):
        mode = request.mode or self.get_parameter("mode").value
        response.response = self._generate_line(mode, request.context or request.prompt, request.prompt)
        return response

    def _generate_line(self, mode: str, context: str, prompt: str = "") -> str:
        mode = mode if mode in self.templates else "coach"
        if self._llm is None:
            return random.choice(self.templates[mode])

        examples = self._persona_examples.get(mode, [])
        dataset_examples = self._dataset_examples.get(mode, [])
        example_lines = "\n".join(
            [f"User: {ex.get('user','')}\nCoach: {ex.get('assistant','')}" for ex in examples][:6]
        )
        dataset_lines = "\n".join(
            [f"User: {ex.get('prompt','')}\nCoach: {ex.get('response','')}" for ex in dataset_examples][:6]
        )

        prompt_text = "You are a playful boxing trainer robot. Reply with one short sentence."
        prompt_text += f" Style: {mode}. Context: {context}.\n"
        if self.get_parameter("use_stats_context").value and self._stats_context:
            prompt_text += f"Stats: {self._stats_context}\n"
        if example_lines:
            prompt_text += f"\n{example_lines}\n"
        if dataset_lines:
            prompt_text += f"\n{dataset_lines}\n"
        if prompt:
            prompt_text += f"User: {prompt}\nCoach:"

        result = self._llm(
            prompt_text,
            max_tokens=int(self.get_parameter("max_tokens").value),
            temperature=float(self.get_parameter("temperature").value),
            stop=["\n"],
        )
        text = result["choices"][0]["text"].strip()
        return text if text else random.choice(self.templates[mode])

    def _publish(self, text: str) -> None:
        msg = TrashTalk()
        msg.stamp = self.get_clock().now().to_msg()
        msg.text = text
        self.pub.publish(msg)


def main() -> None:
    rclpy.init()
    node = TrashTalkNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
