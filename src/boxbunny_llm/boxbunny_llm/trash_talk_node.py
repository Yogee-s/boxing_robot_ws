import os
import sys
import site
import threading
from typing import Dict, List

# Add user site-packages to path (llama_cpp may be installed there)
try:
    user_site = site.getusersitepackages()
    if user_site and user_site not in sys.path:
        sys.path.insert(0, user_site)
except Exception:
    pass

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from boxbunny_msgs.msg import PunchEvent, DrillEvent, TrashTalk
from boxbunny_msgs.srv import GenerateLLM

try:
    import yaml
except Exception:
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
        self.declare_parameter("n_batch", 128)
        self.declare_parameter("singlish", False)
        self.declare_parameter("advice", False)
        self.declare_parameter("memory", False)
        self.declare_parameter("history_turns", 4)
        self.declare_parameter("system_prompt", "You are a helpful boxing coach. Give brief, actionable advice. One sentence only.")
        self.declare_parameter(
            "singlish_prompt_path",
            "/home/boxbunny/Desktop/doomsday_integration/boxing_robot_ws/src/boxbunny_llm/config/singlish_prompt.txt",
        )

        self.pub = self.create_publisher(TrashTalk, "trash_talk", 10)
        self.punch_sub = self.create_subscription(PunchEvent, "punch_events", self._on_punch, 10)
        self.drill_sub = self.create_subscription(DrillEvent, "drill_events", self._on_drill_event, 10)
        self.summary_sub = self.create_subscription(String, "drill_summary", self._on_summary, 10)
        self.stats_sub = self.create_subscription(String, "punch_stats", self._on_stats, 10)
        self.srv = self.create_service(GenerateLLM, "llm/generate", self._on_generate)
        self.stream_pub = self.create_publisher(String, "llm/stream", 10)

        self._llm = None
        self._llm_lock = threading.Lock()
        self._reload_inflight = False
        self._persona_examples = self._load_persona_examples()
        self._dataset_examples = self._load_dataset_examples()
        self._stats_context = ""
        self._singlish_prompt = self._load_singlish_prompt()
        self._history = []
        self._init_llm()
        self.add_on_set_parameters_callback(self._on_params)

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

    def _load_singlish_prompt(self) -> str:
        path = self.get_parameter("singlish_prompt_path").value
        if path and os.path.exists(path):
            try:
                with open(path, "r") as f:
                    return f.read().strip()
            except Exception:
                return "Use Singlish tone."
        return "Use Singlish tone."

    def _init_llm(self) -> None:
        if not self.get_parameter("use_llm_if_available").value:
            self.get_logger().warn("LLM disabled by parameter.")
            return

        model_path = self.get_parameter("model_path").value
        if not model_path or not os.path.exists(model_path):
            self.get_logger().warn(f"LLM model not found at: {model_path}")
            return
            
        # Try to import llama_cpp
        try:
            from llama_cpp import Llama
            self.get_logger().info(f"Loading LLM from: {model_path}")
            self._llm = Llama(
                model_path=model_path,
                n_ctx=int(self.get_parameter("n_ctx").value),
                n_threads=int(self.get_parameter("n_threads").value),
                n_batch=int(self.get_parameter("n_batch").value),
                verbose=False,
            )
            self.get_logger().info("LLM loaded successfully!")
        except ImportError as e:
            self.get_logger().error(f"llama_cpp not found: {e}")
            self.get_logger().error(f"Python path: {sys.path[:3]}...")
        except Exception as exc:
            self.get_logger().error(f"LLM load failed: {exc}")

    def _schedule_reload(self) -> None:
        if self._reload_inflight:
            return
        self._reload_inflight = True

        def _do_reload():
            with self._llm_lock:
                self._llm = None
                self._init_llm()
            self._reload_inflight = False

        threading.Thread(target=_do_reload, daemon=True).start()

    def _on_params(self, params):
        reload_needed = False
        for param in params:
            if param.name in ("n_ctx", "n_threads", "n_batch", "model_path", "use_llm_if_available"):
                reload_needed = True
            if param.name in ("persona_examples_path", "dataset_path"):
                # Reload prompt datasets if paths change
                self._persona_examples = self._load_persona_examples()
                self._dataset_examples = self._load_dataset_examples()
            if param.name in ("singlish_prompt_path",):
                self._singlish_prompt = self._load_singlish_prompt()
        if reload_needed:
            self._schedule_reload()
        return rclpy.parameter.SetParametersResult(successful=True)

    def _on_punch(self, msg: PunchEvent) -> None:
        if self._llm is not None:
            line = self._generate_line("coach", f"punch:{msg.glove}:{msg.punch_type or 'unknown'}")
            if line:
                self._publish(line)

    def _on_drill_event(self, msg: DrillEvent) -> None:
        if self._llm is not None and msg.event_type == "punch_detected":
            line = self._generate_line("coach", f"reaction:{msg.value:.2f}")
            if line:
                self._publish(line)

    def _on_summary(self, msg: String) -> None:
        if self._llm is not None:
            line = self._generate_line("coach", "summary")
            if line:
                self._publish(line)

    def _on_stats(self, msg: String) -> None:
        self._stats_context = msg.data

    def _on_generate(self, request, response):
        """Service handler - returns LLM response or 'not loaded' message."""
        if self._llm is None:
            response.response = "LLM not loaded"
            return response
        
        mode = request.mode or self.get_parameter("mode").value
        result = self._generate_line(mode, request.context or request.prompt, request.prompt)
        response.response = result if result else "No response"
        return response

    def _generate_line(self, mode: str, context: str, prompt: str = "") -> str:
        """Generate LLM response. Returns empty string if LLM not available."""
        if self._llm is None:
            return ""

        examples = self._persona_examples.get(mode, [])
        dataset_examples = self._dataset_examples.get(mode, [])
        example_lines = "\n".join(
            [f"User: {ex.get('user','')}\nCoach: {ex.get('assistant','')}" for ex in examples][:6]
        )
        dataset_lines = "\n".join(
            [f"User: {ex.get('prompt','')}\nCoach: {ex.get('response','')}" for ex in dataset_examples][:6]
        )

        prompt_text = self.get_parameter("system_prompt").value
        prompt_text += f" Style: {mode}.\n"
        if self.get_parameter("singlish").value:
            prompt_text += f"\n{self._singlish_prompt}\n"
        if self.get_parameter("advice").value:
            prompt_text += (
                "\nProvide practical boxing advice and training tips."
                " Avoid medical or injury diagnosis."
            )
        if self.get_parameter("use_stats_context").value and self._stats_context:
            prompt_text += f"Stats: {self._stats_context}\n"
        if self.get_parameter("memory").value:
            turns = int(self.get_parameter("history_turns").value)
            with self._llm_lock:
                history = self._history[-turns * 2 :]
            if history:
                history_text = "\n".join(
                    [f"{item['role']}: {item['text']}" for item in history]
                )
                prompt_text += f"\n{history_text}\n"
        if example_lines:
            prompt_text += f"\n{example_lines}\n"
        if dataset_lines:
            prompt_text += f"\n{dataset_lines}\n"
        if prompt:
            prompt_text += f"User: {prompt}\nCoach:"

        try:
            # Enable streaming
            with self._llm_lock:
                stream = self._llm(
                    prompt_text,
                    max_tokens=int(self.get_parameter("max_tokens").value),
                    temperature=float(self.get_parameter("temperature").value),
                    stop=["\n"],
                    stream=True  # ENABLE STREAMING
                )
            
            full_text = ""
            for chunk in stream:
                token = chunk["choices"][0]["text"]
                full_text += token
                
                # Publish individual token for streaming UI
                stream_msg = String()
                stream_msg.data = token
                self.stream_pub.publish(stream_msg)
                
            text = full_text.strip()
            if prompt:
                with self._llm_lock:
                    self._history.append({"role": "User", "text": prompt})
                    if text:
                        self._history.append({"role": "Coach", "text": text})
            return text if text else ""
        except Exception as e:
            self.get_logger().error(f"LLM generation error: {e}")
            return ""

    def _publish(self, text: str) -> None:
        if not text:
            return
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
