import csv
import json
import os
import random
import time
from datetime import datetime
from collections import deque
from pathlib import Path
from typing import Optional, List

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Int32
from boxbunny_msgs.msg import DrillEvent, PunchEvent, GloveDetections, ActionPrediction
from boxbunny_msgs.srv import StartStopDrill


class ReactionDrillManager(Node):
    def __init__(self) -> None:
        super().__init__("reaction_drill_manager")

        data_root = self._default_data_root()
        self.declare_parameter("countdown_s", 3.0)
        self.declare_parameter("baseline_s", 1.5)
        self.declare_parameter("min_cue_delay_s", 1.5)
        self.declare_parameter("max_cue_delay_s", 4.0)
        self.declare_parameter("max_response_time_s", 2.5)
        self.declare_parameter("min_reaction_time_s", 0.12)
        self.declare_parameter("baseline_velocity_margin", 0.25)
        self.declare_parameter("pose_confirm_window_s", 0.35)
        self.declare_parameter("action_confidence_threshold", 0.45)
        self.declare_parameter("action_confirm_hits", 1)
        self.declare_parameter("early_confidence_threshold", 0.6)
        self.declare_parameter("early_confirm_hits", 2)
        self.declare_parameter("punch_topic", "punch_events")
        self.declare_parameter("glove_topic", "glove_detections")
        self.declare_parameter("num_trials", 5)
        self.declare_parameter("log_dir", str(data_root / "reaction_drill"))

        self.state_pub = self.create_publisher(String, "drill_state", 10)
        self.summary_pub = self.create_publisher(String, "drill_summary", 10)
        self.event_pub = self.create_publisher(DrillEvent, "drill_events", 10)
        self.countdown_pub = self.create_publisher(Int32, "drill_countdown", 10)

        punch_topic = self.get_parameter("punch_topic").value
        glove_topic = self.get_parameter("glove_topic").value
        self.punch_sub = self.create_subscription(PunchEvent, punch_topic, self._on_punch, 10)
        self.det_sub = self.create_subscription(GloveDetections, glove_topic, self._on_detections, 10)
        self.action_sub = self.create_subscription(ActionPrediction, "action_prediction", self._on_action, 10)
        self.service = self.create_service(StartStopDrill, "start_stop_drill", self._on_start_stop)

        self.timer = self.create_timer(0.05, self._tick)

        self._running = False
        self._state = "idle"
        self._next_cue_time: Optional[float] = None
        self._cue_time: Optional[float] = None
        self._trial_index = 0
        self._num_trials = int(self.get_parameter("num_trials").value)
        self._results = []
        self._log_path = None
        self._summary_path = None

        self._countdown_end: Optional[float] = None
        self._next_countdown_tick: Optional[float] = None
        self._baseline_end: Optional[float] = None
        self._baseline_velocity_mps = 0.0
        self._baseline_velocity_mps = 0.0
        self._baseline_samples = deque(maxlen=200)
        self._penalty_end: Optional[float] = None
        self._result_end: Optional[float] = None
        self._result_end: Optional[float] = None
        self._last_state_pub = 0.0
        self._last_penalty_time = 0.0
        self._last_action_pred: Optional[tuple] = None
        self._last_punch_event: Optional[tuple] = None
        self._confirmed_this_cue = False
        self._action_confirm_label: Optional[str] = None
        self._action_confirm_count = 0
        self._action_confirm_start = 0.0

        self.get_logger().info("Reaction drill manager ready")

    @staticmethod
    def _default_data_root() -> Path:
        try:
            here = Path(__file__).resolve()
            for parent in here.parents:
                if parent.name == "boxing_robot_ws":
                    return parent / "data"
        except Exception:
            pass
        return Path(os.path.expanduser("~/boxbunny_data"))

    def _on_start_stop(self, request, response):
        if request.start and not self._running:
            self._start_drill(request.num_trials or self._num_trials)
            response.accepted = True
            response.message = "Drill started"
        elif not request.start and self._running:
            self._stop_drill("Stopped by user")
            response.accepted = True
            response.message = "Drill stopped"
        else:
            response.accepted = False
            response.message = "No change"
        return response

    def _start_drill(self, num_trials: int) -> None:
        self._running = True
        self._trial_index = 0
        self._num_trials = max(1, int(num_trials))
        self._results = []
        self._cue_time = None
        self._state = "countdown"
        self._baseline_velocity_mps = 0.0
        self._baseline_samples.clear()
        now = time.time()
        countdown_s = float(self.get_parameter("countdown_s").value)
        self._countdown_end = now + countdown_s
        self._next_countdown_tick = now
        self._open_log()
        self._publish_state()
        self._publish_event("drill_start")

    def _stop_drill(self, reason: str) -> None:
        self._running = False
        self._state = "idle"
        self._next_cue_time = None
        self._cue_time = None
        self._countdown_end = None
        self._countdown_end = None
        self._baseline_end = None
        self._penalty_end = None
        self._result_end = None
        self._last_action_pred = None
        self._last_punch_event = None
        self._confirmed_this_cue = False
        self._action_confirm_label = None
        self._action_confirm_count = 0
        self._action_confirm_start = 0.0
        self._publish_state()
        self._publish_event("drill_stop", value=0.0)
        self.get_logger().info(f"Drill stopped: {reason}")

    def _random_delay(self) -> float:
        return random.uniform(
            float(self.get_parameter("min_cue_delay_s").value),
            float(self.get_parameter("max_cue_delay_s").value),
        )

    def _tick(self) -> None:
        if not self._running:
            return

        now = time.time()
        
        # Heartbeat: Republish state every 0.5s to keep GUI synced
        if now - self._last_state_pub > 0.5:
             self._publish_state()

        if self._state == "countdown":
            self._update_countdown(now)
            return

        if self._state == "early_penalty":
            if self._penalty_end is not None and now >= self._penalty_end:
                 # Restart Waiting Phase
                 self._state = "waiting"
                 self._last_penalty_time = now
                 self._next_cue_time = now + self._random_delay()
                 self._next_cue_time = now + self._random_delay()
                 self._publish_state()
            return

        if self._state == "result":
            if self._result_end is not None and now >= self._result_end:
                # Transition back to waiting for next trial
                self._state = "waiting"
                self._next_cue_time = now + self._random_delay()
                self._publish_state()
            return
            
        if self._state == "baseline":
            if self._baseline_end is not None and now >= self._baseline_end:
                self._finalize_baseline()
            return

        if self._state == "waiting" and self._next_cue_time is not None and now >= self._next_cue_time:
            self._state = "cue"
            self._cue_time = now
            self._publish_state()
            self._publish_event("cue_on")

        if self._state == "cue" and self._cue_time is not None:
            max_response = float(self.get_parameter("max_response_time_s").value)
            if now - self._cue_time > max_response:
                self._record_result(None, None)
                self._advance_trial()

    def _update_countdown(self, now: float) -> None:
        if self._countdown_end is None:
            self._countdown_end = now
        remaining = max(0, int(self._countdown_end - now + 0.9))
        if self._next_countdown_tick is None or now >= self._next_countdown_tick:
            self._publish_countdown(remaining)
            self._publish_event("countdown_tick", value=float(remaining))
            self._next_countdown_tick = now + 1.0
        if now >= self._countdown_end:
            self._state = "baseline"
            self._baseline_samples.clear()
            self._baseline_end = now + float(self.get_parameter("baseline_s").value)
            self._publish_state()
            self._publish_event("baseline_start")

    def _finalize_baseline(self) -> None:
        if self._baseline_samples:
            self._baseline_velocity_mps = sum(self._baseline_samples) / len(self._baseline_samples)
        else:
            self._baseline_velocity_mps = 0.0
        self._state = "waiting"
        self._cue_time = None
        self._next_cue_time = time.time() + self._random_delay()
        self._publish_state()
        self._publish_event("baseline_done", value=self._baseline_velocity_mps)

    def _on_detections(self, msg: GloveDetections) -> None:
        if self._state != "baseline":
            return
        for det in msg.detections:
            self._baseline_samples.append(abs(det.approach_velocity_mps))

    def _on_action(self, msg: ActionPrediction) -> None:
        if not self._running:
            return

        # Only react to punch actions
        punch_labels = ["jab", "cross", "left_hook", "right_hook", "left_uppercut", "right_uppercut"]
        if msg.action_label not in punch_labels:
             return
        conf = float(msg.confidence)

        # Check for Early Start (Punch before Cue)
        if self._state in ["waiting", "baseline"]:
            # Debounce early penalty to prevent looping
            if (time.time() - self._last_penalty_time) < 1.0:
                return

            if conf < float(self.get_parameter("early_confidence_threshold").value):
                return
            if not self._confirm_action_hit(
                msg.action_label,
                hits_needed=int(self.get_parameter("early_confirm_hits").value),
            ):
                return

            self._publish_event("early_start", glove=msg.action_label)
            # Penalty Logic
            self._state = "early_penalty"
            self._penalty_end = time.time() + 1.5  # 1.5s penalty
            self._publish_state()
            self.get_logger().info(f"Early punch detected: {msg.action_label}")
            return

        if self._state != "cue" or self._cue_time is None:
            return

        if conf < float(self.get_parameter("action_confidence_threshold").value):
            return

        self._last_action_pred = (msg.action_label, time.time())
        if self._confirmed_this_cue:
            return

        reaction_time = time.time() - self._cue_time
        if reaction_time < float(self.get_parameter("min_reaction_time_s").value):
            return
        if not self._confirm_action_hit(
            msg.action_label,
            hits_needed=int(self.get_parameter("action_confirm_hits").value),
        ):
            return

        self._confirmed_this_cue = True
        self._record_result(msg.action_label, reaction_time, None)
        self._publish_event("punch_detected", glove=msg.action_label, value=reaction_time)
        self._advance_trial()

    def _on_punch(self, msg: PunchEvent) -> None:
        return

    def _advance_trial(self) -> None:
        """Advance to next trial or complete the drill."""
        self._trial_index += 1
        if self._trial_index >= self._num_trials:
            self._stop_drill("Completed")
            self._publish_summary_final()
            return
        # Brief "result" state to show feedback, then back to waiting
        self._state = "result"
        self._cue_time = None
        self._penalty_end = None
        self._confirmed_this_cue = False
        self._action_confirm_label = None
        self._action_confirm_count = 0
        self._action_confirm_start = 0.0
        self._publish_state()
        # Schedule transition to waiting after brief delay
        self._result_end = time.time() + 1.0  # 1 second to show result

    def _recent_action_confirm(self, glove: str) -> bool:
        window = float(self.get_parameter("pose_confirm_window_s").value)
        if not self._last_action_pred:
            return False
        label, ts = self._last_action_pred
        if (time.time() - ts) > window:
            return False
        return label == glove

    def _confirm_action_hit(self, label: str, *, hits_needed: int) -> bool:
        """Require short, consistent action hits to avoid noise."""
        window = float(self.get_parameter("pose_confirm_window_s").value)
        now = time.time()
        if self._action_confirm_label != label or (now - self._action_confirm_start) > window:
            self._action_confirm_label = label
            self._action_confirm_count = 1
            self._action_confirm_start = now
            return hits_needed <= 1
        self._action_confirm_count += 1
        return self._action_confirm_count >= hits_needed

    def _recent_punch_confirm(self, glove: str) -> bool:
        window = float(self.get_parameter("pose_confirm_window_s").value)
        if not self._last_punch_event:
            return False
        label, ts, velocity = self._last_punch_event
        if (time.time() - ts) > window:
            return False
        if label != glove:
            return False
        margin = float(self.get_parameter("baseline_velocity_margin").value)
        return velocity >= (self._baseline_velocity_mps + margin)
        
    def _open_log(self) -> None:
        log_dir = Path(os.path.expanduser(str(self.get_parameter("log_dir").value)))
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"reaction_drill_{timestamp}.csv"
        self._log_path = str(log_dir / filename)
        self._summary_path = str(log_dir.parent / "reaction_drill_sessions.csv")
        with open(self._log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "trial_index",
                    "timestamp_unix",
                    "cue_time_unix",
                    "detection_time_unix",
                    "reaction_time_s",
                    "glove",
                    "confidence",
                    "velocity_mps",
                    "punch_type",
                    "imu_confirmed",
                    "baseline_velocity_mps",
                ]
            )

    def _record_result(self, glove: Optional[str], reaction_time: Optional[float], msg: Optional[PunchEvent] = None):
        cue_time = self._cue_time if self._cue_time is not None else time.time()
        detection_time = time.time() if reaction_time is not None else None

        confidence = msg.confidence if msg else 0.0
        velocity = msg.approach_velocity_mps if msg else 0.0
        punch_type = msg.punch_type if msg else ""
        imu_confirmed = msg.imu_confirmed if msg else False

        with open(self._log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    self._trial_index,
                    time.time(),
                    cue_time,
                    detection_time or "",
                    reaction_time or "",
                    glove or "",
                    confidence,
                    velocity,
                    punch_type,
                    imu_confirmed,
                    self._baseline_velocity_mps,
                ]
            )

        if reaction_time is not None:
            self._results.append(reaction_time)
        self._publish_summary(reaction_time)

    def _publish_state(self) -> None:
        msg = String()
        msg.data = self._state
        self.state_pub.publish(msg)
        self._last_state_pub = time.time()

    def _publish_event(self, event_type: str, glove: str = "", value: float = 0.0) -> None:
        event = DrillEvent()
        event.stamp = self.get_clock().now().to_msg()
        event.event_type = event_type
        event.glove = glove
        event.value = float(value)
        event.trial_index = int(self._trial_index)
        self.event_pub.publish(event)

    def _publish_summary(self, last_reaction: Optional[float]) -> None:
        summary = {
            "drill_name": "reaction_drill",
            "trial_index": self._trial_index,
            "total_attempts": len(self._results),
            "reaction_times": self._results,
            "last_reaction_time_s": last_reaction,
            "avg_time": (sum(self._results) / len(self._results)) if self._results else None,
            "best_time": min(self._results) if self._results else None,
            "log_path": self._log_path,
            "is_final": False,
        }
        msg = String()
        msg.data = json.dumps(summary)
        self.summary_pub.publish(msg)

    def _publish_summary_final(self) -> None:
        """Publish final summary when drill completes."""
        summary = {
            "drill_name": "reaction_drill",
            "trial_index": self._trial_index,
            "total_attempts": len(self._results),
            "reaction_times": self._results,
            "last_reaction_time_s": self._results[-1] if self._results else None,
            "avg_time": (sum(self._results) / len(self._results)) if self._results else None,
            "best_time": min(self._results) if self._results else None,
            "log_path": self._log_path,
            "is_final": True,
        }
        msg = String()
        msg.data = json.dumps(summary)
        self.summary_pub.publish(msg)
        
        # Log to CSV
        self._log_result_to_csv(summary)

    def _log_result_to_csv(self, summary: dict) -> None:
        """Append session result to CSV file."""
        file_path = self._summary_path or os.path.expanduser("~/boxbunny_data/reaction_drill_sessions.csv")
        file_exists = os.path.exists(file_path)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(file_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            # Write header if new file
            if not file_exists:
                writer.writerow(["Timestamp", "Drill", "Attempts", "Best (s)", "Avg (s)"])
            
            # Write data row
            writer.writerow([
                timestamp,
                "Reaction Drill",
                summary.get("total_attempts", 0),
                f"{summary.get('best_time', 0):.3f}",
                f"{summary.get('avg_time', 0):.3f}"
            ])
        self.get_logger().info(f"Saved stats to {file_path}")

    def _publish_countdown(self, seconds_left: int) -> None:
        msg = Int32()
        msg.data = int(seconds_left)
        self.countdown_pub.publish(msg)


def main() -> None:
    rclpy.init()
    node = ReactionDrillManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
