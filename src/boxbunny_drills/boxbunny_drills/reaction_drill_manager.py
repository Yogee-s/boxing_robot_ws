import csv
import json
import os
import random
import time
from collections import deque
from typing import Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Int32
from boxbunny_msgs.msg import DrillEvent, PunchEvent, GloveDetections, ActionPrediction
from boxbunny_msgs.srv import StartStopDrill


class ReactionDrillManager(Node):
    def __init__(self) -> None:
        super().__init__("reaction_drill_manager")

        self.declare_parameter("countdown_s", 3.0)
        self.declare_parameter("baseline_s", 1.5)
        self.declare_parameter("min_cue_delay_s", 1.5)
        self.declare_parameter("max_cue_delay_s", 4.0)
        self.declare_parameter("max_response_time_s", 2.5)
        self.declare_parameter("min_reaction_time_s", 0.12)
        self.declare_parameter("baseline_velocity_margin", 0.25)
        self.declare_parameter("punch_topic", "punch_events")
        self.declare_parameter("glove_topic", "glove_detections")
        self.declare_parameter("num_trials", 5)
        self.declare_parameter("log_dir", os.path.expanduser("~/boxbunny_logs"))

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

        self._countdown_end: Optional[float] = None
        self._next_countdown_tick: Optional[float] = None
        self._baseline_end: Optional[float] = None
        self._baseline_velocity_mps = 0.0
        self._baseline_velocity_mps = 0.0
        self._baseline_samples = deque(maxlen=200)
        self._penalty_end: Optional[float] = None
        self._last_state_pub = 0.0

        self.get_logger().info("Reaction drill manager ready")

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
        if not self._running or self._state != "cue" or self._cue_time is None:
            return

        # Only react to punches
        if msg.action_label not in ["jab", "cross", "left_hook", "right_hook", "left_uppercut", "right_uppercut"]:
             return

        reaction_time = time.time() - self._cue_time
        if reaction_time < float(self.get_parameter("min_reaction_time_s").value):
            return

        self._record_result(msg.action_label, reaction_time, None)
        self._publish_event("punch_detected", glove=msg.action_label, value=reaction_time)
        self._advance_trial()

    def _on_punch(self, msg: PunchEvent) -> None:
        if not self._running:
            return

        # Check for Early Start (Punch before Cue)
        if self._state in ["waiting", "countdown", "baseline"]:
            # Use specific threshold to avoid noise triggering early warnings
            if msg.approach_velocity_mps > 0.5:
                self._publish_event("early_start", glove=msg.glove)
                # Penalty Logic
                self._state = "early_penalty"
                self._penalty_end = time.time() + 1.5 # 1.5s penalty
                self._publish_state()
            return

        if self._state != "cue" or self._cue_time is None:
            return

        reaction_time = time.time() - self._cue_time
        if reaction_time < float(self.get_parameter("min_reaction_time_s").value):
            return

        margin = float(self.get_parameter("baseline_velocity_margin").value)
        if msg.approach_velocity_mps < (self._baseline_velocity_mps + margin):
            return

        self._record_result(msg.glove, reaction_time, msg)
        self._publish_event("punch_detected", glove=msg.glove, value=reaction_time)
        self._advance_trial()

        self._trial_index += 1
        if self._trial_index >= self._num_trials:
            self._stop_drill("Completed")
            return
        self._state = "waiting"
        self._cue_time = None
        self._penalty_end = None
        self._next_cue_time = time.time() + self._random_delay()
        self._publish_state()

    def _open_log(self) -> None:
        log_dir = self.get_parameter("log_dir").value
        os.makedirs(log_dir, exist_ok=True)
        filename = time.strftime("reaction_drill_%Y%m%d_%H%M%S.csv")
        self._log_path = os.path.join(log_dir, filename)
        with open(self._log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "trial_index",
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
            "trial_index": self._trial_index,
            "total_trials": self._num_trials,
            "last_reaction_time_s": last_reaction,
            "mean_reaction_time_s": (sum(self._results) / len(self._results)) if self._results else None,
            "median_reaction_time_s": (sorted(self._results)[len(self._results) // 2]) if self._results else None,
            "best_reaction_time_s": min(self._results) if self._results else None,
            "baseline_velocity_mps": self._baseline_velocity_mps,
            "log_path": self._log_path,
        }
        msg = String()
        msg.data = json.dumps(summary)
        self.summary_pub.publish(msg)

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
