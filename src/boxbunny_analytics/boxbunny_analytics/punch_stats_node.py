import json
import time
from collections import deque
from typing import Deque, Dict, Tuple

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from boxbunny_msgs.msg import PunchEvent


class PunchStatsNode(Node):
    def __init__(self) -> None:
        super().__init__("punch_stats_node")

        self.declare_parameter("punch_topic", "punch_events")
        self.declare_parameter("output_topic", "punch_stats")
        self.declare_parameter("window_s", 30.0)
        self.declare_parameter("publish_period_s", 2.0)

        self._window: Deque[Tuple[float, PunchEvent]] = deque(maxlen=3000)

        punch_topic = self.get_parameter("punch_topic").value
        output_topic = self.get_parameter("output_topic").value

        self.sub = self.create_subscription(PunchEvent, punch_topic, self._on_punch, 10)
        self.pub = self.create_publisher(String, output_topic, 10)
        self.timer = self.create_timer(float(self.get_parameter("publish_period_s").value), self._publish_stats)

        self.get_logger().info("Punch stats node ready")

    def _on_punch(self, msg: PunchEvent) -> None:
        stamp = self._to_sec(msg.stamp)
        self._window.append((stamp, msg))

    def _publish_stats(self) -> None:
        window_s = float(self.get_parameter("window_s").value)
        now = time.time()
        while self._window and now - self._window[0][0] > window_s:
            self._window.popleft()

        counts: Dict[str, int] = {}
        total = 0
        avg_velocity = 0.0
        avg_confidence = 0.0
        imu_confirmed = 0

        for _, msg in self._window:
            total += 1
            ptype = msg.punch_type or "unknown"
            counts[ptype] = counts.get(ptype, 0) + 1
            avg_velocity += msg.approach_velocity_mps
            avg_confidence += msg.confidence
            if msg.imu_confirmed:
                imu_confirmed += 1

        if total > 0:
            avg_velocity /= total
            avg_confidence /= total

        summary = {
            "window_s": window_s,
            "total": total,
            "counts": counts,
            "avg_velocity_mps": avg_velocity,
            "avg_confidence": avg_confidence,
            "imu_confirmed_ratio": (imu_confirmed / total) if total > 0 else 0.0,
        }

        out = String()
        out.data = json.dumps(summary)
        self.pub.publish(out)

    @staticmethod
    def _to_sec(stamp) -> float:
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def main() -> None:
    rclpy.init()
    node = PunchStatsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
