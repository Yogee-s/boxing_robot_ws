import time
from typing import Optional

import rclpy
from rclpy.node import Node
from boxbunny_msgs.msg import PunchEvent, ImuPunch


class PunchFusionNode(Node):
    def __init__(self) -> None:
        super().__init__("punch_fusion_node")

        self.declare_parameter("vision_topic", "punch_events_raw")
        self.declare_parameter("imu_topic", "imu/punch")
        self.declare_parameter("output_topic", "punch_events")
        self.declare_parameter("fusion_window_s", 0.25)
        self.declare_parameter("require_imu_confirmation", True)
        self.declare_parameter("allow_vision_fallback", True)

        vision_topic = self.get_parameter("vision_topic").value
        imu_topic = self.get_parameter("imu_topic").value
        output_topic = self.get_parameter("output_topic").value

        self._last_imu: Optional[ImuPunch] = None
        self._last_imu_time = 0.0

        self.create_subscription(PunchEvent, vision_topic, self._on_vision_punch, 10)
        self.create_subscription(ImuPunch, imu_topic, self._on_imu_punch, 10)
        self.pub = self.create_publisher(PunchEvent, output_topic, 10)

        self.get_logger().info("Punch fusion node ready")

    def _on_imu_punch(self, msg: ImuPunch) -> None:
        self._last_imu = msg
        self._last_imu_time = time.time()

    def _on_vision_punch(self, msg: PunchEvent) -> None:
        require_imu = bool(self.get_parameter("require_imu_confirmation").value)
        allow_vision = bool(self.get_parameter("allow_vision_fallback").value)
        window_s = float(self.get_parameter("fusion_window_s").value)

        imu = self._last_imu
        has_imu = imu is not None and (time.time() - self._last_imu_time) <= window_s

        if not has_imu and require_imu and not allow_vision:
            return

        fused = PunchEvent()
        fused.stamp = msg.stamp
        fused.glove = msg.glove
        fused.distance_m = msg.distance_m
        fused.approach_velocity_mps = msg.approach_velocity_mps
        fused.confidence = msg.confidence
        fused.method = msg.method
        fused.is_punch = msg.is_punch

        if has_imu and imu is not None:
            fused.source = "vision+imu"
            fused.punch_type = self._decorate_punch_type(imu.punch_type, msg.glove)
            fused.imu_confirmed = True
        else:
            fused.source = "vision"
            fused.punch_type = "unknown"
            fused.imu_confirmed = False

        self.pub.publish(fused)

    def _decorate_punch_type(self, punch_type: str, glove: str) -> str:
        if punch_type == "straight":
            return "jab" if glove == "left" else "cross"
        if punch_type in {"hook", "uppercut"}:
            side = "left" if glove == "left" else "right"
            return f"{side}_{punch_type}"
        return punch_type or "unknown"


def main() -> None:
    rclpy.init()
    node = PunchFusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
