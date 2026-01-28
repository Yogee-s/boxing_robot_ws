import json
import os
import time
from collections import deque
from typing import Deque, Dict, Optional, Tuple

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from boxbunny_msgs.msg import ImuPunch
from boxbunny_msgs.srv import CalibrateImuPunch


PUNCH_TYPES = ("straight", "hook", "uppercut")
PUNCH_TYPE_ALIASES = {"jab_or_cross": "straight"}


class ImuPunchClassifier(Node):
    def __init__(self) -> None:
        super().__init__("imu_punch_classifier")

        self.declare_parameter("enable_punch_classification", True)
        self.declare_parameter("window_size", 10)
        self.declare_parameter("cooldown_s", 0.5)
        self.declare_parameter("gyro_threshold", 2.5)
        self.declare_parameter("accel_threshold", 6.0)
        self.declare_parameter("accel_peak_ratio", 2.0)
        self.declare_parameter("gyro_peak_ratio", 2.0)
        self.declare_parameter("axis_dominance_ratio", 1.3)
        self.declare_parameter("imu_hand", "right")
        self.declare_parameter("calibration_path", os.path.expanduser("~/.boxbunny/imu_calibration.json"))
        self.declare_parameter("use_calibration", True)

        self.sub = self.create_subscription(Imu, "imu/data", self._on_imu, 10)
        self.pub = self.create_publisher(ImuPunch, "imu/punch", 10)
        self.calib_srv = self.create_service(CalibrateImuPunch, "calibrate_imu_punch", self._on_calibrate)

        self._history: Deque[Imu] = deque(maxlen=int(self.get_parameter("window_size").value))
        self._last_time = 0.0
        self._calibrating: Optional[str] = None
        self._calibration_end = 0.0
        self._calibration_peaks: Dict[str, Tuple[float, float]] = {}

        self._templates = self._load_templates()
        self._calib_timer = self.create_timer(0.05, self._calibration_tick)

        self.get_logger().info("IMU punch classifier initialized")

    def _load_templates(self) -> Dict[str, Dict[str, float]]:
        path = self.get_parameter("calibration_path").value
        if not path or not os.path.exists(path):
            return {}
        try:
            with open(path, "r") as f:
                raw = json.load(f)
            templates = {}
            for key, value in raw.items():
                canonical = PUNCH_TYPE_ALIASES.get(key, key)
                templates[canonical] = value
            return templates
        except Exception:
            return {}

    def _save_templates(self) -> None:
        path = self.get_parameter("calibration_path").value
        if not path:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._templates, f, indent=2)

    def _on_calibrate(self, request, response):
        punch_type = PUNCH_TYPE_ALIASES.get(request.punch_type, request.punch_type)
        duration_s = max(0.5, float(request.duration_s or 2.5))
        if punch_type not in PUNCH_TYPES:
            response.accepted = False
            response.message = f"Unknown punch_type: {punch_type}"
            return response
        self._calibrating = punch_type
        self._calibration_end = time.time() + duration_s
        self._calibration_peaks[punch_type] = (0.0, 0.0)
        response.accepted = True
        response.message = f"Calibrating {punch_type} for {duration_s:.1f}s"
        self.get_logger().info(response.message)
        return response

    def _calibration_tick(self) -> None:
        if self._calibrating is None:
            return
        if time.time() < self._calibration_end:
            return
        punch_type = self._calibrating
        peak_accel, peak_gyro = self._calibration_peaks.get(punch_type, (0.0, 0.0))
        self._templates[punch_type] = {"peak_accel": peak_accel, "peak_gyro": peak_gyro}
        self._save_templates()
        self.get_logger().info(f"Saved calibration for {punch_type}: accel={peak_accel:.2f} gyro={peak_gyro:.2f}")
        self._calibrating = None

    def _on_imu(self, msg: Imu) -> None:
        if self._calibrating:
            peak_accel, peak_gyro = self._calibration_peaks.get(self._calibrating, (0.0, 0.0))
            accel = self._accel_magnitude(msg)
            gyro = self._gyro_magnitude(msg)
            self._calibration_peaks[self._calibrating] = (max(peak_accel, accel), max(peak_gyro, gyro))
            return

        if not self.get_parameter("enable_punch_classification").value:
            return

        self._history.append(msg)
        if len(self._history) < self._history.maxlen:
            return

        now = time.time()
        if now - self._last_time < float(self.get_parameter("cooldown_s").value):
            return

        gyro_thresh = float(self.get_parameter("gyro_threshold").value)
        accel_thresh = float(self.get_parameter("accel_threshold").value)

        peaks = self._window_peaks()
        rms = self._window_rms()
        axis_peaks = self._axis_peaks()
        if not self._passes_filters(peaks, rms, axis_peaks):
            return

        punch_type = self._classify(gyro_thresh, accel_thresh, axis_peaks)
        if punch_type is None:
            return

        self._last_time = now
        confidence = self._estimate_confidence(punch_type, peaks)

        out = ImuPunch()
        out.stamp = self.get_clock().now().to_msg()
        out.glove = str(self.get_parameter("imu_hand").value)
        out.punch_type = punch_type
        out.peak_accel = peaks[0]
        out.peak_gyro = peaks[1]
        out.confidence = confidence
        out.method = "heuristic"
        self.pub.publish(out)

    def _window_peaks(self) -> Tuple[float, float]:
        peak_accel = 0.0
        peak_gyro = 0.0
        for h in self._history:
            peak_accel = max(peak_accel, self._accel_magnitude(h))
            peak_gyro = max(peak_gyro, self._gyro_magnitude(h))
        return peak_accel, peak_gyro

    def _classify(
        self,
        gyro_thresh: float,
        accel_thresh: float,
        axis_peaks: Tuple[Tuple[float, float, float], Tuple[float, float, float]],
    ) -> Optional[str]:
        (ax, ay, az), (gx, gy, gz) = axis_peaks

        if gz > gyro_thresh and ay > accel_thresh:
            return "hook"
        if gy > gyro_thresh and ax > accel_thresh:
            return "straight"
        if az > accel_thresh and gx > gyro_thresh:
            return "uppercut"
        return None

    def _estimate_confidence(self, punch_type: str, peaks: Tuple[float, float]) -> float:
        if not self.get_parameter("use_calibration").value:
            return 0.6
        template = self._templates.get(punch_type)
        if not template:
            return 0.6
        accel_ratio = peaks[0] / max(0.01, template.get("peak_accel", 1.0))
        gyro_ratio = peaks[1] / max(0.01, template.get("peak_gyro", 1.0))
        return float(min(1.0, 0.5 * accel_ratio + 0.5 * gyro_ratio))

    def _window_rms(self) -> Tuple[float, float]:
        if not self._history:
            return 0.0, 0.0
        accel_sum = 0.0
        gyro_sum = 0.0
        for h in self._history:
            accel = self._accel_magnitude(h)
            gyro = self._gyro_magnitude(h)
            accel_sum += accel * accel
            gyro_sum += gyro * gyro
        n = float(len(self._history))
        return (accel_sum / n) ** 0.5, (gyro_sum / n) ** 0.5

    def _axis_peaks(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        ax = max(abs(h.linear_acceleration.x) for h in self._history)
        ay = max(abs(h.linear_acceleration.y) for h in self._history)
        az = max(abs(h.linear_acceleration.z) for h in self._history)
        gx = max(abs(h.angular_velocity.x) for h in self._history)
        gy = max(abs(h.angular_velocity.y) for h in self._history)
        gz = max(abs(h.angular_velocity.z) for h in self._history)
        return (ax, ay, az), (gx, gy, gz)

    def _passes_filters(
        self,
        peaks: Tuple[float, float],
        rms: Tuple[float, float],
        axis_peaks: Tuple[Tuple[float, float, float], Tuple[float, float, float]],
    ) -> bool:
        accel_peak, gyro_peak = peaks
        accel_rms, gyro_rms = rms
        accel_ratio = accel_peak / max(0.01, accel_rms)
        gyro_ratio = gyro_peak / max(0.01, gyro_rms)

        accel_ratio_thresh = float(self.get_parameter("accel_peak_ratio").value)
        gyro_ratio_thresh = float(self.get_parameter("gyro_peak_ratio").value)
        if accel_ratio < accel_ratio_thresh or gyro_ratio < gyro_ratio_thresh:
            return False

        (ax, ay, az), _ = axis_peaks
        axis_sorted = sorted([ax, ay, az], reverse=True)
        dominance = axis_sorted[0] / max(0.01, axis_sorted[1])
        dominance_thresh = float(self.get_parameter("axis_dominance_ratio").value)
        if dominance < dominance_thresh:
            return False

        return True

    @staticmethod
    def _accel_magnitude(msg: Imu) -> float:
        ax = msg.linear_acceleration.x
        ay = msg.linear_acceleration.y
        az = msg.linear_acceleration.z
        return float((ax * ax + ay * ay + az * az) ** 0.5)

    @staticmethod
    def _gyro_magnitude(msg: Imu) -> float:
        gx = msg.angular_velocity.x
        gy = msg.angular_velocity.y
        gz = msg.angular_velocity.z
        return float((gx * gx + gy * gy + gz * gz) ** 0.5)


def main() -> None:
    rclpy.init()
    node = ImuPunchClassifier()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
