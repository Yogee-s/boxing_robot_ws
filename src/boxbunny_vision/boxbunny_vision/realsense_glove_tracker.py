import time
from collections import deque
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer
from sensor_msgs.msg import Image
from boxbunny_msgs.msg import GloveDetection, GloveDetections, PunchEvent


class GloveTracker(Node):
    def __init__(self) -> None:
        super().__init__("realsense_glove_tracker")

        # Topics
        self.declare_parameter("color_topic", "/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("debug_image_topic", "/glove_debug_image")
        self.declare_parameter("punch_topic", "punch_events_raw")

        # HSV thresholds
        self.declare_parameter("hsv_green_lower", [45, 80, 50])
        self.declare_parameter("hsv_green_upper", [85, 255, 255])
        self.declare_parameter("hsv_red_lower1", [0, 90, 50])
        self.declare_parameter("hsv_red_upper1", [10, 255, 255])
        self.declare_parameter("hsv_red_lower2", [160, 90, 50])
        self.declare_parameter("hsv_red_upper2", [180, 255, 255])

        # Detection thresholds
        self.declare_parameter("min_contour_area", 800)
        self.declare_parameter("min_confidence", 0.3)
        self.declare_parameter("depth_threshold_m", 0.75)
        self.declare_parameter("smoothing_window", 5)
        self.declare_parameter("approach_velocity_mps", 0.35)
        self.declare_parameter("approach_frames", 3)
        self.declare_parameter("debounce_time_s", 0.5)

        # Depth scaling
        self.declare_parameter("depth_scale", 0.001)

        # Performance
        self.declare_parameter("resize_scale", 0.7)
        self.declare_parameter("process_every_n", 1)

        # Optional pose verification
        self.declare_parameter("use_pose_verification", False)
        self.declare_parameter("pose_model_path", "")
        self.declare_parameter("pose_min_conf", 0.25)
        self.declare_parameter("pose_process_every_n", 8)

        self.bridge = CvBridge()

        self._frame_count = 0
        self._pose_frame_count = 0
        self._last_punch_time: Dict[str, float] = {"left": 0.0, "right": 0.0}
        self._distance_hist: Dict[str, deque] = {
            "left": deque(maxlen=self.get_parameter("smoothing_window").value),
            "right": deque(maxlen=self.get_parameter("smoothing_window").value),
        }
        self._velocity_hist: Dict[str, deque] = {
            "left": deque(maxlen=self.get_parameter("approach_frames").value),
            "right": deque(maxlen=self.get_parameter("approach_frames").value),
        }

        self._pose_enabled = False
        self._pose_model = None
        self._init_pose_model()

        # Publishers
        self.detections_pub = self.create_publisher(GloveDetections, "glove_detections", 10)
        self.punch_pub = self.create_publisher(
            PunchEvent, self.get_parameter("punch_topic").value, 10
        )
        self.debug_pub = self.create_publisher(Image, self.get_parameter("debug_image_topic").value, 5)

        # Subscribers (sync color + depth)
        self.color_sub = Subscriber(self, Image, self.get_parameter("color_topic").value)
        self.depth_sub = Subscriber(self, Image, self.get_parameter("depth_topic").value)
        self.sync = ApproximateTimeSynchronizer([self.color_sub, self.depth_sub], queue_size=10, slop=0.05)
        self.sync.registerCallback(self._on_frames)

        self.add_on_set_parameters_callback(self._on_params)

        self.get_logger().info("Glove tracker initialized")

    def _init_pose_model(self) -> None:
        if not self.get_parameter("use_pose_verification").value:
            return
        model_path = self.get_parameter("pose_model_path").value
        if not model_path:
            self.get_logger().warn("Pose verification enabled but pose_model_path is empty")
            return
        try:
            from ultralytics import YOLO  # type: ignore

            self._pose_model = YOLO(model_path)
            self._pose_enabled = True
            self.get_logger().info("Pose verification enabled")
        except Exception as exc:  # pragma: no cover - optional
            self.get_logger().warn(f"Pose model load failed: {exc}")
            self._pose_enabled = False

    def _on_params(self, params):
        for param in params:
            if param.name in ("smoothing_window", "approach_frames") and param.type_ == Parameter.Type.INTEGER:
                value = max(1, int(param.value))
                if param.name == "smoothing_window":
                    self._distance_hist["left"] = deque(self._distance_hist["left"], maxlen=value)
                    self._distance_hist["right"] = deque(self._distance_hist["right"], maxlen=value)
                else:
                    self._velocity_hist["left"] = deque(self._velocity_hist["left"], maxlen=value)
                    self._velocity_hist["right"] = deque(self._velocity_hist["right"], maxlen=value)
        return rclpy.parameter.SetParametersResult(successful=True)

    def _on_frames(self, color_msg: Image, depth_msg: Image) -> None:
        self._frame_count += 1
        process_every_n = int(self.get_parameter("process_every_n").value)
        if process_every_n > 1 and (self._frame_count % process_every_n) != 0:
            return

        color = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding="bgr8")
        depth = self.bridge.imgmsg_to_cv2(depth_msg)

        self._last_frame = color

        resize_scale = float(self.get_parameter("resize_scale").value)
        if resize_scale != 1.0:
            color = cv2.resize(color, None, fx=resize_scale, fy=resize_scale, interpolation=cv2.INTER_LINEAR)
            depth = cv2.resize(depth, None, fx=resize_scale, fy=resize_scale, interpolation=cv2.INTER_NEAREST)

        hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)

        detections = []
        debug_img = color.copy()

        for glove, mask in self._build_masks(hsv).items():
            det = self._detect_glove(glove, mask, depth, color.shape)
            if det:
                detections.append(det)
                self._draw_detection(debug_img, det)

        # Publish detections
        det_msg = GloveDetections()
        det_msg.stamp = color_msg.header.stamp
        det_msg.detections = detections
        self.detections_pub.publish(det_msg)

        # Publish debug image
        try:
            debug_msg = self.bridge.cv2_to_imgmsg(debug_img, encoding="bgr8")
            debug_msg.header = color_msg.header
            self.debug_pub.publish(debug_msg)
        except Exception:
            pass

    def _build_masks(self, hsv: np.ndarray) -> Dict[str, np.ndarray]:
        green_lower = np.array(self.get_parameter("hsv_green_lower").value, dtype=np.uint8)
        green_upper = np.array(self.get_parameter("hsv_green_upper").value, dtype=np.uint8)

        red_lower1 = np.array(self.get_parameter("hsv_red_lower1").value, dtype=np.uint8)
        red_upper1 = np.array(self.get_parameter("hsv_red_upper1").value, dtype=np.uint8)
        red_lower2 = np.array(self.get_parameter("hsv_red_lower2").value, dtype=np.uint8)
        red_upper2 = np.array(self.get_parameter("hsv_red_upper2").value, dtype=np.uint8)

        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
        red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)

        kernel = np.ones((5, 5), np.uint8)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

        return {"left": green_mask, "right": red_mask}

    def _detect_glove(
        self, glove: str, mask: np.ndarray, depth: np.ndarray, shape: Tuple[int, int, int]
    ) -> Optional[GloveDetection]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        min_area = float(self.get_parameter("min_contour_area").value)
        if area < min_area:
            return None

        x, y, w, h = cv2.boundingRect(contour)
        roi_depth = depth[y : y + h, x : x + w]
        distance_m = self._median_depth_m(roi_depth)
        if distance_m is None:
            return None

        confidence = min(1.0, area / (min_area * 3.0))

        smoothed_distance = self._smooth_distance(glove, distance_m)
        approach_velocity = self._estimate_velocity(glove, smoothed_distance)

        det = GloveDetection()
        det.glove = glove
        det.distance_m = float(smoothed_distance)
        det.approach_velocity_mps = float(approach_velocity)
        det.confidence = float(confidence)
        det.x = int(x)
        det.y = int(y)
        det.w = int(w)
        det.h = int(h)

        self._maybe_publish_punch(det)

        return det

    def _median_depth_m(self, roi_depth: np.ndarray) -> Optional[float]:
        if roi_depth.size == 0:
            return None

        # RealSense depth is often uint16 in millimeters
        depth_scale = float(self.get_parameter("depth_scale").value)
        valid = roi_depth[np.where(roi_depth > 0)]
        if valid.size < 10:
            return None

        median = float(np.median(valid))
        if roi_depth.dtype == np.uint16:
            return median * depth_scale
        return median

    def _smooth_distance(self, glove: str, distance: float) -> float:
        self._distance_hist[glove].append(distance)
        return float(np.mean(self._distance_hist[glove]))

    def _estimate_velocity(self, glove: str, distance: float) -> float:
        # Positive velocity means moving toward camera (distance decreasing)
        if len(self._distance_hist[glove]) < 2:
            return 0.0
        prev = self._distance_hist[glove][-2]
        dt = 1.0 / 30.0  # approximate; frame sync is near camera FPS
        velocity = (prev - distance) / dt
        self._velocity_hist[glove].append(velocity)
        return velocity

    def _maybe_publish_punch(self, det: GloveDetection) -> None:
        now = time.time()
        min_conf = float(self.get_parameter("min_confidence").value)
        if det.confidence < min_conf:
            return

        debounce = float(self.get_parameter("debounce_time_s").value)
        if now - self._last_punch_time[det.glove] < debounce:
            return

        depth_threshold = float(self.get_parameter("depth_threshold_m").value)
        approach_velocity = float(self.get_parameter("approach_velocity_mps").value)
        approach_frames = int(self.get_parameter("approach_frames").value)

        velocity_hits = sum(1 for v in self._velocity_hist[det.glove] if v > approach_velocity)
        velocity_ok = velocity_hits >= approach_frames
        distance_ok = det.distance_m <= depth_threshold

        pose_ok = self._verify_pose() if self.get_parameter("use_pose_verification").value else True

        if (velocity_ok or distance_ok) and pose_ok:
            self._last_punch_time[det.glove] = now
            event = PunchEvent()
            event.stamp = self.get_clock().now().to_msg()
            event.glove = det.glove
            event.distance_m = det.distance_m
            event.approach_velocity_mps = det.approach_velocity_mps
            event.confidence = det.confidence
            event.method = "velocity" if velocity_ok else "threshold"
            event.is_punch = True
            event.punch_type = "unknown"
            event.imu_confirmed = False
            event.source = "vision"
            self.punch_pub.publish(event)

    def _verify_pose(self) -> bool:
        # Optional pose verification, disabled by default
        if not self._pose_enabled or self._pose_model is None:
            return True

        self._pose_frame_count += 1
        if (self._pose_frame_count % int(self.get_parameter("pose_process_every_n").value)) != 0:
            return True

        try:
            result = self._pose_model(self._last_frame, verbose=False) if hasattr(self, "_last_frame") else None
            if not result or len(result) == 0:
                return False
            kps = result[0].keypoints
            if kps is None or kps.conf is None:
                return False
            return bool(np.max(kps.conf) >= float(self.get_parameter("pose_min_conf").value))
        except Exception:
            return True

    def _draw_detection(self, img: np.ndarray, det: GloveDetection) -> None:
        color = (0, 255, 0) if det.glove == "left" else (0, 0, 255)
        cv2.rectangle(img, (det.x, det.y), (det.x + det.w, det.y + det.h), color, 2)
        label = f"{det.glove} {det.distance_m:.2f}m v={det.approach_velocity_mps:.2f}"
        cv2.putText(img, label, (det.x, max(det.y - 6, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def main() -> None:
    rclpy.init()
    node = GloveTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
