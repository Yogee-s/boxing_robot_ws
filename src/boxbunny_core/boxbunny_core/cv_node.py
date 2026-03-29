"""Computer vision node for BoxBunny.

Wraps the action_prediction/lib/inference_runtime.InferenceEngine to provide
real-time punch detection, pose estimation, and user tracking via ROS 2 topics.
Subscribes to RealSense RGB+depth streams, publishes detections.
"""

import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

from boxbunny_msgs.msg import PoseEstimate, PunchDetection, SessionState, UserTracking

logger = logging.getLogger("boxbunny.cv_node")

# Add action_prediction to path
_WS_ROOT = Path(__file__).resolve().parents[3]
_AP_ROOT = _WS_ROOT / "action_prediction"
if str(_AP_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_AP_ROOT.parent))


class CvNode(Node):
    """ROS 2 node wrapping the action prediction inference engine."""

    def __init__(self) -> None:
        super().__init__("cv_node")

        # Parameters
        self.declare_parameter("checkpoint_path",
                               str(_AP_ROOT / "model" / "best_model.pth"))
        self.declare_parameter("yolo_model_path",
                               str(_AP_ROOT / "model" / "yolo26n-pose.pt"))
        self.declare_parameter("device", "cuda:0")
        self.declare_parameter("inference_interval", 1)
        self.declare_parameter("window_size", 12)
        self.declare_parameter("enabled", True)

        self._checkpoint_path = self.get_parameter("checkpoint_path").value
        self._yolo_model_path = self.get_parameter("yolo_model_path").value
        self._device = self.get_parameter("device").value
        self._inference_interval = self.get_parameter("inference_interval").value
        self._window_size = self.get_parameter("window_size").value
        self._enabled = self.get_parameter("enabled").value

        # State
        self._engine = None
        self._initialized = False
        self._frame_count = 0
        self._session_active = False
        self._bridge = CvBridge()
        self._latest_rgb: Optional[np.ndarray] = None
        self._latest_depth: Optional[np.ndarray] = None
        self._last_bbox: Optional[dict] = None
        self._baseline_bbox_x: Optional[float] = None
        self._baseline_depth: Optional[float] = None

        # Subscribers
        self.create_subscription(
            Image, "/camera/color/image_raw", self._on_color, 5
        )
        self.create_subscription(
            Image, "/camera/aligned_depth_to_color/image_raw", self._on_depth, 5
        )
        self.create_subscription(
            SessionState, "/boxbunny/session/state", self._on_session_state, 10
        )

        # Publishers
        self._pub_detection = self.create_publisher(
            PunchDetection, "/boxbunny/cv/detection", 10
        )
        self._pub_pose = self.create_publisher(
            PoseEstimate, "/boxbunny/cv/pose", 10
        )
        self._pub_tracking = self.create_publisher(
            UserTracking, "/boxbunny/cv/user_tracking", 10
        )

        # Inference timer (runs at camera rate when active)
        self.create_timer(1.0 / 30.0, self._inference_tick)

        logger.info("CV node initialized (checkpoint=%s, device=%s)",
                     self._checkpoint_path, self._device)

    def _lazy_init(self) -> bool:
        """Lazy-load the inference engine on first use."""
        if self._initialized:
            return True
        if not self._enabled:
            return False
        try:
            from action_prediction.lib.inference_runtime import InferenceEngine
            self._engine = InferenceEngine(
                checkpoint_path=self._checkpoint_path,
                device=self._device,
                window_size=self._window_size,
            )
            self._engine.initialize()
            self._initialized = True
            logger.info("Inference engine loaded successfully")
            return True
        except Exception as e:
            logger.error("Failed to load inference engine: %s", e)
            self._enabled = False
            return False

    def _on_color(self, msg: Image) -> None:
        """Receive RGB frame from RealSense."""
        try:
            self._latest_rgb = self._bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            logger.debug("Color frame decode error: %s", e)

    def _on_depth(self, msg: Image) -> None:
        """Receive depth frame from RealSense."""
        try:
            self._latest_depth = self._bridge.imgmsg_to_cv2(msg, "passthrough")
        except Exception as e:
            logger.debug("Depth frame decode error: %s", e)

    def _on_session_state(self, msg: SessionState) -> None:
        """Track session state for baseline management."""
        was_active = self._session_active
        self._session_active = msg.state in ("countdown", "active", "rest")
        if self._session_active and not was_active:
            self._baseline_bbox_x = None
            self._baseline_depth = None
            if not self._initialized:
                self._lazy_init()

    def _inference_tick(self) -> None:
        """Run inference on the latest frame pair."""
        if self._latest_rgb is None or self._latest_depth is None:
            return
        if not self._enabled:
            return

        self._frame_count += 1
        if self._frame_count % self._inference_interval != 0:
            return

        if not self._lazy_init():
            return

        rgb = self._latest_rgb
        depth = self._latest_depth

        try:
            result = self._engine.process_frame(rgb, depth)
        except Exception as e:
            logger.error("Inference error: %s", e)
            return

        if result is None:
            return

        now = time.time()

        # Publish punch detection
        det_msg = PunchDetection()
        det_msg.timestamp = now
        det_msg.punch_type = result.action
        det_msg.confidence = result.confidence
        det_msg.raw_class = result.raw_action
        self._pub_detection.publish(det_msg)

        # Publish user tracking (from YOLO pose bbox)
        self._publish_tracking(result, now)

    def _publish_tracking(self, result, timestamp: float) -> None:
        """Publish user tracking data from inference result."""
        tracking = UserTracking()
        tracking.timestamp = timestamp

        # Get bbox from the engine's internal YOLO results
        bbox = getattr(result, "bbox", None)
        if bbox is None:
            tracking.user_detected = False
            self._pub_tracking.publish(tracking)
            return

        tracking.user_detected = True
        tracking.bbox_centre_x = bbox.get("cx", 0.0)
        tracking.bbox_centre_y = bbox.get("cy", 0.0)
        tracking.bbox_top_y = bbox.get("top_y", 0.0)
        tracking.bbox_width = bbox.get("width", 0.0)
        tracking.bbox_height = bbox.get("height", 0.0)
        tracking.depth = bbox.get("depth", 0.0)

        # Compute displacement from baseline
        if self._baseline_bbox_x is None and tracking.user_detected:
            self._baseline_bbox_x = tracking.bbox_centre_x
            self._baseline_depth = tracking.depth

        if self._baseline_bbox_x is not None:
            tracking.lateral_displacement = tracking.bbox_centre_x - self._baseline_bbox_x
        if self._baseline_depth is not None:
            tracking.depth_displacement = tracking.depth - self._baseline_depth

        self._pub_tracking.publish(tracking)


def main(args=None) -> None:
    """Entry point for the CV node."""
    rclpy.init(args=args)
    node = CvNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
