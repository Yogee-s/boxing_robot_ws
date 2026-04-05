#!/usr/bin/env python3
"""Headless CV inference with ROS publishing.

Runs the InferenceEngine in a tight loop (same as the 4c fusion test)
and publishes PunchDetection + debug_info to ROS topics.  No GUI window.

Usage (from conda boxing_ai with ROS sourced):
    python3 notebooks/scripts/cv_ros_headless.py
"""
import json
import logging
import os
import sys
import threading
import time

_WS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_AP = os.path.join(_WS, "action_prediction")
if _AP not in sys.path:
    sys.path.insert(0, _AP)
if _WS not in sys.path:
    sys.path.insert(0, _WS)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
logger = logging.getLogger("cv_ros_headless")

import numpy as np

# ROS setup
import rclpy
from rclpy.node import Node
from std_msgs.msg import String as StdString

try:
    from boxbunny_msgs.msg import PunchDetection, SessionState
except ImportError:
    logger.error("boxbunny_msgs not found. Source install/setup.bash first.")
    sys.exit(1)

try:
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge
    _HAS_CV_BRIDGE = True
except ImportError:
    _HAS_CV_BRIDGE = False


class CVPublisher(Node):
    """ROS node that publishes CV predictions and camera frames."""

    def __init__(self) -> None:
        super().__init__("cv_node")
        self._pub_detection = self.create_publisher(
            PunchDetection, "/boxbunny/cv/detection", 10)
        self._pub_debug = self.create_publisher(
            StdString, "/boxbunny/cv/debug_info", 10)
        self._pub_direction = self.create_publisher(
            StdString, "/boxbunny/cv/person_direction", 10)

        # Camera frame republishing for other consumers (reaction test, etc.)
        if _HAS_CV_BRIDGE:
            self._pub_color = self.create_publisher(
                Image, "/camera/color/image_raw", 5)
            self._pub_depth = self.create_publisher(
                Image, "/camera/aligned_depth_to_color/image_raw", 5)
            self._cv_bridge = CvBridge()
        else:
            self._pub_color = None

        self._last_action = "idle"
        self._consec = 0
        self._last_direction = "centre"
        self.get_logger().info("CV headless node ready")

    def publish_prediction(
        self, action: str, confidence: float, fps: float,
        consecutive: int, raw_action: str,
    ) -> None:
        """Publish PunchDetection and debug_info."""
        if action == self._last_action:
            self._consec += 1
        else:
            self._last_action = action
            self._consec = 1

        det = PunchDetection()
        det.timestamp = time.time()
        det.punch_type = action
        det.confidence = float(confidence)
        det.raw_class = raw_action
        det.consecutive_frames = self._consec
        self._pub_detection.publish(det)

        debug = StdString()
        debug.data = json.dumps({
            "action": action,
            "confidence": round(float(confidence), 3),
            "consecutive": consecutive,
            "raw": raw_action,
            "fps": round(fps, 1),
            "movement_delta": 0.0,
        })
        self._pub_debug.publish(debug)

    def publish_direction(self, bbox_cx: float, frame_width: float) -> None:
        """Publish person direction from YOLO bbox centre."""
        w = frame_width
        left_b, right_b = w * 0.35, w * 0.65
        hyst = 20.0
        d = self._last_direction
        if d == "centre":
            new = "left" if bbox_cx < left_b - hyst else (
                "right" if bbox_cx > right_b + hyst else "centre")
        elif d == "left":
            new = "left" if bbox_cx < left_b + hyst else (
                "right" if bbox_cx > right_b else "centre")
        elif d == "right":
            new = "right" if bbox_cx > right_b - hyst else (
                "left" if bbox_cx < left_b else "centre")
        else:
            new = "centre"
        self._last_direction = new
        msg = StdString()
        msg.data = new
        self._pub_direction.publish(msg)

    def publish_frames(
        self, rgb: np.ndarray, depth: np.ndarray,
    ) -> None:
        """Re-publish camera frames for other ROS consumers."""
        if self._pub_color is None:
            return
        try:
            self._pub_color.publish(
                self._cv_bridge.cv2_to_imgmsg(rgb, "bgr8"))
            self._pub_depth.publish(
                self._cv_bridge.cv2_to_imgmsg(depth, "passthrough"))
        except Exception:
            pass


def main() -> None:
    import pyrealsense2 as rs
    from action_prediction.lib.inference_runtime import InferenceEngine

    # Resolve model paths
    model_dir = os.path.join(_AP, "model")
    checkpoint = os.path.join(model_dir, "best_model.pth")
    yolo_engine = os.path.join(model_dir, "yolo26n-pose.engine")
    yolo_pt = os.path.join(model_dir, "yolo26n-pose.pt")
    yolo_path = yolo_engine if os.path.exists(yolo_engine) else yolo_pt

    # Init engine (same params as run_with_ros / 4c test)
    logger.info("Loading inference engine: %s", checkpoint)
    engine = InferenceEngine(
        checkpoint_path=checkpoint,
        yolo_model_path=yolo_path,
        device="cuda:0",
        window_size=12,
        inference_interval=1,
        yolo_interval=1,
        optimize_gpu=True,
        downscale_width=384,
        ema_alpha=0.65,
        hysteresis_margin=0.04,
        min_hold_frames=1,
        min_confidence=0.8,
    )
    engine.initialize()
    logger.info("Engine loaded")

    # Init ROS
    rclpy.init()
    node = CVPublisher()
    ros_thread = threading.Thread(
        target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    # Open camera (same config as 4c test)
    pipeline = rs.pipeline()
    rs_cfg = rs.config()
    rs_cfg.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
    rs_cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    align = rs.align(rs.stream.color)
    pipeline.start(rs_cfg)
    logger.info("Camera started (960x540@30fps)")

    frame_count = 0
    fps_timer = time.time()
    fps = 0.0

    try:
        while rclpy.ok():
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            rgb = np.asanyarray(color_frame.get_data())
            depth = np.asanyarray(depth_frame.get_data())

            # Run inference (frames stay in scope — asanyarray is safe)
            result = engine.process_frame(rgb, depth)

            if result is not None:
                node.publish_prediction(
                    action=result.action,
                    confidence=result.confidence,
                    fps=fps,
                    consecutive=result.consecutive_frames,
                    raw_action=result.raw_action,
                )
                # Person direction from YOLO bbox
                if result.bbox is not None:
                    cx = result.bbox.get("cx", 0.0)
                    node.publish_direction(cx, float(rgb.shape[1]))

            # Re-publish camera frames every 3rd frame (saves bandwidth)
            if frame_count % 3 == 0:
                node.publish_frames(rgb, depth)

            # FPS tracking
            frame_count += 1
            elapsed = time.time() - fps_timer
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                fps_timer = time.time()

    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        node.destroy_node()
        rclpy.try_shutdown()
        logger.info("CV headless stopped")


if __name__ == "__main__":
    main()
