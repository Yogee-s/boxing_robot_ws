#!/usr/bin/env python3
"""Wrapper: runs the original LiveVoxelGUI + publishes all CV data to ROS.

Exact same inference as run.py, plus a background ROS node that publishes:
  - PunchDetection (for punch_processor fusion)
  - debug_info (for GUI CV display + FPS)
  - person_direction (for yaw motor)
  - camera frames (for reaction test page)
  - UserTracking (for defense detection + height adjustment)

Usage: cd action_prediction && python3 ../notebooks/scripts/run_with_ros.py
"""
import json
import os
import sys
import threading
import time

_WS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_AP = os.path.join(_WS, "action_prediction")
if _AP not in sys.path:
    sys.path.insert(0, _AP)

# Initialize ROS BEFORE importing Tkinter (avoids conflicts)
import rclpy
from rclpy.node import Node
import numpy as np

from std_msgs.msg import String as _StdString

try:
    from boxbunny_msgs.msg import PunchDetection, PoseEstimate, UserTracking
except ImportError:
    print("ERROR: boxbunny_msgs not found. Source install/setup.bash first.")
    sys.exit(1)

try:
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge
    _HAS_CV_BRIDGE = True
except ImportError:
    _HAS_CV_BRIDGE = False

import cv2

rclpy.init()


class _CVPub(Node):
    def __init__(self):
        super().__init__("cv_ros_bridge")
        # Predictions
        self._pub = self.create_publisher(
            PunchDetection, "/boxbunny/cv/detection", 10)
        self._pub_debug = self.create_publisher(
            _StdString, "/boxbunny/cv/debug_info", 10)
        self._pub_direction = self.create_publisher(
            _StdString, "/boxbunny/cv/person_direction", 10)
        # Tracking
        self._pub_tracking = self.create_publisher(
            UserTracking, "/boxbunny/cv/user_tracking", 10)
        # Camera frames (for reaction test + other consumers)
        if _HAS_CV_BRIDGE:
            self._pub_color = self.create_publisher(
                Image, "/camera/color/image_raw", 5)
            self._pub_depth = self.create_publisher(
                Image, "/camera/aligned_depth_to_color/image_raw", 5)
            self._cv_bridge = CvBridge()
        else:
            self._pub_color = None

        self._last = "idle"
        self._consec = 0
        self._last_direction = "centre"
        # Pose frame (grayscale + skeleton overlay for reaction test)
        # Best-effort QoS with depth 1 for low-latency streaming
        if _HAS_CV_BRIDGE:
            from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
            _stream_qos = QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
            )
            self._pub_pose_frame = self.create_publisher(
                Image, "/boxbunny/cv/pose_frame", _stream_qos)
        else:
            self._pub_pose_frame = None

        self._baseline_cx = None
        self._baseline_depth = None
        self._frame_pub_counter = 0
        self._prev_kps = None
        self._last_kps_id = None  # track when keypoints change
        self.get_logger().info(
            "CV ROS bridge ready (predictions + tracking + camera + pose)")

    def send(self, action: str, confidence: float):
        if action == self._last:
            self._consec += 1
        else:
            self._last = action
            self._consec = 1
        msg = PunchDetection()
        msg.timestamp = time.time()
        msg.punch_type = action
        msg.confidence = float(confidence)
        msg.raw_class = action
        msg.consecutive_frames = self._consec
        self._pub.publish(msg)

    def send_debug(self, action: str, conf: float, fps: float, consec: int):
        msg = _StdString()
        msg.data = json.dumps({
            "action": action,
            "confidence": round(conf, 3),
            "consecutive": consec,
            "raw": action,
            "fps": round(float(fps), 1),
            "movement_delta": 0.0,
        })
        self._pub_debug.publish(msg)

    def send_direction(self, bbox_cx: float, frame_width: float):
        w = frame_width
        # Centre dead-zone = middle 10% of the frame. Tighter than before
        # (was 30%) so the robot tracks lateral movement earlier.
        # Hysteresis of 20 px still prevents chatter at the boundary.
        left_b, right_b = w * 0.45, w * 0.55
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
        m = _StdString()
        m.data = new
        self._pub_direction.publish(m)

    def send_tracking(self, bbox, depth_frame, dsw=384.0, rgb_shape=None):
        """Publish UserTracking from YOLO bbox + depth frame.

        Bbox is in downscaled coordinates. We publish the normalized
        position and scale to full-res only for the depth lookup.
        """
        msg = UserTracking()
        msg.timestamp = time.time()
        if bbox is None or len(bbox) < 4:
            msg.user_detected = False
            self._pub_tracking.publish(msg)
            return
        x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        msg.bbox_centre_x = cx
        msg.bbox_centre_y = cy
        msg.bbox_top_y = y1
        msg.bbox_width = x2 - x1
        msg.bbox_height = y2 - y1
        msg.user_detected = True
        # Depth lookup: scale bbox centre to full-res depth frame coordinates
        if depth_frame is not None and rgb_shape is not None:
            try:
                scale = float(rgb_shape[1]) / max(dsw, 1)
                dx = int(cx * scale)
                dy = int(cy * scale)
                if 0 <= dy < depth_frame.shape[0] and 0 <= dx < depth_frame.shape[1]:
                    depth_mm = float(depth_frame[dy, dx])
                    msg.depth = depth_mm * 0.001
            except Exception:
                pass
        # Lateral displacement from baseline (in downscaled coords — consistent)
        if self._baseline_cx is None and msg.user_detected:
            self._baseline_cx = cx
            self._baseline_depth = msg.depth
        if self._baseline_cx is not None:
            msg.lateral_displacement = cx - self._baseline_cx
        if self._baseline_depth is not None and msg.depth > 0:
            msg.depth_displacement = msg.depth - self._baseline_depth
        self._pub_tracking.publish(msg)

    def send_frames(self, rgb, depth):
        """Publish camera frames (every 3rd call to save bandwidth)."""
        if self._pub_color is None:
            return
        self._frame_pub_counter += 1
        if self._frame_pub_counter % 3 != 0:
            return
        try:
            self._pub_color.publish(
                self._cv_bridge.cv2_to_imgmsg(rgb, "bgr8"))
            if depth is not None:
                self._pub_depth.publish(
                    self._cv_bridge.cv2_to_imgmsg(depth, "passthrough"))
        except Exception:
            pass

    def send_pose_frame(self, rgb, kps, confs, downscale_width=384):
        """Publish half-res grayscale frame with pose dots.

        Keypoints are from the engine's YOLO (1-2 frame lag, acceptable).
        Only dots, no skeleton lines — keeps it clean and fast.
        """
        if self._pub_pose_frame is None:
            return
        h, w = rgb.shape[:2]
        out_w, out_h = w // 2, h // 2
        small = cv2.resize(rgb, (out_w, out_h), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        if kps is not None:
            scale = float(out_w) / max(downscale_width, 1)
            for i in range(len(kps)):
                c = float(confs[i]) if confs is not None and i < len(confs) else 0.5
                if c > 0.3:
                    x, y = int(kps[i][0] * scale), int(kps[i][1] * scale)
                    cv2.circle(gray, (x, y), 3, 255, -1)

        try:
            self._pub_pose_frame.publish(
                self._cv_bridge.cv2_to_imgmsg(gray, "mono8"))
        except Exception:
            pass

    def reset_baseline(self):
        self._baseline_cx = None
        self._baseline_depth = None


_ros_node = _CVPub()
_ros_thread = threading.Thread(target=rclpy.spin, args=(_ros_node,), daemon=True)
_ros_thread.start()


def main():
    import tkinter as tk
    from live_voxelflow_inference import LiveVoxelGUI
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='model/best_model.pth')
    parser.add_argument('--pose-weights', dest='fusion_pose_weights', default='model/yolo26n-pose.engine')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--show-video', action='store_true', default=False)
    parser.add_argument('--no-video', action='store_true', default=True)
    parser.add_argument('--optimize-gpu', action='store_true', default=True)
    parser.add_argument('--no-optimize-gpu', action='store_false', dest='optimize_gpu')
    parser.add_argument('--inference-interval', type=int, default=1)
    parser.add_argument('--yolo-interval', type=int, default=1)
    parser.add_argument('--downscale-width', type=int, default=384)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--temporal-smooth-window', type=int, default=1)
    parser.add_argument('--min-confidence', type=float, default=0.8)
    parser.add_argument('--min-action-prob', type=float, default=0.0)
    parser.add_argument('--min-class-margin', type=float, default=0.0)
    parser.add_argument('--min-voxel-active-ratio', type=float, default=0.0)
    parser.add_argument('--ema-alpha', type=float, default=0.65)
    parser.add_argument('--hysteresis-margin', type=float, default=0.04)
    parser.add_argument('--min-hold-frames', type=int, default=1)
    parser.add_argument('--use-action-state-machine', action='store_true')
    parser.add_argument('--state-enter-consecutive', type=int, default=2)
    parser.add_argument('--state-exit-consecutive', type=int, default=3)
    parser.add_argument('--state-min-hold-steps', type=int, default=3)
    parser.add_argument('--state-sustain-confidence', type=float, default=0.35)
    parser.add_argument('--state-peak-drop-threshold', type=float, default=0.40)
    parser.add_argument('--rgb-res', type=str, default='960x540')
    parser.add_argument('--depth-res', type=str, default='848x480')
    parser.add_argument('--processing-mode', type=str, default='strict')
    parser.add_argument('--camera-pitch', type=float, default=5.0)
    parser.add_argument('--no-auto-pitch', action='store_true')
    parser.add_argument('--frame-sample-rate', type=int, default=1)
    parser.add_argument('--window-size', type=int, default=12)
    parser.add_argument('--feature-queue-size', type=int, default=0)
    parser.add_argument('--result-queue-size', type=int, default=0)
    parser.add_argument('--sensor-queue-size', type=int, default=0)
    parser.add_argument('--yolo-checkpoint', type=str, default='')
    parser.add_argument('--no-yolo', action='store_true')
    args = parser.parse_args()

    root = tk.Tk()
    # Hide the Tkinter window when running as a background service.
    # The LiveVoxelGUI event loop still runs internally for threading.
    if os.environ.get("CV_HEADLESS") == "1":
        root.withdraw()
    app = LiveVoxelGUI(
        root,
        checkpoint_path=args.checkpoint,
        device=args.device,
        window_size=args.window_size,
        frame_sample_rate=args.frame_sample_rate,
        inference_interval=args.inference_interval,
        temporal_smooth_window=args.temporal_smooth_window,
        min_confidence=args.min_confidence,
        min_action_prob=args.min_action_prob,
        min_class_margin=args.min_class_margin,
        min_voxel_active_ratio=args.min_voxel_active_ratio,
        use_action_state_machine=args.use_action_state_machine,
        state_enter_consecutive=args.state_enter_consecutive,
        state_exit_consecutive=args.state_exit_consecutive,
        state_min_hold_steps=args.state_min_hold_steps,
        state_sustain_confidence=args.state_sustain_confidence,
        state_peak_drop_threshold=args.state_peak_drop_threshold,
        camera_pitch=args.camera_pitch,
        auto_pitch=not args.no_auto_pitch,
        rgb_res=args.rgb_res,
        depth_res=args.depth_res,
        downscale_width=args.downscale_width,
        processing_mode=args.processing_mode,
        feature_queue_size=args.feature_queue_size,
        result_queue_size=args.result_queue_size,
        sensor_queue_size=args.sensor_queue_size,
        num_workers=args.num_workers,
        yolo_checkpoint=args.yolo_checkpoint,
        use_yolo=not args.no_yolo,
        yolo_interval=args.yolo_interval,
        no_video=not args.show_video and args.no_video,
        fusion_pose_weights=args.fusion_pose_weights,
        optimize_gpu=args.optimize_gpu,
        ema_alpha=args.ema_alpha,
        hysteresis_margin=args.hysteresis_margin,
        min_hold_frames=args.min_hold_frames,
    )

    # Poll LiveVoxelGUI state and publish everything to ROS at ~30Hz
    def _pub_loop():
        # ── Predictions ──────────────────────────────────────────────
        # Publish the RAW model prediction (before EMA smoothing) for
        # fusion.  smooth_probs transitions slowly between classes due
        # to EMA + windowed mean, which causes the fusion node to see
        # stale predictions (e.g. "jab" when you're already throwing a
        # "cross").  The fusion node has its own frame-counting to
        # handle single-frame noise, so it doesn't need pre-smoothed
        # input.  The GUI still uses smooth_probs for stable display.
        try:
            labels = getattr(app, 'labels', None)
            # Use raw (unsmoothed) probs for ROS fusion publishing
            raw_deque = getattr(app, 'recent_probs', None)
            raw_probs = None
            if raw_deque and len(raw_deque) > 0:
                raw_probs = raw_deque[-1]

            if raw_probs is not None and labels is not None and len(raw_probs) == len(labels):
                idx = int(np.argmax(raw_probs))
                action = labels[idx]
                conf = float(raw_probs[idx])
                if conf < 0.2:
                    action = "idle"
                    conf = 0.0
                _ros_node.send(action, conf)
            elif labels is not None:
                # Fallback to smooth_probs if raw not available yet
                probs = getattr(app, 'smooth_probs', None)
                if probs is not None and len(probs) == len(labels):
                    idx = int(np.argmax(probs))
                    action = labels[idx]
                    conf = float(probs[idx])
                    if conf < 0.2:
                        action = "idle"
                        conf = 0.0
                    _ros_node.send(action, conf)
                else:
                    pred = getattr(app, 'current_prediction', None)
                    conf = getattr(app, 'current_confidence', 0.0)
                    if pred and isinstance(pred, str) and pred != "Initializing...":
                        _ros_node.send(pred, conf)

            # Debug info (always uses smooth for stable FPS display)
            probs = getattr(app, 'smooth_probs', None)
            if probs is not None and labels is not None and len(probs) == len(labels):
                inf_hist = getattr(app, 'inference_time_history', None)
                fps = 0.0
                if inf_hist and len(inf_hist) >= 2:
                    t0 = float(inf_hist[0])
                    t1 = float(inf_hist[-1])
                    if t1 > t0:
                        fps = (len(inf_hist) - 1) / (t1 - t0)
                # Show the smoothed prediction in debug (matches GUI)
                s_idx = int(np.argmax(probs))
                s_action = labels[s_idx]
                s_conf = float(probs[s_idx])
                if s_conf < 0.2:
                    s_action = "idle"
                    s_conf = 0.0
                consec = _ros_node._consec
                _ros_node.send_debug(s_action, s_conf, fps, consec)
        except Exception:
            pass

        # ── Grab latest frame + pose data ────────────────────────────
        bbox = None
        rgb_frame = None
        depth_frame = None
        kps = None
        kps_confs = None
        try:
            bbox = getattr(app, '_latest_pose_bbox', None)
            kps = getattr(app, '_latest_pose_kps', None)
            kps_confs = getattr(app, '_latest_pose_confs', None)
            packet = getattr(app, '_latest_frame_packet', None)
            if packet is not None and len(packet) >= 4:
                rgb_frame = packet[2]
                depth_frame = packet[3]
        except Exception:
            pass

        # ── Person direction + tracking from YOLO bbox ───────────────
        # Bbox is in downscaled coordinates — use downscale_width as
        # frame_width for direction so relative position is correct.
        try:
            dsw = float(getattr(app, 'downscale_width', 384) or 384)
            if bbox is not None and len(bbox) >= 4:
                cx = (bbox[0] + bbox[2]) / 2.0
                _ros_node.send_direction(cx, dsw)
                _ros_node.send_tracking(bbox, depth_frame, dsw,
                                        rgb_frame.shape if rgb_frame is not None else None)
            else:
                _ros_node.send_tracking(None, None)
        except Exception:
            pass

        # ── Pose frame (grayscale + pose dots for reaction test) ─────
        try:
            if rgb_frame is not None:
                dsw = getattr(app, 'downscale_width', 384) or 384
                _ros_node.send_pose_frame(rgb_frame, kps, kps_confs, dsw)
        except Exception:
            pass

        # ── Camera frames (for other consumers) ─────────────────────
        try:
            if rgb_frame is not None:
                _ros_node.send_frames(rgb_frame, depth_frame)
        except Exception:
            pass

        try:
            root.after(33, _pub_loop)
        except Exception:
            pass

    root.after(2000, _pub_loop)  # wait 2s for model to load
    root.mainloop()

    _ros_node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
