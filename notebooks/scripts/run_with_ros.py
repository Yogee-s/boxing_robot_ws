#!/usr/bin/env python3
"""Wrapper: runs the original LiveVoxelGUI + publishes predictions to ROS.

Exact same inference as run.py, plus a background ROS node that publishes
PunchDetection at 30Hz so punch_processor can fuse with IMU data.

Usage: cd action_prediction && python3 ../notebooks/scripts/run_with_ros.py
"""
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

try:
    from boxbunny_msgs.msg import PunchDetection
except ImportError:
    print("ERROR: boxbunny_msgs not found. Source install/setup.bash first.")
    sys.exit(1)

rclpy.init()


class _CVPub(Node):
    def __init__(self):
        super().__init__("cv_ros_bridge")
        self._pub = self.create_publisher(PunchDetection, "/boxbunny/cv/detection", 10)
        self._last = "idle"
        self._consec = 0
        self.get_logger().info("Publishing CV predictions to /boxbunny/cv/detection")

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
    parser.add_argument('--show-video', action='store_false', dest='no_video')
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
        no_video=args.no_video,
        fusion_pose_weights=args.fusion_pose_weights,
        optimize_gpu=args.optimize_gpu,
        ema_alpha=args.ema_alpha,
        hysteresis_margin=args.hysteresis_margin,
        min_hold_frames=args.min_hold_frames,
    )

    # Poll predictions and publish to ROS at 30Hz
    # Use raw (ungated) probabilities so the CV buffer has real predictions.
    # The inference GUI gates predictions below min-confidence to "idle",
    # but we need the raw prediction for IMU pad fusion to filter properly.
    def _pub_loop():
        try:
            import numpy as np
            probs = getattr(app, 'smooth_probs', None)
            labels = getattr(app, 'labels', None)
            if probs is not None and labels is not None and len(probs) == len(labels):
                idx = int(np.argmax(probs))
                action = labels[idx]
                conf = float(probs[idx])
                _ros_node.send(action, conf)
            else:
                # Fallback to gated prediction during startup
                pred = getattr(app, 'current_prediction', None)
                conf = getattr(app, 'current_confidence', 0.0)
                if pred and isinstance(pred, str) and pred != "Initializing...":
                    _ros_node.send(pred, conf)
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
