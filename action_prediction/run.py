#!/usr/bin/env python3
"""
Standalone Boxing Action Recognition — Live Inference

Self-contained launcher that runs real-time action recognition from an
Intel RealSense D435i camera. All model and feature extraction code is
bundled in this folder — no external project imports needed.

Requirements:
    pip install torch numpy opencv-python pyrealsense2 ultralytics

Usage:
    # Minimal (all defaults from checkpoint):
    python run.py --checkpoint model/best_model.pth --pose-weights model/yolo26n-pose.pt

    # Full control:
    python run.py \\
        --checkpoint model/best_model.pth \\
        --pose-weights model/yolo26n-pose.pt \\
        --device cuda:0 \\
        --inference-interval 1 \\
        --yolo-interval 5 \\
        --downscale-width 384 \\
        --temporal-smooth-window 1 \\
        --min-confidence 0.7 \\
        --ema-alpha 0.65 \\
        --hysteresis-margin 0.04 \\
        --min-hold-frames 1 \\
        --processing-mode strict \\
        --depth-res 848x480 \\
        --optimize-gpu \\
        --no-video
"""

import argparse
import os
import sys

# Add parent directory to path so action_prediction.* imports resolve
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def main():
    parser = argparse.ArgumentParser(
        description='Boxing Action Recognition — Live Inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic:
    python run.py --checkpoint model.pth --pose-weights yolo26n-pose.pt

    # Optimized for Jetson (ONNX+TensorRT auto-conversion):
    python run.py --checkpoint model.pth --pose-weights yolo26n-pose.pt --optimize-gpu --no-video

    # Responsive (for detecting repeated punches like jab-jab-jab):
    python run.py --checkpoint model.pth --pose-weights yolo26n-pose.pt \\
        --ema-alpha 0.65 --hysteresis-margin 0.04 --min-hold-frames 1
""")

    # =====================================================================
    # MODEL (required)
    # =====================================================================
    parser.add_argument('--checkpoint', default='model/best_model.pth',
                        help='Trained model checkpoint (.pth). All model config '
                             'is auto-read from checkpoint metadata. '
                             '(default: model/best_model.pth)')
    parser.add_argument('--pose-weights', dest='fusion_pose_weights',
                        default='model/yolo26n-pose.pt',
                        help='YOLO Pose model weights for skeleton detection '
                             '(default: yolo26n-pose.pt)')
    parser.add_argument('--device', default='cuda:0',
                        help='Inference device (default: cuda:0)')

    # =====================================================================
    # SPEED vs ACCURACY
    # =====================================================================
    parser.add_argument('--inference-interval', type=int, default=1,
                        help='Predict every Nth sampled frame. '
                             '1=every frame (most responsive), 2=skip half. Default: 1')
    parser.add_argument('--yolo-interval', type=int, default=5,
                        help='Run YOLO pose every Nth frame, reuse cached pose between. '
                             'Higher=faster but less accurate pose. Default: 5')
    parser.add_argument('--downscale-width', type=int, default=384,
                        help='Downscale depth frames to this width for voxel extraction. '
                             'Lower=faster. 256/384/480. Default: 384')
    parser.add_argument('--num-workers', type=int, default=1,
                        help='Parallel feature extraction workers. Default: 1')

    # =====================================================================
    # POST-PROCESSING / RESPONSIVENESS
    # =====================================================================
    parser.add_argument('--temporal-smooth-window', type=int, default=1,
                        help='Smooth predictions over N frames. '
                             '1=raw (responsive), 3-5=more stable. Default: 1')
    parser.add_argument('--min-confidence', type=float, default=0.7,
                        help='Below this confidence -> predict idle. '
                             '0.0=disabled, 0.9=very strict. Default: 0.7')
    parser.add_argument('--min-action-prob', type=float, default=0.0,
                        help='Min non-idle probability. 0.0=disabled. Default: 0.0')
    parser.add_argument('--min-class-margin', type=float, default=0.0,
                        help='Min top1-top2 probability gap. 0.0=disabled. Default: 0.0')
    parser.add_argument('--min-voxel-active-ratio', type=float, default=0.0,
                        help='Min active voxel ratio. 0.0=disabled. Default: 0.0')
    parser.add_argument('--ema-alpha', type=float, default=0.65,
                        help='EMA weight for new predictions. '
                             '0.35=smooth, 0.65=responsive, 1.0=raw. Default: 0.65')
    parser.add_argument('--hysteresis-margin', type=float, default=0.04,
                        help='Margin needed to switch prediction class. '
                             '0.12=sticky, 0.04=responsive, 0.0=instant. Default: 0.04')
    parser.add_argument('--min-hold-frames', type=int, default=1,
                        help='Hold current prediction for at least N frames. '
                             '3=sticky, 1=responsive. Default: 1')

    # =====================================================================
    # STATE MACHINE (optional — disabled by default)
    # =====================================================================
    parser.add_argument('--use-action-state-machine', action='store_true',
                        help='Enable causal action state machine filter')
    parser.add_argument('--state-enter-consecutive', type=int, default=2,
                        help='Consecutive frames needed to enter an action. Default: 2')
    parser.add_argument('--state-exit-consecutive', type=int, default=3,
                        help='Consecutive frames needed to exit an action. Default: 3')
    parser.add_argument('--state-min-hold-steps', type=int, default=3,
                        help='Min steps to hold an action before exit allowed. Default: 3')
    parser.add_argument('--state-sustain-confidence', type=float, default=0.35,
                        help='Confidence below this counts toward exit. Default: 0.35')
    parser.add_argument('--state-peak-drop-threshold', type=float, default=0.40,
                        help='Drop from peak confidence to trigger exit. Default: 0.40')

    # =====================================================================
    # CAMERA
    # =====================================================================
    parser.add_argument('--rgb-res', type=str, default='960x540',
                        help='RGB stream resolution WxH. Default: 960x540')
    parser.add_argument('--depth-res', type=str, default='848x480',
                        help='Depth stream resolution WxH. Default: 848x480')
    parser.add_argument('--processing-mode', type=str, default='strict',
                        choices=['latest', 'strict'],
                        help='strict=preserve frame order, latest=low latency. Default: strict')
    parser.add_argument('--camera-pitch', type=float, default=0.0,
                        help='Camera pitch in degrees (0=auto-detect from IMU). Default: 0')
    parser.add_argument('--no-auto-pitch', action='store_true',
                        help='Disable automatic IMU pitch detection')

    # =====================================================================
    # GPU OPTIMIZATION
    # =====================================================================
    parser.add_argument('--optimize-gpu', action='store_true', default=True,
                        help='Auto-convert model to ONNX + TensorRT (FP16). '
                             'First run takes 1-2 min, subsequent runs load cached engine. '
                             '(default: on)')
    parser.add_argument('--no-optimize-gpu', action='store_false', dest='optimize_gpu',
                        help='Disable GPU optimization, use raw PyTorch')

    # =====================================================================
    # OUTPUT
    # =====================================================================
    parser.add_argument('--no-video', action='store_true', default=True,
                        help='Disable video rendering for maximum throughput (default: on)')
    parser.add_argument('--show-video', action='store_false', dest='no_video',
                        help='Enable video rendering')
    parser.add_argument('--no-yolo', action='store_true',
                        help='Disable YOLO detection overlay')

    # =====================================================================
    # ADVANCED (rarely needed)
    # =====================================================================
    parser.add_argument('--frame-sample-rate', type=int, default=1,
                        help='Sample every Nth camera frame. Default: 1')
    parser.add_argument('--window-size', type=int, default=12,
                        help='Context window size (auto from checkpoint). Default: 12')
    parser.add_argument('--feature-queue-size', type=int, default=0,
                        help='Feature job queue size (0=auto). Default: 0')
    parser.add_argument('--result-queue-size', type=int, default=0,
                        help='Result queue size (0=auto). Default: 0')
    parser.add_argument('--sensor-queue-size', type=int, default=0,
                        help='RealSense internal queue size (0=auto). Default: 0')
    parser.add_argument('--yolo-checkpoint', type=str, default='',
                        help='YOLO person detection model (separate from pose). Default: none')

    args = parser.parse_args()

    # Import the live inference GUI (from the tools package)
    import tkinter as tk
    from action_prediction.live_voxelflow_inference import LiveVoxelGUI

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
    root.mainloop()


if __name__ == '__main__':
    main()
