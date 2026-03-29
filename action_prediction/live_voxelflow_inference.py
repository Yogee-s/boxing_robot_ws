#!/usr/bin/env python3
"""
Live Boxing Action Recognition with Intel RealSense D435i Camera.

Real-time action recognition using fused depth voxel + 2D pose features:
- 2-channel voxel occupancy deltas (12^3 volumetric motion, gravity-aligned)
- 42-dim YOLO Pose features (joint coords, velocities, elbow angles)
- FusionVoxelPoseTransformerModel with bidirectional temporal attention

Optimized for Jetson Orin Nano:
- 30fps native capture and inference
- Optional ONNX + TensorRT (FP16) auto-conversion
- Batch YOLO pose with configurable interval
- Configurable EMA smoothing and hysteresis for responsiveness

Usage:
    python action_prediction/run.py \\
        --checkpoint model/best_model.pth \\
        --pose-weights model/yolo26n-pose.pt
"""

import argparse
import json
import os
import pickle
import queue
import sys
import threading
import time
import tkinter as tk
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import cv2
import numpy as np
from PIL import Image, ImageTk

try:
    import torch
except ImportError:
    print("Error: PyTorch not found", file=sys.stderr)
    sys.exit(1)

try:
    import pyrealsense2 as rs
except ImportError:
    print("Error: pyrealsense2 not found", file=sys.stderr)
    sys.exit(1)

# Optional YOLO for person-only depth masking
_YOLO_AVAILABLE = False
try:
    from ultralytics import YOLO as _UltralyticsYOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _UltralyticsYOLO = None

# Optional ONNX Runtime for accelerated inference
_ORT_AVAILABLE = False
try:
    import onnxruntime as ort
    # Only count ORT as available if it has a GPU provider
    if any('CUDA' in p or 'Tensorrt' in p for p in ort.get_available_providers()):
        _ORT_AVAILABLE = True
    else:
        _ORT_AVAILABLE = False
except ImportError:
    ort = None

# Optional TensorRT for direct engine inference (Jetson)
_TRT_AVAILABLE = False
try:
    import tensorrt as trt
    _TRT_AVAILABLE = True
except ImportError:
    trt = None

from action_prediction.lib.voxel_features import (
    VoxelFeatureConfig,
    BackgroundModel,
    VoxelOccupancyExtractor,
)
from action_prediction.lib.voxel_model import count_parameters
from action_prediction.lib.fusion_model import (
    FusionVoxelPoseTransformerModel,
    extract_pose_features,
    POSE_FEATURE_DIM,
)
from action_prediction.lib.pose import YOLOPoseEstimator


# Default 8-class label set (matches training)
DEFAULT_LABELS = ['block', 'cross', 'idle', 'jab',
                  'left_hook', 'left_uppercut', 'right_hook', 'right_uppercut']


def _coerce_voxel_size(value, fallback: Optional[Tuple[int, int, int]] = None) -> Tuple[int, int, int]:
    """Normalize voxel size metadata to a 3D tuple."""
    if value is None:
        if fallback is None:
            raise ValueError("voxel size metadata is missing")
        return fallback
    if isinstance(value, int):
        n = int(value)
        return (n, n, n)
    if isinstance(value, (list, tuple, np.ndarray)):
        seq = tuple(int(v) for v in list(value))
        if len(seq) == 1:
            return (seq[0], seq[0], seq[0])
        if len(seq) >= 3:
            return tuple(seq[:3])
    if fallback is not None:
        return fallback
    raise ValueError(f"Invalid voxel size metadata: {value!r}")


def _resolve_checkpoint_feature_layout(checkpoint: dict, default_voxel_size: int = 24) -> Dict[str, object]:
    """Extract model + feature metadata needed for live voxel extraction."""
    config = checkpoint.get('config', {})
    if not isinstance(config, dict):
        config = {}
    dataset_config = checkpoint.get('dataset_config', config.get('dataset_config', {}))
    if not isinstance(dataset_config, dict):
        dataset_config = {}

    fallback_size = (int(default_voxel_size), int(default_voxel_size), int(default_voxel_size))
    voxel_size = _coerce_voxel_size(
        checkpoint.get('voxel_size', dataset_config.get('voxel_grid_size')),
        fallback=fallback_size,
    )
    directional_gradients = bool(dataset_config.get('directional_gradients', False))
    multi_scale_delta_frames = tuple(int(v) for v in (dataset_config.get('multi_scale_delta_frames') or []))
    inferred_in_channels = (4 if directional_gradients else 1) * (
        len(multi_scale_delta_frames) if multi_scale_delta_frames else 1
    )
    in_channels = int(config.get('in_channels', dataset_config.get('in_channels', inferred_in_channels)))

    # Infer velocity_magnitude_channel from actual vs base channel count.
    # With directional_gradients: 4 ch/scale without velocity, 5 ch/scale with.
    velocity_magnitude_channel = bool(dataset_config.get('velocity_magnitude_channel', False))
    if not velocity_magnitude_channel and directional_gradients and in_channels > inferred_in_channels:
        num_scales = max(1, len(multi_scale_delta_frames))
        channels_per_scale = in_channels // num_scales
        if channels_per_scale == 5:
            velocity_magnitude_channel = True

    return {
        'config': config,
        'dataset_config': dataset_config,
        'voxel_size': voxel_size,
        'in_channels': in_channels,
        'voxel_normalization': str(
            checkpoint.get('voxel_normalization', config.get('voxel_normalization', 'clip_p90'))
        ),
        'directional_gradients': directional_gradients,
        'velocity_magnitude_channel': velocity_magnitude_channel,
        'multi_scale_delta_frames': multi_scale_delta_frames,
        'voxel_delta_frames': int(dataset_config.get('voxel_delta_frames', 3)),
        'voxel_depth_weighted': bool(dataset_config.get('voxel_depth_weighted', True)),
    }


def _debug_voxel_grid(voxel_flat: np.ndarray, voxel_size: Tuple[int, int, int], in_channels: int) -> np.ndarray:
    """Return a single 3D grid for debug visualization from a flattened feature frame."""
    voxel_flat = np.asarray(voxel_flat, dtype=np.float32).reshape(-1)
    if in_channels > 1:
        return voxel_flat.reshape(in_channels, *voxel_size)[0]
    return voxel_flat.reshape(voxel_size)


def _resolve_runtime_device(device: str) -> str:
    """Resolve the requested device to a usable runtime device string."""
    requested = str(device).strip() if device is not None else 'cpu'
    if requested.startswith('cuda'):
        if torch.cuda.is_available():
            return requested
        print(f"[WARN] Requested device {requested} but CUDA is unavailable. Falling back to cpu.")
        return 'cpu'
    return requested or 'cpu'


def _select_prediction(
    probs: np.ndarray,
    labels: List[str],
    min_confidence: float,
    min_action_prob: float = 0.0,
    min_class_margin: float = 0.0,
    voxel_active_ratio: float = 0.0,
    min_voxel_active_ratio: float = 0.0,
) -> dict:
    """Apply non-idle gating on top of smoothed class probabilities."""
    probs = np.asarray(probs, dtype=np.float32)
    top_idx = int(np.argmax(probs))
    top_conf = float(probs[top_idx])
    second_conf = float(np.partition(probs, -2)[-2]) if probs.size > 1 else 0.0
    class_margin = top_conf - second_conf
    idle_idx = labels.index('idle') if 'idle' in labels else None
    action_prob = 1.0 - float(probs[idle_idx]) if idle_idx is not None else top_conf

    gate_reasons = []
    pred_idx = top_idx
    if idle_idx is not None and top_idx != idle_idx:
        if top_conf < float(min_confidence):
            gate_reasons.append('confidence')
        if action_prob < float(min_action_prob):
            gate_reasons.append('action_prob')
        if class_margin < float(min_class_margin):
            gate_reasons.append('class_margin')
        if voxel_active_ratio < float(min_voxel_active_ratio):
            gate_reasons.append('voxel_activity')
        if gate_reasons:
            pred_idx = idle_idx

    return {
        'pred_idx': pred_idx,
        'confidence': float(probs[pred_idx]),
        'top_idx': top_idx,
        'top_conf': top_conf,
        'action_prob': action_prob,
        'class_margin': class_margin,
        'idle_idx': idle_idx,
        'gated': bool(gate_reasons),
        'gate_reasons': gate_reasons,
    }


class _CausalActionStateMachine:
    """Causal event filter for live-style action outputs."""

    def __init__(
        self,
        labels: List[str],
        enter_consecutive: int,
        exit_consecutive: int,
        min_hold_steps: int,
        sustain_confidence: float,
        peak_drop_threshold: float,
    ):
        self.labels = list(labels)
        self.idle_idx = self.labels.index('idle') if 'idle' in self.labels else None
        self.enter_consecutive = max(1, int(enter_consecutive))
        self.exit_consecutive = max(1, int(exit_consecutive))
        self.min_hold_steps = max(0, int(min_hold_steps))
        self.sustain_confidence = max(0.0, float(sustain_confidence))
        self.peak_drop_threshold = max(0.0, float(peak_drop_threshold))
        self.reset()

    def reset(self):
        self.active_idx: Optional[int] = None
        self.active_steps = 0
        self.active_peak_conf = 0.0
        self.enter_candidate_idx: Optional[int] = None
        self.enter_count = 0
        self.exit_count = 0

    def update(
        self,
        probs: np.ndarray,
        proposed_idx: int,
        proposed_conf: float,
    ) -> Dict[str, object]:
        probs = np.asarray(probs, dtype=np.float32)
        idle_idx = self.idle_idx
        if idle_idx is None:
            return {
                'pred_idx': int(proposed_idx),
                'confidence': float(proposed_conf),
                'state': 'passthrough',
            }

        if self.active_idx is None:
            if int(proposed_idx) != idle_idx:
                if self.enter_candidate_idx == int(proposed_idx):
                    self.enter_count += 1
                else:
                    self.enter_candidate_idx = int(proposed_idx)
                    self.enter_count = 1

                if self.enter_count >= self.enter_consecutive:
                    self.active_idx = int(proposed_idx)
                    self.active_steps = 1
                    self.active_peak_conf = float(probs[self.active_idx])
                    self.exit_count = 0
                    self.enter_candidate_idx = None
                    self.enter_count = 0
                    return {
                        'pred_idx': self.active_idx,
                        'confidence': float(probs[self.active_idx]),
                        'state': 'activated',
                    }
            else:
                self.enter_candidate_idx = None
                self.enter_count = 0

            return {
                'pred_idx': idle_idx,
                'confidence': float(probs[idle_idx]),
                'state': 'idle',
            }

        active_idx = int(self.active_idx)
        active_conf = float(probs[active_idx])
        self.active_steps += 1
        self.active_peak_conf = max(self.active_peak_conf, active_conf)

        can_exit = self.active_steps >= self.min_hold_steps
        exit_signal = False
        exit_reasons: List[str] = []
        if int(proposed_idx) == idle_idx:
            exit_signal = True
            exit_reasons.append('idle')
        elif int(proposed_idx) != active_idx:
            exit_signal = True
            exit_reasons.append('switch')
        if self.sustain_confidence > 0.0 and active_conf < self.sustain_confidence:
            exit_signal = True
            exit_reasons.append('sustain')
        if self.peak_drop_threshold > 0.0 and active_conf <= (self.active_peak_conf - self.peak_drop_threshold):
            exit_signal = True
            exit_reasons.append('peak_drop')

        if can_exit and exit_signal:
            self.exit_count += 1
        else:
            self.exit_count = 0

        if can_exit and self.exit_count >= self.exit_consecutive:
            self.active_idx = None
            self.active_steps = 0
            self.active_peak_conf = 0.0
            self.exit_count = 0
            if int(proposed_idx) != idle_idx:
                self.enter_candidate_idx = int(proposed_idx)
                self.enter_count = 1
            else:
                self.enter_candidate_idx = None
                self.enter_count = 0
            return {
                'pred_idx': idle_idx,
                'confidence': float(probs[idle_idx]),
                'state': 'deactivated',
                'exit_reasons': exit_reasons,
            }

        return {
            'pred_idx': active_idx,
            'confidence': active_conf,
            'state': 'active',
            'exit_reasons': exit_reasons,
        }

class DepthPunchDetector:
    """Detect punch motion from foreground depth approach velocity.

    Instead of tracking color, this watches the *nearest* foreground depth.
    When someone throws a punch, the closest body part (fist -> elbow ->
    shoulder) rapidly moves toward the camera.  This works even when hands
    leave the frame -- the next-closest body part still shows the approach.

    Nearly zero compute cost: just a percentile on the existing foreground
    depth pixels that the voxel pipeline already computes.

    Design: biased toward allowing punches through (high recall).  The model
    + state machine already handle precision; this detector only blocks when
    the body is clearly stationary (no depth approach at all).
    """

    def __init__(
        self,
        near_percentile: float = 5.0,
        velocity_threshold: float = 0.01,
        history_len: int = 4,
    ):
        # Which percentile of foreground depth to track as "nearest surface".
        # 5% = the closest ~5% of body pixels (fist/elbow region).
        self.near_percentile = max(1.0, min(50.0, float(near_percentile)))
        # Depth velocity (m/frame).  Positive = approaching camera.
        # Default 0.01 is very permissive — only blocks truly stationary body.
        self.velocity_threshold = float(velocity_threshold)
        self.history_len = max(2, int(history_len))

        self.depth_history: deque = deque(maxlen=self.history_len)

        # Exposed state for overlay / gating
        self.nearest_depth: float = 0.0
        self.punch_signal: float = 0.0
        self.retract_signal: float = 0.0
        self.punch_active: bool = False
        self.retracting: bool = False

    def update(self, depth_m: np.ndarray, fg_mask: Optional[np.ndarray]) -> dict:
        """Compute punch signal from one frame's depth + foreground mask.

        Args:
            depth_m: Depth map in meters (H, W).
            fg_mask: Binary foreground mask (H, W), uint8 0/1.  If None the
                     full frame is treated as foreground.

        Returns:
            dict with nearest_depth, punch_signal, punch_active.
        """
        # Get valid foreground depth pixels
        if fg_mask is not None:
            fg_depth = depth_m[fg_mask > 0]
        else:
            fg_depth = depth_m.ravel()
        valid = fg_depth[(fg_depth > 0.15) & (fg_depth < 4.0)]

        if len(valid) < 20:
            # Not enough foreground — can't measure.
            # Default to ALLOWING predictions (don't block when unsure).
            self.punch_active = True
            return {
                'nearest_depth': self.nearest_depth,
                'punch_signal': self.punch_signal,
                'punch_active': True,
            }

        # Nearest surface = low percentile of valid foreground depth
        self.nearest_depth = float(np.percentile(valid, self.near_percentile))
        self.depth_history.append(self.nearest_depth)

        # Velocity: compare latest depth to the oldest in our short window.
        # Positive punch_signal = body approaching camera (punch).
        # Positive retract_signal = body moving away (retraction).
        if len(self.depth_history) >= 2:
            oldest = self.depth_history[0]
            newest = self.depth_history[-1]
            self.punch_signal = max(oldest - newest, 0.0)
            self.retract_signal = max(newest - oldest, 0.0)
        else:
            self.punch_signal = 0.0
            self.retract_signal = 0.0

        self.punch_active = self.punch_signal >= self.velocity_threshold
        self.retracting = self.retract_signal >= self.velocity_threshold and self.punch_signal < self.velocity_threshold

        return {
            'nearest_depth': self.nearest_depth,
            'punch_signal': self.punch_signal,
            'retract_signal': self.retract_signal,
            'punch_active': self.punch_active,
            'retracting': self.retracting,
        }


class PunchSegmentClassifier:
    """Detect-then-classify: buffer voxel features during a punch, classify when done.

    Uses total voxel activity (abs delta magnitude) to detect motion start/end,
    which works for ALL punch types (jab, hook, uppercut) — not just approach.

    States:  IDLE → ACTIVE → COOLDOWN → (classify) → DISPLAY → IDLE
    """

    def __init__(
        self,
        activity_start: float = 0.002,
        activity_end: float = 0.001,
        cooldown_frames: int = 6,
        min_segment_frames: int = 4,
        max_segment_frames: int = 120,
        display_hold_sec: float = 2.0,
    ):
        self.activity_start = activity_start
        self.activity_end = activity_end
        self.cooldown_frames = cooldown_frames
        self.min_segment_frames = min_segment_frames
        self.max_segment_frames = max_segment_frames
        self.display_hold_sec = display_hold_sec

        # State
        self._state = "IDLE"          # IDLE | ACTIVE | COOLDOWN | DISPLAY
        self._buffer: list = []        # list of (feature_dim,) arrays
        self._fg_ratios: list = []
        self._cooldown_count = 0
        self._display_start = 0.0

        # Latest result
        self.last_label: Optional[str] = None
        self.last_confidence: float = 0.0
        self.last_segment_frames: int = 0
    def feed(self, voxel_features: np.ndarray, fg_ratio: float) -> Optional[np.ndarray]:
        """Feed one frame of voxel features.

        Returns the completed segment as (T, feature_dim) when a punch just ended,
        or None if still buffering / idle.
        """
        activity = float(np.abs(voxel_features).mean())

        if self._state == "IDLE":
            if activity >= self.activity_start and fg_ratio > 0.01:
                self._state = "ACTIVE"
                self._buffer = [voxel_features.copy()]
                self._fg_ratios = [fg_ratio]
            return None

        elif self._state == "ACTIVE":
            self._buffer.append(voxel_features.copy())
            self._fg_ratios.append(fg_ratio)
            if activity < self.activity_end:
                self._state = "COOLDOWN"
                self._cooldown_count = 1
            elif len(self._buffer) >= self.max_segment_frames:
                # Force-end very long segments
                return self._finish_segment()
            return None

        elif self._state == "COOLDOWN":
            self._buffer.append(voxel_features.copy())
            self._fg_ratios.append(fg_ratio)
            if activity >= self.activity_start:
                # Motion resumed — go back to active
                self._state = "ACTIVE"
                self._cooldown_count = 0
                return None
            self._cooldown_count += 1
            if self._cooldown_count >= self.cooldown_frames:
                return self._finish_segment()
            return None

        elif self._state == "DISPLAY":
            if time.time() - self._display_start >= self.display_hold_sec:
                self._state = "IDLE"
                self.last_label = None
            return None

        return None

    def _finish_segment(self) -> Optional[np.ndarray]:
        """Finalise the buffered segment. Returns (T, F) array or None if too short."""
        segment = self._buffer
        self._buffer = []
        self._fg_ratios = []
        self._cooldown_count = 0

        if len(segment) < self.min_segment_frames:
            self._state = "IDLE"
            return None

        self._state = "DISPLAY"
        self._display_start = time.time()
        self.last_segment_frames = len(segment)
        return np.stack(segment, axis=0)

    def set_result(self, label: str, confidence: float):
        """Called after classification to store the display result."""
        self.last_label = label
        self.last_confidence = confidence

    @property
    def is_active(self) -> bool:
        return self._state in ("ACTIVE", "COOLDOWN")

    @property
    def is_displaying(self) -> bool:
        return self._state == "DISPLAY" and self.last_label is not None


# Class colours for probability bars (RGB)
CLASS_COLORS = [
    (255, 100, 100),   # jab - red
    (100, 100, 255),   # cross - blue
    (255, 180, 100),   # left_hook - orange
    (100, 255, 180),   # right_hook - cyan
    (255, 100, 255),   # left_uppercut - magenta
    (100, 255, 100),   # right_uppercut - green
    (200, 200, 200),   # block - gray
    (150, 150, 150),   # idle - dark gray
]

# GUI colour palette
COLORS = {
    'bg': '#0d1117',
    'panel': '#161b22',
    'accent': '#ff6b35',
    'text': '#e6edf3',
    'text_dim': '#8b949e',
    'success': '#3fb950',
    'warning': '#d29922',
    'danger': '#f85149',
}


def _load_label_names(checkpoint: dict, checkpoint_path: str, num_classes: int) -> list:
    """
    Resolve class names in the same order used during training.

    Priority:
      1) checkpoint['label_map']
      2) ann_file from checkpoint config or sibling config.json
      3) DEFAULT_LABELS fallback
    """
    # 1) Directly from checkpoint if present.
    if "label_map" in checkpoint:
        lm = checkpoint["label_map"]
        if isinstance(lm, list) and len(lm) == num_classes:
            return [str(x) for x in lm]
        if isinstance(lm, dict):
            try:
                idx_to_name = {int(v): str(k) for k, v in lm.items()}
                return [idx_to_name.get(i, f"class_{i}") for i in range(num_classes)]
            except Exception:
                pass

    # 2) Try ann_file from checkpoint config or run config.json.
    candidate_ann = None
    cfg = checkpoint.get("config", {})
    if isinstance(cfg, dict):
        candidate_ann = cfg.get("ann_file")

    if not candidate_ann:
        cfg_path = Path(checkpoint_path).resolve().parent / "config.json"
        if cfg_path.exists():
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    run_cfg = json.load(f)
                candidate_ann = run_cfg.get("ann_file")
            except Exception:
                pass

    if candidate_ann:
        ann_path = Path(candidate_ann).expanduser()
        if not ann_path.is_absolute():
            ann_path = (Path.cwd() / ann_path).resolve()
        if ann_path.exists():
            try:
                with open(ann_path, "rb") as f:
                    pkl_data = pickle.load(f)
                label_map = pkl_data.get("label_map", {})
                if isinstance(label_map, dict):
                    idx_to_name = {int(v): str(k) for k, v in label_map.items()}
                    return [idx_to_name.get(i, f"class_{i}") for i in range(num_classes)]
            except Exception:
                pass

    # 3) Safe fallback.
    if num_classes == len(DEFAULT_LABELS):
        return list(DEFAULT_LABELS)
    return [f"class_{i}" for i in range(num_classes)]


# ---------------------------------------------------------------------------
#  IMU helper — read camera pitch from D435i accelerometer
# ---------------------------------------------------------------------------

def _read_imu_pitch_separate(duration: float = 1.5) -> tuple:
    """
    Read camera pitch/roll from IMU using a SEPARATE pipeline.

    This runs BEFORE the main RGB+depth pipeline to avoid stream conflicts.
    Matches the recording script's proven approach: tries multiple accel rates,
    collects samples, computes pitch from averaged gravity vector.

    Returns:
        (pitch_deg, roll_deg) or (0.0, 0.0) if no IMU available.
    """
    import math

    try:
        ctx = rs.context()
        devices = list(ctx.query_devices())
        if not devices:
            return 0.0, 0.0

        device = devices[0]
        serial = device.get_info(rs.camera_info.serial_number)

        # Check for motion sensor
        has_motion = False
        for sensor in device.query_sensors():
            name = sensor.get_info(rs.camera_info.name)
            if 'Motion' in name or 'IMU' in name:
                has_motion = True
                break

        if not has_motion:
            print("  Device has no IMU — skipping auto pitch detection")
            return 0.0, 0.0

        # Try multiple sampling rates (matching recording script)
        imu_pipeline = rs.pipeline()
        imu_profile = None

        sampling_rates = [
            (200, 200),   # Both 200Hz
            (250, 200),   # Common D435i config
            (400, 200),
            (63, 200),    # Low power
            (200, None),  # Accel only
            (250, None),
        ]

        for accel_rate, gyro_rate in sampling_rates:
            try:
                imu_config = rs.config()
                imu_config.enable_device(serial)
                imu_config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, accel_rate)
                if gyro_rate is not None:
                    imu_config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, gyro_rate)

                imu_profile = imu_pipeline.start(imu_config)
                print(f"  IMU pipeline started (accel={accel_rate}Hz)")
                break
            except RuntimeError as e:
                if "resolve" in str(e).lower():
                    continue
                else:
                    break

        if imu_profile is None:
            print("  Could not start IMU pipeline (all rates failed)")
            return 0.0, 0.0

        try:
            accel_samples = []
            start_time = time.time()

            print("  Collecting accelerometer data...")
            while time.time() - start_time < duration:
                try:
                    frames = imu_pipeline.wait_for_frames(timeout_ms=100)
                    accel_frame = frames.first_or_default(rs.stream.accel)
                    if accel_frame:
                        d = accel_frame.as_motion_frame().get_motion_data()
                        accel_samples.append((d.x, d.y, d.z))
                except Exception:
                    pass
                time.sleep(0.05)

            if not accel_samples:
                print("  No accelerometer data collected")
                return 0.0, 0.0

            # Average gravity vector
            ax = np.mean([s[0] for s in accel_samples])
            ay = np.mean([s[1] for s in accel_samples])
            az = np.mean([s[2] for s in accel_samples])
            mag = np.sqrt(ax**2 + ay**2 + az**2)

            if mag < 5.0 or mag > 15.0:
                print(f"  Invalid gravity magnitude: {mag:.2f}")
                return 0.0, 0.0

            # Pitch/roll from gravity (matching recording script exactly)
            pitch = math.asin(-az / mag) * 180 / math.pi
            roll = math.atan2(ax, -ay) * 180 / math.pi

            print(f"  IMU: pitch={pitch:.1f}°, roll={roll:.1f}° "
                  f"(from {len(accel_samples)} samples)")
            return float(pitch), float(roll)

        finally:
            imu_pipeline.stop()

    except Exception as e:
        print(f"  Could not read IMU: {e}")
        return 0.0, 0.0


class RollingFeatureBuffer:
    """Maintains rolling voxel (+ optional pose) feature buffer for inference."""

    def __init__(
        self,
        window_size: int = 12,
        voxel_size: int = 12,
        voxel_normalization: str = 'clip_p90',
        in_channels: int = 1,
        voxel_grid_size: Tuple[int, int, int] = (20, 20, 20),
        fusion_mode: bool = False,
        pose_dim: int = 0,
    ):
        self.window_size = window_size
        self.voxel_size = voxel_size
        self.voxel_normalization = str(voxel_normalization)
        self.in_channels = in_channels
        self.voxel_grid_size = voxel_grid_size
        self.fusion_mode = fusion_mode
        self.pose_dim = pose_dim

        # Feature buffers
        self.voxel_buffer = deque(maxlen=window_size)
        self.fg_ratio_buffer = deque(maxlen=window_size)
        # Pose buffer (only used in fusion mode)
        self.pose_buffer = deque(maxlen=window_size) if fusion_mode else None

    def add_frame(
        self,
        voxel_features: np.ndarray,
        fg_ratio: float,
        pose_features: Optional[np.ndarray] = None,
    ):
        """Add a frame's flattened voxel (+ optional pose) feature vector to the buffer."""
        self.voxel_buffer.append(np.asarray(voxel_features, dtype=np.float32).reshape(-1))
        self.fg_ratio_buffer.append(fg_ratio)
        if self.fusion_mode and self.pose_buffer is not None:
            if pose_features is not None:
                self.pose_buffer.append(np.asarray(pose_features, dtype=np.float32).reshape(-1))
            else:
                self.pose_buffer.append(np.zeros(self.pose_dim, dtype=np.float32))

    def get_features(self) -> dict:
        """Get feature tensors for model inference."""
        if len(self.voxel_buffer) < self.window_size:
            return None

        # Stack voxel features.
        voxel = np.stack(list(self.voxel_buffer), axis=0)      # (T, N³*C)
        fg_ratio = np.array(list(self.fg_ratio_buffer))        # (T,)

        # Apply normalization to match training preprocessing.
        voxel_f32 = voxel.astype(np.float32, copy=True)
        if self.voxel_normalization == 'frame_l1':
            frame_energy = np.abs(voxel_f32).sum(axis=1, keepdims=True)
            denom = np.maximum(frame_energy, 1e-6)
            voxel_f32 = voxel_f32 / denom
        elif self.voxel_normalization == 'channel_p90':
            T = voxel_f32.shape[0]
            vx, vy, vz = self.voxel_grid_size
            voxel_5d = voxel_f32.reshape(T, self.in_channels, vx, vy, vz)
            for ch in range(self.in_channels):
                ch_energy = np.abs(voxel_5d[:, ch]).sum(axis=(1, 2, 3))
                if ch_energy.size > 0:
                    scale = float(np.percentile(ch_energy, 90))
                    if np.isfinite(scale) and scale > 1e-6:
                        voxel_5d[:, ch] /= scale
            voxel_f32 = voxel_5d.reshape(T, -1)
        elif self.voxel_normalization == 'clip_p90':
            frame_energy = np.abs(voxel_f32).sum(axis=1)
            if frame_energy.size > 0:
                scale = float(np.percentile(frame_energy, 90))
                if np.isfinite(scale) and scale > 1e-6:
                    voxel_f32 = voxel_f32 / scale
        # 'none': no normalization

        # In fusion mode, concatenate pose features after voxel features.
        if self.fusion_mode and self.pose_buffer is not None:
            pose = np.stack(list(self.pose_buffer), axis=0)  # (T, pose_dim)
            combined = np.concatenate([voxel_f32, pose], axis=1)  # (T, voxel_dim + pose_dim)
        else:
            combined = voxel_f32

        return {
            'features': combined,
            'voxel': voxel_f32,
            'fg_ratio': fg_ratio.astype(np.float32, copy=False),
        }

    @property
    def is_ready(self) -> bool:
        return len(self.voxel_buffer) >= self.window_size


class LiveVoxelGUI:
    """Real-time voxel-only inference GUI with debug overlays."""

    def __init__(
        self,
        root,
        checkpoint_path: str,
        device: str = 'cuda:0',
        voxel_size: int = 12,
        window_size: int = 12,
        frame_sample_rate: int = 2,
        inference_interval: int = 4,
        temporal_smooth_window: int = 5,
        min_confidence: float = 0.4,
        min_action_prob: float = 0.0,
        min_class_margin: float = 0.0,
        min_voxel_active_ratio: float = 0.0,
        use_action_state_machine: bool = False,
        state_enter_consecutive: int = 2,
        state_exit_consecutive: int = 2,
        state_min_hold_steps: int = 2,
        state_sustain_confidence: float = 0.78,
        state_peak_drop_threshold: float = 0.02,
        camera_pitch: float = 0.0,
        camera_roll: float = 0.0,
        auto_pitch: bool = True,
        rgb_res: str = '960x540',
        depth_res: str = '848x480',
        downscale_width=None,
        processing_mode: str = 'latest',
        feature_queue_size: int = 0,
        result_queue_size: int = 0,
        sensor_queue_size: int = 0,
        num_workers: int = 2,
        yolo_checkpoint: str = '',
        use_yolo: bool = True,
        yolo_interval: int = 5,
        no_video: bool = False,
        use_depth_punch: bool = False,
        punch_near_percentile: float = 5.0,
        punch_velocity_threshold: float = 0.03,
        punch_history_len: int = 6,
        segment_mode: bool = False,
        segment_activity_start: float = 0.008,
        segment_activity_end: float = 0.004,
        segment_cooldown: int = 6,
        segment_display_hold: float = 2.5,
        fusion_pose_weights: str = 'yolo11m-pose.pt',
        optimize_gpu: bool = False,
        ema_alpha: float = 0.35,
        hysteresis_margin: float = 0.12,
        min_hold_frames: int = 3,
    ):
        self.root = root
        self.checkpoint_path = checkpoint_path
        self.device = _resolve_runtime_device(device)
        self.requested_device = str(device)
        self.voxel_size = voxel_size
        self.window_size = window_size
        self.frame_sample_rate = frame_sample_rate
        self.temporal_smooth_window = temporal_smooth_window
        self.min_confidence = min_confidence
        self.min_action_prob = min_action_prob
        self.min_class_margin = min_class_margin
        self.min_voxel_active_ratio = min_voxel_active_ratio
        self.use_action_state_machine = bool(use_action_state_machine)
        self.state_enter_consecutive = max(1, int(state_enter_consecutive))
        self.state_exit_consecutive = max(1, int(state_exit_consecutive))
        self.state_min_hold_steps = max(0, int(state_min_hold_steps))
        self.state_sustain_confidence = max(0.0, float(state_sustain_confidence))
        self.state_peak_drop_threshold = max(0.0, float(state_peak_drop_threshold))
        self.camera_pitch = camera_pitch
        self.camera_roll = camera_roll
        self.auto_pitch = auto_pitch
        self.source_fps = 60  # Used by camera init before checkpoint metadata is loaded.
        self.processing_mode = str(processing_mode).strip().lower()
        if self.processing_mode not in ('latest', 'strict'):
            print(f"Invalid processing_mode={processing_mode}; using 'latest'")
            self.processing_mode = 'latest'
        self.strict_mode = (self.processing_mode == 'strict')
        self._imu_lock = threading.Lock()
        self._imu_pending_pitch = None
        self._imu_pending_roll = None
        self._imu_stop = threading.Event()
        self.optimize_gpu = optimize_gpu
        self._ema_alpha_cfg = max(0.0, min(1.0, float(ema_alpha)))
        self._hysteresis_margin_cfg = max(0.0, float(hysteresis_margin))
        self._min_hold_frames_cfg = max(0, int(min_hold_frames))

        # Parse resolutions
        try:
            rw, rh = rgb_res.lower().split('x')
            self.rgb_res = (int(rw), int(rh))
        except:
            print(f"Invalid RGB resolution format: {rgb_res}. Defaulting to 960x540.")
            self.rgb_res = (960, 540)
            
        try:
            dw, dh = depth_res.lower().split('x')
            self.depth_res = (int(dw), int(dh))
        except:
            print(f"Invalid depth resolution format: {depth_res}. Defaulting to 848x480.")
            self.depth_res = (848, 480)

        if downscale_width is None:
            self.downscale_width = None
        else:
            try:
                dsw = int(downscale_width)
                self.downscale_width = dsw if dsw > 0 else None
            except Exception:
                self.downscale_width = None

        # YOLO person detection
        self.use_yolo = use_yolo and _YOLO_AVAILABLE
        self.yolo_checkpoint = yolo_checkpoint
        self.yolo_model = None  # loaded in _init_camera
        self.yolo_interval = max(1, int(yolo_interval))
        self._yolo_cache_bbox = None  # cached person bbox
        self._yolo_cache_counter = 0  # counts sampled frames since last YOLO run
        self._yolo_lock = threading.Lock()  # serialize YOLO GPU calls

        # Depth-based punch detection (tracks nearest foreground approaching camera)
        self.use_depth_punch = bool(use_depth_punch)
        self.depth_punch_detector = None
        if self.use_depth_punch:
            self.depth_punch_detector = DepthPunchDetector(
                near_percentile=punch_near_percentile,
                velocity_threshold=punch_velocity_threshold,
                history_len=punch_history_len,
            )
        self._latest_punch_result: Optional[dict] = None
        self._latest_pose_kps = None
        self._latest_pose_confs = None
        self._latest_pose_bbox = None

        # Segment mode: detect-then-classify (one label per complete punch)
        self.segment_mode = bool(segment_mode)
        self.segment_classifier: Optional[PunchSegmentClassifier] = None
        if self.segment_mode:
            self.segment_classifier = PunchSegmentClassifier(
                activity_start=segment_activity_start,
                activity_end=segment_activity_end,
                cooldown_frames=segment_cooldown,
                display_hold_sec=segment_display_hold,
            )

        # Multi-worker support
        self.num_workers = max(1, int(num_workers))

        # State
        self.running = True
        self.paused = False
        self.model_ready = False
        self.camera_ready = False
        self.model_error = None

        # Performance tracking
        self.frame_count = 0
        self.inference_interval = inference_interval
        self.fps_history = deque(maxlen=30)
        self.capture_fps_history = deque(maxlen=60)
        self._last_sensor_ts_ms = None

        # Async feature processing (keeps GUI loop responsive under heavy foreground).
        self._extractor_lock = threading.Lock()
        need_all_frames = self.strict_mode or self.segment_mode
        default_feature_q = 256 if need_all_frames else 1
        default_result_q = 256 if need_all_frames else 1
        default_sensor_q = 16 if need_all_frames else 1
        try:
            feature_queue_size = int(feature_queue_size)
        except Exception:
            feature_queue_size = 0
        try:
            result_queue_size = int(result_queue_size)
        except Exception:
            result_queue_size = 0
        try:
            sensor_queue_size = int(sensor_queue_size)
        except Exception:
            sensor_queue_size = 0
        self.feature_queue_size = (
            feature_queue_size if feature_queue_size > 0 else default_feature_q
        )
        self.result_queue_size = (
            result_queue_size if result_queue_size > 0 else default_result_q
        )
        self.sensor_queue_size = (
            sensor_queue_size if sensor_queue_size > 0 else default_sensor_q
        )
        self._feature_job_queue = queue.Queue(maxsize=self.feature_queue_size)
        self._feature_result_queue = queue.Queue(maxsize=self.result_queue_size)
        self._feature_worker_stop = threading.Event()
        self._feature_worker_threads = []  # list of worker threads
        self._last_applied_feature_frame = -1
        self._sampled_frames_applied = 0
        self._capture_seen_frames = 0
        self._last_queue_warn_t = 0.0

        # Dedicated inference thread (keeps GPU inference off GUI thread).
        self._inference_queue = queue.Queue(maxsize=2)
        self._inference_result_queue = queue.Queue(maxsize=2)
        self._inference_stop = threading.Event()
        self._inference_thread = None
        self.no_video = bool(no_video)

        # Dedicated capture thread (keeps camera polling independent of GUI thread).
        self._frame_lock = threading.Lock()
        self._latest_frame_packet = None  # (frame_no, sensor_ts_ms, rgb_full, depth_full)
        self._last_displayed_frame_no = -1
        self._capture_stop = threading.Event()
        self._capture_thread = None
        self.gui_update_ms = 16  # ~60Hz default redraw cadence
        self._video_photo = None
        self._video_photo_size = (0, 0)
        self.inference_time_history = deque(maxlen=120)

        # Prediction state
        self.current_prediction = "Initializing..."
        self.current_confidence = 0.0
        self.class_probs = None
        self.smooth_probs = None
        self.recent_probs = deque(maxlen=max(1, int(self.temporal_smooth_window)))
        self._ema_probs = None  # Exponential moving average of probabilities
        self._ema_alpha = getattr(self, '_ema_alpha_cfg', 0.35)
        self._held_pred_idx = None  # Current held prediction for hysteresis
        self._held_pred_frames = 0  # How many frames the current prediction has been held
        self._hysteresis_margin = getattr(self, '_hysteresis_margin_cfg', 0.12)
        self._min_hold_frames = getattr(self, '_min_hold_frames_cfg', 3)
        self.prediction_trail = deque(maxlen=30)
        self.action_state_machine = None

        # Debug modes (toggled with keyboard shortcuts)
        self.debug_modes = {
            'show_depth': False,      # D
            'show_voxel': False,      # V
            'show_probs': True,       # P
            'show_features': False,   # X
        }

        # Feature statistics (updated on each inference)
        self.feature_stats = {
            'voxel_activity': 0.0,
            'fg_ratio': 0.0,
            'voxel_active_ratio': 0.0,
            'action_prob': 0.0,
            'class_margin': 0.0,
            'gate_reasons': 'none',
            'state': 'idle',
        }

        # Debug data (updated each frame)
        self.last_voxel_grid = None
        self.last_depth_m = None
        self._last_video_frame = None
        self._resize_redraw_pending = False

        # Labels (overridden when checkpoint is loaded)
        self.labels = list(DEFAULT_LABELS)
        self.feature_mode = 'voxel_only'
        self.model_arch = 'causal_voxel_transformer'
        self.checkpoint_config = {}
        self.dataset_config = {}
        self.voxel_grid_size = (self.voxel_size, self.voxel_size, self.voxel_size)
        self.in_channels = 1
        self.voxel_normalization = 'clip_p90'

        # Fusion mode state (set during _load_model if checkpoint is fusion)
        self.fusion_mode = False
        self.fusion_pose_weights = fusion_pose_weights
        self.pose_dim = POSE_FEATURE_DIM
        self.pose_embed_dim = 64
        self.pose_estimator = None  # YOLOPoseEstimator, initialized when fusion detected
        self._pose_lock = threading.Lock()  # serialize pose GPU calls

        # Setup GUI
        self._setup_gui()

        # Initialize components in background
        self.root.after(100, self._init_async)

    def _setup_gui(self):
        """Setup the Tkinter GUI layout."""
        self.root.title("Voxel Live Inference")
        self.root.configure(bg=COLORS['bg'])

        # Font scale: bigger in no-video mode since the panel IS the UI
        _nv = self.no_video
        _fs = 1.6 if _nv else 1.0  # font scale factor

        if not _nv:
            self.root.geometry("960x520")
        else:
            self.root.geometry("520x680")

        # Main container
        main = tk.Frame(self.root, bg=COLORS['bg'])
        main.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Top bar — status only
        top = tk.Frame(main, bg=COLORS['panel'])
        top.pack(fill=tk.X, pady=(0, 4))

        self.status_label = tk.Label(top, text="Initializing...", font=('DejaVu Sans', int(9 * _fs)),
                                     bg=COLORS['panel'], fg=COLORS['warning'])
        self.status_label.pack(side=tk.RIGHT, padx=10, pady=4)

        # Content area: video (left) + info panel (right)
        content = tk.Frame(main, bg=COLORS['bg'])
        content.pack(fill=tk.BOTH, expand=True)

        # Video panel (hidden in no-video mode)
        if not _nv:
            video_frame = tk.Frame(content, bg=COLORS['panel'])
            video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))
            self.video_frame = video_frame
            self.video_container = tk.Frame(video_frame, bg='black')
            self.video_container.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
            self.video_label = tk.Label(self.video_container, bg='black', bd=0, highlightthickness=0, anchor=tk.CENTER)
            self.video_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
            self.video_container.bind('<Configure>', self._on_video_resize)
        else:
            self.video_frame = None
            self.video_container = None
            self.video_label = None

        # Info panel (fixed width on right, or full-width in no-video mode)
        info_width = 240 if not _nv else None
        info_frame = tk.Frame(content, bg=COLORS['panel'],
                              **(dict(width=info_width) if info_width else {}))
        if _nv:
            info_frame.pack(fill=tk.BOTH, expand=True)
        else:
            info_frame.pack(side=tk.RIGHT, fill=tk.Y)
        info_frame.pack_propagate(False)

        # --- Prediction section ---
        tk.Label(info_frame, text="PREDICTION", font=('DejaVu Sans', int(11 * _fs), 'bold'),
                 bg=COLORS['panel'], fg=COLORS['text_dim']).pack(pady=(8 if _nv else 4, 2))

        self.pred_label = tk.Label(info_frame, text="---",
                                   font=('DejaVu Sans', int(28 if _nv else 16), 'bold'),
                                   bg=COLORS['panel'], fg=COLORS['accent'])
        self.pred_label.pack(pady=4 if _nv else 2)

        self.conf_label = tk.Label(info_frame, text="0%",
                                   font=('DejaVu Sans', int(16 if _nv else 10)),
                                   bg=COLORS['panel'], fg=COLORS['text_dim'])
        self.conf_label.pack()

        # Depth punch status (always visible when depth punch is enabled)
        self.punch_label = None
        if self.use_depth_punch:
            self.punch_label = tk.Label(
                info_frame, text="Depth: --  Vel: --",
                font=('DejaVu Sans', int(13 if _nv else 9)),
                bg=COLORS['panel'], fg=COLORS['text_dim'])
            self.punch_label.pack(pady=(6 if _nv else 2, 0))

        # Gate / state status — large and obvious
        self.gate_label = tk.Label(
            info_frame, text="",
            font=('DejaVu Sans', int(18 if _nv else 10), 'bold'),
            bg=COLORS['panel'], fg=COLORS['text_dim'])
        self.gate_label.pack(pady=(4 if _nv else 1, 0))

        # Prediction trail canvas
        trail_h = 22 if _nv else 14
        self.trail_canvas = tk.Canvas(info_frame, bg=COLORS['panel'],
                                      height=trail_h, highlightthickness=0)
        self.trail_canvas.pack(fill=tk.X, padx=10, pady=(4 if _nv else 2, 0))

        # --- Performance section ---
        tk.Label(info_frame, text="PERFORMANCE", font=('DejaVu Sans', int(10 * _fs), 'bold'),
                 bg=COLORS['panel'], fg=COLORS['text_dim']).pack(pady=(8 if _nv else 6, 2))

        self.fps_label = tk.Label(info_frame, text="FPS: --",
                                  font=('DejaVu Sans', int(9 * _fs)),
                                  bg=COLORS['panel'], fg=COLORS['text'])
        self.fps_label.pack()

        self.latency_label = tk.Label(info_frame, text="Latency: --ms",
                                      font=('DejaVu Sans', int(9 * _fs)),
                                      bg=COLORS['panel'], fg=COLORS['text'])
        self.latency_label.pack()

        # --- Camera tilt section ---
        tk.Label(info_frame, text="CAMERA", font=('DejaVu Sans', int(10 * _fs), 'bold'),
                 bg=COLORS['panel'], fg=COLORS['text_dim']).pack(pady=(6 if _nv else 4, 2))

        pitch_frame = tk.Frame(info_frame, bg=COLORS['panel'])
        pitch_frame.pack(fill=tk.X, pady=2)

        self.pitch_label = tk.Label(pitch_frame, text="Pitch: --°",
                                    font=('DejaVu Sans', int(9 * _fs)),
                                    bg=COLORS['panel'], fg=COLORS['text'])
        self.pitch_label.pack(side=tk.LEFT, padx=(10, 5))

        # Manual IMU Update Button
        self.btn_update_imu = tk.Button(
            pitch_frame, text="Refresh IMU", font=('DejaVu Sans', int(9 * _fs)),
            bg='#444444', fg='white', relief=tk.FLAT, padx=4, pady=1,
            command=self._manual_imu_update
        )
        self.btn_update_imu.pack(side=tk.LEFT)

        # --- Config / status section ---
        tk.Label(info_frame, text="CONFIG", font=('DejaVu Sans', int(9 * _fs), 'bold'),
                 bg=COLORS['panel'], fg=COLORS['text_dim']).pack(pady=(6 if _nv else 4, 1))

        # Model checkpoint name (just the filename)
        ckpt_name = os.path.basename(os.path.dirname(self.checkpoint_path))
        self.config_model_label = tk.Label(
            info_frame, text=f"Model: {ckpt_name}",
            font=('DejaVu Sans', int(8 * _fs)), bg=COLORS['panel'], fg=COLORS['text'],
            wraplength=480 if _nv else 220, justify=tk.LEFT)
        self.config_model_label.pack(anchor=tk.W, padx=8)

        # Feature toggles summary
        config_lines = []
        config_lines.append(f"Window: {self.window_size} | FSR: {int(self.frame_sample_rate)}")
        config_lines.append(f"Workers: {self.num_workers} | Device: {self.device}")
        yolo_status = "ON" if self.use_yolo else "OFF"
        config_lines.append(f"YOLO: {yolo_status}" + (f" (interval={self.yolo_interval})" if self.use_yolo else ""))
        video_status = "OFF" if self.no_video else "ON"
        config_lines.append(f"Video: {video_status}")
        config_lines.append(f"Resolution: {self.downscale_width or 'native'}")
        config_lines.append(f"Smooth: {self.temporal_smooth_window} | Conf: {self.min_confidence}")
        config_lines.append(
            f"ActP: {self.min_action_prob:.2f} | Margin: {self.min_class_margin:.2f} | VoxAct: {self.min_voxel_active_ratio:.3f}"
        )
        if self.use_action_state_machine:
            config_lines.append(
                f"State: E{self.state_enter_consecutive}/X{self.state_exit_consecutive}/"
                f"H{self.state_min_hold_steps} | Sust {self.state_sustain_confidence:.2f}"
            )
        else:
            config_lines.append("State: OFF")
        if self.segment_mode:
            config_lines.append(">> SEGMENT MODE (detect-then-classify)")

        self.config_detail_label = tk.Label(
            info_frame, text="\n".join(config_lines),
            font=('DejaVu Sans Mono', int(7 * _fs)), bg=COLORS['panel'], fg=COLORS['text_dim'],
            justify=tk.LEFT)
        self.config_detail_label.pack(anchor=tk.W, padx=8, pady=(1, 0))

        # --- Debug toggles ---
        tk.Label(info_frame, text="DEBUG OVERLAYS", font=('DejaVu Sans', int(9 * _fs), 'bold'),
                 bg=COLORS['panel'], fg=COLORS['text_dim']).pack(pady=(6 if _nv else 4, 0))

        self.debug_buttons = {}
        debug_btn_frame = tk.Frame(info_frame, bg=COLORS['panel'])
        debug_btn_frame.pack(pady=2)

        debug_options = [
            ('show_depth', 'D', 'Depth'),
            ('show_voxel', 'V', 'Voxel'),
            ('show_probs', 'P', 'Probs'),
            ('show_features', 'X', 'Stats'),
        ]

        row1 = tk.Frame(debug_btn_frame, bg=COLORS['panel'])
        row1.pack()
        row2 = tk.Frame(debug_btn_frame, bg=COLORS['panel'])
        row2.pack(pady=(5, 0))

        btn_font_sz = int(9 * _fs) if _nv else 7
        btn_width = 8 if _nv else 6
        for i, (mode, key, label) in enumerate(debug_options):
            parent = row1 if i < 2 else row2
            is_active = self.debug_modes[mode]
            bg_color = COLORS['accent'] if is_active else '#333333'
            fg_color = 'white' if is_active else COLORS['text_dim']
            btn = tk.Button(
                parent, text=f"{key}:{label}", font=('DejaVu Sans', btn_font_sz),
                bg=bg_color, fg=fg_color,
                activebackground=COLORS['accent'], activeforeground='white',
                relief=tk.FLAT, width=btn_width, padx=2, pady=1,
                command=lambda m=mode: self._toggle_debug(m)
            )
            btn.pack(side=tk.LEFT, padx=2)
            self.debug_buttons[mode] = btn

        # Feature stats text (shown when X toggled)
        self.stats_label = tk.Label(info_frame, text="",
                                    font=('DejaVu Sans Mono', int(8 * _fs)),
                                    bg=COLORS['panel'], fg=COLORS['text_dim'], justify=tk.LEFT)
        self.stats_label.pack(pady=(4, 0))

        # Bind keys
        self.root.protocol("WM_DELETE_WINDOW", self._quit)
        self.root.bind('<Escape>', lambda e: self._quit())
        self.root.bind('<space>', lambda e: self._toggle_pause())
        for key_char, mode in [('d', 'show_depth'), ('v', 'show_voxel'),
                                ('p', 'show_probs'), ('x', 'show_features')]:
            self.root.bind(key_char, lambda e, m=mode: self._toggle_debug(m))
            self.root.bind(key_char.upper(), lambda e, m=mode: self._toggle_debug(m))

    def _toggle_debug(self, mode: str):
        if mode in self.debug_modes:
            self.debug_modes[mode] = not self.debug_modes[mode]
            if mode in self.debug_buttons:
                btn = self.debug_buttons[mode]
                if self.debug_modes[mode]:
                    btn.config(bg=COLORS['accent'], fg='white')
                else:
                    btn.config(bg='#333333', fg=COLORS['text_dim'])

    def _init_async(self):
        """Initialize model first, then camera, to avoid config race conditions."""
        threading.Thread(target=self._init_pipeline, daemon=True).start()

    def _init_pipeline(self):
        self._load_model()
        if not self.model_ready:
            print("Aborting camera initialization because model loading failed.")
            self.current_prediction = "MODEL LOAD ERROR"
            self.root.after(
                0,
                lambda: self.status_label.config(text="Model Load Error", fg=COLORS['danger'])
            )
            return
        self._init_camera()

    # ------------------------------------------------------------------
    #  MODEL LOADING
    # ------------------------------------------------------------------

    def _load_model(self):
        """Load voxel-only inference model from checkpoint."""
        try:
            print(f"Loading model from {self.checkpoint_path}...")
            print(f"  Requested device: {self.requested_device}")
            print(f"  Runtime device:   {self.device}")
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device,
                                    weights_only=False)

            # Training saves config as a flat dict of CLI args plus dataset metadata.
            feature_layout = _resolve_checkpoint_feature_layout(checkpoint, default_voxel_size=self.voxel_size)
            config = feature_layout['config']
            self.checkpoint_config = dict(config)
            self.dataset_config = dict(feature_layout['dataset_config'])
            # Propagate inferred fields that may not exist in the raw checkpoint.
            self.dataset_config.setdefault(
                'velocity_magnitude_channel', feature_layout.get('velocity_magnitude_channel', False)
            )
            self.in_channels = int(feature_layout['in_channels'])
            self.voxel_normalization = str(feature_layout['voxel_normalization'])
            self.model_arch = str(checkpoint.get('model_arch', config.get('model_arch', 'causal_voxel_transformer')))
            self.feature_mode = str(checkpoint.get('feature_mode', config.get('feature_mode', 'voxel_only')))
            if self.model_arch not in {
                'causal_voxel_transformer', 'voxel_mlp', 'voxel_temporal_mlp',
                'fusion_voxel_pose_transformer',
            }:
                raise RuntimeError(
                    f"Unsupported model_arch={self.model_arch}. "
                    "This script supports causal_voxel_transformer, voxel_mlp, "
                    "voxel_temporal_mlp, and fusion_voxel_pose_transformer."
                )
            if self.feature_mode not in {'voxel_only', 'fusion_voxel_pose'}:
                raise RuntimeError(
                    f"Unsupported feature_mode={self.feature_mode}. "
                    "This script supports voxel_only and fusion_voxel_pose checkpoints."
                )

            # Detect fusion mode from checkpoint metadata
            self.fusion_mode = (
                self.feature_mode == 'fusion_voxel_pose'
                or self.model_arch == 'fusion_voxel_pose_transformer'
            )
            if self.fusion_mode:
                self.pose_dim = int(checkpoint.get('pose_dim', config.get('pose_dim', POSE_FEATURE_DIM)))
                self.pose_embed_dim = int(checkpoint.get('pose_embed_dim', config.get('pose_embed_dim', 64)))

            voxel_size = tuple(int(v) for v in feature_layout['voxel_size'])
            self.voxel_grid_size = voxel_size

            state_dict = checkpoint.get('model_state_dict', {})
            num_classes = int(config.get('num_classes', len(self.labels)))
            # Detect num_classes from the final classifier layer in the state dict.
            # Find the highest-indexed classifier weight to get the output layer.
            classifier_keys = sorted(
                [k for k in state_dict if k.startswith('classifier.') and k.endswith('.weight')],
            )
            if classifier_keys:
                last_cls_key = classifier_keys[-1]
                num_classes = int(state_dict[last_cls_key].shape[0])

            if self.model_arch == 'fusion_voxel_pose_transformer':
                self.model = FusionVoxelPoseTransformerModel(
                    voxel_size=voxel_size,
                    num_classes=num_classes,
                    d_model=int(config.get('transformer_d_model', 192)),
                    num_heads=int(config.get('transformer_heads', 8)),
                    num_layers=int(config.get('transformer_layers', 4)),
                    dim_feedforward=int(config.get('transformer_ffn_dim', 576)),
                    dropout=0.0,
                    max_len=int(config.get('transformer_max_len', 256)),
                    in_channels=self.in_channels,
                    pose_dim=self.pose_dim,
                    pose_embed_dim=self.pose_embed_dim,
                    pose_dropout=0.0,
                    dual_voxel_stem=bool(checkpoint.get('dual_voxel_stem', False)),
                )
            else:
                raise RuntimeError(
                    f"Unsupported model_arch={self.model_arch}. "
                    "Supported: fusion_voxel_pose_transformer."
                )

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()

            # --- Jetson / GPU optimizations ---
            if self.device.startswith('cuda'):
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

            # --- Auto ONNX/TensorRT conversion ---
            self._ort_session = None
            self._ort_input_names = None
            self._trt_context = None
            self._trt_engine = None
            self._trt_bindings = None
            if self.optimize_gpu:
                clip_len = checkpoint.get('clip_len', config.get('clip_len', 12))
                self._setup_optimized_inference(
                    self.checkpoint_path, self.model, num_classes,
                    voxel_size, self.in_channels, clip_len,
                )

            self.labels = _load_label_names(checkpoint, self.checkpoint_path, num_classes)

            # Auto-read clip_len / frame_sample_rate from checkpoint
            ckpt_clip = checkpoint.get('clip_len', config.get('clip_len', None))
            ckpt_fsr = checkpoint.get('frame_sample_rate', config.get('frame_sample_rate', None))
            ckpt_fps = checkpoint.get('source_fps', config.get('source_fps', None))

            if ckpt_clip and ckpt_clip != self.window_size:
                print(f"  WARNING: checkpoint clip_len={ckpt_clip} vs CLI --window-size={self.window_size}")
                print(f"           Using checkpoint value {ckpt_clip}")
                self.window_size = ckpt_clip
            if ckpt_fsr and ckpt_fsr != self.frame_sample_rate:
                print(f"  WARNING: checkpoint frame_sample_rate={ckpt_fsr} vs CLI={self.frame_sample_rate}")
                print(f"           Using checkpoint value {ckpt_fsr}")
                self.frame_sample_rate = ckpt_fsr

            self.source_fps = int(ckpt_fps) if ckpt_fps else 30
            if self.strict_mode:
                # In strict mode we drain worker results aggressively to avoid queue buildup.
                target_gui_fps = max(30, min(60, int(round(self.source_fps))))
            else:
                # GUI redraw target follows effective sampled rate by default.
                target_gui_fps = max(
                    15,
                    min(60, int(round(self.source_fps / max(self.frame_sample_rate, 1))))
                )
            self.gui_update_ms = max(1, int(round(1000.0 / float(target_gui_fps))))

            self.voxel_size = int(voxel_size[0])
            self.feature_buffer = RollingFeatureBuffer(
                window_size=self.window_size,
                voxel_size=self.voxel_size,
                voxel_normalization=self.voxel_normalization,
                in_channels=self.in_channels,
                voxel_grid_size=self.voxel_grid_size,
                fusion_mode=self.fusion_mode,
                pose_dim=self.pose_dim,
            )

            # Initialize YOLO pose estimator for fusion mode
            if self.fusion_mode:
                try:
                    self.pose_estimator = YOLOPoseEstimator(
                        weights=self.fusion_pose_weights,
                        device=self.device,
                        conf=0.15,
                        imgsz=320,
                    )
                    print(f"  Fusion mode: YOLO pose estimator loaded ({self.fusion_pose_weights})")
                except Exception as e:
                    print(f"  WARNING: Could not load YOLO pose estimator: {e}")
                    print(f"  Fusion model will run with zero pose features (voxel-only degraded mode)")
                    self.pose_estimator = None

            if self.use_action_state_machine:
                self.action_state_machine = _CausalActionStateMachine(
                    labels=self.labels,
                    enter_consecutive=self.state_enter_consecutive,
                    exit_consecutive=self.state_exit_consecutive,
                    min_hold_steps=self.state_min_hold_steps,
                    sustain_confidence=self.state_sustain_confidence,
                    peak_drop_threshold=self.state_peak_drop_threshold,
                )
            else:
                self.action_state_machine = None
            self.model_ready = True
            print(f"Model loaded: {num_classes} classes, voxel {voxel_size}")
            print(f"  Model arch: {self.model_arch}")
            print(f"  Feature mode: {self.feature_mode}")
            print(f"  In channels: {self.in_channels}")
            print(f"  Voxel normalization: {self.voxel_normalization}")
            print(
                "  Transformer: "
                f"d_model={config.get('transformer_d_model', 256)}, "
                f"heads={config.get('transformer_heads', 8)}, "
                f"layers={config.get('transformer_layers', 4)}"
            )
            print(f"  Window: {self.window_size} | FSR: {self.frame_sample_rate}")
            if self.fusion_mode:
                print(f"  Fusion mode: pose_dim={self.pose_dim}, pose_embed_dim={self.pose_embed_dim}")
            if self.use_action_state_machine:
                print(
                    "  Action state machine: "
                    f"enter={self.state_enter_consecutive}, "
                    f"exit={self.state_exit_consecutive}, "
                    f"hold={self.state_min_hold_steps}, "
                    f"sustain>={self.state_sustain_confidence:.2f}, "
                    f"peak_drop>={self.state_peak_drop_threshold:.3f}"
                )
            if self.fusion_mode:
                print("  Processed streams: voxel, pose, foreground_ratio")
            else:
                print("  Processed streams: voxel, foreground_ratio")
            print(
                "  Processing: "
                f"{self.processing_mode} | queues job={self.feature_queue_size}, "
                f"result={self.result_queue_size}, sensor={self.sensor_queue_size}"
            )
            print(f"  Labels: {self.labels}")

        except Exception as e:
            self.model_error = str(e)
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()

    # ------------------------------------------------------------------
    #  CAMERA INITIALIZATION
    # ------------------------------------------------------------------

    def _init_camera(self):
        """Initialize RealSense camera + feature extractors."""
        try:
            print("Initializing RealSense camera...")

            # ----------------------------------------------------------
            #  AUTO PITCH — read IMU via SEPARATE pipeline BEFORE main
            #  (matches recording script: IMU can't coexist with RGB+depth)
            # ----------------------------------------------------------
            if self.auto_pitch and self.camera_pitch == 0.0:
                imu_pitch, imu_roll = _read_imu_pitch_separate(duration=1.5)
                if abs(imu_pitch) > 0.5:
                    self.camera_pitch = imu_pitch
                    self.camera_roll = imu_roll
                    print(f"  Camera pitch (auto IMU): {self.camera_pitch:.1f}°, "
                          f"roll: {self.camera_roll:.1f}°")
                else:
                    print(f"  IMU pitch near zero ({imu_pitch:.1f}°) — camera is level")
            elif self.camera_pitch != 0.0:
                print(f"  Camera pitch (from CLI): {self.camera_pitch:.1f}°")

            # ----------------------------------------------------------
            #  START MAIN PIPELINE — RGB + Depth only (no accel)
            # ----------------------------------------------------------
            # Longer pause after IMU pipeline to let USB/CSI bus settle (Jetson needs this)
            time.sleep(2.0)

            # Hardware reset the RealSense device to clear any stale state
            try:
                ctx = rs.context()
                devices = ctx.query_devices()
                if len(devices) > 0:
                    print("  Resetting RealSense hardware...")
                    devices[0].hardware_reset()
                    time.sleep(3.0)
            except Exception as e:
                print(f"  Hardware reset skipped: {e}")

            self.pipeline = rs.pipeline()
            rs_config = rs.config()
            
            # Using source_fps from checkpoint (fallback to 30)
            target_fps = getattr(self, 'source_fps', 30)
            if target_fps not in [15, 30, 60, 90]:
                print(f"  Warning: Unusual target FPS {target_fps}. Defaulting to 60 or closest supported.")
                target_fps = 60

            rw, rh = self.rgb_res
            dw, dh = self.depth_res
            
            rs_config.enable_stream(rs.stream.color, rw, rh, rs.format.bgr8, target_fps)
            rs_config.enable_stream(rs.stream.depth, dw, dh, rs.format.z16, target_fps)
            profile = self.pipeline.start(rs_config)

            # Queue depth + exposure settings.
            depth_sensor = None
            try:
                device = profile.get_device()
                color_sensor = None
                for sensor in device.query_sensors():
                    name = sensor.get_info(rs.camera_info.name)
                    lname = name.lower()
                    if ('rgb' in lname or 'color' in lname) and color_sensor is None:
                        color_sensor = sensor
                    if ('stereo' in lname or 'depth' in lname) and depth_sensor is None:
                        depth_sensor = sensor

                    # In strict mode allow queueing to preserve frame continuity.
                    if sensor.supports(rs.option.frames_queue_size):
                        sensor.set_option(rs.option.frames_queue_size, float(self.sensor_queue_size))

                if color_sensor is not None and color_sensor.supports(rs.option.auto_exposure_priority):
                    color_sensor.set_option(rs.option.auto_exposure_priority, 0.0)
                    print("  RGB auto_exposure_priority=0 (fixed FPS mode)")
                if depth_sensor is not None and depth_sensor.supports(rs.option.auto_exposure_priority):
                    depth_sensor.set_option(rs.option.auto_exposure_priority, 0.0)
                    print("  Depth auto_exposure_priority=0 (fixed FPS mode)")
                print(
                    "  Sensor frames_queue_size="
                    f"{self.sensor_queue_size} ({self.processing_mode}-mode)"
                )
            except Exception as e:
                print(f"  Warning: Could not apply camera queue/exposure settings: {e}")

            # Apply 'High Accuracy' visual preset (value 3)
            try:
                if depth_sensor is None:
                    depth_sensor = profile.get_device().first_depth_sensor()
                if depth_sensor.supports(rs.option.visual_preset):
                    depth_sensor.set_option(rs.option.visual_preset, 3) # 3 = High Accuracy
                    print("  Applied 'High Accuracy' visual preset to depth sensor for optimal quality.")
            except Exception as e:
                print(f"  Warning: Could not set visual preset: {e}")

            # Enable alignment from Depth to Color to ensure exact matching
            self.align_to = rs.align(rs.stream.color)
            print("  Depth-to-RGB alignment enabled")

            # Get depth intrinsics (after alignment, the depth intrinsics will match the color intrinsics)
            # Actually, to get intrinsics AFTER alignment for feature config, we will just use color intrinsics
            # or the depth stream profile from the aligned stream, which takes RGB intrinsics anyway.
            # But the simplest is to just get the color stream's intrinsics here.
            color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
            intrinsics = color_stream.get_intrinsics()
            print(f"  Intrinsics: fx={intrinsics.fx:.1f}, fy={intrinsics.fy:.1f}, "
                  f"cx={intrinsics.ppx:.1f}, cy={intrinsics.ppy:.1f}")

            self.pitch_label.config(text=f"Pitch: {self.camera_pitch:.1f}°")

            # ----------------------------------------------------------
            #  FEATURE CONFIG — match training exactly
            # ----------------------------------------------------------
            # Optional downscaling for feature extraction.
            native_w = int(intrinsics.width)
            native_h = int(intrinsics.height)
            requested_w = self.downscale_width

            if requested_w is None:
                target_w = native_w
            elif requested_w >= native_w:
                target_w = native_w
                if requested_w > native_w:
                    print(f"  Downscale disabled: requested width {requested_w} >= source width {native_w}")
            else:
                target_w = int(requested_w)

            self.downscale_width = target_w
            self.scale_factor = float(target_w) / native_w if native_w > 0 else 1.0
            target_h = int(round(native_h * self.scale_factor))

            if self.scale_factor < 0.999:
                print(
                    f"  Feature resolution: downscaled "
                    f"{native_w}x{native_h} -> {target_w}x{target_h} "
                    f"(scale={self.scale_factor:.3f})"
                )
            else:
                print(f"  Feature resolution: full-res {native_w}x{native_h} (no downscaling)")

            self.feature_config = VoxelFeatureConfig(
                fx=intrinsics.fx * self.scale_factor,
                fy=intrinsics.fy * self.scale_factor,
                cx=intrinsics.ppx * self.scale_factor,
                cy=intrinsics.ppy * self.scale_factor,
                depth_scale=0.001,
                camera_pitch_deg=self.camera_pitch,
                camera_roll_deg=self.camera_roll,
                # Voxel settings — MUST match training
                voxel_grid_size=self.voxel_grid_size,
                voxel_person_centric=True,
                voxel_depth_weighted=bool(self.dataset_config.get('voxel_depth_weighted', True)),
                directional_gradients=bool(self.dataset_config.get('directional_gradients', False)),
                velocity_magnitude_channel=bool(self.dataset_config.get('velocity_magnitude_channel', False)),
                multi_scale_delta_frames=tuple(
                    int(v) for v in (self.dataset_config.get('multi_scale_delta_frames') or ())
                ),
                voxel_delta_frames=int(self.dataset_config.get('voxel_delta_frames', 3)),
                include_raw_occupancy=bool(self.dataset_config.get('include_raw_occupancy', False)),
            )

            # Initialize extractors with tilt correction
            # The capture-thread extractor runs at full frame rate (30fps) to keep
            # voxel delta lookbacks consistent with training.  Workers only run YOLO.
            self.bg_model = BackgroundModel(self.feature_config)
            self.voxel_extractor = VoxelOccupancyExtractor(self.feature_config)
            self.voxel_extractor.set_tilt_rotation(self.camera_pitch, self.camera_roll)
            self._capture_voxel_lock = threading.Lock()
            print("  Runtime extraction: voxel=capture-thread (rate-matched), pose=worker-thread")

            # Load YOLO person detection model (if available)
            if self.use_yolo and self.yolo_checkpoint:
                yolo_path = Path(self.yolo_checkpoint)
                if not yolo_path.is_absolute():
                    yolo_path = (Path(_PROJECT_ROOT) / yolo_path).resolve()
                if yolo_path.exists():
                    try:
                        self.yolo_model = _UltralyticsYOLO(str(yolo_path))
                        # Warm up with a dummy frame
                        dummy = np.zeros((64, 64, 3), dtype=np.uint8)
                        self.yolo_model.predict(dummy, verbose=False, device=self.device)
                        print(f"  YOLO person detector loaded: {yolo_path.name}")
                    except Exception as e:
                        print(f"  Warning: Could not load YOLO model: {e}")
                        self.yolo_model = None
                        self.use_yolo = False
                else:
                    print(f"  Warning: YOLO checkpoint not found: {yolo_path}")
                    self.use_yolo = False
            elif self.use_yolo and not self.yolo_checkpoint:
                print("  YOLO: no checkpoint specified, using depth-only foreground")
                self.use_yolo = False

            # Build background model from first 30 frames
            print("  Building background model...")
            bg_frames = []
            
            # Flush first few frames which are often invalid or empty.
            # Use a generous timeout for the very first frame — on Jetson the
            # USB/CSI stack can be slow to deliver after the IMU pipeline stops.
            for i in range(10):
                timeout = 15000 if i == 0 else 5000
                try:
                    self.pipeline.wait_for_frames(timeout_ms=timeout)
                except RuntimeError:
                    if i == 0:
                        print("  Retrying first frame with extended timeout...")
                        time.sleep(1.0)
                        self.pipeline.wait_for_frames(timeout_ms=15000)
                self.root.update()
                
            for _ in range(30):
                frames = self.pipeline.wait_for_frames()
                # Use alignment here too so we get the exact identical intrinsic properties
                if hasattr(self, 'align_to'):
                    frames = self.align_to.process(frames)
                    
                depth_frame = frames.get_depth_frame()
                if depth_frame:
                    depth = np.asanyarray(depth_frame.get_data())
                    if self.scale_factor != 1.0:
                        target_h = int(intrinsics.height * self.scale_factor)
                        depth = cv2.resize(depth, (self.downscale_width, target_h), interpolation=cv2.INTER_NEAREST)

                    depth_m = depth.astype(np.float32) * 0.001
                    bg_frames.append(depth_m)
                    
                self.root.update() # Keep GUI responsive

            if bg_frames:
                bg_stack = np.stack(bg_frames, axis=0)
                bg_depth = np.percentile(bg_stack, 90, axis=0).astype(np.float32)
                self.bg_model.set_background(bg_depth)
                print(f"  Background model built from {len(bg_frames)} frames")

            self.camera_ready = True
            print(f"Camera ready (tilt: pitch={self.camera_pitch:.1f}°, "
                  f"roll={self.camera_roll:.1f}°)")

            if self.use_depth_punch:
                d = self.depth_punch_detector
                print(f"  Depth punch detector enabled (percentile={d.near_percentile:.0f}%, "
                      f"vel_thresh={d.velocity_threshold:.3f}, "
                      f"history={d.history_len})")

            # Start async feature workers (each with independent extractor copies).
            self._feature_worker_stop.clear()
            self._feature_worker_threads = []
            for wid in range(self.num_workers):
                w_extractors = self._make_worker_extractors(wid)
                t = threading.Thread(
                    target=self._feature_worker_loop,
                    args=(wid, w_extractors),
                    daemon=True,
                )
                t.start()
                self._feature_worker_threads.append(t)
            print(f"  {self.num_workers} feature worker(s) started")

            # Start dedicated inference thread (GPU inference off GUI thread).
            self._inference_stop.clear()
            self._inference_thread = threading.Thread(
                target=self._inference_thread_loop, daemon=True
            )
            self._inference_thread.start()
            print("  Inference thread started")

            # Start dedicated capture thread.
            self._capture_stop.clear()
            self._capture_thread = threading.Thread(
                target=self._capture_loop, daemon=True
            )
            self._capture_thread.start()
            print("  Capture thread started")

            # Start background IMU thread for continuous pitch updates
            self._imu_lock = threading.Lock()
            self._imu_pending_pitch = None  # set by bg thread, consumed by main loop
            self._imu_pending_roll = None
            self._imu_stop = threading.Event()
            if self.auto_pitch:
                self._imu_thread = threading.Thread(
                    target=self._imu_background_loop, daemon=True
                )
                self._imu_thread.start()
                print("  Background IMU tracking started")

            # Start frame loop
            self.root.after(10, self._update_frame)

        except Exception as e:
            print(f"Error initializing camera: {e}")
            import traceback
            traceback.print_exc()

    # ------------------------------------------------------------------
    #  CONTINUOUS IMU TRACKING (background thread)
    # ------------------------------------------------------------------

    def _imu_background_loop(self):
        """Background thread: periodically reads IMU via separate pipeline.

        Since the D435i can't stream accel + RGB+depth in the same pipeline,
        this thread creates a short-lived IMU-only pipeline every few seconds,
        reads accel samples, computes pitch/roll, and posts the update.
        """
        import math

        while not self._imu_stop.is_set():
            # Wait for 120 seconds (2 minutes) OR until manually triggered via event
            self._imu_stop.wait(120.0)
            if self._imu_stop.is_set():
                break

            try:
                pitch, roll = _read_imu_pitch_separate(duration=0.5)
                if abs(pitch) > 0.1 or abs(roll) > 0.1:
                    with self._imu_lock:
                        self._imu_pending_pitch = pitch
                        self._imu_pending_roll = roll
            except Exception:
                pass  # Don't crash the thread

    def _manual_imu_update(self):
        """Manually trigger an IMU update in the background thread."""
        if hasattr(self, '_imu_stop') and not self._imu_stop.is_set():
            print("  Manual IMU update requested...")
            # Rather than messing with the wait() event directly, 
            # we just quickly override the sleep and force a poll by restarting the loop timer
            # A cleaner way using threading is to set a separate event or just spawn a one-off thread
            threading.Thread(target=self._force_imu_read, daemon=True).start()
            
    def _force_imu_read(self):
        try:
            pitch, roll = _read_imu_pitch_separate(duration=0.5)
            if abs(pitch) > 0.1 or abs(roll) > 0.1:
                with self._imu_lock:
                    self._imu_pending_pitch = pitch
                    self._imu_pending_roll = roll
        except Exception as e:
            print(f"Could not read IMU: {e}")

    def _enqueue_feature_job(self, job: dict):
        """Queue feature job according to selected processing mode."""
        if self.strict_mode:
            while self.running and not self._capture_stop.is_set():
                try:
                    self._feature_job_queue.put(job, timeout=0.05)
                    return True
                except queue.Full:
                    continue
            return False

        # Segment mode: use large queue but don't block — drop oldest if full.
        # Latest mode: same but with small queue (size 1).
        try:
            self._feature_job_queue.put_nowait(job)
            return True
        except queue.Full:
            pass

        try:
            self._feature_job_queue.get_nowait()
        except queue.Empty:
            pass

        try:
            self._feature_job_queue.put_nowait(job)
            return True
        except queue.Full:
            return False

    def _enqueue_feature_result(self, result: dict):
        """Queue worker output according to selected processing mode."""
        if self.strict_mode:
            while self.running and not self._feature_worker_stop.is_set():
                try:
                    self._feature_result_queue.put(result, timeout=0.05)
                    return True
                except queue.Full:
                    continue
            return False

        # Segment mode: large queue, non-blocking. Latest mode: small queue.
        try:
            self._feature_result_queue.put_nowait(result)
            return True
        except queue.Full:
            pass

        try:
            self._feature_result_queue.get_nowait()
        except queue.Empty:
            pass

        try:
            self._feature_result_queue.put_nowait(result)
            return True
        except queue.Full:
            return False

    def _make_worker_extractors(self, worker_id: int) -> dict:
        """Create independent extractor copies for a worker thread.

        Each worker gets its own BackgroundModel (shared bg data, read-only),
        and VoxelOccupancyExtractor.
        This eliminates locking between workers.
        """
        bg = BackgroundModel(self.feature_config)
        # Copy the already-built background depth (read-only after init)
        if self.bg_model.is_initialized():
            bg.set_background(self.bg_model.get_background().copy())

        voxel = VoxelOccupancyExtractor(self.feature_config)
        voxel.set_tilt_rotation(self.camera_pitch, self.camera_roll)

        return {
            'bg_model': bg,
            'voxel_extractor': voxel,
        }

    def _detect_person_yolo(self, rgb: np.ndarray) -> 'np.ndarray | None':
        """Run YOLO on RGB to get person bounding box [x1, y1, x2, y2].

        Returns None if no person detected or YOLO is disabled.
        """
        if self.yolo_model is None:
            return None

        try:
            with self._yolo_lock:  # serialize GPU access across workers
                results = self.yolo_model.predict(
                    rgb, verbose=False, device=self.device,
                    classes=[0],  # class 0 = person in COCO
                    conf=0.3,
                    imgsz=160,  # minimal input for speed
                )
            if results and len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                if len(boxes) > 0:
                    # Take the largest person detection
                    areas = (boxes.xyxy[:, 2] - boxes.xyxy[:, 0]) * \
                            (boxes.xyxy[:, 3] - boxes.xyxy[:, 1])
                    best_idx = int(areas.argmax())
                    xyxy = boxes.xyxy[best_idx].cpu().numpy()
                    return xyxy.astype(np.float32)
        except Exception:
            pass

        return None

    def _feature_worker_loop(self, worker_id: int, extractors: dict):
        """Background worker for voxel feature extraction.

        Each worker has its own extractor copies (no shared lock needed).
        """
        while not self._feature_worker_stop.is_set():
            try:
                job = self._feature_job_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            if job is None:
                break

            try:
                result = self._compute_feature_result(job, extractors)
                if result is not None:
                    self._enqueue_feature_result(result)
            except Exception as e:
                self._enqueue_feature_result({'error': str(e)})

    def _run_pose_estimation(self, rgb: np.ndarray) -> dict:
        """Run YOLO pose estimation on RGB frame (fusion mode).

        Returns dict with keypoints, confidences, and bbox.
        Thread-safe via _pose_lock to serialize GPU access.
        """
        pose_result = {'keypoints': None, 'confidences': None, 'bbox': None}
        if self.pose_estimator is None:
            return pose_result
        try:
            with self._pose_lock:
                kps, confs, bbox = self.pose_estimator.predict_with_bbox(rgb)
            pose_result['keypoints'] = kps
            pose_result['confidences'] = confs
            pose_result['bbox'] = bbox
        except Exception:
            pass
        return pose_result

    def _compute_feature_result(self, job: dict, extractors: dict) -> dict:
        """Compute one sampled frame's features (worker thread).

        Voxel features are PRE-COMPUTED in the capture thread at full frame rate
        so that delta lookbacks match training.  This worker only runs YOLO pose.
        """
        rgb = job['rgb']
        depth = job['depth']
        depth_m = job['depth_m']
        frame_count = int(job['frame_count'])

        # Use pre-computed voxel features from capture thread
        voxel_flat = job.get('voxel_features_precomputed')
        voxel_grid = job.get('voxel_grid_precomputed')
        fg_ratio = job.get('fg_ratio_precomputed', 0.0)

        # Fallback: if capture thread didn't compute voxels (bg model not ready)
        if voxel_flat is None:
            bg_model = extractors['bg_model']
            voxel_extractor = extractors['voxel_extractor']
            fg_mask = bg_model.get_foreground_mask(depth)
            fg_ratio = float(np.mean(fg_mask)) if fg_mask is not None else 0.0
            voxel_flat = voxel_extractor.extract(
                depth_m, fg_mask, return_debug=False,
            ).astype(np.float32, copy=False)
            voxel_grid = _debug_voxel_grid(
                voxel_flat,
                tuple(int(v) for v in self.feature_config.voxel_grid_size),
                int(self.in_channels),
            )

        # Run YOLO pose on RGB — skip frames via yolo_interval and reuse
        # the last pose result to keep prediction FPS high on Jetson.
        # Velocities are computed from consecutive static pose features.
        pose_features = None
        pose_kps = None
        pose_confs = None
        pose_bbox = None
        if self.fusion_mode:
            from action_prediction.lib.fusion_model import (
                extract_pose_features_static, compute_pose_velocity,
                POSE_STATIC_DIM, POSE_VELOCITY_DIM,
            )
            prev_static = getattr(self, '_prev_pose_static', None)

            run_pose = (frame_count % max(1, self.yolo_interval) == 0)
            if run_pose:
                pose_data = self._run_pose_estimation(rgb)
                kps = pose_data.get('keypoints')
                confs = pose_data.get('confidences')
                # Compute static features (26-dim)
                cur_static = extract_pose_features_static(kps, confs)
                # Compute velocity from previous frame (16-dim)
                vel = compute_pose_velocity(cur_static, prev_static) if prev_static is not None else np.zeros(POSE_VELOCITY_DIM, dtype=np.float32)
                pose_features = np.concatenate([cur_static, vel])
                self._prev_pose_static = cur_static.copy()
                pose_kps = kps
                pose_confs = confs
                pose_bbox = pose_data.get('bbox')
                # Cache for reuse on skipped frames
                self._cached_pose = {
                    'features': pose_features,
                    'kps': pose_kps,
                    'confs': pose_confs,
                    'bbox': pose_bbox,
                }
            else:
                # Reuse cached pose from last YOLO run
                cached = getattr(self, '_cached_pose', None)
                if cached is not None:
                    pose_features = cached['features']
                    pose_kps = cached['kps']
                    pose_confs = cached['confs']
                    pose_bbox = cached['bbox']
                else:
                    pose_features = extract_pose_features(None, None)

        # Depth punch detection
        punch_result = None
        if self.depth_punch_detector is not None:
            fg_mask = extractors['bg_model'].get_foreground_mask(depth)
            punch_result = self.depth_punch_detector.update(depth_m, fg_mask)

        return {
            'frame_count': frame_count,
            'voxel_features': voxel_flat,
            'voxel_grid': voxel_grid,
            'fg_ratio': fg_ratio,
            'punch_result': punch_result,
            'pose_features': pose_features,
            'pose_keypoints': pose_kps,
            'pose_confidences': pose_confs,
            'pose_bbox': pose_bbox,
        }

    def _apply_feature_result(self, result: dict):
        """Apply worker result on main thread and trigger inference if due."""
        frame_count = int(result['frame_count'])
        if frame_count <= self._last_applied_feature_frame:
            # Drop stale/out-of-order feature results.
            return

        # If model load changed runtime clip_len, keep buffer aligned.
        if self.feature_buffer.window_size != self.window_size:
            print(
                f"  [Auto-Fix] Resizing RollingFeatureBuffer from "
                f"{self.feature_buffer.window_size} to {self.window_size}"
            )
            self.feature_buffer = RollingFeatureBuffer(
                window_size=self.window_size,
                voxel_size=self.voxel_size,
                voxel_normalization=self.voxel_normalization,
                in_channels=self.in_channels,
                voxel_grid_size=self.voxel_grid_size,
                fusion_mode=self.fusion_mode,
                pose_dim=self.pose_dim,
            )

        voxel_features = result['voxel_features']
        voxel_grid = result['voxel_grid']
        fg_ratio = result['fg_ratio']
        punch_result = result.get('punch_result')
        pose_features = result.get('pose_features')
        self._last_applied_feature_frame = frame_count
        self._sampled_frames_applied += 1

        # Store latest punch detection result for overlay + inference
        if punch_result is not None:
            self._latest_punch_result = punch_result

        # Store latest pose data for overlay drawing
        self._latest_pose_kps = result.get('pose_keypoints')
        self._latest_pose_confs = result.get('pose_confidences')
        self._latest_pose_bbox = result.get('pose_bbox')

        # Debug/overlay state
        self.last_voxel_grid = voxel_grid

        # --- Segment mode: buffer features, classify when punch ends ---
        if self.segment_mode and self.segment_classifier is not None:
            self.feature_buffer.add_frame(voxel_features, fg_ratio, pose_features=pose_features)
            segment = self.segment_classifier.feed(voxel_features, fg_ratio)
            if segment is not None and self.model_ready:
                self._classify_segment(segment)
            return

        # --- Streaming mode (original): rolling buffer + periodic inference ---
        self.feature_buffer.add_frame(voxel_features, fg_ratio, pose_features=pose_features)

        if (
            self.model_ready and
            self.feature_buffer.is_ready and
            self._sampled_frames_applied % max(1, int(self.inference_interval)) == 0
        ):
            # Submit inference to dedicated thread (non-blocking).
            features = self.feature_buffer.get_features()
            if features is not None:
                # Attach punch signal so inference thread can use it for gating
                if self._latest_punch_result is not None:
                    features['depth_punch_active'] = self._latest_punch_result.get('punch_active', False)
                    features['depth_punch_signal'] = self._latest_punch_result.get('punch_signal', 0.0)
                    features['depth_retracting'] = self._latest_punch_result.get('retracting', False)
                try:
                    self._inference_queue.put_nowait(features)
                except queue.Full:
                    # Drop this inference cycle — previous one still running.
                    try:
                        self._inference_queue.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        self._inference_queue.put_nowait(features)
                    except queue.Full:
                        pass

    def _classify_segment(self, segment: np.ndarray):
        """Classify a complete punch segment on the main thread.

        segment: (T, feature_dim) raw voxel features for the punch.
        Runs model.forward() directly — MLP is <1ms, safe on main thread.
        """
        T, F = segment.shape
        clip_len = self.window_size

        # Proportional sampling (same as training): sample uniformly across segment
        if T >= clip_len:
            indices = np.linspace(0, T - 1, clip_len).astype(int)
            sampled = segment[indices]
            pad_mask = np.zeros(clip_len, dtype=bool)
        else:
            pad_len = clip_len - T
            sampled = np.concatenate([
                np.zeros((pad_len, F), dtype=np.float32),
                segment,
            ], axis=0)
            pad_mask = np.zeros(clip_len, dtype=bool)
            pad_mask[:pad_len] = True

        # Normalise (same as RollingFeatureBuffer)
        sampled = sampled.astype(np.float32)
        voxel_dim = int(np.prod(self.voxel_grid_size))
        total_voxel = voxel_dim * self.in_channels
        voxel_part = sampled[:, :total_voxel]
        if self.voxel_normalization == 'channel_p90':
            ch = self.in_channels
            vox_5d = voxel_part.reshape(clip_len, ch, *self.voxel_grid_size)
            for c in range(ch):
                ch_energy = np.abs(vox_5d[:, c]).sum(axis=(1, 2, 3))
                scale = float(np.percentile(ch_energy[ch_energy > 0], 90)) if np.any(ch_energy > 0) else 1.0
                if scale > 1e-8:
                    vox_5d[:, c] /= scale
            sampled[:, :total_voxel] = vox_5d.reshape(clip_len, -1)
        elif self.voxel_normalization == 'clip_p90':
            vals = np.abs(voxel_part)
            p90 = float(np.percentile(vals[vals > 0], 90)) if np.any(vals > 0) else 1.0
            if p90 > 1e-8:
                sampled[:, :total_voxel] /= p90

        try:
            with torch.no_grad():
                tensor = torch.from_numpy(sampled).float().unsqueeze(0).to(self.device)
                mask = torch.from_numpy(pad_mask).bool().unsqueeze(0).to(self.device)
                with torch.amp.autocast('cuda', enabled=self.device.startswith('cuda')):
                    output = self.model(tensor, padding_mask=mask)
                probs = torch.softmax(output['logits'], dim=1)[0].cpu().numpy()

            pred_idx = int(np.argmax(probs))
            confidence = float(probs[pred_idx])
            label = self.labels[pred_idx] if pred_idx < len(self.labels) else f"class_{pred_idx}"

            # Skip idle predictions
            idle_idx = self.labels.index('idle') if 'idle' in self.labels else None
            if pred_idx == idle_idx:
                # Show the second-best non-idle if confidence is low
                probs_copy = probs.copy()
                if idle_idx is not None:
                    probs_copy[idle_idx] = 0
                alt_idx = int(np.argmax(probs_copy))
                if probs_copy[alt_idx] > 0.15:
                    pred_idx = alt_idx
                    confidence = float(probs[alt_idx])
                    label = self.labels[alt_idx]
                else:
                    self.segment_classifier._state = "IDLE"
                    return

            self.segment_classifier.set_result(label, confidence)
            print(f"  [SEGMENT] {label.upper()} ({confidence:.0%}) — {T} frames")

        except Exception as e:
            print(f"  [SEGMENT] Classification error: {e}")

    def _drain_feature_results(self):
        """Poll completed worker outputs and apply them."""
        if self.strict_mode or self.segment_mode:
            # Segment mode needs every frame to detect motion onset/offset.
            while True:
                try:
                    result = self._feature_result_queue.get_nowait()
                except queue.Empty:
                    break

                if isinstance(result, dict) and 'error' in result:
                    print(f"Feature worker error: {result['error']}")
                    continue

                self._apply_feature_result(result)

            return

        latest = None
        while True:
            try:
                result = self._feature_result_queue.get_nowait()
            except queue.Empty:
                break

            if isinstance(result, dict) and 'error' in result:
                print(f"Feature worker error: {result['error']}")
                continue

            latest = result

        if latest is not None:
            self._apply_feature_result(latest)

    def _capture_loop(self):
        """Continuously read camera frames and enqueue sampled feature jobs."""
        while self.running and not self._capture_stop.is_set():
            try:
                if self.strict_mode:
                    frames = self.pipeline.wait_for_frames(timeout_ms=100)
                else:
                    frames = self.pipeline.poll_for_frames()
                    if not frames:
                        time.sleep(0.001)
                        continue

                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue

                frame_no = int(color_frame.get_frame_number())
                self._capture_seen_frames += 1
                if self.strict_mode:
                    sample_this_frame = (
                        (self._capture_seen_frames - 1) % max(1, int(self.frame_sample_rate)) == 0
                    )
                else:
                    sample_this_frame = (
                        frame_no % max(1, int(self.frame_sample_rate)) == 0
                    )

                # Alignment is costly; only do it on sampled frames used by feature extraction.
                # GUI can render the latest raw frame in between without adding latency.
                if sample_this_frame and hasattr(self, 'align_to'):
                    aligned = self.align_to.process(frames)
                    aligned_color = aligned.get_color_frame()
                    aligned_depth = aligned.get_depth_frame()
                    if aligned_color and aligned_depth:
                        color_frame = aligned_color
                        depth_frame = aligned_depth

                rgb_full = np.asanyarray(color_frame.get_data())
                depth_full = np.asanyarray(depth_frame.get_data())

                sensor_ts_ms = float(color_frame.get_timestamp())

                # Camera-side FPS based on sensor timestamps.
                if self._last_sensor_ts_ms is not None:
                    dt_ms = sensor_ts_ms - self._last_sensor_ts_ms
                    if dt_ms > 0.1:
                        self.capture_fps_history.append(1000.0 / dt_ms)
                self._last_sensor_ts_ms = sensor_ts_ms

                # Keep only latest frame for GUI rendering.
                with self._frame_lock:
                    self._latest_frame_packet = (frame_no, sensor_ts_ms, rgb_full, depth_full)

                # Sampled feature jobs for worker (independent of GUI draw cadence).
                if sample_this_frame:
                    # Always enqueue the newest frame. If the queue is full,
                    # drop the oldest job to make room — this keeps prediction
                    # FPS stable instead of skipping new frames entirely.
                    if self._feature_job_queue.full():
                        try:
                            self._feature_job_queue.get_nowait()
                        except queue.Empty:
                            pass
                        now_t = time.time()
                        if self.feature_queue_size > 8 and (now_t - self._last_queue_warn_t) > 2.0:
                            print(
                                f"  [Queue] feature backlog — dropped oldest to stay current"
                            )
                            self._last_queue_warn_t = now_t

                    if hasattr(self, 'scale_factor') and self.scale_factor != 1.0:
                        target_w = self.downscale_width
                        target_h = int(rgb_full.shape[0] * self.scale_factor)
                        rgb = cv2.resize(rgb_full, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                        depth = cv2.resize(depth_full, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
                    else:
                        rgb = rgb_full
                        depth = depth_full

                    depth_m = depth.astype(np.float32) * 0.001

                    # Run voxel extraction HERE in the capture thread at full
                    # frame rate so delta lookbacks match training (60fps extraction).
                    # This is fast (~8ms) and keeps the voxel history buffer at 30fps.
                    # The worker thread only needs to run YOLO pose (slower, ~5ms).
                    voxel_flat = None
                    voxel_grid = None
                    fg_ratio = 0.0
                    try:
                        with self._capture_voxel_lock:
                            if self.bg_model.is_initialized():
                                fg_mask = self.bg_model.get_foreground_mask(depth)
                                fg_ratio = float(np.mean(fg_mask)) if fg_mask is not None else 0.0
                                voxel_flat = self.voxel_extractor.extract(
                                    depth_m, fg_mask, return_debug=False,
                                ).astype(np.float32, copy=False)
                                voxel_grid = _debug_voxel_grid(
                                    voxel_flat,
                                    tuple(int(v) for v in self.feature_config.voxel_grid_size),
                                    int(self.in_channels),
                                )
                            else:
                                # Still initializing background model
                                self.bg_model.update(depth)
                    except Exception:
                        pass

                    self._enqueue_feature_job({
                        'frame_count': frame_no,
                        'rgb': rgb,
                        'depth': depth,
                        'depth_m': depth_m,
                        'voxel_features_precomputed': voxel_flat,
                        'voxel_grid_precomputed': voxel_grid,
                        'fg_ratio_precomputed': fg_ratio,
                    })

            except Exception as e:
                if self.running and not self._capture_stop.is_set():
                    print(f"Capture thread error: {e}")
                time.sleep(0.01)

    def _schedule_next_update(self, loop_start: float):
        """Schedule next GUI tick while accounting for time already spent this tick."""
        elapsed_ms = (time.time() - loop_start) * 1000.0
        min_delay = 33 if self.no_video else 1  # 30 FPS cap for no-video mode
        delay_ms = max(min_delay, int(self.gui_update_ms - elapsed_ms))
        self.root.after(delay_ms, self._update_frame)

    # ------------------------------------------------------------------
    #  MAIN FRAME LOOP
    # ------------------------------------------------------------------

    def _update_frame(self):
        """Main frame update loop — runs as fast as possible."""
        if not self.running:
            return

        if not self.camera_ready:
            self.root.after(100, self._update_frame)
            return

        loop_start = time.time()

        try:
            with self._frame_lock:
                packet = self._latest_frame_packet

            if packet is None:
                self._drain_feature_results()
                self._schedule_next_update(loop_start)
                return

            frame_no, _sensor_ts_ms, rgb_full, depth_full = packet
            if frame_no == self._last_displayed_frame_no:
                # No new camera frame yet, but still apply finished feature jobs.
                self._drain_feature_results()
                self._schedule_next_update(loop_start)
                return

            self._last_displayed_frame_no = frame_no

            # Check for IMU pitch update from background thread
            if self._imu_pending_pitch is not None:
                with self._imu_lock:
                    new_pitch = self._imu_pending_pitch
                    new_roll = self._imu_pending_roll
                    self._imu_pending_pitch = None
                    self._imu_pending_roll = None
                if new_pitch is not None and abs(new_pitch - self.camera_pitch) > 0.5:
                    self.camera_pitch = new_pitch
                    self.camera_roll = new_roll if new_roll is not None else self.camera_roll
                    # Workers have independent extractors — update primary copies
                    # (workers are not affected; they keep their own tilt until re-created)
                    self.voxel_extractor.set_tilt_rotation(self.camera_pitch, self.camera_roll)
                    self.pitch_label.config(text=f"Pitch: {self.camera_pitch:.1f}°")

            self.frame_count += 1

            # Apply finished feature jobs (may queue inference).
            self._drain_feature_results()

            # Apply finished inference results to GUI labels.
            if not self.segment_mode:
                self._drain_inference_results()

            # Segment mode: update display from segment classifier
            if self.segment_mode and self.segment_classifier is not None:
                sc = self.segment_classifier
                if sc.is_displaying:
                    self.pred_label.config(
                        text=sc.last_label.upper().replace('_', ' '),
                        fg=COLORS['success'] if sc.last_confidence > 0.5 else COLORS['accent'],
                    )
                    self.conf_label.config(text=f"{sc.last_confidence * 100:.0f}%")
                elif sc.is_active:
                    self.pred_label.config(text="...", fg=COLORS['warning'])
                    self.conf_label.config(text="")
                else:
                    self.pred_label.config(text="READY", fg=COLORS['text_dim'])
                    self.conf_label.config(text="")

            # Draw overlay and update display (skip if no-video mode)
            if not self.no_video:
                depth_m_full = depth_full.astype(np.float32) * 0.001
                self.last_depth_m = depth_m_full
                display = self._draw_overlay(rgb_full, depth_m_full)
                self._update_video(display)

            # FPS
            elapsed = time.time() - loop_start
            self.fps_history.append(1.0 / max(elapsed, 0.001))
            avg_fps = np.mean(self.fps_history)
            pred_hz_txt = ""
            if len(self.inference_time_history) >= 2:
                t0 = float(self.inference_time_history[0])
                t1 = float(self.inference_time_history[-1])
                if t1 > t0:
                    pred_hz = (len(self.inference_time_history) - 1) / (t1 - t0)
                    pred_hz_txt = f" | Pred {pred_hz:.1f}"
            if len(self.capture_fps_history) > 0:
                cam_fps = float(np.mean(self.capture_fps_history))
                self.fps_label.config(text=f"FPS: GUI {avg_fps:.1f} | Cam {cam_fps:.1f}{pred_hz_txt}")
            else:
                self.fps_label.config(text=f"FPS: GUI {avg_fps:.1f}{pred_hz_txt}")

            if self.model_ready and self.camera_ready:
                self.status_label.config(text="Running", fg=COLORS['success'])

        except Exception as e:
            print(f"Frame error: {e}")

        self._schedule_next_update(loop_start)

    # ------------------------------------------------------------------
    #  INFERENCE
    # ------------------------------------------------------------------

    def _inference_thread_loop(self):
        """Dedicated thread for model inference (keeps GPU work off GUI thread)."""
        while not self._inference_stop.is_set():
            try:
                features = self._inference_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            if features is None:
                break

            try:
                result = self._run_inference_gpu(features)
                if result is not None:
                    # Replace old result if GUI hasn't consumed it yet.
                    try:
                        self._inference_result_queue.put_nowait(result)
                    except queue.Full:
                        try:
                            self._inference_result_queue.get_nowait()
                        except queue.Empty:
                            pass
                        try:
                            self._inference_result_queue.put_nowait(result)
                        except queue.Full:
                            pass
            except Exception as e:
                print(f"Inference thread error: {e}")

    def _setup_optimized_inference(self, checkpoint_path, model, num_classes,
                                    voxel_size, in_channels, clip_len):
        """Auto-convert PyTorch → ONNX → TensorRT and set up fast inference.

        Priority: TensorRT engine > ORT with GPU > PyTorch fallback.
        All artifacts are cached next to the .pth so subsequent runs are instant.
        """
        pth_path = Path(checkpoint_path)
        onnx_path = pth_path.with_suffix('.onnx')
        trt_path = pth_path.with_suffix('.trt')

        voxel_dim = int(voxel_size[0]) ** 3
        pose_dim = getattr(self, 'pose_dim', POSE_FEATURE_DIM)
        feat_dim = voxel_dim * in_channels + pose_dim
        clip_len = int(clip_len) if clip_len else 12

        # --- Step 1: Export to ONNX if not cached ---
        if not onnx_path.exists():
            print(f"  ONNX model not found — exporting to {onnx_path.name}...")
            try:
                dummy_features = torch.randn(1, clip_len, feat_dim)
                dummy_mask = torch.zeros(1, clip_len, dtype=torch.bool)

                model_cpu = model.cpu()
                torch.onnx.export(
                    model_cpu,
                    (dummy_features, dummy_mask),
                    str(onnx_path),
                    input_names=['features', 'padding_mask'],
                    output_names=['logits'],
                    dynamic_axes={
                        'features': {0: 'batch', 1: 'seq_len'},
                        'padding_mask': {0: 'batch', 1: 'seq_len'},
                        'logits': {0: 'batch'},
                    },
                    opset_version=17,
                    do_constant_folding=True,
                )
                model.to(self.device)
                size_mb = onnx_path.stat().st_size / (1024 * 1024)
                print(f"  ONNX exported: {onnx_path.name} ({size_mb:.1f} MB)")
            except Exception as e:
                print(f"  WARNING: ONNX export failed ({e}), using PyTorch")
                model.to(self.device)
                return
        else:
            size_mb = onnx_path.stat().st_size / (1024 * 1024)
            print(f"  ONNX model found: {onnx_path.name} ({size_mb:.1f} MB)")

        # --- Step 2: Try TensorRT direct engine (best perf on Jetson) ---
        if _TRT_AVAILABLE and self.device.startswith('cuda'):
            try:
                self._setup_tensorrt_engine(onnx_path, trt_path, feat_dim, clip_len)
                if self._trt_context is not None:
                    return  # TRT is live, skip ORT
            except Exception as e:
                print(f"  TensorRT setup failed ({e}), trying ORT...")

        # --- Step 3: Fall back to ORT if GPU provider available ---
        if _ORT_AVAILABLE:
            try:
                self._setup_ort_session(onnx_path, feat_dim, clip_len)
                if self._ort_session is not None:
                    return
            except Exception as e:
                print(f"  ORT setup failed ({e})")

        print("  Falling back to PyTorch eager mode.")

    def _setup_tensorrt_engine(self, onnx_path, trt_path, feat_dim, clip_len):
        """Build or load a TensorRT FP16 engine and allocate IO buffers.

        Uses PyTorch CUDA tensors as GPU buffers (no pycuda dependency).
        """
        if not trt_path.exists():
            print(f"  Building TensorRT engine (FP16) — this takes 1-2 minutes on first run...")
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(TRT_LOGGER)
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            parser = trt.OnnxParser(network, TRT_LOGGER)

            with open(str(onnx_path), 'rb') as f:
                if not parser.parse(f.read()):
                    for i in range(parser.num_errors):
                        print(f"    Parse error: {parser.get_error(i)}")
                    return

            config = builder.create_builder_config()
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)

            # Dynamic shape profile
            profile = builder.create_optimization_profile()
            feat_input = network.get_input(0)
            profile.set_shape(feat_input.name,
                              min=(1, 4, feat_dim),
                              opt=(1, clip_len, feat_dim),
                              max=(1, 64, feat_dim))
            mask_input = network.get_input(1)
            profile.set_shape(mask_input.name,
                              min=(1, 4),
                              opt=(1, clip_len),
                              max=(1, 64))
            config.add_optimization_profile(profile)

            t0 = time.time()
            engine_bytes = builder.build_serialized_network(network, config)
            if engine_bytes is None:
                print("  ERROR: TensorRT engine build failed")
                return

            with open(str(trt_path), 'wb') as f:
                f.write(engine_bytes)
            size_mb = trt_path.stat().st_size / (1024 * 1024)
            print(f"  TensorRT engine built: {trt_path.name} "
                  f"({size_mb:.1f} MB, {time.time() - t0:.0f}s)")
        else:
            size_mb = trt_path.stat().st_size / (1024 * 1024)
            print(f"  TensorRT engine found: {trt_path.name} ({size_mb:.1f} MB)")

        # Load engine and create execution context
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(TRT_LOGGER)
        with open(str(trt_path), 'rb') as f:
            engine = runtime.deserialize_cuda_engine(f.read())

        context = engine.create_execution_context()
        context.set_input_shape('features', (1, clip_len, feat_dim))
        context.set_input_shape('padding_mask', (1, clip_len))

        # Pre-allocate GPU buffers using PyTorch tensors (no pycuda needed)
        num_classes = len(self.labels)
        device = self.device
        d_features = torch.empty((1, clip_len, feat_dim), dtype=torch.float32, device=device)
        d_mask = torch.empty((1, clip_len), dtype=torch.bool, device=device)
        d_logits = torch.empty((1, num_classes), dtype=torch.float32, device=device)

        context.set_tensor_address('features', d_features.data_ptr())
        context.set_tensor_address('padding_mask', d_mask.data_ptr())
        context.set_tensor_address('logits', d_logits.data_ptr())

        # Create a CUDA stream for async TRT execution
        trt_stream = torch.cuda.Stream(device=device)

        self._trt_engine = engine
        self._trt_context = context
        self._trt_bindings = {
            'd_features': d_features,
            'd_mask': d_mask,
            'd_logits': d_logits,
            'feat_dim': feat_dim,
            'clip_len': clip_len,
            'num_classes': num_classes,
            'stream': trt_stream,
        }

        # Warmup
        d_features.normal_()
        d_mask.zero_()
        with torch.cuda.stream(trt_stream):
            context.execute_async_v3(stream_handle=trt_stream.cuda_stream)
        trt_stream.synchronize()

        print("  TensorRT inference ready (FP16)")

    def _setup_ort_session(self, onnx_path, feat_dim, clip_len):
        """Create an ORT inference session with the best GPU provider."""
        available_providers = ort.get_available_providers()
        providers = []

        if 'TensorrtExecutionProvider' in available_providers:
            providers.append((
                'TensorrtExecutionProvider', {
                    'device_id': int(self.device.split(':')[-1]) if ':' in self.device else 0,
                    'trt_max_workspace_size': 1 << 30,
                    'trt_fp16_enable': True,
                    'trt_engine_cache_enable': True,
                    'trt_engine_cache_path': str(onnx_path.parent),
                }
            ))
        if 'CUDAExecutionProvider' in available_providers:
            providers.append((
                'CUDAExecutionProvider', {
                    'device_id': int(self.device.split(':')[-1]) if ':' in self.device else 0,
                    'arena_extend_strategy': 'kSameAsRequested',
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                }
            ))
        providers.append('CPUExecutionProvider')

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4

        session = ort.InferenceSession(
            str(onnx_path), sess_options=sess_options, providers=providers,
        )

        active_provider = session.get_providers()[0]
        print(f"  ONNX Runtime session ready (provider: {active_provider})")

        # Warmup
        dummy_f = np.random.randn(1, clip_len, feat_dim).astype(np.float32)
        dummy_m = np.zeros((1, clip_len), dtype=bool)
        input_names = [inp.name for inp in session.get_inputs()]
        session.run(None, {input_names[0]: dummy_f, input_names[1]: dummy_m})
        print("  ONNX Runtime warmup complete")

        self._ort_session = session
        self._ort_input_names = input_names

    def _run_inference_gpu(self, features: dict) -> 'dict | None':
        """Run model forward pass on GPU (inference thread only).

        Priority: TensorRT > ONNX Runtime > PyTorch.
        Returns dict with prediction data for the GUI thread to consume.
        """
        self.inference_time_history.append(time.time())

        feat_np = features['features']  # (T, feat_dim)
        mask_np = features.get('padding_mask', None)

        if self._trt_context is not None:
            # --- TensorRT path (fastest on Jetson) ---
            b = self._trt_bindings
            T = feat_np.shape[0]

            # Update dynamic shapes if seq_len changed from pre-allocated size
            if T != b['clip_len']:
                self._trt_context.set_input_shape('features', (1, T, b['feat_dim']))
                self._trt_context.set_input_shape('padding_mask', (1, T))
                # Re-allocate buffers for new size
                b['d_features'] = torch.empty(
                    (1, T, b['feat_dim']), dtype=torch.float32, device=self.device)
                b['d_mask'] = torch.empty((1, T), dtype=torch.bool, device=self.device)
                self._trt_context.set_tensor_address('features', b['d_features'].data_ptr())
                self._trt_context.set_tensor_address('padding_mask', b['d_mask'].data_ptr())
                b['clip_len'] = T

            # Copy input data into pre-allocated GPU tensors
            feat_tensor = torch.from_numpy(feat_np).float().unsqueeze(0)
            b['d_features'][:, :T, :].copy_(feat_tensor, non_blocking=True)
            if mask_np is not None:
                mask_tensor = torch.from_numpy(mask_np).bool().unsqueeze(0)
                b['d_mask'][:, :T].copy_(mask_tensor, non_blocking=True)
            else:
                b['d_mask'].zero_()

            # Execute TRT on its own CUDA stream
            stream = b['stream']
            with torch.cuda.stream(stream):
                self._trt_context.execute_async_v3(stream_handle=stream.cuda_stream)
            stream.synchronize()

            # Read logits from GPU
            raw_probs = torch.softmax(b['d_logits'][0], dim=0).cpu().numpy()

        elif self._ort_session is not None:
            # --- ONNX Runtime path ---
            feat_input = feat_np[np.newaxis, :, :].astype(np.float32)
            if mask_np is not None:
                mask_input = mask_np[np.newaxis, :].astype(bool)
            else:
                mask_input = np.zeros((1, feat_input.shape[1]), dtype=bool)

            ort_inputs = {
                self._ort_input_names[0]: feat_input,
                self._ort_input_names[1]: mask_input,
            }
            logits_np = self._ort_session.run(None, ort_inputs)[0]

            logits_shifted = logits_np[0] - logits_np[0].max()
            exp_logits = np.exp(logits_shifted)
            raw_probs = exp_logits / exp_logits.sum()

        else:
            # --- PyTorch path (fallback) ---
            with torch.no_grad():
                combined = torch.from_numpy(feat_np).float().unsqueeze(0)
                combined = combined.to(self.device, non_blocking=True)
                padding_mask = None
                if mask_np is not None:
                    padding_mask = torch.from_numpy(mask_np).bool().unsqueeze(0)
                    padding_mask = padding_mask.to(self.device, non_blocking=True)

                with torch.amp.autocast('cuda', enabled=self.device.startswith('cuda')):
                    output = self.model(combined, padding_mask=padding_mask)

                logits = output['logits']
                probs = torch.softmax(logits, dim=1)
                raw_probs = probs[0].cpu().numpy()

        # Temporal smoothing: EMA for stability, windowed mean as fallback
        self.recent_probs.append(raw_probs)
        if self._ema_probs is None:
            self._ema_probs = raw_probs.copy()
        else:
            self._ema_probs = self._ema_alpha * raw_probs + (1.0 - self._ema_alpha) * self._ema_probs
        # Blend EMA with windowed mean for extra stability
        windowed_mean = np.mean(np.stack(self.recent_probs, axis=0), axis=0)
        smooth_probs = 0.6 * self._ema_probs + 0.4 * windowed_mean
        smooth_probs = smooth_probs / max(float(smooth_probs.sum()), 1e-8)

        # Feature stats
        fg_ratio = features.get('fg_ratio', None)
        if fg_ratio is None:
            fg_ratio = np.zeros((features['voxel'].shape[0],), dtype=np.float32)
        voxel_abs_mean = float(np.abs(features['voxel']).mean())
        voxel_active_ratio = float((np.abs(features['voxel']) > 0.01).mean())
        decision = _select_prediction(
            probs=smooth_probs,
            labels=self.labels,
            min_confidence=self.min_confidence,
            min_action_prob=self.min_action_prob,
            min_class_margin=self.min_class_margin,
            voxel_active_ratio=voxel_active_ratio,
            min_voxel_active_ratio=self.min_voxel_active_ratio,
        )

        idle_idx = decision.get('idle_idx')
        pred_idx = int(decision['pred_idx'])
        conf = float(decision['confidence'])

        # Prediction hysteresis: prevent flickering by requiring a margin
        # to switch away from the current held prediction.
        if self._held_pred_idx is not None:
            self._held_pred_frames += 1
            if pred_idx != self._held_pred_idx:
                held_conf = float(smooth_probs[self._held_pred_idx])
                new_conf = float(smooth_probs[pred_idx])
                # Only switch if: held prediction has been shown long enough
                # AND new class is clearly stronger than current.
                if (
                    self._held_pred_frames < self._min_hold_frames
                    or (new_conf - held_conf) < self._hysteresis_margin
                ):
                    # Stick with current prediction
                    pred_idx = self._held_pred_idx
                    conf = held_conf
                else:
                    # Switch to new prediction
                    self._held_pred_idx = pred_idx
                    self._held_pred_frames = 0
            else:
                # Same prediction, keep holding
                pass
        else:
            self._held_pred_idx = pred_idx
            self._held_pred_frames = 0

        depth_punch_active = features.get('depth_punch_active', None)
        depth_punch_signal = features.get('depth_punch_signal', 0.0)
        depth_gated = False
        block_idx = self.labels.index('block') if 'block' in self.labels else None

        # --- Depth punch filter (disabled — model predictions are reliable) ---
        # depth_punch_active = features.get('depth_punch_active', None)

        # --- Block consecutive filter ---
        # Block needs N consecutive frames before showing (prevents flicker).
        if pred_idx == block_idx:
            if not hasattr(self, '_block_consec_count'):
                self._block_consec_count = 0
                self._block_consec_needed = 4
            self._block_consec_count += 1
            if self._block_consec_count < self._block_consec_needed:
                pred_idx = idle_idx
                conf = float(smooth_probs[idle_idx])
        else:
            if hasattr(self, '_block_consec_count'):
                self._block_consec_count = 0

        # --- State machine (optional, runs after filters) ---
        state_name = 'passthrough'
        state_exit_reasons = 'none'
        if self.action_state_machine is not None:
            state_decision = self.action_state_machine.update(
                probs=smooth_probs,
                proposed_idx=pred_idx,
                proposed_conf=conf,
            )
            pred_idx = int(state_decision['pred_idx'])
            conf = float(state_decision['confidence'])
            state_name = str(state_decision.get('state', 'passthrough'))
            exit_reasons = state_decision.get('exit_reasons', [])
            if exit_reasons:
                state_exit_reasons = ",".join(str(reason) for reason in exit_reasons)

        prediction = self.labels[pred_idx]
        is_final_idle = (idle_idx is not None and pred_idx == idle_idx)
        feat_stats = {
            'voxel_activity': voxel_abs_mean,
            'voxel_active_ratio': voxel_active_ratio,
            'fg_ratio': float(np.mean(fg_ratio)),
            'action_prob': float(decision['action_prob']),
            'class_margin': float(decision['class_margin']),
            'gated': (bool(decision['gated']) or depth_gated) and is_final_idle,
            'gate_reasons': ",".join(
                (decision['gate_reasons'] if decision['gate_reasons'] else [])
                + (['depth_punch'] if (depth_gated and is_final_idle) else [])
            ) or 'none',
            'state': state_name,
            'state_exit_reasons': state_exit_reasons,
            'depth_punch_signal': depth_punch_signal,
            'depth_punch_active': bool(depth_punch_active) if depth_punch_active is not None else None,
        }

        latency_ms = (time.time() - self.inference_time_history[-1]) * 1000

        return {
            'prediction': prediction,
            'confidence': conf,
            'pred_idx': pred_idx,
            'smooth_probs': smooth_probs,
            'feature_stats': feat_stats,
            'latency_ms': latency_ms,
        }

    def _drain_inference_results(self):
        """Apply inference results to GUI labels (main thread)."""
        latest = None
        while True:
            try:
                result = self._inference_result_queue.get_nowait()
            except queue.Empty:
                break
            latest = result

        if latest is None:
            return

        prev_prediction = self.current_prediction
        self.current_prediction = latest['prediction']
        self.current_confidence = latest['confidence']
        self.class_probs = latest['smooth_probs']
        self.smooth_probs = latest['smooth_probs']
        self.feature_stats = latest['feature_stats']
        self.prediction_trail.append(latest['pred_idx'])

        # --- Console punch log (filtered) ---
        # Only log punches held for at least 3 consecutive predictions
        pred = self.current_prediction
        conf = self.current_confidence
        is_action = pred != 'idle'

        if not hasattr(self, '_punch_log_state'):
            self._punch_log_state = {
                'pending_action': None,   # action waiting to be confirmed
                'pending_count': 0,       # consecutive frames of pending action
                'pending_conf_sum': 0.0,  # sum of confidences for averaging
                'confirmed_action': None, # currently confirmed punch
                'confirmed_start': 0.0,   # start time of confirmed punch
                'min_confirm': 3,         # frames needed to confirm a punch
                'warmup_idle_count': 0,   # consecutive idle frames at startup
                'warmup_done': False,     # True after 30 consecutive idle frames
            }
        s = self._punch_log_state

        # Wait for 30 consecutive idle frames before logging anything
        if not s['warmup_done']:
            if not is_action:
                s['warmup_idle_count'] += 1
                if s['warmup_idle_count'] >= 30:
                    s['warmup_done'] = True
                    print("  [Ready] Stable idle detected — punch logging active")
            else:
                s['warmup_idle_count'] = 0
            return

        if is_action:
            if pred == s['pending_action']:
                s['pending_count'] += 1
                s['pending_conf_sum'] += conf
            else:
                # New action — reset pending
                s['pending_action'] = pred
                s['pending_count'] = 1
                s['pending_conf_sum'] = conf

            # Confirm punch after N consecutive frames
            if s['pending_count'] >= s['min_confirm'] and s['confirmed_action'] != pred:
                if s['confirmed_action'] is not None:
                    # End previous confirmed punch
                    duration = time.time() - s['confirmed_start']
                    print(f"      {'':20s}  ended after {duration:.2f}s")
                avg_conf = s['pending_conf_sum'] / s['pending_count']
                s['confirmed_action'] = pred
                s['confirmed_start'] = time.time()
                print(f"  >>> {pred.upper():20s}  conf={avg_conf:.0%}")
        else:
            # Idle — end any confirmed punch
            s['pending_action'] = None
            s['pending_count'] = 0
            s['pending_conf_sum'] = 0.0
            if s['confirmed_action'] is not None:
                duration = time.time() - s['confirmed_start']
                print(f"      {'':20s}  ended after {duration:.2f}s")
                s['confirmed_action'] = None

        self.latency_label.config(text=f"Latency: {latest['latency_ms']:.1f}ms")
        self.pred_label.config(text=self.current_prediction.upper())
        self.conf_label.config(text=f"{self.current_confidence * 100:.0f}%")

        if self.current_confidence > 0.7:
            self.pred_label.config(fg=COLORS['success'])
        elif self.current_confidence > 0.4:
            self.pred_label.config(fg=COLORS['accent'])
        else:
            self.pred_label.config(fg=COLORS['warning'])

        # Update depth punch status label
        fs = self.feature_stats
        if self.punch_label is not None:
            punch_result = self._latest_punch_result
            if punch_result is not None:
                nd = punch_result.get('nearest_depth', 0.0)
                ps = fs.get('depth_punch_signal', 0.0)
                pa = fs.get('depth_punch_active')
                if pa:
                    self.punch_label.config(
                        text=f"Depth: {nd:.2f}m  Vel: {ps:.3f}  APPROACHING",
                        fg=COLORS['success'])
                else:
                    self.punch_label.config(
                        text=f"Depth: {nd:.2f}m  Vel: {ps:.3f}  still",
                        fg=COLORS['text_dim'])

        # Update gate / state status — big and obvious
        gate_reasons = fs.get('gate_reasons', 'none')
        state = fs.get('state', 'idle')
        is_idle = self.current_prediction == 'idle'
        if gate_reasons and gate_reasons != 'none' and is_idle:
            self.gate_label.config(text=f"[FILTERED: {gate_reasons}]", fg=COLORS['danger'])
        elif state in ('active', 'activated') and not is_idle:
            self.gate_label.config(text=f">> {self.current_prediction.upper()} <<", fg=COLORS['success'])
        else:
            self.gate_label.config(text="", fg=COLORS['text_dim'])

        if self.debug_modes['show_features']:
            fs = self.feature_stats
            punch_lines = ""
            if self.use_depth_punch:
                dps = fs.get('depth_punch_signal', 0.0)
                dpa = fs.get('depth_punch_active')
                punch_lines = (
                    f"\nDpVel: {dps:.3f}"
                    f"\nPunch: {'YES' if dpa else 'no'}"
                )
            self.stats_label.config(text=(
                f"Voxel: {fs['voxel_activity']:.4f}\n"
                f"Act%:  {fs['voxel_active_ratio']:.1%}\n"
                f"AProb: {fs['action_prob']:.2f}\n"
                f"Marg:  {fs['class_margin']:.2f}\n"
                f"Gate:  {fs['gate_reasons']}\n"
                f"State: {fs.get('state', 'idle')}\n"
                f"FG:    {fs['fg_ratio']:.1%}"
                f"{punch_lines}"
            ))
        else:
            self.stats_label.config(text="")

        self._draw_prediction_trail()

    # ------------------------------------------------------------------
    #  GUI DRAWING
    # ------------------------------------------------------------------

    def _draw_prediction_trail(self):
        self.trail_canvas.delete("all")
        trail = list(self.prediction_trail)
        if not trail:
            return
        canvas_w = self.trail_canvas.winfo_width() or 260
        dot_size = 6
        spacing = min(8, (canvas_w - dot_size) / max(len(trail), 1))
        start_x = canvas_w - dot_size
        for i, pred_idx in enumerate(reversed(trail)):
            x = start_x - i * spacing
            if x < 0:
                break
            c = CLASS_COLORS[pred_idx % len(CLASS_COLORS)]
            self.trail_canvas.create_oval(
                x, 4, x + dot_size, 4 + dot_size,
                fill=f'#{c[0]:02x}{c[1]:02x}{c[2]:02x}', outline='')

    def _get_visible_source_roi(self, src_w: int, src_h: int) -> tuple:
        """Approximate visible ROI in source coordinates after fill+center-crop."""
        if self.video_container is None:
            return 0, 0, src_w, src_h
        target_w = max(1, int(self.video_container.winfo_width()))
        target_h = max(1, int(self.video_container.winfo_height()))
        if target_w <= 1 or target_h <= 1:
            return 0, 0, src_w, src_h

        scale = max(target_w / max(src_w, 1), target_h / max(src_h, 1))
        new_w = max(1, int(round(src_w * scale)))
        new_h = max(1, int(round(src_h * scale)))

        crop_w = max(0, new_w - target_w)
        crop_h = max(0, new_h - target_h)
        crop_l = crop_w // 2
        crop_r = crop_w - crop_l
        crop_t = crop_h // 2
        crop_b = crop_h - crop_t

        vis_left = int(np.floor(crop_l / scale))
        vis_top = int(np.floor(crop_t / scale))
        vis_right = src_w - int(np.ceil(crop_r / scale))
        vis_bottom = src_h - int(np.ceil(crop_b / scale))

        vis_left = max(0, min(vis_left, src_w - 1))
        vis_top = max(0, min(vis_top, src_h - 1))
        vis_right = max(vis_left + 1, min(vis_right, src_w))
        vis_bottom = max(vis_top + 1, min(vis_bottom, src_h))
        return vis_left, vis_top, vis_right, vis_bottom

    def _draw_overlay(self, rgb: np.ndarray, depth_m: np.ndarray) -> np.ndarray:
        """Draw debug overlays on the RGB frame."""
        display = rgb.copy()
        h, w = display.shape[:2]

        # Draw pose skeleton (only the 7 joints the model uses)
        if self._latest_pose_kps is not None:
            kps = self._latest_pose_kps
            confs = self._latest_pose_confs if self._latest_pose_confs is not None else np.ones(17)

            USED_JOINTS = {0: 'N', 5: 'LS', 6: 'RS', 7: 'LE', 8: 'RE', 9: 'LW', 10: 'RW'}
            USED_CONNECTIONS = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (0, 5), (0, 6)]
            CONN_COLORS = {
                (5, 6): (0, 220, 0), (5, 7): (255, 165, 0), (7, 9): (255, 165, 0),
                (6, 8): (100, 180, 255), (8, 10): (100, 180, 255),
                (0, 5): (180, 120, 255), (0, 6): (180, 120, 255),
            }

            # Draw connections
            for (i, j) in USED_CONNECTIONS:
                if i >= len(kps) or j >= len(kps):
                    continue
                x1, y1 = int(kps[i][0]), int(kps[i][1])
                x2, y2 = int(kps[j][0]), int(kps[j][1])
                if x1 <= 0 or y1 <= 0 or x2 <= 0 or y2 <= 0:
                    continue
                if x1 >= w or y1 >= h or x2 >= w or y2 >= h:
                    continue
                color = CONN_COLORS.get((i, j), (200, 200, 200))
                thickness = 3 if min(confs[i], confs[j]) > 0.3 else 1
                cv2.line(display, (x1, y1), (x2, y2), color, thickness)

            # Draw joints with labels
            for idx, label in USED_JOINTS.items():
                if idx >= len(kps):
                    continue
                x, y = int(kps[idx][0]), int(kps[idx][1])
                if not (0 < x < w and 0 < y < h):
                    continue
                c = confs[idx] if idx < len(confs) else 0.0
                color = (0, 255, 0) if c > 0.7 else (0, 255, 255) if c > 0.3 else (0, 0, 255)
                radius = max(4, int(4 + c * 5))
                cv2.circle(display, (x, y), radius, color, -1)
                cv2.circle(display, (x, y), radius, (0, 0, 0), 1)
                cv2.putText(display, label, (x + radius + 3, y + 4),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

            # Draw YOLO bbox
            if self._latest_pose_bbox is not None:
                bbox = self._latest_pose_bbox
                if not np.allclose(bbox, 0):
                    bx1, by1, bx2, by2 = bbox.astype(int)
                    cv2.rectangle(display, (bx1, by1), (bx2, by2), (0, 255, 0), 2)

        # Class probability bars (P)
        if self.debug_modes['show_probs'] and self.class_probs is not None:
            bar_w = 100
            bar_h = 12
            sx = w - bar_w - 20
            sy = h - (len(self.labels) * (bar_h + 4)) - 20
            for i, (label, prob) in enumerate(zip(self.labels, self.class_probs)):
                y = sy + i * (bar_h + 4)
                cv2.rectangle(display, (sx, y), (sx + bar_w, y + bar_h), (50, 50, 50), -1)
                fill = int(prob * bar_w)
                cv2.rectangle(display, (sx, y), (sx + fill, y + bar_h),
                              CLASS_COLORS[i % len(CLASS_COLORS)], -1)
                cv2.putText(display, f"{label[:4]}", (sx - 45, y + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
                cv2.putText(display, f"{prob*100:.0f}%", (sx + bar_w + 5, y + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

        # Prediction box (always). Move to top-left when depth minimap is hidden.
        pred_text = f"{self.current_prediction.upper()}"
        conf_text = f"{self.current_confidence*100:.0f}%"
        pred_font = cv2.FONT_HERSHEY_SIMPLEX
        pred_scale = 0.8
        pred_thickness = 2
        conf_scale = 0.6
        conf_thickness = 1
        pred_sz, _ = cv2.getTextSize(pred_text, pred_font, pred_scale, pred_thickness)
        conf_sz, _ = cv2.getTextSize(conf_text, pred_font, conf_scale, conf_thickness)
        box_w = max(220, pred_sz[0] + 24, conf_sz[0] + 24)
        box_h = 65

        vis_l, vis_t, vis_r, vis_b = self._get_visible_source_roi(w, h)
        desired_x = (180 if self.debug_modes['show_depth'] else 10) + vis_l
        min_x = vis_l + 10
        max_x = max(min_x, vis_r - box_w - 10)
        box_x = min(max(desired_x, min_x), max_x)
        min_y = vis_t + 10
        max_y = max(min_y, vis_b - box_h - 10)
        box_y = min(max(vis_t + 15, min_y), max_y)

        cv2.rectangle(display, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 0, 0), -1)
        cv2.rectangle(display, (box_x, box_y), (box_x + box_w, box_y + box_h), (100, 100, 100), 1)
        if self.current_confidence > 0.7:
            color = (0, 255, 128)
        elif self.current_confidence > 0.4:
            color = (0, 165, 255)
        else:
            color = (0, 200, 255)
        cv2.putText(display, pred_text, (box_x + 10, box_y + 30),
                    pred_font, pred_scale, color, pred_thickness)
        cv2.putText(display, conf_text, (box_x + 10, box_y + 55),
                    pred_font, conf_scale, (200, 200, 200), conf_thickness)

        # Depth punch detector overlay — nearest depth + velocity indicator
        punch = self._latest_punch_result
        if self.use_depth_punch and punch is not None:
            ps = punch.get('punch_signal', 0.0)
            pa = punch.get('punch_active', False)
            nd = punch.get('nearest_depth', 0.0)

            # Punch signal bar (bottom-left)
            bar_x, bar_y = 10, h - 35
            bar_max_w = 180
            bar_h_px = 16
            # Background
            cv2.rectangle(display, (bar_x, bar_y), (bar_x + bar_max_w, bar_y + bar_h_px),
                          (40, 40, 40), -1)
            # Fill proportional to signal (capped at 0.15 m/frame for display)
            fill_w = min(bar_max_w, int(ps / 0.15 * bar_max_w))
            fill_color = (0, 255, 0) if pa else (0, 120, 200)
            cv2.rectangle(display, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h_px),
                          fill_color, -1)
            cv2.rectangle(display, (bar_x, bar_y), (bar_x + bar_max_w, bar_y + bar_h_px),
                          (100, 100, 100), 1)

            status = "PUNCH" if pa else "idle"
            cv2.putText(display, f"Depth {nd:.2f}m | vel {ps:.3f} {status}",
                        (bar_x, bar_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        fill_color, 1)

        return display

    def _update_video(self, frame: np.ndarray):
        self._last_video_frame = frame
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        target_w = max(1, int(self.video_container.winfo_width()))
        target_h = max(1, int(self.video_container.winfo_height()))
        if target_w <= 1 or target_h <= 1:
            return

        h, w = rgb.shape[:2]
        
        # Calculate dynamic canvas width based on active debug modes
        # Base video is 1x. If depth is on, it goes right. If voxel is on, it goes below.
        panels_w = w
        panels_h = h
        if self.debug_modes['show_depth'] and self.last_depth_m is not None:
            dh_tmp, dw_tmp = self.last_depth_m.shape[:2]
            # It will be scaled to height `h` at 1.0x, so its width at 1.0x is:
            panels_w += int(round(dw_tmp * (h / max(dh_tmp, 1))))
            
        if self.debug_modes['show_voxel'] and self.last_voxel_grid is not None:
            vsize_1x = min(h, 150)
            panels_h += vsize_1x + 4  # Side + Top + gap goes below, adding height
            panels_w = max(panels_w, vsize_1x * 2 + 4) # ensure width covers at least voxel views
            
        # Scale to fit container without hard cap — avoids black space
        scale = min(target_w / max(panels_w, 1), target_h / max(panels_h, 1))
        
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        interp = cv2.INTER_LINEAR if scale >= 1.0 else cv2.INTER_AREA
        
        # Resize main RGB
        main_resized = cv2.resize(rgb, (new_w, new_h), interpolation=interp)
        
        # Create final canvas sized exactly to content (no excess black)
        canvas_w = new_w  # start with just the video width
        canvas_h = new_h
        
        # Pre-calculate side panel sizes so we can size canvas correctly
        depth_panel = None
        if self.debug_modes['show_depth'] and self.last_depth_m is not None:
            depth_m = self.last_depth_m
            
            # Map up to 3m to 255. Use TURBO for better structure, and paint invalid (0) as black.
            d_clip = np.clip(depth_m, 0, 3.0)
            depth_norm = (d_clip / 3.0 * 255).astype(np.uint8)
            depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_TURBO)
            depth_color[depth_m == 0] = [0, 0, 0]  # Black for invalid
            
            # Make depth panel match video height with correct aspect ratio
            dh_orig, dw_orig = depth_color.shape[:2]
            d_h = new_h
            d_w = max(1, int(round(dw_orig * (d_h / max(dh_orig, 1)))))
            depth_scaled = cv2.resize(depth_color, (d_w, d_h))
            depth_panel = cv2.cvtColor(depth_scaled, cv2.COLOR_BGR2RGB)
            canvas_w += d_w
        
        voxel_panels = None
        voxel_activity = 0.0
        if self.debug_modes['show_voxel'] and self.last_voxel_grid is not None:
            vox = np.abs(self.last_voxel_grid)
            vsize = max(1, min(new_h, int(150 * scale)))
            gap = max(1, int(4 * scale))
            
            side_proj = np.fliplr(vox.sum(axis=0))
            side_proj = (side_proj / (side_proj.max() + 1e-6) * 255).astype(np.uint8)
            side_color = cv2.applyColorMap(side_proj, cv2.COLORMAP_HOT)
            side_img = cv2.resize(side_color, (vsize, vsize), interpolation=cv2.INTER_NEAREST)
            
            top_proj = np.flipud(vox.sum(axis=1).T)
            top_proj = (top_proj / (top_proj.max() + 1e-6) * 255).astype(np.uint8)
            top_color = cv2.applyColorMap(top_proj, cv2.COLORMAP_HOT)
            top_img = cv2.resize(top_color, (vsize, vsize), interpolation=cv2.INTER_NEAREST)
            
            voxel_panels = (side_img, top_img, vsize, gap)
            voxel_total_w = vsize * 2 + gap
            voxel_activity = float((vox > 0.01).mean()) * 100
            canvas_w = max(canvas_w, voxel_total_w)
            canvas_h += vsize + gap
        
        # Now create canvas sized exactly to content
        canvas = np.zeros((canvas_h, max(1, canvas_w), 3), dtype=np.uint8)
        
        # Place main video at top-left
        canvas[0:new_h, 0:new_w] = main_resized
        
        current_x = new_w
        
        # Draw Depth Panel beside video
        if depth_panel is not None:
            dh, dw = depth_panel.shape[:2]
            canvas[0:dh, current_x:current_x+dw] = depth_panel
            cv2.rectangle(canvas, (current_x, 0), (current_x+dw, dh), (100, 100, 100), 1)
            cv2.putText(canvas, "Depth", (current_x + 5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            current_x += dw
                
        # Draw Voxel Panel BELOW video
        if voxel_panels is not None:
            side_img, top_img, vsize, gap = voxel_panels
            vy = new_h + gap // 2
            vx = 0
            if vy + vsize <= canvas_h and vx + vsize * 2 + gap <= canvas_w:
                # Side view
                canvas[vy:vy+vsize, vx:vx+vsize] = cv2.cvtColor(side_img, cv2.COLOR_BGR2RGB)
                cv2.rectangle(canvas, (vx, vy), (vx+vsize, vy+vsize), (100, 100, 100), 1)
                cv2.putText(canvas, "Side(YZ)", (vx+3, vy+vsize-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                
                # Top view
                tx = vx + vsize + gap
                canvas[vy:vy+vsize, tx:tx+vsize] = cv2.cvtColor(top_img, cv2.COLOR_BGR2RGB)
                cv2.rectangle(canvas, (tx, vy), (tx+vsize, vy+vsize), (100, 100, 100), 1)
                cv2.putText(canvas, "Top(XZ)", (tx+3, vy+vsize-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                
                cv2.putText(canvas, f"Act: {voxel_activity:.0f}%", (tx, vy+vsize+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1)

        # Embed final canvas
        img = Image.fromarray(canvas)
        img_size = (int(img.width), int(img.height))
        if self._video_photo is None or self._video_photo_size != img_size:
            self._video_photo = ImageTk.PhotoImage(img)
            self._video_photo_size = img_size
            self.video_label.config(image=self._video_photo)
            self.video_label._photo = self._video_photo
        else:
            self._video_photo.paste(img)

    def _on_video_resize(self, _event=None):
        """Redraw immediately on resize so fill/crop updates without lag."""
        if self.video_label is None:
            return
        if self._resize_redraw_pending:
            return
        self._resize_redraw_pending = True

        def _redraw():
            self._resize_redraw_pending = False
            if self.running and self._last_video_frame is not None:
                self._update_video(self._last_video_frame)

        self.root.after_idle(_redraw)

    def _toggle_pause(self):
        self.paused = not self.paused
        if self.paused:
            self.status_label.config(text="● Paused", fg=COLORS['warning'])
        else:
            self.status_label.config(text="Running", fg=COLORS['success'])

    def _quit(self):
        self.running = False
        # Stop IMU background thread
        if hasattr(self, '_imu_stop'):
            self._imu_stop.set()
        if hasattr(self, '_feature_worker_stop'):
            self._feature_worker_stop.set()
        # Send sentinel to ALL workers
        if hasattr(self, '_feature_job_queue'):
            for _ in range(max(1, self.num_workers)):
                try:
                    self._feature_job_queue.put_nowait(None)
                except Exception:
                    pass
        # Stop inference thread
        if hasattr(self, '_inference_stop'):
            self._inference_stop.set()
        if hasattr(self, '_inference_queue'):
            try:
                self._inference_queue.put_nowait(None)
            except Exception:
                pass
        if hasattr(self, '_capture_stop'):
            self._capture_stop.set()
        if hasattr(self, 'pipeline'):
            try:
                self.pipeline.stop()
            except Exception:
                pass
        self.root.quit()
        self.root.destroy()


# ======================================================================
#  CLI
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Live voxel-only inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Voxel-only inference with auto IMU pitch detection
  python tools/inference/live_voxelflow_inference.py \\
      --checkpoint work_dirs/voxel_transformer_eye_level_v4/run_20260310_111908/best_model.pth

  # Specify camera pitch manually (e.g. -20° = tilted down)
  python tools/inference/live_voxelflow_inference.py \\
      --checkpoint work_dirs/voxel_transformer_eye_level_v4/run_20260310_111908/best_model.pth \\
      --camera-pitch -20
""")

    # Required
    parser.add_argument('--checkpoint', required=True,
                        help='Model checkpoint path (.pth)')

    # Device
    parser.add_argument('--device', default='cuda:0',
                        help='Inference device (default: cuda:0)')

    # Model / feature parameters (auto-read from checkpoint)
    parser.add_argument('--voxel-size', type=int, default=12,
                        help='Voxel grid size (auto from checkpoint, default: 12)')
    parser.add_argument('--window-size', type=int, default=12,
                        help='Context window = training clip_len (auto from checkpoint)')
    parser.add_argument('--frame-sample-rate', type=int, default=1,
                        help='Sample every Nth camera frame (1=every frame, matches 30fps training)')

    # Camera orientation — CRITICAL for matching training data
    parser.add_argument('--camera-pitch', type=float, default=0.0,
                        help='Camera pitch in degrees. Positive=tilted DOWN. '
                             'If 0 and D435i IMU is available, pitch is auto-detected. '
                             'Must match the pitch used during training data collection.')
    parser.add_argument('--camera-roll', type=float, default=0.0,
                        help='Camera roll in degrees (default: 0)')
    parser.add_argument('--no-auto-pitch', action='store_true',
                        help='Disable auto IMU pitch detection')

    # Inference tuning
    parser.add_argument('--inference-interval', type=int, default=4,
                        help='Run model every N sampled frames (default: 4)')
    parser.add_argument('--temporal-smooth-window', type=int, default=3,
                        help='Moving-average smoothing window (default: 3)')
    parser.add_argument('--min-confidence', type=float, default=0.4,
                        help='Confidence threshold for idle fallback (default: 0.4)')
    parser.add_argument('--min-action-prob', type=float, default=0.0,
                        help='Minimum non-idle probability (= 1 - idle_prob) before allowing non-idle output')
    parser.add_argument('--min-class-margin', type=float, default=0.0,
                        help='Minimum top1-top2 probability margin before allowing non-idle output')
    parser.add_argument('--min-voxel-active-ratio', type=float, default=0.0,
                        help='Minimum active-voxel ratio in the current window before allowing non-idle output')
    parser.add_argument('--use-action-state-machine', action='store_true',
                        help='Apply a causal action-event filter on live predictions')
    parser.add_argument('--state-enter-consecutive', type=int, default=2,
                        help='Consecutive sampled predictions required to enter a non-idle action')
    parser.add_argument('--state-exit-consecutive', type=int, default=2,
                        help='Consecutive sampled exit signals required to return to idle')
    parser.add_argument('--state-min-hold-steps', type=int, default=2,
                        help='Minimum sampled steps to hold an action before exit is allowed')
    parser.add_argument('--state-sustain-confidence', type=float, default=0.78,
                        help='If active-class confidence drops below this, count toward action exit')
    parser.add_argument('--state-peak-drop-threshold', type=float, default=0.02,
                        help='If active-class confidence falls this far below its event peak, count toward exit')

    # Responsiveness tuning — controls how fast predictions react to changes
    parser.add_argument('--ema-alpha', type=float, default=0.35,
                        help='EMA weight for new predictions (higher=more responsive, lower=smoother). '
                             '0.35=default, 0.6-0.8=responsive for repeated punches, 1.0=raw')
    parser.add_argument('--hysteresis-margin', type=float, default=0.12,
                        help='New class must exceed current by this margin to switch. '
                             '0.12=default, 0.02-0.05=responsive, 0.0=instant switching')
    parser.add_argument('--min-hold-frames', type=int, default=3,
                        help='Hold current prediction for at least N frames. '
                             '3=default, 1=responsive for repeated punches')

    parser.add_argument('--processing-mode', type=str, default='latest',
                        choices=['latest', 'strict'],
                        help='latest: low-latency with frame dropping; strict: preserve sampled frame continuity')
    parser.add_argument('--feature-queue-size', type=int, default=0,
                        help='Feature job queue size. <=0 uses mode default (latest=1, strict=256).')
    parser.add_argument('--result-queue-size', type=int, default=0,
                        help='Feature result queue size. <=0 uses mode default (latest=1, strict=256).')
    parser.add_argument('--sensor-queue-size', type=int, default=0,
                        help='RealSense internal frames_queue_size. <=0 uses mode default (latest=1, strict=16).')

    # Multi-worker / YOLO
    parser.add_argument('--num-workers', type=int, default=2,
                        help='Number of parallel feature extraction workers (default: 2)')
    parser.add_argument('--yolo-checkpoint', type=str, default='checkpoints/yolo26n.pt',
                        help='YOLO model for person detection (default: checkpoints/yolo26n.pt)')
    parser.add_argument('--no-yolo', action='store_true',
                        help='Disable YOLO person detection, use depth-only foreground')
    parser.add_argument('--yolo-interval', type=int, default=5,
                        help='Run YOLO every N sampled frames, reuse bbox between (default: 5)')

    # Camera settings
    parser.add_argument('--rgb-res', type=str, default='960x540',
                        help='RGB stream resolution WxH (default: 960x540)')
    parser.add_argument('--depth-res', type=str, default='848x480',
                        help='Depth stream resolution WxH (default: 848x480)')
    parser.add_argument('--downscale-width', type=int, default=None,
                        help='Internal feature width. Omit to use full source width (no downscale).')
    parser.add_argument('--no-video', action='store_true',
                        help='Disable camera feed rendering for maximum throughput (predictions still shown)')
    parser.add_argument('--optimize-gpu', action='store_true',
                        help='Enable Jetson/GPU optimizations: cudnn.benchmark, torch.compile, '
                             'and TensorRT backend if available. Adds ~30s warmup but faster inference.')

    # Segment mode: detect-then-classify (one label per punch, not per frame)
    parser.add_argument('--segment-mode', action='store_true',
                        help='Detect punch segments via voxel activity, classify complete punches. '
                             'Gives one clean label per punch instead of noisy per-frame predictions.')
    parser.add_argument('--segment-activity-start', type=float, default=0.002,
                        help='Voxel activity threshold to START buffering a punch segment')
    parser.add_argument('--segment-activity-end', type=float, default=0.001,
                        help='Voxel activity threshold below which punch is considered over')
    parser.add_argument('--segment-cooldown', type=int, default=6,
                        help='Frames of low activity before confirming punch end')
    parser.add_argument('--segment-display-hold', type=float, default=2.5,
                        help='Seconds to display the punch classification result')

    # Fusion mode (auto-detected from checkpoint, but pose weights can be overridden)
    parser.add_argument('--fusion-pose-weights', type=str, default='yolo11m-pose.pt',
                        help='YOLO pose model weights for fusion mode (default: yolo11m-pose.pt). '
                             'Only used when checkpoint contains a fusion_voxel_pose_transformer model.')

    # Depth punch detection (nearest foreground approach velocity)
    parser.add_argument('--use-depth-punch', action='store_true',
                        help='Enable depth-based punch detection: gate predictions by foreground approach velocity')
    parser.add_argument('--punch-near-percentile', type=float, default=5.0,
                        help='Percentile of foreground depth to track as "nearest surface" (default: 5.0)')
    parser.add_argument('--punch-velocity-threshold', type=float, default=0.01,
                        help='Depth velocity (m) to consider a punch approaching. Very low = permissive (default: 0.01)')
    parser.add_argument('--punch-history-len', type=int, default=4,
                        help='Frames of depth history for velocity calculation (default: 4)')

    args = parser.parse_args()

    root = tk.Tk()
    app = LiveVoxelGUI(
        root,
        checkpoint_path=args.checkpoint,
        device=args.device,
        voxel_size=args.voxel_size,
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
        camera_roll=args.camera_roll,
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
        use_depth_punch=args.use_depth_punch,
        punch_near_percentile=args.punch_near_percentile,
        punch_velocity_threshold=args.punch_velocity_threshold,
        punch_history_len=args.punch_history_len,
        segment_mode=args.segment_mode,
        segment_activity_start=args.segment_activity_start,
        segment_activity_end=args.segment_activity_end,
        segment_cooldown=args.segment_cooldown,
        segment_display_hold=args.segment_display_hold,
        fusion_pose_weights=args.fusion_pose_weights,
        optimize_gpu=args.optimize_gpu,
        ema_alpha=args.ema_alpha,
        hysteresis_margin=args.hysteresis_margin,
        min_hold_frames=args.min_hold_frames,
    )
    root.mainloop()


if __name__ == '__main__':
    main()
