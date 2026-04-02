"""Headless inference engine for BoxBunny action recognition.

Extracts the core inference pipeline from ``live_voxelflow_inference.py``
into a clean, reusable wrapper that ``cv_node.py`` can import directly.
No GUI / Tkinter code — only model loading, feature extraction, inference,
and post-processing.
"""
from __future__ import annotations

import logging
import os
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("boxbunny.inference_runtime")

try:
    import torch
except ImportError:
    raise ImportError("PyTorch is required for inference_runtime")

# Ensure action_prediction/ is on sys.path so that `from lib.xxx` imports
# inside fusion_model.py / voxel_model.py resolve correctly.  These files
# use `from lib.voxel_model import ...` which requires the action_prediction
# directory (the parent of lib/) to be importable.
_LIB_DIR = Path(__file__).resolve().parent          # action_prediction/lib/
_AP_DIR = _LIB_DIR.parent                           # action_prediction/
if str(_AP_DIR) not in sys.path:
    sys.path.insert(0, str(_AP_DIR))

# Now use absolute imports (matching how live_voxelflow_inference.py does it)
from lib.voxel_features import BackgroundModel, VoxelFeatureConfig, VoxelOccupancyExtractor
from lib.fusion_model import (
    FusionVoxelPoseTransformerModel,
    POSE_FEATURE_DIM,
    extract_pose_features,
)

_POSE_AVAILABLE = False
try:
    from lib.pose import YOLOPoseEstimator
    _POSE_AVAILABLE = True
except ImportError:
    YOLOPoseEstimator = None

# Optional accelerated runtimes
_ORT_AVAILABLE = False
try:
    import onnxruntime as ort
    if any("CUDA" in p or "Tensorrt" in p for p in ort.get_available_providers()):
        _ORT_AVAILABLE = True
except ImportError:
    ort = None

_TRT_AVAILABLE = False
try:
    import tensorrt as trt
    _TRT_AVAILABLE = True
except ImportError:
    trt = None


# ── Default class labels ────────────────────────────────────────────────────
DEFAULT_LABELS = [
    "jab", "cross", "left_hook", "right_hook",
    "left_uppercut", "right_uppercut", "block", "idle",
]


# ── Result dataclass ────────────────────────────────────────────────────────

@dataclass
class InferenceResult:
    """Result from a single frame inference."""

    action: str = "idle"
    confidence: float = 0.0
    raw_action: str = "idle"
    bbox: Optional[Dict] = None
    keypoints: Optional[np.ndarray] = None
    keypoint_confidences: Optional[np.ndarray] = None
    movement_delta: float = 0.0
    consecutive_frames: int = 0
    smooth_probs: Optional[np.ndarray] = None
    fps: float = 0.0


# ── Rolling feature buffer ──────────────────────────────────────────────────

class RollingFeatureBuffer:
    """Rolling temporal buffer for voxel (+ optional pose) features."""

    def __init__(
        self,
        window_size: int = 12,
        voxel_size: int = 12,
        voxel_normalization: str = "clip_p90",
        in_channels: int = 1,
        voxel_grid_size: Tuple[int, int, int] = (20, 20, 20),
        fusion_mode: bool = False,
        pose_dim: int = 0,
    ) -> None:
        self.window_size = window_size
        self.voxel_size = voxel_size
        self.voxel_normalization = str(voxel_normalization)
        self.in_channels = in_channels
        self.voxel_grid_size = voxel_grid_size
        self.fusion_mode = fusion_mode
        self.pose_dim = pose_dim

        self.voxel_buffer: deque = deque(maxlen=window_size)
        self.fg_ratio_buffer: deque = deque(maxlen=window_size)
        self.pose_buffer: Optional[deque] = (
            deque(maxlen=window_size) if fusion_mode else None
        )

    def add_frame(
        self,
        voxel_features: np.ndarray,
        fg_ratio: float,
        pose_features: Optional[np.ndarray] = None,
    ) -> None:
        self.voxel_buffer.append(
            np.asarray(voxel_features, dtype=np.float32).reshape(-1)
        )
        self.fg_ratio_buffer.append(fg_ratio)
        if self.fusion_mode and self.pose_buffer is not None:
            if pose_features is not None:
                self.pose_buffer.append(
                    np.asarray(pose_features, dtype=np.float32).reshape(-1)
                )
            else:
                self.pose_buffer.append(np.zeros(self.pose_dim, dtype=np.float32))

    def get_features(self) -> Optional[Dict]:
        if len(self.voxel_buffer) < self.window_size:
            return None

        voxel = np.stack(list(self.voxel_buffer), axis=0)
        fg_ratio = np.array(list(self.fg_ratio_buffer), dtype=np.float32)

        voxel_f32 = voxel.astype(np.float32, copy=True)
        if self.voxel_normalization == "clip_p90":
            frame_energy = np.abs(voxel_f32).sum(axis=1)
            if frame_energy.size > 0:
                scale = float(np.percentile(frame_energy, 90))
                if np.isfinite(scale) and scale > 1e-6:
                    voxel_f32 = voxel_f32 / scale
        elif self.voxel_normalization == "frame_l1":
            frame_energy = np.abs(voxel_f32).sum(axis=1, keepdims=True)
            voxel_f32 = voxel_f32 / np.maximum(frame_energy, 1e-6)
        elif self.voxel_normalization == "channel_p90":
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

        if self.fusion_mode and self.pose_buffer is not None:
            pose = np.stack(list(self.pose_buffer), axis=0)
            combined = np.concatenate([voxel_f32, pose], axis=1)
        else:
            combined = voxel_f32

        return {
            "features": combined,
            "voxel": voxel_f32,
            "fg_ratio": fg_ratio,
        }

    @property
    def is_ready(self) -> bool:
        return len(self.voxel_buffer) >= self.window_size


# ── Prediction selection helper ─────────────────────────────────────────────

def _select_prediction(
    probs: np.ndarray,
    labels: List[str],
    min_confidence: float = 0.0,
    min_action_prob: float = 0.0,
    min_class_margin: float = 0.0,
    voxel_active_ratio: float = 1.0,
    min_voxel_active_ratio: float = 0.0,
) -> Dict:
    """Select the best prediction from smoothed probabilities."""
    idle_idx = labels.index("idle") if "idle" in labels else None
    sorted_indices = np.argsort(probs)[::-1]
    pred_idx = int(sorted_indices[0])
    conf = float(probs[pred_idx])

    gate_reasons: List[str] = []
    gated = False

    if conf < min_confidence:
        gate_reasons.append("low_confidence")
        gated = True
    if min_action_prob > 0 and idle_idx is not None:
        action_prob = 1.0 - float(probs[idle_idx])
        if action_prob < min_action_prob:
            gate_reasons.append("low_action_prob")
            gated = True
    else:
        action_prob = 1.0
    if min_class_margin > 0 and len(sorted_indices) >= 2:
        margin = float(probs[sorted_indices[0]] - probs[sorted_indices[1]])
        if margin < min_class_margin:
            gate_reasons.append("low_margin")
            gated = True
    else:
        margin = 1.0
    if min_voxel_active_ratio > 0 and voxel_active_ratio < min_voxel_active_ratio:
        gate_reasons.append("low_voxel_activity")
        gated = True

    if gated and idle_idx is not None:
        pred_idx = idle_idx
        conf = float(probs[idle_idx])

    return {
        "pred_idx": pred_idx,
        "confidence": conf,
        "idle_idx": idle_idx,
        "gated": gated,
        "gate_reasons": gate_reasons,
        "action_prob": action_prob,
        "class_margin": margin,
    }


# ── Label loader ────────────────────────────────────────────────────────────

def _load_label_names(
    checkpoint: Dict, checkpoint_path: str, num_classes: int,
) -> List[str]:
    """Extract label names from checkpoint metadata."""
    labels = checkpoint.get("label_names") or checkpoint.get("labels")
    if labels and len(labels) == num_classes:
        return list(labels)

    config = checkpoint.get("config", {})
    if isinstance(config, dict):
        labels = config.get("label_names") or config.get("labels")
        if labels and len(labels) == num_classes:
            return list(labels)

    # Try loading from label file next to checkpoint
    label_path = Path(checkpoint_path).with_suffix(".labels")
    if label_path.exists():
        with open(label_path) as f:
            labels = [line.strip() for line in f if line.strip()]
        if len(labels) == num_classes:
            return labels

    return list(DEFAULT_LABELS[:num_classes])


# ── Checkpoint feature layout resolver ──────────────────────────────────────

def _resolve_checkpoint_feature_layout(
    checkpoint: Dict, default_voxel_size: int = 12,
) -> Dict:
    """Extract voxel/feature config from checkpoint metadata."""
    config = checkpoint.get("config", {})
    if not isinstance(config, dict):
        config = {}
    dataset_config = checkpoint.get("dataset_config", config.get("dataset_config", {}))
    if not isinstance(dataset_config, dict):
        dataset_config = {}

    in_channels = int(checkpoint.get("in_channels", config.get("in_channels", 1)))
    voxel_norm = str(
        checkpoint.get("voxel_normalization", config.get("voxel_normalization", "clip_p90"))
    )
    voxel_size_raw = checkpoint.get(
        "voxel_grid_size", config.get("voxel_grid_size", None)
    )
    if voxel_size_raw and hasattr(voxel_size_raw, "__len__") and len(voxel_size_raw) == 3:
        voxel_size = tuple(int(v) for v in voxel_size_raw)
    else:
        n = int(voxel_size_raw) if voxel_size_raw else default_voxel_size
        voxel_size = (n, n, n)

    return {
        "config": config,
        "dataset_config": dataset_config,
        "in_channels": in_channels,
        "voxel_normalization": voxel_norm,
        "voxel_size": voxel_size,
        "velocity_magnitude_channel": dataset_config.get(
            "velocity_magnitude_channel", False
        ),
    }


# ── Main InferenceEngine ────────────────────────────────────────────────────

class InferenceEngine:
    """Headless action recognition inference engine.

    Wraps the full pipeline: RealSense capture → voxel extraction →
    pose estimation → model inference → post-processing.

    Usage::

        engine = InferenceEngine(checkpoint_path="model/best_model.pth")
        engine.initialize()  # loads model, builds background

        while True:
            rgb, depth = get_camera_frames()
            result = engine.process_frame(rgb, depth)
            if result:
                print(result.action, result.confidence)
    """

    def __init__(
        self,
        checkpoint_path: str = "",
        yolo_model_path: str = "",
        device: str = "cuda:0",
        window_size: int = 12,
        # Post-processing
        ema_alpha: float = 0.65,
        hysteresis_margin: float = 0.04,
        min_hold_frames: int = 1,
        min_confidence: float = 0.8,
        min_action_prob: float = 0.0,
        min_class_margin: float = 0.0,
        min_voxel_active_ratio: float = 0.0,
        # Camera
        camera_pitch: float = 5.0,
        camera_roll: float = 0.0,
        downscale_width: int = 384,
        # Speed
        inference_interval: int = 1,
        yolo_interval: int = 1,
        optimize_gpu: bool = True,
        block_consecutive_needed: int = 4,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.yolo_model_path = yolo_model_path
        self.device = device
        self.window_size = window_size

        # Post-processing params
        self._ema_alpha = max(0.0, min(1.0, float(ema_alpha)))
        self._hysteresis_margin = max(0.0, float(hysteresis_margin))
        self._min_hold_frames = max(0, int(min_hold_frames))
        self.min_confidence = min_confidence
        self.min_action_prob = min_action_prob
        self.min_class_margin = min_class_margin
        self.min_voxel_active_ratio = min_voxel_active_ratio

        # Camera params
        self.camera_pitch = camera_pitch
        self.camera_roll = camera_roll
        self.downscale_width = downscale_width

        # Speed params
        self.inference_interval = max(1, inference_interval)
        self.yolo_interval = max(1, yolo_interval)
        self.optimize_gpu = optimize_gpu
        self._block_consecutive_needed = block_consecutive_needed

        # Model state (set during initialize)
        self.model = None
        self.labels: List[str] = list(DEFAULT_LABELS)
        self.fusion_mode = False
        self.in_channels = 1
        self.voxel_grid_size = (12, 12, 12)
        self.voxel_normalization = "clip_p90"
        self.pose_dim = POSE_FEATURE_DIM
        self.pose_embed_dim = 64
        self.dataset_config: Dict = {}

        # Runtime state
        self._initialized = False
        self._bg_model: Optional[BackgroundModel] = None
        self._voxel_extractor: Optional[VoxelOccupancyExtractor] = None
        self._feature_buffer: Optional[RollingFeatureBuffer] = None
        self._pose_estimator: Optional[object] = None
        self._pose_lock = threading.Lock()
        self._feature_config: Optional[VoxelFeatureConfig] = None

        # Smoothing state
        self._ema_probs: Optional[np.ndarray] = None
        self._recent_probs: deque = deque(maxlen=5)
        self._held_pred_idx: Optional[int] = None
        self._held_pred_frames: int = 0
        self._block_consec_count: int = 0

        # Consecutive frame tracking
        self._prev_action: str = "idle"
        self._consecutive_count: int = 0

        # Pose tracking for movement delta
        self._prev_keypoints: Optional[np.ndarray] = None
        self._cached_pose: Optional[Dict] = None
        self._prev_pose_static: Optional[np.ndarray] = None

        # Frame counting
        self._frame_count: int = 0
        self._inference_times: deque = deque(maxlen=30)

        # Accelerated runtime handles
        self._ort_session = None
        self._ort_input_names = None
        self._trt_context = None
        self._trt_engine = None
        self._trt_bindings = None

        # Scale factor (computed during first frame)
        self._scale_factor: Optional[float] = None

    # ── Initialization ──────────────────────────────────────────────────

    def initialize(self) -> None:
        """Load model checkpoint, initialize feature extractors."""
        self._load_model()
        self._initialized = True
        logger.info(
            "InferenceEngine initialized: labels=%s, fusion=%s, device=%s",
            self.labels, self.fusion_mode, self.device,
        )

    def _load_model(self) -> None:
        """Load the PyTorch checkpoint and instantiate the model."""
        logger.info("Loading model from %s ...", self.checkpoint_path)

        checkpoint = torch.load(
            self.checkpoint_path, map_location=self.device, weights_only=False
        )

        layout = _resolve_checkpoint_feature_layout(checkpoint, default_voxel_size=12)
        config = layout["config"]
        self.dataset_config = dict(layout["dataset_config"])
        self.in_channels = int(layout["in_channels"])
        self.voxel_normalization = str(layout["voxel_normalization"])
        self.voxel_grid_size = tuple(int(v) for v in layout["voxel_size"])

        model_arch = str(
            checkpoint.get("model_arch", config.get("model_arch", "fusion_voxel_pose_transformer"))
        )
        feature_mode = str(
            checkpoint.get("feature_mode", config.get("feature_mode", "fusion_voxel_pose"))
        )

        self.fusion_mode = (
            feature_mode == "fusion_voxel_pose"
            or model_arch == "fusion_voxel_pose_transformer"
        )
        if self.fusion_mode:
            self.pose_dim = int(
                checkpoint.get("pose_dim", config.get("pose_dim", POSE_FEATURE_DIM))
            )
            self.pose_embed_dim = int(
                checkpoint.get("pose_embed_dim", config.get("pose_embed_dim", 64))
            )

        state_dict = checkpoint.get("model_state_dict", {})
        num_classes = int(config.get("num_classes", len(self.labels)))
        classifier_keys = sorted(
            k for k in state_dict if k.startswith("classifier.") and k.endswith(".weight")
        )
        if classifier_keys:
            num_classes = int(state_dict[classifier_keys[-1]].shape[0])

        self.model = FusionVoxelPoseTransformerModel(
            voxel_size=self.voxel_grid_size,
            num_classes=num_classes,
            d_model=int(config.get("transformer_d_model", 192)),
            num_heads=int(config.get("transformer_heads", 8)),
            num_layers=int(config.get("transformer_layers", 4)),
            dim_feedforward=int(config.get("transformer_ffn_dim", 576)),
            dropout=0.0,
            max_len=int(config.get("transformer_max_len", 256)),
            in_channels=self.in_channels,
            pose_dim=self.pose_dim,
            pose_embed_dim=self.pose_embed_dim,
            pose_dropout=0.0,
            dual_voxel_stem=bool(checkpoint.get("dual_voxel_stem", False)),
        )
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        if self.device.startswith("cuda"):
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.labels = _load_label_names(checkpoint, self.checkpoint_path, num_classes)
        logger.info(
            "Model loaded: %d classes, arch=%s, voxel=%s, channels=%d",
            num_classes, model_arch, self.voxel_grid_size, self.in_channels,
        )

    def _init_feature_pipeline(
        self, frame_h: int, frame_w: int,
    ) -> None:
        """Initialize feature extractors using frame dimensions."""
        self._scale_factor = self.downscale_width / frame_w

        # Use default RealSense D435i intrinsics scaled to our resolution
        fx = 424.0 * self._scale_factor
        fy = 424.0 * self._scale_factor
        cx = (frame_w / 2.0) * self._scale_factor
        cy = (frame_h / 2.0) * self._scale_factor

        self._feature_config = VoxelFeatureConfig(
            fx=fx, fy=fy, cx=cx, cy=cy,
            depth_scale=0.001,
            camera_pitch_deg=self.camera_pitch,
            camera_roll_deg=self.camera_roll,
            voxel_grid_size=self.voxel_grid_size,
            voxel_person_centric=True,
            voxel_depth_weighted=bool(self.dataset_config.get("voxel_depth_weighted", True)),
            directional_gradients=bool(self.dataset_config.get("directional_gradients", False)),
            velocity_magnitude_channel=bool(
                self.dataset_config.get("velocity_magnitude_channel", False)
            ),
            multi_scale_delta_frames=tuple(
                int(v) for v in (self.dataset_config.get("multi_scale_delta_frames") or ())
            ),
            voxel_delta_frames=int(self.dataset_config.get("voxel_delta_frames", 3)),
            include_raw_occupancy=bool(self.dataset_config.get("include_raw_occupancy", False)),
        )

        self._bg_model = BackgroundModel(self._feature_config)
        self._voxel_extractor = VoxelOccupancyExtractor(self._feature_config)
        self._voxel_extractor.set_tilt_rotation(self.camera_pitch, self.camera_roll)

        self._feature_buffer = RollingFeatureBuffer(
            window_size=self.window_size,
            voxel_size=self.voxel_grid_size[0],
            voxel_normalization=self.voxel_normalization,
            in_channels=self.in_channels,
            voxel_grid_size=self.voxel_grid_size,
            fusion_mode=self.fusion_mode,
            pose_dim=self.pose_dim,
        )

        # Initialize pose estimator if in fusion mode
        if self.fusion_mode and _POSE_AVAILABLE and self.yolo_model_path:
            try:
                self._pose_estimator = YOLOPoseEstimator(
                    model_path=self.yolo_model_path,
                    device=self.device,
                )
                logger.info("YOLO pose estimator loaded: %s", self.yolo_model_path)
            except Exception as e:
                logger.warning("Failed to load YOLO pose estimator: %s", e)

        logger.info(
            "Feature pipeline initialized: scale=%.2f, bg_model ready=%s",
            self._scale_factor, self._bg_model.is_initialized(),
        )

    # ── Main process_frame entry point ──────────────────────────────────

    def process_frame(
        self, rgb: np.ndarray, depth: np.ndarray,
    ) -> Optional[InferenceResult]:
        """Process a single RGB+depth frame pair.

        Args:
            rgb: BGR8 image from RealSense (H, W, 3)
            depth: 16UC1 depth image in millimeters (H, W)

        Returns:
            InferenceResult or None if not ready yet (building background).
        """
        if not self._initialized:
            return None

        # Lazy-initialize feature pipeline on first frame
        if self._scale_factor is None:
            self._init_feature_pipeline(rgb.shape[0], rgb.shape[1])

        self._frame_count += 1

        # Downscale
        sf = self._scale_factor
        if sf != 1.0:
            target_w = self.downscale_width
            target_h = int(rgb.shape[0] * sf)
            rgb_ds = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            depth_ds = cv2.resize(depth, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        else:
            rgb_ds = rgb
            depth_ds = depth

        depth_m = depth_ds.astype(np.float32) * 0.001

        # Build background model from initial frames
        if not self._bg_model.is_initialized():
            self._bg_model.update(depth_ds)
            return None

        # Extract voxel features
        fg_mask = self._bg_model.get_foreground_mask(depth_ds)
        fg_ratio = float(np.mean(fg_mask)) if fg_mask is not None else 0.0

        try:
            voxel_flat = self._voxel_extractor.extract(
                depth_m, fg_mask, return_debug=False,
            ).astype(np.float32, copy=False)
        except Exception as e:
            logger.debug("Voxel extraction error: %s", e)
            return None

        # Extract pose features (if fusion mode)
        pose_features = None
        pose_kps = None
        pose_confs = None
        pose_bbox = None

        if self.fusion_mode and self._pose_estimator is not None:
            from lib.fusion_model import (
                extract_pose_features_static,
                compute_pose_velocity,
                POSE_STATIC_DIM,
                POSE_VELOCITY_DIM,
            )

            run_pose = (self._frame_count % max(1, self.yolo_interval) == 0)
            if run_pose:
                try:
                    with self._pose_lock:
                        kps, confs, bbox = self._pose_estimator.predict_with_bbox(rgb_ds)
                    cur_static = extract_pose_features_static(kps, confs)
                    vel = (
                        compute_pose_velocity(cur_static, self._prev_pose_static)
                        if self._prev_pose_static is not None
                        else np.zeros(POSE_VELOCITY_DIM, dtype=np.float32)
                    )
                    pose_features = np.concatenate([cur_static, vel])
                    self._prev_pose_static = cur_static.copy()
                    pose_kps = kps
                    pose_confs = confs
                    pose_bbox = bbox
                    self._cached_pose = {
                        "features": pose_features,
                        "kps": pose_kps,
                        "confs": pose_confs,
                        "bbox": pose_bbox,
                    }
                except Exception:
                    if self._cached_pose:
                        pose_features = self._cached_pose["features"]
                        pose_kps = self._cached_pose["kps"]
                        pose_confs = self._cached_pose["confs"]
                        pose_bbox = self._cached_pose["bbox"]
            else:
                if self._cached_pose:
                    pose_features = self._cached_pose["features"]
                    pose_kps = self._cached_pose["kps"]
                    pose_confs = self._cached_pose["confs"]
                    pose_bbox = self._cached_pose["bbox"]

        # Add to feature buffer
        self._feature_buffer.add_frame(voxel_flat, fg_ratio, pose_features=pose_features)

        # Only run inference at configured interval
        if self._frame_count % self.inference_interval != 0:
            return None

        if not self._feature_buffer.is_ready:
            return None

        # Run inference
        features = self._feature_buffer.get_features()
        if features is None:
            return None

        t0 = time.time()
        result = self._run_inference(features)
        inference_ms = (time.time() - t0) * 1000
        self._inference_times.append(inference_ms)

        if result is None:
            return None

        # Compute movement delta from keypoints
        movement_delta = 0.0
        if pose_kps is not None and self._prev_keypoints is not None:
            movement_delta = self._compute_movement_delta(
                self._prev_keypoints, pose_kps,
            )
        if pose_kps is not None:
            self._prev_keypoints = pose_kps.copy() if hasattr(pose_kps, "copy") else pose_kps

        # Track consecutive frames
        action = result["prediction"]
        if action == self._prev_action:
            self._consecutive_count += 1
        else:
            self._prev_action = action
            self._consecutive_count = 1

        # Build bbox dict from pose bbox
        bbox_dict = None
        if pose_bbox is not None:
            try:
                x1, y1, x2, y2 = pose_bbox[:4]
                w = x2 - x1
                h = y2 - y1
                # Scale back to original resolution
                inv_sf = 1.0 / sf if sf and sf > 0 else 1.0
                bbox_dict = {
                    "cx": float((x1 + x2) / 2) * inv_sf,
                    "cy": float((y1 + y2) / 2) * inv_sf,
                    "top_y": float(y1) * inv_sf,
                    "width": float(w) * inv_sf,
                    "height": float(h) * inv_sf,
                    "depth": float(
                        np.median(depth_m[
                            max(0, int(y1)):min(depth_m.shape[0], int(y2)),
                            max(0, int(x1)):min(depth_m.shape[1], int(x2)),
                        ]) if depth_m is not None else 0.0
                    ),
                }
            except Exception:
                pass

        # Compute FPS
        fps = 0.0
        if len(self._inference_times) >= 2:
            avg_ms = sum(self._inference_times) / len(self._inference_times)
            fps = 1000.0 / max(avg_ms, 1.0)

        return InferenceResult(
            action=action,
            confidence=result["confidence"],
            raw_action=result["prediction"],
            bbox=bbox_dict,
            keypoints=pose_kps,
            keypoint_confidences=pose_confs,
            movement_delta=movement_delta,
            consecutive_frames=self._consecutive_count,
            smooth_probs=result.get("smooth_probs"),
            fps=fps,
        )

    # ── Core inference ──────────────────────────────────────────────────

    def _run_inference(self, features: Dict) -> Optional[Dict]:
        """Run model forward pass with post-processing."""
        feat_np = features["features"]

        # PyTorch inference (primary path)
        with torch.no_grad():
            combined = torch.from_numpy(feat_np).float().unsqueeze(0)
            combined = combined.to(self.device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=self.device.startswith("cuda")):
                output = self.model(combined, padding_mask=None)

            logits = output["logits"]
            raw_probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        # EMA smoothing
        self._recent_probs.append(raw_probs)
        if self._ema_probs is None:
            self._ema_probs = raw_probs.copy()
        else:
            self._ema_probs = (
                self._ema_alpha * raw_probs + (1.0 - self._ema_alpha) * self._ema_probs
            )

        windowed_mean = np.mean(np.stack(self._recent_probs, axis=0), axis=0)
        smooth_probs = 0.6 * self._ema_probs + 0.4 * windowed_mean
        smooth_probs = smooth_probs / max(float(smooth_probs.sum()), 1e-8)

        # Feature stats for gating
        voxel_active_ratio = float((np.abs(features["voxel"]) > 0.01).mean())

        decision = _select_prediction(
            probs=smooth_probs,
            labels=self.labels,
            min_confidence=self.min_confidence,
            min_action_prob=self.min_action_prob,
            min_class_margin=self.min_class_margin,
            voxel_active_ratio=voxel_active_ratio,
            min_voxel_active_ratio=self.min_voxel_active_ratio,
        )

        pred_idx = int(decision["pred_idx"])
        conf = float(decision["confidence"])
        idle_idx = decision["idle_idx"]

        # Hysteresis: prevent flickering
        if self._held_pred_idx is not None:
            self._held_pred_frames += 1
            if pred_idx != self._held_pred_idx:
                held_conf = float(smooth_probs[self._held_pred_idx])
                new_conf = float(smooth_probs[pred_idx])
                if (
                    self._held_pred_frames < self._min_hold_frames
                    or (new_conf - held_conf) < self._hysteresis_margin
                ):
                    pred_idx = self._held_pred_idx
                    conf = held_conf
                else:
                    self._held_pred_idx = pred_idx
                    self._held_pred_frames = 0
        else:
            self._held_pred_idx = pred_idx
            self._held_pred_frames = 0

        # Block consecutive filter
        block_idx = self.labels.index("block") if "block" in self.labels else None
        if pred_idx == block_idx:
            self._block_consec_count += 1
            if self._block_consec_count < self._block_consecutive_needed:
                if idle_idx is not None:
                    pred_idx = idle_idx
                    conf = float(smooth_probs[idle_idx])
        else:
            self._block_consec_count = 0

        prediction = self.labels[pred_idx]

        return {
            "prediction": prediction,
            "confidence": conf,
            "pred_idx": pred_idx,
            "smooth_probs": smooth_probs,
        }

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _compute_movement_delta(
        prev_kps: np.ndarray, curr_kps: np.ndarray, conf_threshold: float = 0.3,
    ) -> float:
        """Compute max keypoint displacement between frames."""
        if prev_kps is None or curr_kps is None:
            return 0.0
        max_dist = 0.0
        n = min(len(prev_kps), len(curr_kps))
        for i in range(n):
            p, c = prev_kps[i], curr_kps[i]
            if len(p) >= 3 and len(c) >= 3:
                if p[2] < conf_threshold or c[2] < conf_threshold:
                    continue
            dist = float(np.sqrt((c[0] - p[0]) ** 2 + (c[1] - p[1]) ** 2))
            max_dist = max(max_dist, dist)
        return max_dist

    @property
    def is_initialized(self) -> bool:
        return self._initialized
