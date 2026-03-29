"""
Production inference engine for boxing action recognition.

Provides a self-contained ``InferenceEngine`` class that encapsulates the
full prediction pipeline: model loading, feature extraction, temporal
buffering, model forward pass (TensorRT > ONNX > PyTorch), and
post-processing (smoothing, hysteresis, state machine, block filter).

Usage::

    engine = InferenceEngine(checkpoint_path='model/best_model.pth')
    engine.initialize()
    result = engine.process_frame(rgb, depth)
    print(result.action, result.confidence)
"""

from __future__ import annotations

import logging
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Ensure project root is importable
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Local lib imports
from action_prediction.lib.checkpoint_utils import (
    DEFAULT_LABELS,
    _coerce_voxel_size,
    _resolve_checkpoint_feature_layout,
    _resolve_runtime_device,
    load_checkpoint,
    load_labels,
)
from action_prediction.lib.feature_buffer import RollingFeatureBuffer, debug_voxel_grid
from action_prediction.lib.prediction_utils import (
    BlockConsecutiveFilter,
    PredictionHysteresis,
    ProbabilitySmoothing,
    select_prediction,
)
from action_prediction.lib.state_machine import (
    CausalActionStateMachine,
    DepthPunchDetector,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional accelerator availability
# ---------------------------------------------------------------------------

_ORT_AVAILABLE = False
try:
    import onnxruntime as ort

    if any('CUDA' in p or 'Tensorrt' in p for p in ort.get_available_providers()):
        _ORT_AVAILABLE = True
except ImportError:
    ort = None  # type: ignore[assignment]

_TRT_AVAILABLE = False
try:
    import tensorrt as trt  # type: ignore[import-untyped]

    _TRT_AVAILABLE = True
except ImportError:
    trt = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: Dict[str, Any] = {
    'ema_alpha': 0.35,
    'hysteresis_margin': 0.12,
    'min_hold_frames': 3,
    'min_confidence': 0.4,
    'min_action_prob': 0.0,
    'min_class_margin': 0.0,
    'min_voxel_active_ratio': 0.0,
    'state_enter_consecutive': 2,
    'state_exit_consecutive': 2,
    'state_min_hold_steps': 2,
    'state_sustain_confidence': 0.78,
    'state_peak_drop_threshold': 0.02,
    'block_consecutive_needed': 4,
    'temporal_smooth_window': 5,
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class PredictionResult:
    """Outcome of a single frame processed by :class:`InferenceEngine`.

    Attributes:
        action: Predicted action label (after all filtering).
        confidence: Confidence score for *action*.
        probabilities: Mapping of every label to its smoothed probability.
        is_idle: ``True`` when the predicted action is ``'idle'``.
        raw_action: Pre-filtered action label (before state machine).
        state_machine_state: Current state of the causal state machine.
        voxel_active_ratio: Fraction of voxel cells with significant
            activity.
        frame_count: Total number of frames processed so far.
    """

    action: str
    confidence: float
    probabilities: Dict[str, float]
    is_idle: bool
    raw_action: str
    state_machine_state: str
    voxel_active_ratio: float
    frame_count: int


# ---------------------------------------------------------------------------
# Inference engine
# ---------------------------------------------------------------------------


class InferenceEngine:
    """Self-contained production inference engine for boxing action recognition.

    Wraps the full prediction pipeline: checkpoint loading, voxel feature
    extraction, optional pose estimation, rolling feature buffering, model
    inference (TensorRT > ONNX > PyTorch fallback), temporal smoothing,
    hysteresis, consecutive-block filtering, and causal state machine.

    Args:
        checkpoint_path: Path to the ``.pth`` model checkpoint.
        device: PyTorch device string (e.g. ``'cuda:0'``).
        window_size: Temporal window length (number of frames per
            inference).
        config: Optional dict of tuneable parameters (merged with
            :data:`DEFAULT_CONFIG`).
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = 'cuda:0',
        window_size: int = 12,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.checkpoint_path: str = checkpoint_path
        self._requested_device: str = device
        self.device: str = _resolve_runtime_device(device)
        self.window_size: int = window_size

        # Merge user config with defaults
        self.cfg: Dict[str, Any] = dict(DEFAULT_CONFIG)
        if config is not None:
            self.cfg.update(config)

        # State flags
        self._initialized: bool = False
        self.frame_count: int = 0

        # Components (populated by initialize())
        self.labels: List[str] = list(DEFAULT_LABELS)
        self.model: Any = None
        self.feature_buffer: Optional[RollingFeatureBuffer] = None
        self.smoother: Optional[ProbabilitySmoothing] = None
        self.hysteresis: Optional[PredictionHysteresis] = None
        self.block_filter: Optional[BlockConsecutiveFilter] = None
        self.state_machine: Optional[CausalActionStateMachine] = None

        # Voxel extraction components
        self.voxel_extractor: Any = None
        self.bg_model: Any = None
        self.pose_estimator: Any = None

        # Model metadata
        self.fusion_mode: bool = False
        self.model_arch: str = ''
        self.feature_mode: str = ''
        self.in_channels: int = 1
        self.voxel_normalization: str = 'clip_p90'
        self.voxel_grid_size: Tuple[int, int, int] = (12, 12, 12)
        self.pose_dim: int = 0
        self.pose_embed_dim: int = 64
        self.dataset_config: Dict[str, Any] = {}
        self.num_classes: int = 8

        # Accelerated inference sessions
        self._ort_session: Any = None
        self._ort_input_names: Optional[List[str]] = None
        self._trt_context: Any = None
        self._trt_engine: Any = None
        self._trt_bindings: Optional[Dict[str, Any]] = None

        # Pose caching for skipped frames
        self._prev_pose_static: Optional[np.ndarray] = None
        self._cached_pose_features: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Lazy-load model, create all internal components.

        Loads the checkpoint, builds the model, sets up the feature buffer,
        smoother, hysteresis filter, state machine, and block filter.
        Attempts TensorRT > ONNX > PyTorch fallback for inference.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
            RuntimeError: If the model architecture is unsupported.
        """
        import torch

        from action_prediction.lib.fusion_model import (
            POSE_FEATURE_DIM,
            FusionVoxelPoseTransformerModel,
        )
        from action_prediction.lib.pose import YOLOPoseEstimator
        from action_prediction.lib.voxel_features import (
            BackgroundModel,
            VoxelFeatureConfig,
            VoxelOccupancyExtractor,
        )

        logger.info("Loading model from %s ...", self.checkpoint_path)
        logger.info("  Requested device: %s", self._requested_device)
        logger.info("  Runtime device:   %s", self.device)

        checkpoint = load_checkpoint(self.checkpoint_path, device=self.device)

        # --- Extract feature layout ---
        feature_layout = _resolve_checkpoint_feature_layout(
            checkpoint, default_voxel_size=self.window_size,
        )
        config = feature_layout['config']
        self.dataset_config = dict(feature_layout['dataset_config'])
        self.dataset_config.setdefault(
            'velocity_magnitude_channel',
            feature_layout.get('velocity_magnitude_channel', False),
        )
        self.in_channels = int(feature_layout['in_channels'])
        self.voxel_normalization = str(feature_layout['voxel_normalization'])
        self.model_arch = str(
            checkpoint.get('model_arch', config.get('model_arch', 'causal_voxel_transformer'))
        )
        self.feature_mode = str(
            checkpoint.get('feature_mode', config.get('feature_mode', 'voxel_only'))
        )

        if self.model_arch not in {
            'causal_voxel_transformer', 'voxel_mlp', 'voxel_temporal_mlp',
            'fusion_voxel_pose_transformer',
        }:
            raise RuntimeError(
                f"Unsupported model_arch={self.model_arch}. "
                "Supported: causal_voxel_transformer, voxel_mlp, "
                "voxel_temporal_mlp, fusion_voxel_pose_transformer."
            )
        if self.feature_mode not in {'voxel_only', 'fusion_voxel_pose'}:
            raise RuntimeError(
                f"Unsupported feature_mode={self.feature_mode}. "
                "Supported: voxel_only, fusion_voxel_pose."
            )

        # Detect fusion mode
        self.fusion_mode = (
            self.feature_mode == 'fusion_voxel_pose'
            or self.model_arch == 'fusion_voxel_pose_transformer'
        )
        if self.fusion_mode:
            self.pose_dim = int(
                checkpoint.get('pose_dim', config.get('pose_dim', POSE_FEATURE_DIM))
            )
            self.pose_embed_dim = int(
                checkpoint.get('pose_embed_dim', config.get('pose_embed_dim', 64))
            )

        voxel_size = tuple(int(v) for v in feature_layout['voxel_size'])
        self.voxel_grid_size = voxel_size

        # --- Detect num_classes ---
        state_dict = checkpoint.get('model_state_dict', {})
        self.num_classes = int(config.get('num_classes', len(self.labels)))
        classifier_keys = sorted(
            [k for k in state_dict if k.startswith('classifier.') and k.endswith('.weight')],
        )
        if classifier_keys:
            last_cls_key = classifier_keys[-1]
            self.num_classes = int(state_dict[last_cls_key].shape[0])

        # --- Build model ---
        if self.model_arch == 'fusion_voxel_pose_transformer':
            self.model = FusionVoxelPoseTransformerModel(
                voxel_size=voxel_size,
                num_classes=self.num_classes,
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

        # GPU optimizations
        if self.device.startswith('cuda'):
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # --- Accelerated inference setup ---
        self._ort_session = None
        self._ort_input_names = None
        self._trt_context = None
        self._trt_engine = None
        self._trt_bindings = None

        clip_len = checkpoint.get('clip_len', config.get('clip_len', 12))
        self._setup_optimized_inference(clip_len)

        # --- Labels ---
        self.labels = load_labels(checkpoint, self.checkpoint_path, self.num_classes)

        # Auto-read clip_len from checkpoint
        ckpt_clip = checkpoint.get('clip_len', config.get('clip_len', None))
        if ckpt_clip and ckpt_clip != self.window_size:
            logger.warning(
                "Checkpoint clip_len=%d vs configured window_size=%d. "
                "Using checkpoint value.",
                ckpt_clip, self.window_size,
            )
            self.window_size = ckpt_clip

        # --- Feature buffer ---
        self.feature_buffer = RollingFeatureBuffer(
            window_size=self.window_size,
            voxel_size=int(voxel_size[0]),
            voxel_normalization=self.voxel_normalization,
            in_channels=self.in_channels,
            voxel_grid_size=self.voxel_grid_size,
            fusion_mode=self.fusion_mode,
            pose_dim=self.pose_dim,
        )

        # --- Post-processing pipeline ---
        self.smoother = ProbabilitySmoothing(
            alpha=float(self.cfg['ema_alpha']),
            window_size=int(self.cfg['temporal_smooth_window']),
        )
        self.hysteresis = PredictionHysteresis(
            hysteresis_margin=float(self.cfg['hysteresis_margin']),
            min_hold_frames=int(self.cfg['min_hold_frames']),
        )
        self.block_filter = BlockConsecutiveFilter(
            block_consecutive_needed=int(self.cfg['block_consecutive_needed']),
            labels=self.labels,
        )
        self.state_machine = CausalActionStateMachine(
            labels=self.labels,
            enter_consecutive=int(self.cfg['state_enter_consecutive']),
            exit_consecutive=int(self.cfg['state_exit_consecutive']),
            min_hold_steps=int(self.cfg['state_min_hold_steps']),
            sustain_confidence=float(self.cfg['state_sustain_confidence']),
            peak_drop_threshold=float(self.cfg['state_peak_drop_threshold']),
        )

        # --- Voxel extraction components (created but NOT configured with
        #     camera intrinsics yet -- caller must supply frames) ---
        # These are placeholders; actual VoxelFeatureConfig requires camera
        # intrinsics which are only known at frame-processing time or from
        # an external camera setup call.
        self._voxel_feature_config: Optional[VoxelFeatureConfig] = None
        self._BackgroundModel = BackgroundModel
        self._VoxelOccupancyExtractor = VoxelOccupancyExtractor
        self._VoxelFeatureConfig = VoxelFeatureConfig

        # YOLO pose estimator for fusion mode
        if self.fusion_mode:
            try:
                self.pose_estimator = YOLOPoseEstimator(
                    device=self.device,
                    conf=0.15,
                    imgsz=320,
                )
                logger.info("Fusion mode: YOLO pose estimator loaded")
            except Exception as e:
                logger.warning(
                    "Could not load YOLO pose estimator: %s. "
                    "Fusion model will run with zero pose features.",
                    e,
                )
                self.pose_estimator = None

        self._initialized = True
        logger.info(
            "InferenceEngine initialized: %d classes, voxel %s, arch=%s, "
            "fusion=%s, labels=%s",
            self.num_classes, voxel_size, self.model_arch,
            self.fusion_mode, self.labels,
        )

    # ------------------------------------------------------------------
    # Camera / Extraction Setup
    # ------------------------------------------------------------------

    def setup_camera(
        self,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        depth_scale: float = 0.001,
        camera_pitch_deg: float = 0.0,
        camera_roll_deg: float = 0.0,
    ) -> None:
        """Configure voxel extraction with camera intrinsics.

        Must be called before :meth:`process_frame` if voxel extraction
        is desired.  If not called, :meth:`process_frame` expects
        pre-extracted features.

        Args:
            fx: Focal length X (pixels).
            fy: Focal length Y (pixels).
            cx: Principal point X (pixels).
            cy: Principal point Y (pixels).
            depth_scale: Multiplier to convert raw depth to meters.
            camera_pitch_deg: Camera pitch from IMU (degrees).
            camera_roll_deg: Camera roll from IMU (degrees).
        """
        self._voxel_feature_config = self._VoxelFeatureConfig(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            depth_scale=depth_scale,
            camera_pitch_deg=camera_pitch_deg,
            camera_roll_deg=camera_roll_deg,
            voxel_grid_size=self.voxel_grid_size,
            voxel_person_centric=True,
            voxel_depth_weighted=bool(self.dataset_config.get('voxel_depth_weighted', True)),
            directional_gradients=bool(self.dataset_config.get('directional_gradients', False)),
            velocity_magnitude_channel=bool(
                self.dataset_config.get('velocity_magnitude_channel', False)
            ),
            multi_scale_delta_frames=tuple(
                int(v) for v in (self.dataset_config.get('multi_scale_delta_frames') or ())
            ),
            voxel_delta_frames=int(self.dataset_config.get('voxel_delta_frames', 3)),
            include_raw_occupancy=bool(self.dataset_config.get('include_raw_occupancy', False)),
        )
        self.bg_model = self._BackgroundModel(self._voxel_feature_config)
        self.voxel_extractor = self._VoxelOccupancyExtractor(self._voxel_feature_config)
        self.voxel_extractor.set_tilt_rotation(camera_pitch_deg, camera_roll_deg)
        logger.info(
            "Camera configured: fx=%.1f fy=%.1f cx=%.1f cy=%.1f pitch=%.1f roll=%.1f",
            fx, fy, cx, cy, camera_pitch_deg, camera_roll_deg,
        )

    def set_background(self, bg_depth_m: np.ndarray) -> None:
        """Set the background depth model directly.

        Args:
            bg_depth_m: Background depth image in meters, shape ``(H, W)``.
        """
        if self.bg_model is not None:
            self.bg_model.set_background(bg_depth_m.astype(np.float32))

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def process_frame(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
    ) -> PredictionResult:
        """Process one RGB + depth frame through the full inference pipeline.

        Extracts voxel features, optionally runs pose estimation, buffers
        features, runs model inference when the buffer is ready, and
        applies all post-processing.

        Args:
            rgb: RGB image, shape ``(H, W, 3)``, dtype ``uint8``.
            depth: Raw depth image, shape ``(H, W)``, dtype ``uint16``.
                Converted to meters internally using the configured
                depth scale.

        Returns:
            :class:`PredictionResult` with the predicted action and
            metadata.

        Raises:
            RuntimeError: If :meth:`initialize` has not been called.
        """
        if not self._initialized:
            raise RuntimeError("InferenceEngine.initialize() must be called first.")

        self.frame_count += 1
        depth_m = depth.astype(np.float32) * 0.001

        # --- Voxel feature extraction ---
        voxel_flat: Optional[np.ndarray] = None
        fg_ratio: float = 0.0

        if self.voxel_extractor is not None and self.bg_model is not None:
            if self.bg_model.is_initialized():
                fg_mask = self.bg_model.get_foreground_mask(depth)
                fg_ratio = float(np.mean(fg_mask)) if fg_mask is not None else 0.0
                voxel_flat = self.voxel_extractor.extract(
                    depth_m, fg_mask, return_debug=False,
                ).astype(np.float32, copy=False)
            else:
                # Background model still initializing
                self.bg_model.update(depth)

        if voxel_flat is None:
            # No voxel features available -- produce zeros
            total_dim = int(np.prod(self.voxel_grid_size)) * self.in_channels
            voxel_flat = np.zeros(total_dim, dtype=np.float32)

        # --- Pose estimation (fusion mode) ---
        pose_features: Optional[np.ndarray] = None
        if self.fusion_mode:
            pose_features = self._extract_pose(rgb)

        # --- Buffer features ---
        self.feature_buffer.add_frame(voxel_flat, fg_ratio, pose_features=pose_features)

        # --- Run inference if buffer ready ---
        if not self.feature_buffer.is_ready:
            return PredictionResult(
                action='idle',
                confidence=0.0,
                probabilities={label: 0.0 for label in self.labels},
                is_idle=True,
                raw_action='idle',
                state_machine_state='buffering',
                voxel_active_ratio=0.0,
                frame_count=self.frame_count,
            )

        features = self.feature_buffer.get_features()
        if features is None:
            return self._idle_result('buffer_empty')

        # --- Model forward pass ---
        raw_probs = self._run_inference(features)

        # --- Temporal smoothing ---
        smooth_probs = self.smoother.update(raw_probs)

        # --- Feature stats ---
        voxel_active_ratio = float((np.abs(features['voxel']) > 0.01).mean())

        # --- Non-idle gating ---
        decision = select_prediction(
            probs=smooth_probs,
            labels=self.labels,
            min_confidence=float(self.cfg['min_confidence']),
            min_action_prob=float(self.cfg['min_action_prob']),
            min_class_margin=float(self.cfg['min_class_margin']),
            voxel_active_ratio=voxel_active_ratio,
            min_voxel_active_ratio=float(self.cfg['min_voxel_active_ratio']),
        )

        pred_idx = int(decision['pred_idx'])
        conf = float(decision['confidence'])

        # --- Prediction hysteresis ---
        pred_idx, conf = self.hysteresis.update(pred_idx, smooth_probs)

        # --- Block consecutive filter ---
        pred_idx, conf = self.block_filter.update(pred_idx, conf, smooth_probs)

        raw_action = self.labels[pred_idx]

        # --- State machine ---
        state_name = 'passthrough'
        state_decision = self.state_machine.update(
            probs=smooth_probs,
            proposed_idx=pred_idx,
            proposed_conf=conf,
        )
        pred_idx = int(state_decision['pred_idx'])
        conf = float(state_decision['confidence'])
        state_name = str(state_decision.get('state', 'passthrough'))

        # --- Build result ---
        idle_idx = self.labels.index('idle') if 'idle' in self.labels else None
        is_idle = (idle_idx is not None and pred_idx == idle_idx)

        probabilities = {
            label: float(smooth_probs[i])
            for i, label in enumerate(self.labels)
        }

        return PredictionResult(
            action=self.labels[pred_idx],
            confidence=conf,
            probabilities=probabilities,
            is_idle=is_idle,
            raw_action=raw_action,
            state_machine_state=state_name,
            voxel_active_ratio=voxel_active_ratio,
            frame_count=self.frame_count,
        )

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all internal state (buffers, state machine, etc.).

        The model remains loaded; only per-session state is cleared.
        """
        self.frame_count = 0
        if self.feature_buffer is not None:
            self.feature_buffer.reset()
        if self.smoother is not None:
            self.smoother.reset()
        if self.hysteresis is not None:
            self.hysteresis.reset()
        if self.block_filter is not None:
            self.block_filter.reset()
        if self.state_machine is not None:
            self.state_machine.reset()
        self._prev_pose_static = None
        self._cached_pose_features = None
        logger.info("InferenceEngine state reset.")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _idle_result(self, state: str) -> PredictionResult:
        """Construct an idle PredictionResult."""
        return PredictionResult(
            action='idle',
            confidence=0.0,
            probabilities={label: 0.0 for label in self.labels},
            is_idle=True,
            raw_action='idle',
            state_machine_state=state,
            voxel_active_ratio=0.0,
            frame_count=self.frame_count,
        )

    def _extract_pose(self, rgb: np.ndarray) -> Optional[np.ndarray]:
        """Run pose estimation and produce the fusion pose feature vector.

        Args:
            rgb: RGB image, shape ``(H, W, 3)``.

        Returns:
            Pose feature vector of shape ``(pose_dim,)`` or ``None``.
        """
        from action_prediction.lib.fusion_model import (
            POSE_VELOCITY_DIM,
            compute_pose_velocity,
            extract_pose_features,
            extract_pose_features_static,
        )

        if self.pose_estimator is None:
            return extract_pose_features(None, None)

        try:
            kps, confs, bbox = self.pose_estimator.predict_with_bbox(rgb)
            cur_static = extract_pose_features_static(kps, confs)
            vel = (
                compute_pose_velocity(cur_static, self._prev_pose_static)
                if self._prev_pose_static is not None
                else np.zeros(POSE_VELOCITY_DIM, dtype=np.float32)
            )
            pose_features = np.concatenate([cur_static, vel])
            self._prev_pose_static = cur_static.copy()
            self._cached_pose_features = pose_features
            return pose_features
        except Exception as e:
            logger.debug("Pose estimation failed: %s", e)
            if self._cached_pose_features is not None:
                return self._cached_pose_features
            return extract_pose_features(None, None)

    def _run_inference(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        """Run model forward pass and return raw softmax probabilities.

        Priority: TensorRT > ONNX Runtime > PyTorch.

        Args:
            features: Dict from ``RollingFeatureBuffer.get_features()``.

        Returns:
            Raw probability vector of shape ``(num_classes,)``.
        """
        import torch

        feat_np = features['features']  # (T, feat_dim)
        mask_np = features.get('padding_mask', None)

        if self._trt_context is not None:
            return self._run_inference_trt(feat_np, mask_np)
        elif self._ort_session is not None:
            return self._run_inference_ort(feat_np, mask_np)
        else:
            return self._run_inference_pytorch(feat_np, mask_np)

    def _run_inference_pytorch(
        self,
        feat_np: np.ndarray,
        mask_np: Optional[np.ndarray],
    ) -> np.ndarray:
        """PyTorch eager-mode inference (fallback)."""
        import torch

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
            return probs[0].cpu().numpy()

    def _run_inference_ort(
        self,
        feat_np: np.ndarray,
        mask_np: Optional[np.ndarray],
    ) -> np.ndarray:
        """ONNX Runtime inference path."""
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
        return exp_logits / exp_logits.sum()

    def _run_inference_trt(
        self,
        feat_np: np.ndarray,
        mask_np: Optional[np.ndarray],
    ) -> np.ndarray:
        """TensorRT inference path (fastest on Jetson)."""
        import torch

        b = self._trt_bindings
        T = feat_np.shape[0]

        # Update dynamic shapes if seq_len changed
        if T != b['clip_len']:
            self._trt_context.set_input_shape('features', (1, T, b['feat_dim']))
            self._trt_context.set_input_shape('padding_mask', (1, T))
            b['d_features'] = torch.empty(
                (1, T, b['feat_dim']), dtype=torch.float32, device=self.device,
            )
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

        return torch.softmax(b['d_logits'][0], dim=0).cpu().numpy()

    # ------------------------------------------------------------------
    # Optimized inference setup (TRT > ONNX > PyTorch)
    # ------------------------------------------------------------------

    def _setup_optimized_inference(self, clip_len: int) -> None:
        """Auto-convert PyTorch -> ONNX -> TensorRT and set up fast inference.

        Priority: TensorRT engine > ORT with GPU > PyTorch fallback.
        All artifacts are cached next to the .pth so subsequent runs are
        instant.

        Args:
            clip_len: Temporal clip length for the model.
        """
        import torch

        from action_prediction.lib.fusion_model import POSE_FEATURE_DIM

        pth_path = Path(self.checkpoint_path)
        onnx_path = pth_path.with_suffix('.onnx')
        trt_path = pth_path.with_suffix('.trt')

        voxel_dim = int(self.voxel_grid_size[0]) ** 3
        pose_dim = self.pose_dim if self.fusion_mode else 0
        feat_dim = voxel_dim * self.in_channels + pose_dim
        clip_len = int(clip_len) if clip_len else 12

        # --- Step 1: Export to ONNX if not cached ---
        if not onnx_path.exists():
            logger.info("ONNX model not found -- exporting to %s ...", onnx_path.name)
            try:
                dummy_features = torch.randn(1, clip_len, feat_dim)
                dummy_mask = torch.zeros(1, clip_len, dtype=torch.bool)

                model_cpu = self.model.cpu()
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
                self.model.to(self.device)
                size_mb = onnx_path.stat().st_size / (1024 * 1024)
                logger.info("ONNX exported: %s (%.1f MB)", onnx_path.name, size_mb)
            except Exception as e:
                logger.warning("ONNX export failed (%s), using PyTorch", e)
                self.model.to(self.device)
                return
        else:
            size_mb = onnx_path.stat().st_size / (1024 * 1024)
            logger.info("ONNX model found: %s (%.1f MB)", onnx_path.name, size_mb)

        # --- Step 2: Try TensorRT direct engine ---
        if _TRT_AVAILABLE and self.device.startswith('cuda'):
            try:
                self._setup_tensorrt_engine(onnx_path, trt_path, feat_dim, clip_len)
                if self._trt_context is not None:
                    return  # TRT is live, skip ORT
            except Exception as e:
                logger.warning("TensorRT setup failed (%s), trying ORT...", e)

        # --- Step 3: Fall back to ORT if GPU provider available ---
        if _ORT_AVAILABLE:
            try:
                self._setup_ort_session(onnx_path, feat_dim, clip_len)
                if self._ort_session is not None:
                    return
            except Exception as e:
                logger.warning("ORT setup failed (%s)", e)

        logger.info("Falling back to PyTorch eager mode.")

    def _setup_tensorrt_engine(
        self,
        onnx_path: Path,
        trt_path: Path,
        feat_dim: int,
        clip_len: int,
    ) -> None:
        """Build or load a TensorRT FP16 engine and allocate IO buffers.

        Args:
            onnx_path: Path to the ONNX model file.
            trt_path: Path to cache the TensorRT engine.
            feat_dim: Input feature dimension.
            clip_len: Temporal clip length.
        """
        import torch

        if not trt_path.exists():
            logger.info("Building TensorRT engine (FP16) -- this may take 1-2 minutes ...")
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(TRT_LOGGER)
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            parser = trt.OnnxParser(network, TRT_LOGGER)

            with open(str(onnx_path), 'rb') as f:
                if not parser.parse(f.read()):
                    for i in range(parser.num_errors):
                        logger.error("TRT parse error: %s", parser.get_error(i))
                    return

            config = builder.create_builder_config()
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)

            # Dynamic shape profile
            profile = builder.create_optimization_profile()
            feat_input = network.get_input(0)
            profile.set_shape(
                feat_input.name,
                min=(1, 4, feat_dim),
                opt=(1, clip_len, feat_dim),
                max=(1, 64, feat_dim),
            )
            mask_input = network.get_input(1)
            profile.set_shape(
                mask_input.name,
                min=(1, 4),
                opt=(1, clip_len),
                max=(1, 64),
            )
            config.add_optimization_profile(profile)

            t0 = time.time()
            engine_bytes = builder.build_serialized_network(network, config)
            if engine_bytes is None:
                logger.error("TensorRT engine build failed")
                return

            with open(str(trt_path), 'wb') as f:
                f.write(engine_bytes)
            size_mb = trt_path.stat().st_size / (1024 * 1024)
            logger.info(
                "TensorRT engine built: %s (%.1f MB, %.0fs)",
                trt_path.name, size_mb, time.time() - t0,
            )
        else:
            size_mb = trt_path.stat().st_size / (1024 * 1024)
            logger.info("TensorRT engine found: %s (%.1f MB)", trt_path.name, size_mb)

        # Load engine and create execution context
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(TRT_LOGGER)
        with open(str(trt_path), 'rb') as f:
            engine = runtime.deserialize_cuda_engine(f.read())

        context = engine.create_execution_context()
        context.set_input_shape('features', (1, clip_len, feat_dim))
        context.set_input_shape('padding_mask', (1, clip_len))

        # Pre-allocate GPU buffers using PyTorch tensors
        num_classes = self.num_classes
        device = self.device
        d_features = torch.empty((1, clip_len, feat_dim), dtype=torch.float32, device=device)
        d_mask = torch.empty((1, clip_len), dtype=torch.bool, device=device)
        d_logits = torch.empty((1, num_classes), dtype=torch.float32, device=device)

        context.set_tensor_address('features', d_features.data_ptr())
        context.set_tensor_address('padding_mask', d_mask.data_ptr())
        context.set_tensor_address('logits', d_logits.data_ptr())

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

        logger.info("TensorRT inference ready (FP16)")

    def _setup_ort_session(
        self,
        onnx_path: Path,
        feat_dim: int,
        clip_len: int,
    ) -> None:
        """Create an ORT inference session with the best GPU provider.

        Args:
            onnx_path: Path to the ONNX model file.
            feat_dim: Input feature dimension.
            clip_len: Temporal clip length.
        """
        available_providers = ort.get_available_providers()
        providers: list = []

        if 'TensorrtExecutionProvider' in available_providers:
            providers.append((
                'TensorrtExecutionProvider', {
                    'device_id': int(self.device.split(':')[-1]) if ':' in self.device else 0,
                    'trt_max_workspace_size': 1 << 30,
                    'trt_fp16_enable': True,
                    'trt_engine_cache_enable': True,
                    'trt_engine_cache_path': str(onnx_path.parent),
                },
            ))
        if 'CUDAExecutionProvider' in available_providers:
            providers.append((
                'CUDAExecutionProvider', {
                    'device_id': int(self.device.split(':')[-1]) if ':' in self.device else 0,
                    'arena_extend_strategy': 'kSameAsRequested',
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                },
            ))
        providers.append('CPUExecutionProvider')

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4

        session = ort.InferenceSession(
            str(onnx_path), sess_options=sess_options, providers=providers,
        )

        active_provider = session.get_providers()[0]
        logger.info("ONNX Runtime session ready (provider: %s)", active_provider)

        # Warmup
        dummy_f = np.random.randn(1, clip_len, feat_dim).astype(np.float32)
        dummy_m = np.zeros((1, clip_len), dtype=bool)
        input_names = [inp.name for inp in session.get_inputs()]
        session.run(None, {input_names[0]: dummy_f, input_names[1]: dummy_m})
        logger.info("ONNX Runtime warmup complete")

        self._ort_session = session
        self._ort_input_names = input_names
