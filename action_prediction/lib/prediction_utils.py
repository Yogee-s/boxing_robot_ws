"""
Prediction post-processing utilities.

Provides modular components for the inference post-processing pipeline:
- select_prediction: non-idle gating based on confidence thresholds
- ProbabilitySmoothing: EMA + windowed mean blending for stable outputs
- PredictionHysteresis: prevents rapid flickering between predictions
- BlockConsecutiveFilter: requires consecutive block frames before showing
"""

import logging
from collections import deque
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def select_prediction(
    probs: np.ndarray,
    labels: List[str],
    min_confidence: float,
    min_action_prob: float = 0.0,
    min_class_margin: float = 0.0,
    voxel_active_ratio: float = 0.0,
    min_voxel_active_ratio: float = 0.0,
) -> Dict[str, object]:
    """Apply non-idle gating on top of smoothed class probabilities.

    Checks whether the top prediction meets several thresholds before
    allowing it through.  If any threshold is not met, the prediction is
    gated to idle.

    Args:
        probs: Smoothed probability vector over all classes.
        labels: Ordered list of class label strings.
        min_confidence: Minimum top-class confidence to allow non-idle.
        min_action_prob: Minimum ``1 - P(idle)`` to allow non-idle.
        min_class_margin: Minimum gap between top and second-best class.
        voxel_active_ratio: Current voxel activity ratio from features.
        min_voxel_active_ratio: Minimum voxel activity to allow non-idle.

    Returns:
        Dict with ``pred_idx``, ``confidence``, ``top_idx``, ``top_conf``,
        ``action_prob``, ``class_margin``, ``idle_idx``, ``gated``,
        ``gate_reasons``.
    """
    probs = np.asarray(probs, dtype=np.float32)
    top_idx = int(np.argmax(probs))
    top_conf = float(probs[top_idx])
    second_conf = float(np.partition(probs, -2)[-2]) if probs.size > 1 else 0.0
    class_margin = top_conf - second_conf
    idle_idx = labels.index('idle') if 'idle' in labels else None
    action_prob = 1.0 - float(probs[idle_idx]) if idle_idx is not None else top_conf

    gate_reasons: List[str] = []
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


class ProbabilitySmoothing:
    """Temporal probability smoothing using EMA + windowed mean blending.

    Combines an exponential moving average (EMA) with a short windowed
    mean for extra stability.  The final smoothed probabilities are a
    weighted blend of the two: ``0.6 * EMA + 0.4 * windowed_mean``.

    Args:
        alpha: EMA smoothing factor. Higher values weight recent frames
            more heavily. Must be in ``[0, 1]``.
        window_size: Number of recent frames for the windowed mean.
    """

    def __init__(
        self,
        alpha: float = 0.35,
        window_size: int = 5,
    ) -> None:
        self.alpha: float = max(0.0, min(1.0, float(alpha)))
        self.window_size: int = max(1, int(window_size))
        self._ema_probs: Optional[np.ndarray] = None
        self._recent_probs: deque = deque(maxlen=self.window_size)

    def update(self, raw_probs: np.ndarray) -> np.ndarray:
        """Incorporate a new frame's raw probabilities and return smoothed output.

        Args:
            raw_probs: Raw softmax probability vector from the model.

        Returns:
            Smoothed and re-normalized probability vector.
        """
        raw_probs = np.asarray(raw_probs, dtype=np.float32)
        self._recent_probs.append(raw_probs)

        if self._ema_probs is None:
            self._ema_probs = raw_probs.copy()
        else:
            self._ema_probs = (
                self.alpha * raw_probs + (1.0 - self.alpha) * self._ema_probs
            )

        # Blend EMA with windowed mean for extra stability
        windowed_mean = np.mean(np.stack(self._recent_probs, axis=0), axis=0)
        smooth_probs = 0.6 * self._ema_probs + 0.4 * windowed_mean
        smooth_probs = smooth_probs / max(float(smooth_probs.sum()), 1e-8)
        return smooth_probs

    def reset(self) -> None:
        """Clear all smoothing state."""
        self._ema_probs = None
        self._recent_probs.clear()


class PredictionHysteresis:
    """Prevents rapid flickering between predictions via a margin-based hold.

    Once a prediction is established, a competing prediction must exceed
    the current one by at least *hysteresis_margin* AND the current
    prediction must have been held for at least *min_hold_frames* before
    a switch is allowed.

    Args:
        hysteresis_margin: Minimum confidence gap required to switch
            predictions.
        min_hold_frames: Minimum number of frames a prediction must be
            held before switching is allowed.
    """

    def __init__(
        self,
        hysteresis_margin: float = 0.12,
        min_hold_frames: int = 3,
    ) -> None:
        self.hysteresis_margin: float = max(0.0, float(hysteresis_margin))
        self.min_hold_frames: int = max(0, int(min_hold_frames))
        self._held_pred_idx: Optional[int] = None
        self._held_pred_frames: int = 0

    def update(
        self,
        pred_idx: int,
        smooth_probs: np.ndarray,
    ) -> tuple:
        """Apply hysteresis filtering to the proposed prediction.

        Args:
            pred_idx: Proposed prediction index (after gating).
            smooth_probs: Full smoothed probability vector.

        Returns:
            Tuple of ``(filtered_pred_idx, filtered_confidence)``.
        """
        smooth_probs = np.asarray(smooth_probs, dtype=np.float32)

        if self._held_pred_idx is not None:
            self._held_pred_frames += 1
            if pred_idx != self._held_pred_idx:
                held_conf = float(smooth_probs[self._held_pred_idx])
                new_conf = float(smooth_probs[pred_idx])
                # Only switch if held prediction has been shown long enough
                # AND new class is clearly stronger than current.
                if (
                    self._held_pred_frames < self.min_hold_frames
                    or (new_conf - held_conf) < self.hysteresis_margin
                ):
                    # Stick with current prediction
                    pred_idx = self._held_pred_idx
                    conf = held_conf
                else:
                    # Switch to new prediction
                    self._held_pred_idx = pred_idx
                    self._held_pred_frames = 0
                    conf = new_conf
            else:
                # Same prediction, keep holding
                conf = float(smooth_probs[pred_idx])
        else:
            self._held_pred_idx = pred_idx
            self._held_pred_frames = 0
            conf = float(smooth_probs[pred_idx])

        return pred_idx, conf

    def reset(self) -> None:
        """Clear held prediction state."""
        self._held_pred_idx = None
        self._held_pred_frames = 0


class BlockConsecutiveFilter:
    """Requires N consecutive block frames before allowing a block prediction.

    Prevents brief block flickers from being displayed.  When the
    prediction is 'block' but the consecutive count has not yet been
    reached, the prediction is forced to idle.

    Args:
        block_consecutive_needed: Number of consecutive block-predicted
            frames required before the block prediction is emitted.
        labels: Ordered list of class label strings (used to find the
            block and idle indices).
    """

    def __init__(
        self,
        block_consecutive_needed: int = 4,
        labels: Optional[List[str]] = None,
    ) -> None:
        self.block_consecutive_needed: int = max(1, int(block_consecutive_needed))
        self._block_consec_count: int = 0

        self._block_idx: Optional[int] = None
        self._idle_idx: Optional[int] = None
        if labels is not None:
            self.set_labels(labels)

    def set_labels(self, labels: List[str]) -> None:
        """Configure block and idle indices from the label list.

        Args:
            labels: Ordered list of class label strings.
        """
        self._block_idx = labels.index('block') if 'block' in labels else None
        self._idle_idx = labels.index('idle') if 'idle' in labels else None

    def update(
        self,
        pred_idx: int,
        conf: float,
        smooth_probs: np.ndarray,
    ) -> tuple:
        """Apply the consecutive block filter.

        Args:
            pred_idx: Current prediction index.
            conf: Current prediction confidence.
            smooth_probs: Full smoothed probability vector.

        Returns:
            Tuple of ``(filtered_pred_idx, filtered_confidence)``.
        """
        smooth_probs = np.asarray(smooth_probs, dtype=np.float32)

        if pred_idx == self._block_idx and self._block_idx is not None:
            self._block_consec_count += 1
            if self._block_consec_count < self.block_consecutive_needed:
                if self._idle_idx is not None:
                    return self._idle_idx, float(smooth_probs[self._idle_idx])
                # No idle class -- pass through unchanged
                return pred_idx, conf
        else:
            self._block_consec_count = 0

        return pred_idx, conf

    def reset(self) -> None:
        """Clear consecutive count."""
        self._block_consec_count = 0
