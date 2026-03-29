"""
Causal action state machine and punch detection classifiers.

Provides:
- CausalActionStateMachine: hysteresis-based event filter for live action outputs
- DepthPunchDetector: detects punch motion from foreground depth approach velocity
- PunchSegmentClassifier: detect-then-classify segmenter for complete punch events
"""

import logging
import time
from collections import deque
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class CausalActionStateMachine:
    """Causal event filter for live-style action outputs.

    Prevents flickering by requiring consecutive frames of the same
    prediction before activating an action, and requiring multiple exit
    signals before deactivating.

    Args:
        labels: Ordered list of class label strings.
        enter_consecutive: Number of consecutive frames required to
            activate a new action.
        exit_consecutive: Number of consecutive exit-signal frames
            required to deactivate the current action.
        min_hold_steps: Minimum number of steps an action must be held
            before it can be deactivated.
        sustain_confidence: Minimum confidence to sustain an active action.
        peak_drop_threshold: Maximum allowed drop from peak confidence
            before triggering an exit signal.
    """

    def __init__(
        self,
        labels: List[str],
        enter_consecutive: int,
        exit_consecutive: int,
        min_hold_steps: int,
        sustain_confidence: float,
        peak_drop_threshold: float,
    ) -> None:
        self.labels: List[str] = list(labels)
        self.idle_idx: Optional[int] = (
            self.labels.index('idle') if 'idle' in self.labels else None
        )
        self.enter_consecutive: int = max(1, int(enter_consecutive))
        self.exit_consecutive: int = max(1, int(exit_consecutive))
        self.min_hold_steps: int = max(0, int(min_hold_steps))
        self.sustain_confidence: float = max(0.0, float(sustain_confidence))
        self.peak_drop_threshold: float = max(0.0, float(peak_drop_threshold))
        self.reset()

    def reset(self) -> None:
        """Reset all internal state to idle."""
        self.active_idx: Optional[int] = None
        self.active_steps: int = 0
        self.active_peak_conf: float = 0.0
        self.enter_candidate_idx: Optional[int] = None
        self.enter_count: int = 0
        self.exit_count: int = 0

    def update(
        self,
        probs: np.ndarray,
        proposed_idx: int,
        proposed_conf: float,
    ) -> Dict[str, object]:
        """Process one frame and return the filtered prediction.

        Args:
            probs: Full probability vector over all classes.
            proposed_idx: Index of the proposed (pre-state-machine) prediction.
            proposed_conf: Confidence of the proposed prediction.

        Returns:
            Dict with ``pred_idx``, ``confidence``, ``state``, and
            optionally ``exit_reasons``.
        """
        probs = np.asarray(probs, dtype=np.float32)
        idle_idx = self.idle_idx
        if idle_idx is None:
            return {
                'pred_idx': int(proposed_idx),
                'confidence': float(proposed_conf),
                'state': 'passthrough',
            }

        # --- No action currently active ---
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

        # --- Action is currently active ---
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
        if (
            self.peak_drop_threshold > 0.0
            and active_conf <= (self.active_peak_conf - self.peak_drop_threshold)
        ):
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

    Args:
        near_percentile: Which percentile of foreground depth to track
            as the nearest surface.
        velocity_threshold: Minimum depth velocity (m/frame) to consider
            as punch motion.
        history_len: Number of frames of depth history to keep.
    """

    def __init__(
        self,
        near_percentile: float = 5.0,
        velocity_threshold: float = 0.01,
        history_len: int = 4,
    ) -> None:
        self.near_percentile: float = max(1.0, min(50.0, float(near_percentile)))
        self.velocity_threshold: float = float(velocity_threshold)
        self.history_len: int = max(2, int(history_len))

        self.depth_history: deque = deque(maxlen=self.history_len)

        # Exposed state for overlay / gating
        self.nearest_depth: float = 0.0
        self.punch_signal: float = 0.0
        self.retract_signal: float = 0.0
        self.punch_active: bool = False
        self.retracting: bool = False

    def update(
        self,
        depth_m: np.ndarray,
        fg_mask: Optional[np.ndarray],
    ) -> Dict[str, object]:
        """Compute punch signal from one frame's depth + foreground mask.

        Args:
            depth_m: Depth map in meters, shape ``(H, W)``.
            fg_mask: Binary foreground mask, shape ``(H, W)``, uint8 0/1.
                If ``None``, the full frame is treated as foreground.

        Returns:
            Dict with ``nearest_depth``, ``punch_signal``,
            ``retract_signal``, ``punch_active``, ``retracting``.
        """
        # Get valid foreground depth pixels
        if fg_mask is not None:
            fg_depth = depth_m[fg_mask > 0]
        else:
            fg_depth = depth_m.ravel()
        valid = fg_depth[(fg_depth > 0.15) & (fg_depth < 4.0)]

        if len(valid) < 20:
            # Not enough foreground -- default to ALLOWING predictions.
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
        if len(self.depth_history) >= 2:
            oldest = self.depth_history[0]
            newest = self.depth_history[-1]
            self.punch_signal = max(oldest - newest, 0.0)
            self.retract_signal = max(newest - oldest, 0.0)
        else:
            self.punch_signal = 0.0
            self.retract_signal = 0.0

        self.punch_active = self.punch_signal >= self.velocity_threshold
        self.retracting = (
            self.retract_signal >= self.velocity_threshold
            and self.punch_signal < self.velocity_threshold
        )

        return {
            'nearest_depth': self.nearest_depth,
            'punch_signal': self.punch_signal,
            'retract_signal': self.retract_signal,
            'punch_active': self.punch_active,
            'retracting': self.retracting,
        }


class PunchSegmentClassifier:
    """Detect-then-classify: buffer voxel features during a punch, classify when done.

    Uses total voxel activity (abs delta magnitude) to detect motion
    start/end, which works for ALL punch types (jab, hook, uppercut) --
    not just approach.

    States:  ``IDLE`` -> ``ACTIVE`` -> ``COOLDOWN`` -> (classify) -> ``DISPLAY`` -> ``IDLE``

    Args:
        activity_start: Minimum mean absolute voxel activity to start
            buffering.
        activity_end: Activity threshold below which cooldown begins.
        cooldown_frames: Number of low-activity frames before finalizing
            the segment.
        min_segment_frames: Minimum frames for a valid punch segment.
        max_segment_frames: Maximum frames before force-ending.
        display_hold_sec: Seconds to hold the classification result
            before returning to idle.
    """

    def __init__(
        self,
        activity_start: float = 0.002,
        activity_end: float = 0.001,
        cooldown_frames: int = 6,
        min_segment_frames: int = 4,
        max_segment_frames: int = 120,
        display_hold_sec: float = 2.0,
    ) -> None:
        self.activity_start: float = activity_start
        self.activity_end: float = activity_end
        self.cooldown_frames: int = cooldown_frames
        self.min_segment_frames: int = min_segment_frames
        self.max_segment_frames: int = max_segment_frames
        self.display_hold_sec: float = display_hold_sec

        # State
        self._state: str = "IDLE"
        self._buffer: list = []
        self._fg_ratios: list = []
        self._cooldown_count: int = 0
        self._display_start: float = 0.0

        # Latest result
        self.last_label: Optional[str] = None
        self.last_confidence: float = 0.0
        self.last_segment_frames: int = 0

    def feed(self, voxel_features: np.ndarray, fg_ratio: float) -> Optional[np.ndarray]:
        """Feed one frame of voxel features.

        Args:
            voxel_features: Flattened voxel feature vector for this frame.
            fg_ratio: Foreground pixel ratio (0-1).

        Returns:
            The completed segment as ``(T, feature_dim)`` when a punch
            just ended, or ``None`` if still buffering / idle.
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
                return self._finish_segment()
            return None

        elif self._state == "COOLDOWN":
            self._buffer.append(voxel_features.copy())
            self._fg_ratios.append(fg_ratio)
            if activity >= self.activity_start:
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
        """Finalize the buffered segment.

        Returns:
            ``(T, F)`` array if the segment meets the minimum length,
            or ``None`` if too short.
        """
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

    def set_result(self, label: str, confidence: float) -> None:
        """Store the classification result for display.

        Args:
            label: Predicted action label.
            confidence: Classification confidence score.
        """
        self.last_label = label
        self.last_confidence = confidence

    @property
    def is_active(self) -> bool:
        """Whether the segmenter is currently buffering a punch."""
        return self._state in ("ACTIVE", "COOLDOWN")

    @property
    def is_displaying(self) -> bool:
        """Whether a classification result is currently being displayed."""
        return self._state == "DISPLAY" and self.last_label is not None
