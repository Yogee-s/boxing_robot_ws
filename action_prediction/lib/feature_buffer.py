"""
Rolling feature buffer for temporal model inference.

Provides:
- RollingFeatureBuffer: maintains a sliding window of voxel (+ optional pose)
  features, applies normalization matching training, and produces batched
  tensors ready for model forward pass.
- debug_voxel_grid: helper to reshape flattened voxel features for visualization.
"""

import logging
from collections import deque
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def debug_voxel_grid(
    voxel_flat: np.ndarray,
    voxel_size: Tuple[int, int, int],
    in_channels: int,
) -> np.ndarray:
    """Return a single 3D grid for debug visualization from a flattened feature frame.

    When multiple channels exist, returns only the first channel's 3D grid.

    Args:
        voxel_flat: Flattened voxel feature vector for one frame.
        voxel_size: Spatial dimensions of the voxel grid ``(X, Y, Z)``.
        in_channels: Number of feature channels per voxel cell.

    Returns:
        3D numpy array of shape *voxel_size*.
    """
    voxel_flat = np.asarray(voxel_flat, dtype=np.float32).reshape(-1)
    if in_channels > 1:
        return voxel_flat.reshape(in_channels, *voxel_size)[0]
    return voxel_flat.reshape(voxel_size)


class RollingFeatureBuffer:
    """Maintains a rolling voxel (+ optional pose) feature buffer for inference.

    Accumulates per-frame flattened voxel features (and optional pose features
    in fusion mode) in a fixed-size deque.  When the buffer is full, produces
    a batch of features with normalization applied to match the training
    preprocessing pipeline.

    Args:
        window_size: Number of frames (temporal window) for one inference.
        voxel_size: Spatial dimension of the voxel grid (assumes cubic).
        voxel_normalization: Normalization mode -- one of ``'clip_p90'``,
            ``'channel_p90'``, ``'frame_l1'``, or ``'none'``.
        in_channels: Number of voxel feature channels.
        voxel_grid_size: Full 3D voxel grid dimensions ``(X, Y, Z)``.
        fusion_mode: If ``True``, also buffer pose features.
        pose_dim: Dimensionality of pose feature vectors.
    """

    def __init__(
        self,
        window_size: int = 12,
        voxel_size: int = 12,
        voxel_normalization: str = 'clip_p90',
        in_channels: int = 1,
        voxel_grid_size: Tuple[int, int, int] = (20, 20, 20),
        fusion_mode: bool = False,
        pose_dim: int = 0,
    ) -> None:
        self.window_size: int = window_size
        self.voxel_size: int = voxel_size
        self.voxel_normalization: str = str(voxel_normalization)
        self.in_channels: int = in_channels
        self.voxel_grid_size: Tuple[int, int, int] = voxel_grid_size
        self.fusion_mode: bool = fusion_mode
        self.pose_dim: int = pose_dim

        # Feature buffers
        self.voxel_buffer: deque = deque(maxlen=window_size)
        self.fg_ratio_buffer: deque = deque(maxlen=window_size)
        # Pose buffer (only used in fusion mode)
        self.pose_buffer: Optional[deque] = (
            deque(maxlen=window_size) if fusion_mode else None
        )

    def add_frame(
        self,
        voxel_features: np.ndarray,
        fg_ratio: float,
        pose_features: Optional[np.ndarray] = None,
    ) -> None:
        """Add a frame's flattened voxel (+ optional pose) feature vector to the buffer.

        Args:
            voxel_features: Flattened voxel feature vector for one frame.
            fg_ratio: Foreground pixel ratio for this frame.
            pose_features: Optional pose feature vector (fusion mode only).
        """
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

    def get_features(self) -> Optional[Dict[str, np.ndarray]]:
        """Get feature tensors for model inference.

        Returns:
            Dict with ``'features'`` (combined voxel+pose), ``'voxel'``
            (normalized voxel only), and ``'fg_ratio'`` arrays.  Returns
            ``None`` if the buffer is not yet full.
        """
        if len(self.voxel_buffer) < self.window_size:
            return None

        # Stack voxel features.
        voxel = np.stack(list(self.voxel_buffer), axis=0)       # (T, N^3*C)
        fg_ratio = np.array(list(self.fg_ratio_buffer))         # (T,)

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

    def reset(self) -> None:
        """Clear all buffered features."""
        self.voxel_buffer.clear()
        self.fg_ratio_buffer.clear()
        if self.pose_buffer is not None:
            self.pose_buffer.clear()

    @property
    def is_ready(self) -> bool:
        """Whether the buffer has accumulated enough frames for inference."""
        return len(self.voxel_buffer) >= self.window_size
