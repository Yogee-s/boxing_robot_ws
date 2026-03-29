"""
Checkpoint loading and metadata extraction utilities.

Provides functions for:
- Loading PyTorch checkpoints
- Resolving voxel size metadata
- Extracting feature layout configuration from checkpoint metadata
- Resolving runtime device strings
- Extracting label names from checkpoints
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Default 8-class label set (matches training)
DEFAULT_LABELS: List[str] = [
    'block', 'cross', 'idle', 'jab',
    'left_hook', 'left_uppercut', 'right_hook', 'right_uppercut',
]


def _coerce_voxel_size(
    value: object,
    fallback: Optional[Tuple[int, int, int]] = None,
) -> Tuple[int, int, int]:
    """Normalize voxel size metadata to a 3D tuple.

    Args:
        value: Raw voxel size from checkpoint metadata. May be int, list,
            tuple, ndarray, or None.
        fallback: Default tuple to use when *value* is None.

    Returns:
        A 3-tuple of ints representing (X, Y, Z) voxel grid dimensions.

    Raises:
        ValueError: If *value* cannot be coerced and no *fallback* is given.
    """
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


def _resolve_checkpoint_feature_layout(
    checkpoint: dict,
    default_voxel_size: int = 24,
) -> Dict[str, object]:
    """Extract model and feature metadata needed for live voxel extraction.

    Parses the checkpoint's ``config`` and ``dataset_config`` dicts to
    determine voxel grid size, channel count, normalization mode, and
    other settings that must match the training pipeline exactly.

    Args:
        checkpoint: The loaded checkpoint dictionary.
        default_voxel_size: Fallback voxel grid dimension when metadata
            is missing.

    Returns:
        Dict with keys: ``config``, ``dataset_config``, ``voxel_size``,
        ``in_channels``, ``voxel_normalization``, ``directional_gradients``,
        ``velocity_magnitude_channel``, ``multi_scale_delta_frames``,
        ``voxel_delta_frames``, ``voxel_depth_weighted``.
    """
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
    multi_scale_delta_frames = tuple(
        int(v) for v in (dataset_config.get('multi_scale_delta_frames') or [])
    )
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


def _resolve_runtime_device(device: Optional[str]) -> str:
    """Resolve the requested device to a usable runtime device string.

    If CUDA is requested but unavailable, falls back to ``'cpu'``.

    Args:
        device: Requested device string (e.g. ``'cuda:0'``, ``'cpu'``).

    Returns:
        A valid PyTorch device string.
    """
    import torch

    requested = str(device).strip() if device is not None else 'cpu'
    if requested.startswith('cuda'):
        if torch.cuda.is_available():
            return requested
        logger.warning(
            "Requested device %s but CUDA is unavailable. Falling back to cpu.",
            requested,
        )
        return 'cpu'
    return requested or 'cpu'


def load_checkpoint(checkpoint_path: str, device: str = 'cpu') -> dict:
    """Load a PyTorch ``.pth`` checkpoint file and return the checkpoint dict.

    Args:
        checkpoint_path: Path to the ``.pth`` file.
        device: Device to map tensors onto during loading.

    Returns:
        The checkpoint dictionary.

    Raises:
        FileNotFoundError: If *checkpoint_path* does not exist.
        RuntimeError: If the file cannot be loaded by PyTorch.
    """
    import torch

    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info("Loading checkpoint from %s", checkpoint_path)
    checkpoint = torch.load(str(path), map_location=device, weights_only=False)
    return checkpoint


def load_labels(checkpoint: dict, checkpoint_path: str, num_classes: int) -> List[str]:
    """Extract ordered label names from a checkpoint.

    Resolution priority:
      1. ``checkpoint['label_map']`` (list or index-keyed dict)
      2. ``ann_file`` referenced in checkpoint config or sibling ``config.json``
      3. :data:`DEFAULT_LABELS` fallback (when *num_classes* matches)
      4. Generic ``class_0 .. class_N`` names

    Args:
        checkpoint: The loaded checkpoint dictionary.
        checkpoint_path: Filesystem path to the checkpoint (used to find
            sibling config files).
        num_classes: Expected number of output classes.

    Returns:
        Ordered list of *num_classes* label strings.
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
    candidate_ann: Optional[str] = None
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
