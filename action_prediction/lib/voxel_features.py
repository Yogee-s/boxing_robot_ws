"""
Voxel-only depth feature extraction utilities.

The supported pipeline in this repository uses:
- gravity-aligned background subtraction
- person-centric voxel occupancy deltas
- optional debug metadata for overlays and visualization
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class VoxelFeatureConfig:
    """Configuration for voxel-only depth feature extraction."""
    
    # Camera intrinsics (depth camera)
    fx: float = 424.0
    fy: float = 424.0
    cx: float = 424.0
    cy: float = 240.0
    depth_scale: float = 0.001  # Convert raw depth to meters
    
    # Camera orientation (from IMU, for gravity alignment)
    camera_pitch_deg: float = 0.0  # Tilt forward/back (positive = looking down)
    camera_roll_deg: float = 0.0   # Tilt left/right
    
    # Background subtraction
    bg_threshold_m: float = 0.15  # Threshold for foreground segmentation
    bg_min_depth_m: float = 0.3   # Minimum valid depth
    bg_max_depth_m: float = 4.0   # Maximum valid depth
    
    # Voxel grid settings (in GRAVITY-ALIGNED coordinates)
    voxel_grid_size: Tuple[int, int, int] = (20, 20, 20)  # Default grid (8000 total)
    # Person-centric bounds (relative to detected person center)
    voxel_extent_x: float = 0.8   # Half-width in X (left/right) meters
    voxel_extent_y: float = 1.0   # Half-height in Y (up/down) meters  
    voxel_extent_z: float = 0.6   # Half-depth in Z (forward/back) meters
    voxel_delta_frames: int = 3   # Frames to look back for occupancy delta
    voxel_person_centric: bool = True  # If True, center voxel grid on person
    voxel_depth_weighted: bool = True  # Weight voxels by inverse depth (closer = stronger)
    
    # Directional voxel flow: add spatial gradient channels (Δx, Δy, Δz)
    # to the occupancy delta so the model sees explicit motion direction.
    directional_gradients: bool = False

    # Velocity magnitude channel: add abs(delta) as an extra channel after
    # the directional gradients.  Gives the model an explicit "activity map"
    # (high where the glove is moving, near-zero on the static torso) without
    # requiring it to learn abs() from the signed delta.
    # Only active when directional_gradients=True.
    # Channel layout per scale: [delta, grad_x, grad_y, grad_z, abs_delta]
    velocity_magnitude_channel: bool = False
    
    # Multi-scale temporal delta: compute deltas at multiple lookback
    # distances instead of a single voxel_delta_frames.  When non-empty
    # this overrides voxel_delta_frames and the final feature becomes
    # the concatenation of deltas at each scale (multiplied by channels
    # if directional_gradients is also enabled).
    multi_scale_delta_frames: Tuple[int, ...] = ()

    # Include raw occupancy grid as the first channel (before deltas).
    # Gives the model "where the body is" (static spatial context) in
    # addition to "what moved" (delta).  Adds 1 channel to in_channels.
    include_raw_occupancy: bool = False
    
    # Optional fixed bounds (used when person_centric=False)
    voxel_x_range: Tuple[float, float] = (-0.8, 0.8)   # Fighter's reach envelope (left/right)
    voxel_y_range: Tuple[float, float] = (-1.2, 0.6)   # Fighter's body (up/down from center)
    voxel_z_range: Tuple[float, float] = (0.5, 3.0)    # Depth range in front of camera


def build_tilt_rotation_matrix(pitch_deg: float, roll_deg: float) -> np.ndarray:
    """
    Build rotation matrix to correct for camera tilt.
    
    Transforms from camera coordinates to gravity-aligned coordinates.
    Assumes camera is tilted DOWN by pitch_deg (positive = looking down).
    
    Args:
        pitch_deg: Camera pitch in degrees (positive = tilted forward/down)
        roll_deg: Camera roll in degrees (positive = tilted right)
    
    Returns:
        3x3 rotation matrix
    """
    if abs(pitch_deg) < 1e-6 and abs(roll_deg) < 1e-6:
        return np.eye(3, dtype=np.float32)
    
    # Convert to radians - NEGATIVE because we want to UN-tilt
    # Camera tilted DOWN by pitch_deg → rotate by -pitch to bring back to horizontal
    alpha = np.radians(-pitch_deg)  # Rotation around X-axis
    beta = np.radians(-roll_deg)    # Rotation around Z-axis
    
    # Rotation matrices
    # X-axis rotation (Pitch correction)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(alpha), -np.sin(alpha)],
        [0, np.sin(alpha), np.cos(alpha)]
    ], dtype=np.float32)
    
    # Z-axis rotation (Roll correction)
    Rz = np.array([
        [np.cos(beta), -np.sin(beta), 0],
        [np.sin(beta), np.cos(beta), 0],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # Combined: first undo roll, then undo pitch
    return Rx @ Rz


@dataclass
class ExtractionDebugInfo:
    """Debug information from feature extraction for visualization."""
    
    # Background model
    background_depth_m: Optional[np.ndarray] = None
    
    # Foreground segmentation
    foreground_mask: Optional[np.ndarray] = None
    foreground_ratio: float = 0.0
    
    # Point cloud (subsampled for viz)
    points_camera: Optional[np.ndarray] = None  # Original camera coords
    points_gravity: Optional[np.ndarray] = None  # After tilt correction
    
    # Voxel grid bounds (in gravity-aligned coords)
    voxel_center: Optional[np.ndarray] = None  # (3,) center of voxel grid
    voxel_bounds_min: Optional[np.ndarray] = None  # (3,) min corner
    voxel_bounds_max: Optional[np.ndarray] = None  # (3,) max corner
    
    # Current and previous occupancy grids
    voxel_current: Optional[np.ndarray] = None  # (8,8,8) current occupancy
    voxel_previous: Optional[np.ndarray] = None  # (8,8,8) previous occupancy

    # Camera info
    tilt_rotation: Optional[np.ndarray] = None  # 3x3 rotation matrix


class BackgroundModel:
    """
    Depth-based background model for foreground segmentation.
    
    Uses 90th percentile of initial frames to estimate static background.
    Much more robust than RGB-based segmentation for boxing.
    """
    
    def __init__(self, config: VoxelFeatureConfig):
        self.config = config
        self.background: Optional[np.ndarray] = None
        self._init_frames: List[np.ndarray] = []
        self._num_init_frames: int = 30
        self._is_initialized: bool = False
    
    def update(self, depth_frame: np.ndarray) -> bool:
        """
        Add frame to background model during initialization.
        Returns True when model is ready.
        """
        if self._is_initialized:
            return True
        
        # Filter valid depth
        valid_depth = depth_frame.copy().astype(np.float32) * self.config.depth_scale
        valid_depth[(valid_depth < self.config.bg_min_depth_m) | 
                   (valid_depth > self.config.bg_max_depth_m)] = 0
        
        self._init_frames.append(valid_depth)
        
        if len(self._init_frames) >= self._num_init_frames:
            # Compute 90th percentile background (robust to occasional foreground)
            stack = np.stack(self._init_frames, axis=0)
            self.background = np.percentile(stack, 90, axis=0)
            self._is_initialized = True
            self._init_frames = []  # Free memory
            return True
        
        return False
    
    def set_background(self, background_m: np.ndarray):
        """Directly set background depth (in meters)."""
        self.background = background_m.astype(np.float32)
        self._is_initialized = True
    
    def get_background(self) -> Optional[np.ndarray]:
        """Get the background depth image in meters."""
        return self.background
    
    def is_initialized(self) -> bool:
        """Check if background model is ready."""
        return self._is_initialized
    
    def get_foreground_mask(self, depth_frame: np.ndarray) -> np.ndarray:
        """
        Segment foreground using depth background subtraction.
        Returns binary mask (1 = foreground, 0 = background).
        """
        if not self._is_initialized or self.background is None:
            return np.zeros(depth_frame.shape[:2], dtype=np.uint8)
        
        # Convert to meters
        depth_m = depth_frame.astype(np.float32) * self.config.depth_scale
        
        # Foreground = significantly closer than background
        diff = self.background - depth_m
        
        # Valid foreground: closer to camera than background by threshold
        # AND within valid depth range
        foreground = (
            (diff > self.config.bg_threshold_m) &
            (depth_m > self.config.bg_min_depth_m) &
            (depth_m < self.config.bg_max_depth_m)
        )
        
        return foreground.astype(np.uint8)


class VoxelOccupancyExtractor:
    """
    Extracts volumetric motion encoding via voxel occupancy changes.
    
    Divides 3D space into a coarse voxel grid and tracks occupancy changes
    over time. This captures punch motion, body lean, and stance changes
    without relying on pose or glove-specific heuristics.
    
    Key improvements for tilted camera:
    - Applies camera tilt correction to align point cloud with gravity
    - Uses person-centric bounds that follow the detected foreground
    - Computes occupancy delta (motion) rather than absolute occupancy
    """
    
    def __init__(self, config: VoxelFeatureConfig):
        self.config = config
        self._grid_size = np.array(config.voxel_grid_size, dtype=np.int32)
        
        # Tilt correction matrix
        self._tilt_rotation = build_tilt_rotation_matrix(
            config.camera_pitch_deg, config.camera_roll_deg
        )
        
        # Fixed bounds (used when person_centric=False)
        self._x_range = np.array(config.voxel_x_range, dtype=np.float32)
        self._y_range = np.array(config.voxel_y_range, dtype=np.float32)
        self._z_range = np.array(config.voxel_z_range, dtype=np.float32)
        
        # Person-centric extents
        self._extent_x = config.voxel_extent_x
        self._extent_y = config.voxel_extent_y
        self._extent_z = config.voxel_extent_z
        
        # History buffer for occupancy
        self._history: List[np.ndarray] = []
        self._center_history: List[np.ndarray] = []  # Track person center
        # History must hold enough frames for the largest delta lookback.
        if config.multi_scale_delta_frames:
            self._max_history = max(config.multi_scale_delta_frames) + 1
        else:
            self._max_history = config.voxel_delta_frames + 1
        
        # Debug info for visualization
        self.last_debug_info: Optional[ExtractionDebugInfo] = None
    
    def set_tilt_rotation(self, pitch_deg: float, roll_deg: float):
        """Update tilt correction matrix."""
        self._tilt_rotation = build_tilt_rotation_matrix(pitch_deg, roll_deg)
    
    def deproject_to_pointcloud(
        self,
        depth_m: np.ndarray,
        mask: Optional[np.ndarray] = None,
        apply_tilt: bool = True,
        subsample: int = 1,
        return_depths: bool = False,
    ) -> np.ndarray:
        """
        Convert depth image to 3D point cloud.

        Optimised for the common case where a foreground mask is provided:
        only foreground pixels are back-projected, avoiding a full-resolution
        (H, W, 3) intermediate allocation.

        Args:
            depth_m: Depth image in meters
            mask: Optional binary mask (1 = include)
            apply_tilt: If True, rotate point cloud to gravity-aligned coords
            subsample: Subsample factor (1 = no subsampling, 2 = every 2nd pixel)
            return_depths: If True, also return original depth values

        Returns:
            (N, 3) array of 3D points, or ((N, 3), (N,)) if return_depths=True
        """
        _empty = np.zeros((0, 3), dtype=np.float32)

        # Subsample depth and mask
        depth_sub = depth_m[::subsample, ::subsample]
        if mask is not None:
            mask_sub = mask[::subsample, ::subsample]
            valid = (mask_sub > 0) & (depth_sub > 0)
        else:
            valid = depth_sub > 0

        if not np.any(valid):
            if return_depths:
                return _empty, np.zeros(0, dtype=np.float32)
            return _empty

        # Only compute coordinates for valid (foreground) pixels — avoids
        # creating a full (H', W', 3) intermediate array.
        vi, ui = np.where(valid)                         # row, col indices
        u_px = ui.astype(np.float32) * subsample         # back to original pixel coords
        v_px = vi.astype(np.float32) * subsample
        z = depth_sub[vi, ui]

        x = (u_px - self.config.cx) * z / self.config.fx
        y = (v_px - self.config.cy) * z / self.config.fy
        points = np.stack([x, y, z], axis=-1)            # (N, 3) — foreground only

        # Apply tilt correction
        if apply_tilt and len(points) > 0:
            points = (points @ self._tilt_rotation.T).astype(np.float32)

        if return_depths:
            return points, z
        return points
    
    def compute_foreground_center(self, points: np.ndarray) -> np.ndarray:
        """
        Compute centroid of foreground point cloud.
        
        Returns:
            (3,) center position
        """
        if len(points) == 0:
            return np.zeros(3, dtype=np.float32)
        
        # Use median for robustness to outliers
        return np.median(points, axis=0).astype(np.float32)
    
    def compute_occupancy_person_centric(
        self, 
        points: np.ndarray,
        center: np.ndarray,
        depths: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute voxel occupancy grid centered on person.
        
        Args:
            points: (N, 3) array of 3D points (gravity-aligned)
            center: (3,) person center position
            depths: (N,) original depth values for weighting (optional)
        
        Returns:
            Flattened occupancy grid (grid_x * grid_y * grid_z,)
        """
        if len(points) == 0:
            return np.zeros(np.prod(self._grid_size), dtype=np.float32)
        
        # Compute bounds relative to person center
        x_min = center[0] - self._extent_x
        x_max = center[0] + self._extent_x
        y_min = center[1] - self._extent_y
        y_max = center[1] + self._extent_y
        z_min = center[2] - self._extent_z
        z_max = center[2] + self._extent_z
        
        # Voxel size
        voxel_size = np.array([
            (x_max - x_min) / self._grid_size[0],
            (y_max - y_min) / self._grid_size[1],
            (z_max - z_min) / self._grid_size[2],
        ], dtype=np.float32)
        
        # Compute voxel indices for each point
        voxel_indices = np.zeros((len(points), 3), dtype=np.int32)
        voxel_indices[:, 0] = np.floor(
            (points[:, 0] - x_min) / voxel_size[0]
        ).astype(np.int32)
        voxel_indices[:, 1] = np.floor(
            (points[:, 1] - y_min) / voxel_size[1]
        ).astype(np.int32)
        voxel_indices[:, 2] = np.floor(
            (points[:, 2] - z_min) / voxel_size[2]
        ).astype(np.int32)
        
        # Filter out-of-bounds points
        valid = (
            (voxel_indices[:, 0] >= 0) & (voxel_indices[:, 0] < self._grid_size[0]) &
            (voxel_indices[:, 1] >= 0) & (voxel_indices[:, 1] < self._grid_size[1]) &
            (voxel_indices[:, 2] >= 0) & (voxel_indices[:, 2] < self._grid_size[2])
        )
        voxel_indices = voxel_indices[valid]
        # Create occupancy grid
        occupancy = np.zeros(self._grid_size, dtype=np.float32)
        if len(voxel_indices) > 0:
            # Compute weights based on depth if enabled
            if self.config.voxel_depth_weighted and depths is not None:
                depths_valid = depths[valid]
                # Inverse depth weighting: closer = higher weight
                # Normalize relative to person center depth
                center_depth = center[2]
                # Weight = 1 + (center_depth - depth) / center_depth
                # Objects closer than center get weight > 1
                # Objects further get weight < 1 (but always positive)
                weights = np.clip(1.0 + (center_depth - depths_valid) / max(center_depth, 0.5), 0.5, 3.0)
            else:
                weights = np.ones(len(voxel_indices), dtype=np.float32)
            
            # Aggregate weighted contributions to voxels using bincount
            flat_indices = (
                voxel_indices[:, 0] * (self._grid_size[1] * self._grid_size[2]) +
                voxel_indices[:, 1] * self._grid_size[2] +
                voxel_indices[:, 2]
            )
            weighted_counts = np.bincount(flat_indices, weights=weights, minlength=np.prod(self._grid_size))
            occupancy = weighted_counts.reshape(self._grid_size).astype(np.float32)
            
            # Normalize to [0, 1]
            max_val = occupancy.max()
            if max_val > 0:
                occupancy = occupancy / max_val
        
        return occupancy.flatten()
    
    def compute_occupancy(self, points: np.ndarray) -> np.ndarray:
        """
        Compute voxel occupancy grid from point cloud (fixed bounds).
        
        Args:
            points: (N, 3) array of 3D points
        
        Returns:
            Flattened occupancy grid (grid_x * grid_y * grid_z,)
        """
        if len(points) == 0:
            return np.zeros(np.prod(self._grid_size), dtype=np.float32)
        
        # Voxel size
        voxel_size = np.array([
            (self._x_range[1] - self._x_range[0]) / self._grid_size[0],
            (self._y_range[1] - self._y_range[0]) / self._grid_size[1],
            (self._z_range[1] - self._z_range[0]) / self._grid_size[2],
        ], dtype=np.float32)
        
        # Compute voxel indices for each point
        voxel_indices = np.zeros((len(points), 3), dtype=np.int32)
        voxel_indices[:, 0] = np.floor(
            (points[:, 0] - self._x_range[0]) / voxel_size[0]
        ).astype(np.int32)
        voxel_indices[:, 1] = np.floor(
            (points[:, 1] - self._y_range[0]) / voxel_size[1]
        ).astype(np.int32)
        voxel_indices[:, 2] = np.floor(
            (points[:, 2] - self._z_range[0]) / voxel_size[2]
        ).astype(np.int32)
        
        # Filter out-of-bounds points
        valid = (
            (voxel_indices[:, 0] >= 0) & (voxel_indices[:, 0] < self._grid_size[0]) &
            (voxel_indices[:, 1] >= 0) & (voxel_indices[:, 1] < self._grid_size[1]) &
            (voxel_indices[:, 2] >= 0) & (voxel_indices[:, 2] < self._grid_size[2])
        )
        voxel_indices = voxel_indices[valid]
        
        # Create occupancy grid
        occupancy = np.zeros(self._grid_size, dtype=np.float32)
        if len(voxel_indices) > 0:
            # Use bincount for efficiency
            flat_indices = (
                voxel_indices[:, 0] * (self._grid_size[1] * self._grid_size[2]) +
                voxel_indices[:, 1] * self._grid_size[2] +
                voxel_indices[:, 2]
            )
            counts = np.bincount(flat_indices, minlength=np.prod(self._grid_size))
            occupancy = counts.reshape(self._grid_size).astype(np.float32)
            
            # Normalize to [0, 1]
            max_count = occupancy.max()
            if max_count > 0:
                occupancy = occupancy / max_count
        
        return occupancy.flatten()
    
    def extract(
        self,
        depth_m: np.ndarray,
        foreground_mask: np.ndarray,
        return_debug: bool = False,
        bbox_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Extract voxel occupancy delta feature.

        Args:
            depth_m: Depth image in meters
            foreground_mask: Binary foreground mask
            return_debug: If True, populate self.last_debug_info
            bbox_mask: Optional 2D binary mask from person bounding box.
                       When provided, AND-ed with foreground_mask to eliminate
                       background people before point cloud creation.

        Returns:
            Voxel occupancy delta (flattened grid)
        """
        debug_info = ExtractionDebugInfo() if return_debug else None

        # Combine foreground mask with optional bbox mask to isolate target person
        combined_mask = foreground_mask
        if bbox_mask is not None:
            combined_mask = foreground_mask & bbox_mask

        # Get foreground point cloud (gravity-aligned) with depths for weighting
        if self.config.voxel_depth_weighted:
            points, depths = self.deproject_to_pointcloud(
                depth_m, combined_mask, apply_tilt=True, return_depths=True
            )
        else:
            points = self.deproject_to_pointcloud(depth_m, combined_mask, apply_tilt=True)
            depths = None
        
        if return_debug:
            # Also get camera-space points for visualization
            points_cam = self.deproject_to_pointcloud(depth_m, combined_mask, apply_tilt=False, subsample=4)
            points_grav = self.deproject_to_pointcloud(depth_m, combined_mask, apply_tilt=True, subsample=4)
            debug_info.points_camera = points_cam
            debug_info.points_gravity = points_grav
            debug_info.tilt_rotation = self._tilt_rotation.copy()
            debug_info.foreground_mask = combined_mask.copy()
            debug_info.foreground_ratio = np.mean(combined_mask) if combined_mask is not None else 0
        
        # Compute person center
        center = self.compute_foreground_center(points)
        
        # Use smoothed center for stability
        if len(self._center_history) > 0:
            # Exponential smoothing
            alpha = 0.3
            smoothed_center = alpha * center + (1 - alpha) * self._center_history[-1]
        else:
            smoothed_center = center
        
        if return_debug:
            debug_info.voxel_center = smoothed_center.copy()
            debug_info.voxel_bounds_min = np.array([
                smoothed_center[0] - self._extent_x,
                smoothed_center[1] - self._extent_y,
                smoothed_center[2] - self._extent_z,
            ])
            debug_info.voxel_bounds_max = np.array([
                smoothed_center[0] + self._extent_x,
                smoothed_center[1] + self._extent_y,
                smoothed_center[2] + self._extent_z,
            ])
        
        # Compute current occupancy
        if self.config.voxel_person_centric:
            current_occ = self.compute_occupancy_person_centric(points, smoothed_center, depths)
        else:
            current_occ = self.compute_occupancy(points)
        
        grid_size = self.config.voxel_grid_size
        if isinstance(grid_size, int):
            grid_size = (grid_size, grid_size, grid_size)

        if return_debug:
            debug_info.voxel_current = current_occ.reshape(grid_size)
        
        # Add to history
        self._history.append(current_occ)
        self._center_history.append(smoothed_center)
        if len(self._history) > self._max_history:
            self._history.pop(0)
            self._center_history.pop(0)
        
        # Compute delta (change from N frames ago)
        # Determine which lookback distances to use.
        delta_frame_list = list(self.config.multi_scale_delta_frames) if self.config.multi_scale_delta_frames else [self.config.voxel_delta_frames]

        all_deltas = []
        for delta_idx in delta_frame_list:
            if len(self._history) > delta_idx:
                past_occ = self._history[-(delta_idx + 1)]
                single_delta = current_occ - past_occ

                if return_debug and delta_idx == delta_frame_list[0]:
                    debug_info.voxel_previous = past_occ.reshape(grid_size)
            else:
                single_delta = current_occ

            if self.config.directional_gradients:
                # Reshape to 3D grid for spatial gradient computation.
                delta_3d = single_delta.reshape(grid_size)
                # Central differences along each spatial axis.
                grad_x = np.roll(delta_3d, -1, axis=0) - np.roll(delta_3d, 1, axis=0)
                grad_y = np.roll(delta_3d, -1, axis=1) - np.roll(delta_3d, 1, axis=1)
                grad_z = np.roll(delta_3d, -1, axis=2) - np.roll(delta_3d, 1, axis=2)
                # Base 4-channel: [scalar_delta, grad_x, grad_y, grad_z]
                channels = [delta_3d, grad_x, grad_y, grad_z]
                # Optional 5th channel: velocity magnitude (abs_delta).
                # High where anything is moving (glove), near-zero on static torso.
                if self.config.velocity_magnitude_channel:
                    channels.append(np.abs(delta_3d))
                multi_channel = np.stack(channels, axis=0)  # (4 or 5, N, N, N)
                all_deltas.append(multi_channel.flatten())
            else:
                all_deltas.append(single_delta)

        # Prepend raw occupancy channel if configured.
        if self.config.include_raw_occupancy:
            all_deltas.insert(0, current_occ)

        # Concatenate across channels (raw_occ + deltas at each scale).
        combined = np.concatenate(all_deltas) if len(all_deltas) > 1 else all_deltas[0]

        if return_debug:
            self.last_debug_info = debug_info

        return combined
    
    def reset(self):
        """Reset history buffer."""
        self._history = []
        self._center_history = []


__all__ = [
    "VoxelFeatureConfig",
    "build_tilt_rotation_matrix",
    "ExtractionDebugInfo",
    "BackgroundModel",
    "VoxelOccupancyExtractor",
]
