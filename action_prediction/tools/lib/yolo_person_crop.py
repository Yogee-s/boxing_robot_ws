#!/usr/bin/env python3
"""
YOLO Person Cropping Utility for RGB-D Boxing Action Recognition.

Crops the human region from RGB-D frames using YOLO object detection,
reducing background noise and focusing the model on the boxer.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Optional, Union

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


class YOLOPersonCrop:
    """
    Crops person region from RGB-D frames using YOLO detection.
    
    Features:
    - Detects person bounding box using YOLO
    - Crops both RGB and depth to the same region
    - Returns square crops (padded if needed)
    - Temporal smoothing for stable bounding boxes
    - Handles missing detections gracefully
    """
    
    def __init__(
        self,
        model_path: str = 'checkpoints/yolo26m.pt',
        device: str = 'cuda:0',
        conf_threshold: float = 0.5,
        padding_ratio: float = 0.15,
        temporal_smoothing: float = 0.3,
        detect_interval: int = 1,
    ):
        """
        Initialize YOLO person cropper.
        
        Args:
            model_path: Path to YOLO model weights
            device: Device to run inference on
            conf_threshold: Minimum confidence for person detection
            padding_ratio: Extra padding around detected person (0.15 = 15%)
            temporal_smoothing: Smoothing factor for bounding box (0 = no smoothing, 1 = full smoothing)
            detect_interval: Run detection every N frames (1 = every frame)
        """
        if YOLO is None:
            raise ImportError("ultralytics not installed. Run: pip install ultralytics")
        
        self.device = device
        self.conf_threshold = conf_threshold
        self.padding_ratio = padding_ratio
        self.temporal_smoothing = temporal_smoothing
        self.detect_interval = detect_interval
        self.frame_count = 0
        
        # Load YOLO model
        self.model = YOLO(model_path)
        self.model.to(device)
        
        
        # Disable saving/logging to prevent 'runs/' folder creation
        # We redirect to a temp dir just in case it creates it anyway
        import tempfile
        self.temp_dir = tempfile.mkdtemp(prefix='yolo_runs_')
        self.model.overrides['save'] = False
        self.model.overrides['project'] = self.temp_dir
        
        # Previous bounding box for temporal smoothing
        self._prev_bbox: Optional[np.ndarray] = None
        
        # COCO class ID for 'person' is 0
        self.person_class_id = 0
    
    def reset(self):
        """Reset temporal state (call when starting a new video/clip)."""
        self._prev_bbox = None
        self.frame_count = 0
    
    def _detect_batch(self, rgb_list: list) -> list:
        """
        Detect person in a batch of RGB frames.
        
        Args:
            rgb_list: List of RGB images (H, W, 3)
            
        Returns:
            List of (bounding box [x1, y1, x2, y2] or None)
        """
        if not rgb_list:
            return []
            
        # Run YOLO inference on batch
        # verbose=False, save=False, project=None, name=None to prevent dirs
        results = self.model(
            rgb_list, 
            verbose=False, 
            conf=self.conf_threshold, 
            save=False,
            project=self.temp_dir,
            name=None
        )
        
        bboxes = []
        for res in results:
            if len(res) == 0 or res.boxes is None:
                bboxes.append(None)
                continue
                
            boxes = res.boxes
            person_mask = boxes.cls == self.person_class_id
            
            if not person_mask.any():
                bboxes.append(None)
                continue
            
            # Get person boxes
            person_boxes = boxes.xyxy[person_mask].cpu().numpy()
            person_confs = boxes.conf[person_mask].cpu().numpy()
            
            if len(person_boxes) == 0:
                bboxes.append(None)
            else:
                best_idx = np.argmax(person_confs)
                bboxes.append(person_boxes[best_idx])
                
        return bboxes

    def _detect_person(self, rgb: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect person in RGB frame.
        
        Args:
            rgb: RGB image (H, W, 3) uint8
            
        Returns:
            Bounding box [x1, y1, x2, y2] or None if no person detected
        """
        # Run YOLO inference (save=False prevents runs/ folder... hopefully)
        # Note: setting project/name to None typically defaults to runs/detect, 
        # but combined with save=False it minimizes artifacts.
        results = self.model(
            rgb, 
            verbose=False, 
            conf=self.conf_threshold, 
            save=False,
            project=self.temp_dir,
            name=None
        )
        
        if len(results) == 0 or results[0].boxes is None:
            return None
        
        boxes = results[0].boxes
        
        # Filter for person class
        person_mask = boxes.cls == self.person_class_id
        if not person_mask.any():
            return None
        
        # Get person boxes
        person_boxes = boxes.xyxy[person_mask].cpu().numpy()
        person_confs = boxes.conf[person_mask].cpu().numpy()
        
        if len(person_boxes) == 0:
            return None
        
        # Select the person with highest confidence
        best_idx = np.argmax(person_confs)
        bbox = person_boxes[best_idx]  # [x1, y1, x2, y2]
        
        return bbox
    
    def _smooth_bbox(self, bbox: np.ndarray) -> np.ndarray:
        """Apply temporal smoothing to bounding box."""
        if self._prev_bbox is None:
            self._prev_bbox = bbox.copy()
            return bbox
        
        # Exponential moving average
        smoothed = (1 - self.temporal_smoothing) * bbox + self.temporal_smoothing * self._prev_bbox
        self._prev_bbox = smoothed.copy()
        return smoothed
    
    def _make_square_with_padding(
        self,
        bbox: np.ndarray,
        img_h: int,
        img_w: int,
    ) -> np.ndarray:
        """
        Convert bbox to square and add padding, clamping to image bounds.
        
        Args:
            bbox: [x1, y1, x2, y2]
            img_h, img_w: Image dimensions
            
        Returns:
            Square bbox [x1, y1, x2, y2] with padding
        """
        x1, y1, x2, y2 = bbox
        
        # Current dimensions
        w = x2 - x1
        h = y2 - y1
        
        # Make square (use larger dimension)
        size = max(w, h)
        
        # Add padding
        size = size * (1 + self.padding_ratio)
        
        # Center of current bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        
        # New square bbox
        half_size = size / 2
        x1_new = cx - half_size
        y1_new = cy - half_size
        x2_new = cx + half_size
        y2_new = cy + half_size
        
        # Clamp to image bounds
        x1_new = max(0, x1_new)
        y1_new = max(0, y1_new)
        x2_new = min(img_w, x2_new)
        y2_new = min(img_h, y2_new)
        
        return np.array([x1_new, y1_new, x2_new, y2_new])
    
    def _crop_and_resize(
        self,
        img: np.ndarray,
        bbox: np.ndarray,
        output_size: int,
        is_depth: bool = False,
    ) -> np.ndarray:
        """
        Crop image to bbox and resize to output_size.
        
        Args:
            img: Input image (H, W, C) or (H, W) for depth
            bbox: [x1, y1, x2, y2]
            output_size: Output dimension (square)
            is_depth: If True, use nearest neighbor interpolation
            
        Returns:
            Cropped and resized image
        """
        x1, y1, x2, y2 = bbox.astype(int)
        h, w = img.shape[:2]
        
        # Ensure bounds are valid
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        # Crop
        if img.ndim == 3:
            crop = img[y1:y2, x1:x2, :]
        else:
            crop = img[y1:y2, x1:x2]
        
        # Handle edge case where crop is empty
        if crop.size == 0:
            if img.ndim == 3:
                return np.zeros((output_size, output_size, img.shape[2]), dtype=img.dtype)
            else:
                return np.zeros((output_size, output_size), dtype=img.dtype)
        
        # Resize
        interpolation = cv2.INTER_NEAREST if is_depth else cv2.INTER_LINEAR
        resized = cv2.resize(crop, (output_size, output_size), interpolation=interpolation)
        
        return resized
    
    def __call__(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        output_size: int = 224,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Crop person region from RGB-D frame.
        
        Args:
            rgb: RGB image (H, W, 3) uint8
            depth: Depth map (H, W) float32 in meters
            output_size: Output crop size (square)
            
        Returns:
            rgb_crop: Cropped RGB (output_size, output_size, 3)
            depth_crop: Cropped depth (output_size, output_size)
            bbox: Detected bounding box [x1, y1, x2, y2] or None
        """
        h, w = rgb.shape[:2]
        
        # Detect person (with frame skipping)
        should_detect = (self.frame_count % self.detect_interval == 0)
        self.frame_count += 1
        
        bbox = None
        if should_detect:
            bbox = self._detect_person(rgb)
        
        if bbox is None:
            # No detection - use previous bbox or center crop
            if self._prev_bbox is not None:
                bbox = self._prev_bbox.copy()
            else:
                # Center crop fallback
                size = min(h, w) * 0.8
                cx, cy = w / 2, h / 2
                bbox = np.array([cx - size/2, cy - size/2, cx + size/2, cy + size/2])
        else:
            # Apply temporal smoothing
            bbox = self._smooth_bbox(bbox)
        
        # Make square with padding
        bbox = self._make_square_with_padding(bbox, h, w)
        
        # Crop and resize
        rgb_crop = self._crop_and_resize(rgb, bbox, output_size, is_depth=False)
        depth_crop = self._crop_and_resize(depth, bbox, output_size, is_depth=True)
        
        return rgb_crop, depth_crop, bbox
    
    def crop_batch(
        self,
        rgb_frames: list,
        depth_frames: list,
        output_size: int = 224,
    ) -> Tuple[np.ndarray, np.ndarray, list]:
        """
        Crop person region from a batch of RGB-D frames.
        
        Resets temporal state before processing and uses consistent
        bounding boxes across the batch for stability.
        
        Args:
            rgb_frames: List of RGB images (H, W, 3)
            depth_frames: List of depth maps (H, W)
            output_size: Output crop size
            
        Returns:
            rgb_crops: (N, output_size, output_size, 3) uint8
            depth_crops: (N, output_size, output_size) float32
            bboxes: List of bounding boxes
        """
        # Note: We do NOT reset here anymore to allow stateful batch processing
        # self.reset() should be called manually by consumer at start of clip
        
        # 1. Run detection on entire batch (Much faster than loop)
        batch_bboxes = self._detect_batch(rgb_frames)
        
        rgb_crops = []
        depth_crops = []
        final_bboxes = []
        
        h, w = rgb_frames[0].shape[:2]
        
        # 2. Process results sequentially (needed for temporal smoothing)
        for i, (rgb, depth) in enumerate(zip(rgb_frames, depth_frames)):
            bbox = batch_bboxes[i]
            
            if bbox is None:
                # No detection - use previous bbox or center crop
                if self._prev_bbox is not None:
                    bbox = self._prev_bbox.copy()
                else:
                    # Center crop fallback
                    size = min(h, w) * 0.8
                    cx, cy = w / 2, h / 2
                    bbox = np.array([cx - size/2, cy - size/2, cx + size/2, cy + size/2])
            else:
                # Apply temporal smoothing
                bbox = self._smooth_bbox(bbox)
            
            # Make square with padding
            bbox = self._make_square_with_padding(bbox, h, w)
            
            # Crop and resize
            rgb_crop = self._crop_and_resize(rgb, bbox, output_size, is_depth=False)
            depth_crop = self._crop_and_resize(depth, bbox, output_size, is_depth=True)
            
            rgb_crops.append(rgb_crop)
            depth_crops.append(depth_crop)
            final_bboxes.append(bbox)
        
        return np.stack(rgb_crops), np.stack(depth_crops), final_bboxes


def normalize_depth(depth: np.ndarray, max_depth: float = 4.0) -> np.ndarray:
    """
    Normalize depth values to [0, 1] range.
    
    Args:
        depth: Depth map in meters (H, W) or (N, H, W)
        max_depth: Maximum depth value for clipping
        
    Returns:
        Normalized depth in [0, 1] range
    """
    depth_clipped = np.clip(depth, 0, max_depth)
    depth_norm = depth_clipped / max_depth
    return depth_norm.astype(np.float32)


def create_rgbd_tensor(
    rgb: np.ndarray,
    depth: np.ndarray,
    max_depth: float = 4.0,
) -> np.ndarray:
    """
    Create 4-channel RGB-D tensor from RGB and depth.
    
    Args:
        rgb: RGB image (H, W, 3) uint8 or (N, H, W, 3)
        depth: Depth map (H, W) or (N, H, W) in meters
        max_depth: Maximum depth for normalization
        
    Returns:
        RGB-D tensor (H, W, 4) or (N, H, W, 4) float32 normalized to [0, 1]
    """
    # Normalize RGB to [0, 1]
    rgb_norm = rgb.astype(np.float32) / 255.0
    
    # Normalize depth to [0, 1]
    depth_norm = normalize_depth(depth, max_depth)
    
    # Expand depth dims if needed
    if depth_norm.ndim == 2:
        depth_norm = depth_norm[..., np.newaxis]
    elif depth_norm.ndim == 3 and rgb_norm.ndim == 4:
        depth_norm = depth_norm[..., np.newaxis]
    
    # Concatenate
    rgbd = np.concatenate([rgb_norm, depth_norm], axis=-1)
    
    return rgbd


if __name__ == '__main__':
    # Quick test
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='checkpoints/yolo26m.pt')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    
    # Create cropper
    cropper = YOLOPersonCrop(args.model, args.device)
    
    # Test with dummy data
    rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    depth = np.random.rand(480, 640).astype(np.float32) * 4.0
    
    rgb_crop, depth_crop, bbox = cropper(rgb, depth, output_size=224)
    
    print(f"RGB crop shape: {rgb_crop.shape}")
    print(f"Depth crop shape: {depth_crop.shape}")
    print(f"Bounding box: {bbox}")
    
    # Test RGBD tensor creation
    rgbd = create_rgbd_tensor(rgb_crop, depth_crop)
    print(f"RGBD tensor shape: {rgbd.shape}")
    print(f"RGBD tensor range: [{rgbd.min():.3f}, {rgbd.max():.3f}]")
    
    print("✅ YOLO person crop utility works!")
