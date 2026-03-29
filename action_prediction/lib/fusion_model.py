"""
Fusion model: 2D Pose + Depth Voxel for boxing action recognition.

Architecture:
    VOXEL BRANCH  Conv3DStem(2ch, 12^3) -> 192-dim per-frame embedding
    POSE BRANCH   PoseEncoder(42 -> 64) with confidence gating
    FUSION        concat(192, 64) = 256 -> Linear(256, 192) + LayerNorm
    TEMPORAL      CausalTransformerEncoder (d=192, 4 layers, 8 heads)
    CLASSIFIER    2-layer head -> num_classes

Voxel channels: 2 (occupancy delta at fast 4-frame + slow 16-frame scales).
Gradients and |delta| are dropped because pose provides explicit directional
and structural information.  This reduces per-frame voxel features from
17,280 to 3,456 dims, cutting extraction and inference cost ~5x.

When pose is unavailable (all zeros), the model degrades gracefully to
voxel-only performance because the pose branch output is zero and the
fusion projection learns to pass voxel information through.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from action_prediction.lib.voxel_model import Conv3DStem, PositionalEncoding


# ---------------------------------------------------------------------------
# Pose feature extraction (NumPy, runs on CPU during data processing)
# ---------------------------------------------------------------------------

# COCO-17 joint indices used for boxing
JOINT_NOSE = 0
JOINT_L_SHOULDER = 5
JOINT_R_SHOULDER = 6
JOINT_L_ELBOW = 7
JOINT_R_ELBOW = 8
JOINT_L_WRIST = 9
JOINT_R_WRIST = 10
JOINT_L_HIP = 11
JOINT_R_HIP = 12

USED_JOINTS: List[int] = [
    JOINT_NOSE,
    JOINT_L_SHOULDER, JOINT_R_SHOULDER,
    JOINT_L_ELBOW, JOINT_R_ELBOW,
    JOINT_L_WRIST, JOINT_R_WRIST,
]
NUM_USED_JOINTS = len(USED_JOINTS)  # 7

# Static pose: 14 (coords) + 7 (conf) + 2 (arm ext) + 1 (shoulder rot) + 2 (elbow angles) = 26
POSE_STATIC_DIM = NUM_USED_JOINTS * 2 + NUM_USED_JOINTS + 2 + 1 + 2  # 26
# Temporal pose: 14 (joint velocities) + 2 (arm ext rate) = 16
POSE_VELOCITY_DIM = NUM_USED_JOINTS * 2 + 2  # 16
# Total pose feature dimension
POSE_FEATURE_DIM = POSE_STATIC_DIM + POSE_VELOCITY_DIM  # 42


def extract_pose_features_static(
    keypoints_xy: np.ndarray,
    confidences: np.ndarray,
    min_shoulder_px: float = 15.0,
) -> np.ndarray:
    """Extract 26-dim static pose feature vector from COCO-17 keypoints.

    Layout: [14 coords | 7 conf | 2 arm_ext | 1 shoulder_rot | 2 elbow_angles]

    Args:
        keypoints_xy: (17, 2) pixel coordinates from YOLO Pose.
        confidences: (17,) per-joint confidence scores.
        min_shoulder_px: Minimum shoulder width in pixels to trust the detection.

    Returns:
        (26,) feature vector.  All zeros when detection is invalid.
    """
    out = np.zeros(POSE_STATIC_DIM, dtype=np.float32)

    if keypoints_xy is None or confidences is None:
        return out
    if keypoints_xy.shape[0] < 13 or confidences.shape[0] < 13:
        return out

    # Selected joints
    joints_xy = keypoints_xy[USED_JOINTS].astype(np.float32)   # (7, 2)
    confs = confidences[USED_JOINTS].astype(np.float32)         # (7,)

    # Shoulder midpoint as reference origin
    l_sh = keypoints_xy[JOINT_L_SHOULDER].astype(np.float32)
    r_sh = keypoints_xy[JOINT_R_SHOULDER].astype(np.float32)
    shoulder_mid = (l_sh + r_sh) * 0.5

    # Shoulder width as scale reference
    shoulder_width = float(np.linalg.norm(r_sh - l_sh))
    if shoulder_width < min_shoulder_px:
        return out

    # 1) Normalized (x, y) coordinates: 7 joints x 2 = 14 dims
    norm_coords = (joints_xy - shoulder_mid) / shoulder_width   # (7, 2)

    # 2) Confidence scores: 7 dims (indices 14:21)

    # 3) Arm extension ratios: 2 dims (left, right)
    l_wrist = keypoints_xy[JOINT_L_WRIST].astype(np.float32)
    r_wrist = keypoints_xy[JOINT_R_WRIST].astype(np.float32)
    l_elbow = keypoints_xy[JOINT_L_ELBOW].astype(np.float32)
    r_elbow = keypoints_xy[JOINT_R_ELBOW].astype(np.float32)
    l_arm_ext = float(np.linalg.norm(l_wrist - l_sh)) / shoulder_width
    r_arm_ext = float(np.linalg.norm(r_wrist - r_sh)) / shoulder_width

    # 4) Shoulder rotation: 1 dim
    shoulder_rot = (r_sh[0] - l_sh[0]) / shoulder_width

    # 5) Elbow angles: 2 dims (0=fully bent, pi=fully straight)
    #    Angle at elbow between upper arm (shoulder→elbow) and forearm (elbow→wrist)
    def _elbow_angle(shoulder, elbow, wrist):
        v1 = shoulder - elbow
        v2 = wrist - elbow
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            return 0.0
        cos_angle = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        return float(np.arccos(cos_angle)) / np.pi  # Normalize to [0, 1]

    l_elbow_angle = _elbow_angle(l_sh, l_elbow, l_wrist)
    r_elbow_angle = _elbow_angle(r_sh, r_elbow, r_wrist)

    # Pack features
    out[:14] = norm_coords.flatten()
    out[14:21] = confs
    out[21] = l_arm_ext
    out[22] = r_arm_ext
    out[23] = shoulder_rot
    out[24] = l_elbow_angle
    out[25] = r_elbow_angle

    return out


def compute_pose_velocity(
    current_static: np.ndarray,
    prev_static: np.ndarray,
) -> np.ndarray:
    """Compute 16-dim velocity features from two consecutive static pose frames.

    Layout: [14 joint velocities (dx,dy per joint) | 2 arm extension rate]

    Returns:
        (16,) velocity vector. Zeros if either input is all-zero (invalid).
    """
    vel = np.zeros(POSE_VELOCITY_DIM, dtype=np.float32)

    # If either frame is invalid (all zeros), return zero velocity
    if np.abs(current_static[:14]).sum() < 1e-6 or np.abs(prev_static[:14]).sum() < 1e-6:
        return vel

    # Joint coordinate velocities: current_coords - prev_coords
    vel[:14] = current_static[:14] - prev_static[:14]
    # Arm extension rate: current_ext - prev_ext
    vel[14] = current_static[21] - prev_static[21]  # left arm ext rate
    vel[15] = current_static[22] - prev_static[22]  # right arm ext rate

    return vel


def extract_pose_features(
    keypoints_xy: np.ndarray,
    confidences: np.ndarray,
    prev_static: np.ndarray = None,
    min_shoulder_px: float = 15.0,
) -> np.ndarray:
    """Extract 42-dim pose feature vector (26 static + 16 velocity).

    Args:
        keypoints_xy: (17, 2) pixel coordinates from YOLO Pose.
        confidences: (17,) per-joint confidence scores.
        prev_static: (26,) previous frame's static features for velocity.
                     If None, velocity dims are zeros.

    Returns:
        (42,) feature vector.
    """
    static = extract_pose_features_static(keypoints_xy, confidences, min_shoulder_px)

    if prev_static is not None:
        vel = compute_pose_velocity(static, prev_static)
    else:
        vel = np.zeros(POSE_VELOCITY_DIM, dtype=np.float32)

    return np.concatenate([static, vel])


def extract_pose_features_batch(
    keypoints_xy: Optional[np.ndarray],
    confidences: Optional[np.ndarray],
    num_frames: int,
) -> np.ndarray:
    """Extract pose features for a sequence of frames (with velocities).

    Args:
        keypoints_xy: (T, 17, 2) or None
        confidences: (T, 17) or None
        num_frames: expected number of frames T

    Returns:
        (T, 42) pose features (26 static + 16 velocity per frame)
    """
    out = np.zeros((num_frames, POSE_FEATURE_DIM), dtype=np.float32)
    if keypoints_xy is None or confidences is None:
        return out

    T = min(num_frames, keypoints_xy.shape[0], confidences.shape[0])

    # First pass: compute static features for all frames
    statics = np.zeros((T, POSE_STATIC_DIM), dtype=np.float32)
    for t in range(T):
        statics[t] = extract_pose_features_static(keypoints_xy[t], confidences[t])

    # Second pass: compute velocities and pack full features
    for t in range(T):
        prev = statics[t - 1] if t > 0 else None
        vel = compute_pose_velocity(statics[t], prev) if prev is not None else np.zeros(POSE_VELOCITY_DIM, dtype=np.float32)
        out[t, :POSE_STATIC_DIM] = statics[t]
        out[t, POSE_STATIC_DIM:] = vel

    return out


def flip_pose_features_horizontal(pose_features: np.ndarray) -> np.ndarray:
    """Flip pose features for horizontal voxel flip augmentation.

    Negates x-coordinates and swaps left/right joint pairs.
    Handles both static (26-dim) and full (42-dim) pose features.

    Args:
        pose_features: (..., 42) pose features

    Returns:
        Flipped pose features with same shape.
    """
    out = pose_features.copy()
    dim = out.shape[-1]
    if dim < POSE_STATIC_DIM:
        return out

    # --- Static features (indices 0:26) ---

    # 1) Negate x-coordinates (indices 0, 2, 4, 6, 8, 10, 12)
    out[..., 0:14:2] *= -1.0

    # 2) Swap left/right joint pairs in coordinates
    #    L_Sh (idx 2,3) <-> R_Sh (idx 4,5)
    #    L_Elb (idx 6,7) <-> R_Elb (idx 8,9)
    #    L_Wr (idx 10,11) <-> R_Wr (idx 12,13)
    for l_start, r_start in [(2, 4), (6, 8), (10, 12)]:
        tmp = out[..., l_start:l_start + 2].copy()
        out[..., l_start:l_start + 2] = out[..., r_start:r_start + 2]
        out[..., r_start:r_start + 2] = tmp

    # 3) Swap left/right confidence scores (indices 14-20)
    for l_idx, r_idx in [(15, 16), (17, 18), (19, 20)]:
        tmp = out[..., l_idx].copy()
        out[..., l_idx] = out[..., r_idx]
        out[..., r_idx] = tmp

    # 4) Swap arm extension ratios: idx 21 (L) <-> 22 (R)
    tmp = out[..., 21].copy()
    out[..., 21] = out[..., 22]
    out[..., 22] = tmp

    # 5) Negate shoulder rotation: idx 23
    out[..., 23] *= -1.0

    # 6) Swap elbow angles: idx 24 (L) <-> 25 (R)
    tmp = out[..., 24].copy()
    out[..., 24] = out[..., 25]
    out[..., 25] = tmp

    # --- Velocity features (indices 26:42) ---
    if dim >= POSE_FEATURE_DIM:
        v_off = POSE_STATIC_DIM  # 26

        # 7) Negate velocity x-components (indices v_off+0, +2, +4, +6, +8, +10, +12)
        out[..., v_off:v_off + 14:2] *= -1.0

        # 8) Swap left/right joint velocity pairs
        for l_start, r_start in [(2, 4), (6, 8), (10, 12)]:
            tmp = out[..., v_off + l_start:v_off + l_start + 2].copy()
            out[..., v_off + l_start:v_off + l_start + 2] = out[..., v_off + r_start:v_off + r_start + 2]
            out[..., v_off + r_start:v_off + r_start + 2] = tmp

        # 9) Swap arm extension rate: idx v_off+14 (L) <-> v_off+15 (R)
        tmp = out[..., v_off + 14].copy()
        out[..., v_off + 14] = out[..., v_off + 15]
        out[..., v_off + 15] = tmp

    return out


# Label swap map for horizontal flip augmentation
# left_hook <-> right_hook, left_uppercut <-> right_uppercut, jab <-> cross
FLIP_LABEL_MAP = {
    "jab": "cross",
    "cross": "jab",
    "left_hook": "right_hook",
    "right_hook": "left_hook",
    "left_uppercut": "right_uppercut",
    "right_uppercut": "left_uppercut",
    "block": "block",
    "idle": "idle",
}


def build_flip_label_indices(label_map: Dict[str, int]) -> Dict[int, int]:
    """Build integer label index swap map for horizontal flip augmentation.

    Args:
        label_map: {class_name: class_index}

    Returns:
        {old_index: new_index} mapping
    """
    idx_map = {}
    for name, idx in label_map.items():
        flipped_name = FLIP_LABEL_MAP.get(name, name)
        if flipped_name in label_map:
            idx_map[idx] = label_map[flipped_name]
        else:
            idx_map[idx] = idx
    return idx_map


# ---------------------------------------------------------------------------
# PyTorch modules
# ---------------------------------------------------------------------------

class PoseEncoder(nn.Module):
    """Lightweight MLP encoder for normalized pose features.

    Includes confidence-based gating: the output embedding is scaled by the
    mean confidence of visible joints so that noisy/missing pose detections
    are automatically suppressed.
    """

    def __init__(
        self,
        pose_dim: int = POSE_FEATURE_DIM,
        embed_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pose_dim = pose_dim
        self.embed_dim = embed_dim

        self.mlp = nn.Sequential(
            nn.Linear(pose_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Confidence scores are at indices 14:21 of the RAW pose (first 24 dims)
        self._conf_start = NUM_USED_JOINTS * 2   # 14
        self._conf_end = self._conf_start + NUM_USED_JOINTS  # 21

    def forward(self, pose_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pose_features: (batch, seq_len, pose_dim)
                Raw 24-dim pose features. Velocity is computed internally.

        Returns:
            (batch, seq_len, embed_dim) with confidence gating applied
        """
        # Extract mean confidence for gating (from raw pose, before velocity)
        confs = pose_features[..., self._conf_start:self._conf_end]  # (B, T, 7)
        mean_conf = confs.mean(dim=-1, keepdim=True)  # (B, T, 1)

        emb = self.mlp(pose_features)  # (B, T, embed_dim)

        # Gate by mean confidence: noisy frames get suppressed
        return emb * mean_conf


class FusionVoxelPoseTransformerModel(nn.Module):
    """Causal transformer with fused voxel + pose input.

    Architecture:
        Voxel branch:  Conv3DStem(in_channels, grid_size) -> d_model dim
                       When dual_voxel_stem=True and in_channels=2:
                         Stem A (ch0, occupancy) -> d_model//2
                         Stem B (ch1, delta)     -> d_model//2
                         concat -> d_model
        Pose branch:   PoseEncoder(pose_dim -> pose_embed_dim)
        Fusion:        concat -> Linear(d_model + pose_embed_dim, d_model) + LN
        Temporal:      CausalTransformerEncoder
        Classifier:    2-layer head -> num_classes

    When all pose features are zero (missing detection or dropout), the pose
    branch outputs zero and the fusion layer passes voxel information through.
    """

    def __init__(
        self,
        num_classes: int = 8,
        voxel_size: Tuple[int, int, int] = (12, 12, 12),
        in_channels: int = 2,
        d_model: int = 192,
        num_heads: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 576,
        dropout: float = 0.2,
        max_len: int = 256,
        pose_dim: int = POSE_FEATURE_DIM,
        pose_embed_dim: int = 64,
        pose_dropout: float = 0.1,
        causal: bool = True,
        dual_voxel_stem: bool = False,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.voxel_size = tuple(int(v) for v in voxel_size)
        self.voxel_dim = int(math.prod(self.voxel_size))
        self.in_channels = in_channels
        self.pose_dim = pose_dim
        self.pose_embed_dim = pose_embed_dim
        self.d_model = d_model
        self.causal = causal
        self.dual_voxel_stem = dual_voxel_stem and in_channels == 2

        # Voxel branch
        grid_n = self.voxel_size[0]
        if self.dual_voxel_stem:
            # Separate Conv3D stems for occupancy (ch0) and delta (ch1)
            half_d = d_model // 2
            self.conv3d_stem_occ = Conv3DStem(
                in_channels=1, d_model=half_d, grid_size=grid_n,
            )
            self.conv3d_stem_delta = Conv3DStem(
                in_channels=1, d_model=half_d, grid_size=grid_n,
            )
            self.conv3d_stem = None  # not used
        else:
            self.conv3d_stem = Conv3DStem(
                in_channels=in_channels, d_model=d_model, grid_size=grid_n,
            )

        # Pose branch: MLP with confidence gating -> pose_embed_dim
        self.pose_encoder = PoseEncoder(
            pose_dim=pose_dim,
            embed_dim=pose_embed_dim,
            dropout=pose_dropout,
        )

        # Fusion: concat(d_model, pose_embed_dim) -> d_model
        self.fusion_proj = nn.Sequential(
            nn.Linear(d_model + pose_embed_dim, d_model),
            nn.LayerNorm(d_model),
        )

        # Temporal transformer (causal or bidirectional)
        self.pos_encoder = PositionalEncoding(
            d_model=d_model, max_len=max_len, dropout=dropout,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
        )

        # Classifier head — uses mean+max pooling for full-window context
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, self.num_classes),
        )

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1,
        )

    def _last_valid_token(
        self, x: torch.Tensor, padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if padding_mask is None:
            return x[:, -1, :]
        valid_counts = (~padding_mask).sum(dim=1).clamp(min=1)
        last_idx = (valid_counts - 1).long()
        batch_idx = torch.arange(x.size(0), device=x.device)
        return x[batch_idx, last_idx, :]

    @staticmethod
    def _mean_max_pool(
        x: torch.Tensor, padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Pool via concat(mean, max) over valid timesteps -> (B, 2*d_model)."""
        if padding_mask is None:
            mean_p = x.mean(dim=1)
            max_p = x.max(dim=1).values
        else:
            valid = (~padding_mask).float().unsqueeze(-1)  # (B, T, 1)
            denom = valid.sum(dim=1).clamp(min=1.0)
            mean_p = (x * valid).sum(dim=1) / denom
            masked = x.masked_fill(padding_mask.unsqueeze(-1), float('-inf'))
            max_p = masked.max(dim=1).values
            max_p = torch.where(torch.isfinite(max_p), max_p, torch.zeros_like(max_p))
        return torch.cat([mean_p, max_p], dim=-1)

    def forward(
        self,
        features: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        return_embeddings: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: (B, T, voxel_dim*in_channels + pose_dim)
                      First voxel_dim*in_channels columns are voxel features,
                      last pose_dim columns are pose features.
            padding_mask: (B, T) bool, True = padded timestep
            return_embeddings: if True, include sequence_embedding in output

        Returns:
            dict with 'logits' and optionally 'sequence_embedding'
        """
        B, T, _ = features.shape
        voxel_feat_dim = self.voxel_dim * self.in_channels

        # Split voxel and pose
        voxel_flat = features[:, :, :voxel_feat_dim]
        pose_flat = features[:, :, voxel_feat_dim:voxel_feat_dim + self.pose_dim]

        # Voxel branch: reshape to 5D and encode
        N = self.voxel_size[0]
        C = self.in_channels

        if self.dual_voxel_stem:
            # Split channels and process through separate stems
            voxel_all = voxel_flat.reshape(B * T, C, N, N, N)
            occ_emb = self.conv3d_stem_occ(voxel_all[:, 0:1])    # (B*T, d_model//2)
            delta_emb = self.conv3d_stem_delta(voxel_all[:, 1:2])  # (B*T, d_model//2)
            voxel_emb = torch.cat([occ_emb, delta_emb], dim=-1)   # (B*T, d_model)
        else:
            voxel_5d = voxel_flat.reshape(B * T, C, N, N, N)
            voxel_emb = self.conv3d_stem(voxel_5d)  # (B*T, d_model)
        voxel_emb = voxel_emb.reshape(B, T, -1)  # (B, T, d_model)

        # Pose branch: encode with confidence gating
        pose_emb = self.pose_encoder(pose_flat)  # (B, T, pose_embed_dim)

        # Fusion: concatenate and project
        fused = torch.cat([voxel_emb, pose_emb], dim=-1)  # (B, T, d_model + pose_embed_dim)
        x = self.fusion_proj(fused)  # (B, T, d_model)

        # Temporal: positional encoding + transformer
        x = self.pos_encoder(x)

        attn_mask = self._causal_mask(x.size(1), x.device) if self.causal else None
        attn_padding_mask = padding_mask
        if padding_mask is not None:
            attn_padding_mask = padding_mask.clone()
            attn_padding_mask[:, 0] = False

        x = self.transformer(
            x, mask=attn_mask, src_key_padding_mask=attn_padding_mask,
        )
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        # Pool: mean + max over valid timesteps (captures full action shape)
        pooled = self._mean_max_pool(x, padding_mask)

        # Classify
        logits = self.classifier(pooled)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)

        out: Dict[str, torch.Tensor] = {"logits": logits}
        if return_embeddings:
            out["sequence_embedding"] = pooled
        return out


class VoxelPoseCNNModel(nn.Module):
    """Two-branch CNN model adapted from the 3d_multimodel VoxelFlowModel.

    Each branch has its own temporal Conv1D processing before late fusion.
    Proven architecture that previously achieved best results with glove
    tracking — now adapted for pose features.

    Voxel branch:
        3D CNN per frame -> temporal Conv1D -> pool -> 128d
    Pose branch:
        Per-frame MLP (24 -> 96) -> temporal Conv1D×2 -> pool -> 64d
    Fusion:
        concat(128, 64) = 192d -> MLP(256, 256) -> classifier(8)
    Auxiliary losses on each branch for independent discriminability.
    """

    def __init__(
        self,
        num_classes: int = 8,
        voxel_size: Tuple[int, int, int] = (12, 12, 12),
        in_channels: int = 2,
        voxel_embed_dim: int = 128,
        pose_embed_dim: int = 64,
        fusion_hidden_dim: int = 256,
        pose_dim: int = POSE_FEATURE_DIM,
        dropout: float = 0.3,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.voxel_size = tuple(int(v) for v in voxel_size)
        self.voxel_dim = int(math.prod(self.voxel_size))
        self.in_channels = in_channels
        self.pose_dim = pose_dim
        self.voxel_embed_dim = voxel_embed_dim
        self.pose_embed_dim = pose_embed_dim

        # === Voxel Branch: 3D CNN + temporal Conv1D ===
        self.voxel_conv3d = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((4, 4, 4)),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((2, 2, 2)),
        )
        voxel_flat_dim = 128 * 8  # 128 channels * 2*2*2

        # Project down before temporal conv to reduce params
        self.voxel_spatial_proj = nn.Sequential(
            nn.Linear(voxel_flat_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
        )
        self.voxel_temporal = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.voxel_proj = nn.Sequential(
            nn.Linear(256, voxel_embed_dim),
            nn.LayerNorm(voxel_embed_dim),
            nn.Dropout(dropout),
        )

        # === Pose Branch: MLP + temporal Conv1D ===
        pose_hidden = 96
        self.pose_frame_mlp = nn.Sequential(
            nn.Linear(pose_dim, pose_hidden),
            nn.LayerNorm(pose_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(pose_hidden, pose_hidden),
            nn.LayerNorm(pose_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(pose_hidden, pose_hidden),
            nn.LayerNorm(pose_hidden),
            nn.ReLU(inplace=True),
        )
        self.pose_temporal = nn.Sequential(
            nn.Conv1d(pose_hidden, pose_hidden * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(pose_hidden * 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(pose_hidden * 2, pose_hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(pose_hidden),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.pose_proj = nn.Sequential(
            nn.Linear(pose_hidden, pose_embed_dim),
            nn.LayerNorm(pose_embed_dim),
        )

        # Confidence gating
        self._conf_start = NUM_USED_JOINTS * 2  # 14
        self._conf_end = self._conf_start + NUM_USED_JOINTS  # 21

        # === Fusion: concat + MLP + classifier ===
        total_embed = voxel_embed_dim + pose_embed_dim
        self.fusion = nn.Sequential(
            nn.Linear(total_embed, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(fusion_hidden_dim, num_classes)

        # Auxiliary classifiers (branch-level supervision)
        self.voxel_aux = nn.Linear(voxel_embed_dim, num_classes)
        self.pose_aux = nn.Linear(pose_embed_dim, num_classes)

    def forward(
        self,
        features: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        return_embeddings: bool = False,
    ) -> Dict[str, torch.Tensor]:
        B, T, _ = features.shape
        voxel_feat_dim = self.voxel_dim * self.in_channels

        voxel_flat = features[:, :, :voxel_feat_dim]
        pose_flat = features[:, :, voxel_feat_dim:voxel_feat_dim + self.pose_dim]

        # === Voxel branch ===
        N = self.voxel_size[0]
        C = self.in_channels
        voxel_5d = voxel_flat.reshape(B * T, C, N, N, N)
        voxel_cnn = self.voxel_conv3d(voxel_5d)  # (B*T, 128, 2, 2, 2)
        voxel_cnn = voxel_cnn.reshape(B, T, -1)  # (B, T, 1024)
        voxel_cnn = self.voxel_spatial_proj(voxel_cnn)  # (B, T, 256)
        voxel_cnn = voxel_cnn.transpose(1, 2)  # (B, 256, T)
        voxel_cnn = self.voxel_temporal(voxel_cnn).squeeze(-1)  # (B, 256)
        voxel_emb = self.voxel_proj(voxel_cnn)  # (B, 128)

        # === Pose branch with confidence gating ===
        confs = pose_flat[..., self._conf_start:self._conf_end]  # (B, T, 7)
        mean_conf = confs.mean(dim=-1, keepdim=True)  # (B, T, 1)
        pose_frame = self.pose_frame_mlp(pose_flat) * mean_conf  # (B, T, 96)
        pose_temp = pose_frame.transpose(1, 2)  # (B, 96, T)
        pose_temp = self.pose_temporal(pose_temp).squeeze(-1)  # (B, 96)
        pose_emb = self.pose_proj(pose_temp)  # (B, 64)

        # === Late fusion ===
        fused = torch.cat([voxel_emb, pose_emb], dim=-1)  # (B, 192)
        fused = self.fusion(fused)  # (B, 256)
        logits = self.classifier(fused)  # (B, 8)

        out: Dict[str, torch.Tensor] = {"logits": logits}
        if return_embeddings:
            out["sequence_embedding"] = fused
            out["voxel_embedding"] = voxel_emb
            out["pose_embedding"] = pose_emb
            out["voxel_aux_logits"] = self.voxel_aux(voxel_emb)
            out["pose_aux_logits"] = self.pose_aux(pose_emb)
        return out


class TwoStreamFusionModel(nn.Module):
    """Two-stream late fusion: separate voxel and pose temporal processing.

    Instead of a transformer, this model computes temporal statistics over
    per-frame embeddings and fuses them late.  Much fewer parameters (~100K
    vs ~730K transformer), harder to overfit on small datasets.

    Voxel stream:
        Conv3D stem per frame -> d_voxel embedding
        Temporal stats [mean, max, last-first] -> 3*d_voxel
        Stream MLP -> d_voxel

    Pose stream:
        Raw 24-dim per frame + frame-to-frame velocity
        Temporal stats [mean, max, std, first, last, last-first] -> 6*pose_dim
        Velocity stats [mean, max] -> 2*pose_dim
        Stream MLP -> d_pose

    Fusion:
        concat(d_voxel, d_pose) -> classifier
    """

    def __init__(
        self,
        num_classes: int = 8,
        voxel_size: Tuple[int, int, int] = (12, 12, 12),
        in_channels: int = 2,
        d_voxel: int = 128,
        d_pose: int = 64,
        pose_dim: int = POSE_FEATURE_DIM,
        dropout: float = 0.2,
        # Accept but ignore transformer args so the same CLI works
        **kwargs,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.voxel_size = tuple(int(v) for v in voxel_size)
        self.voxel_dim = int(math.prod(self.voxel_size))
        self.in_channels = in_channels
        self.pose_dim = pose_dim
        self.d_voxel = d_voxel
        self.d_pose = d_pose

        # Voxel per-frame encoder
        grid_n = self.voxel_size[0]
        self.voxel_stem = Conv3DStem(
            in_channels=in_channels, d_model=d_voxel, grid_size=grid_n,
        )

        # Voxel temporal head: [mean, max, last-first] = 3 * d_voxel
        self.voxel_head = nn.Sequential(
            nn.Linear(d_voxel * 3, d_voxel),
            nn.LayerNorm(d_voxel),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Pose temporal head: [mean, max, std, first, last, Δ] + [vel_mean, vel_max] = 8 * pose_dim
        self.pose_head = nn.Sequential(
            nn.Linear(pose_dim * 8, d_pose * 2),
            nn.LayerNorm(d_pose * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_pose * 2, d_pose),
            nn.LayerNorm(d_pose),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Classifier
        fused_dim = d_voxel + d_pose
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, fused_dim // 2),
            nn.LayerNorm(fused_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim // 2, num_classes),
        )

    @staticmethod
    def _temporal_stats_voxel(
        x: torch.Tensor, valid: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute [mean, max, last-first] over time -> (B, 3*D)."""
        if valid is not None:
            v = valid.float().unsqueeze(-1)  # (B, T, 1)
            denom = v.sum(dim=1).clamp(min=1.0)
            mean_p = (x * v).sum(dim=1) / denom
            masked = x.masked_fill(~valid.unsqueeze(-1), float('-inf'))
            max_p = masked.max(dim=1).values
            max_p = torch.where(torch.isfinite(max_p), max_p, torch.zeros_like(max_p))
            # last - first valid
            valid_counts = valid.sum(dim=1).clamp(min=1).long()
            batch_idx = torch.arange(x.size(0), device=x.device)
            last_p = x[batch_idx, valid_counts - 1]
            # first valid
            first_idx = valid.float().argmax(dim=1).long()
            first_p = x[batch_idx, first_idx]
        else:
            mean_p = x.mean(dim=1)
            max_p = x.max(dim=1).values
            first_p = x[:, 0]
            last_p = x[:, -1]
        delta_p = last_p - first_p
        return torch.cat([mean_p, max_p, delta_p], dim=-1)

    @staticmethod
    def _temporal_stats_pose(
        x: torch.Tensor, valid: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute [mean, max, std, first, last, delta, vel_mean, vel_max] -> (B, 8*D)."""
        if valid is not None:
            v = valid.float().unsqueeze(-1)
            denom = v.sum(dim=1).clamp(min=1.0)
            mean_p = (x * v).sum(dim=1) / denom
            masked = x.masked_fill(~valid.unsqueeze(-1), float('-inf'))
            max_p = masked.max(dim=1).values
            max_p = torch.where(torch.isfinite(max_p), max_p, torch.zeros_like(max_p))
            # std
            var = ((x - mean_p.unsqueeze(1)) ** 2 * v).sum(dim=1) / denom
            std_p = var.clamp(min=1e-8).sqrt()
            # first / last
            valid_counts = valid.sum(dim=1).clamp(min=1).long()
            batch_idx = torch.arange(x.size(0), device=x.device)
            last_p = x[batch_idx, valid_counts - 1]
            first_idx = valid.float().argmax(dim=1).long()
            first_p = x[batch_idx, first_idx]
        else:
            mean_p = x.mean(dim=1)
            max_p = x.max(dim=1).values
            std_p = x.std(dim=1)
            first_p = x[:, 0]
            last_p = x[:, -1]
        delta_p = last_p - first_p

        # Frame-to-frame velocity
        vel = x[:, 1:] - x[:, :-1]  # (B, T-1, D)
        vel_mean = vel.mean(dim=1)
        vel_max = vel.max(dim=1).values

        return torch.cat([mean_p, max_p, std_p, first_p, last_p, delta_p,
                          vel_mean, vel_max], dim=-1)

    def forward(
        self,
        features: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        return_embeddings: bool = False,
    ) -> Dict[str, torch.Tensor]:
        B, T, _ = features.shape
        voxel_feat_dim = self.voxel_dim * self.in_channels

        voxel_flat = features[:, :, :voxel_feat_dim]
        pose_flat = features[:, :, voxel_feat_dim:voxel_feat_dim + self.pose_dim]

        # Voxel stream: per-frame Conv3D encoding
        N = self.voxel_size[0]
        C = self.in_channels
        voxel_5d = voxel_flat.reshape(B * T, C, N, N, N)
        voxel_emb = self.voxel_stem(voxel_5d).reshape(B, T, -1)  # (B, T, d_voxel)

        # Temporal statistics
        valid = ~padding_mask if padding_mask is not None else None
        voxel_stats = self._temporal_stats_voxel(voxel_emb, valid)  # (B, 3*d_voxel)
        pose_stats = self._temporal_stats_pose(pose_flat, valid)     # (B, 8*pose_dim)

        # Stream heads
        voxel_out = self.voxel_head(voxel_stats)  # (B, d_voxel)
        pose_out = self.pose_head(pose_stats)      # (B, d_pose)

        # Late fusion
        fused = torch.cat([voxel_out, pose_out], dim=-1)  # (B, d_voxel + d_pose)
        logits = self.classifier(fused)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)

        out: Dict[str, torch.Tensor] = {"logits": logits}
        if return_embeddings:
            out["sequence_embedding"] = fused
        return out


__all__ = [
    "POSE_FEATURE_DIM",
    "USED_JOINTS",
    "NUM_USED_JOINTS",
    "FLIP_LABEL_MAP",
    "extract_pose_features",
    "extract_pose_features_batch",
    "flip_pose_features_horizontal",
    "build_flip_label_indices",
    "PoseEncoder",
    "FusionVoxelPoseTransformerModel",
    "TwoStreamFusionModel",
]
