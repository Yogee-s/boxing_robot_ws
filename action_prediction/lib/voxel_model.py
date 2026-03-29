"""
Shared model components for voxel-based boxing action recognition.

Provides building blocks used by the fusion model:
- `PositionalEncoding` — sinusoidal positional encoding for transformer
- `Conv3DStem` — 3D convolutional encoder for voxel grids
- `count_parameters` — utility to count trainable parameters
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int = 200, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class Conv3DStem(nn.Module):
    """Small 3D convolutional encoder for voxel grids.

    Each frame's (C, N, N, N) voxel grid is compressed to a d_model
    embedding via strided 3D convolutions.

    Output spatial size after 3 stride-2 convs: ceil(N/2) -> ceil(N/4) -> ceil(N/8).
    For N=12: 12 -> 6 -> 3 -> 2 -> flatten 64*8=512 -> Linear -> d_model
    """

    def __init__(
        self,
        in_channels: int = 2,
        d_model: int = 192,
        grid_size: int = 12,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.grid_size = grid_size

        self.convs = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(16),
            nn.GELU(),
            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.GELU(),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.GELU(),
        )

        # Compute output spatial size after 3 stride-2 convs
        s = grid_size
        for _ in range(3):
            s = (s + 1) // 2
        self._flat_dim = 64 * s * s * s

        self.proj = nn.Sequential(
            nn.Linear(self._flat_dim, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch * seq_len, in_channels, N, N, N)

        Returns:
            (batch * seq_len, d_model)
        """
        x = self.convs(x)
        x = x.flatten(1)
        return self.proj(x)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
