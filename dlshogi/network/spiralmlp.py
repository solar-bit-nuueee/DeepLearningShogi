"""SpiralMLP (adaptation for dlshogi).

Paper: SpiralMLP: A Lightweight Vision MLP Architecture (arXiv:2404.00648).
This module provides a SpiralMLP-inspired block that performs token mixing along a
2D spiral ordering using a depthwise 1D convolution (lightweight), followed by a
channel MLP.

Notes:
- dlshogi inputs are typically (B, C, H, W) with H=W=9 for shogi boards.
- This implementation is intentionally self-contained and has no dependencies on
  other network definitions.

The goal is to offer a practical building block under dlshogi/network.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn


# Cache spiral indices per (H, W) on CPU; moved to device on demand.
_SPIRAL_CACHE: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}


def _spiral_indices(h: int, w: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (idx, inv_idx) for spiral order flatten indices.

    Spiral order is generated in the common 'peel layers' manner:
    left-to-right across top row, top-to-bottom on right col,
    right-to-left across bottom row, bottom-to-top on left col, repeat.

    idx maps from row-major flatten to spiral-order positions.
    inv_idx is the inverse mapping (spiral -> row-major).
    """
    key = (h, w)
    if key in _SPIRAL_CACHE:
        return _SPIRAL_CACHE[key]

    top, left = 0, 0
    bottom, right = h - 1, w - 1
    order = []

    while left <= right and top <= bottom:
        for j in range(left, right + 1):
            order.append((top, j))
        top += 1

        for i in range(top, bottom + 1):
            order.append((i, right))
        right -= 1

        if top <= bottom:
            for j in range(right, left - 1, -1):
                order.append((bottom, j))
            bottom -= 1

        if left <= right:
            for i in range(bottom, top - 1, -1):
                order.append((i, left))
            left += 1

    # Convert (i, j) to row-major flatten index.
    idx = torch.tensor([i * w + j for (i, j) in order], dtype=torch.long)
    inv = torch.empty_like(idx)
    inv[idx] = torch.arange(idx.numel(), dtype=torch.long)

    _SPIRAL_CACHE[key] = (idx, inv)
    return idx, inv


class LayerNorm2d(nn.Module):
    """LayerNorm-like normalization for (B, C, H, W) tensors.

    Implemented as GroupNorm with 1 group, which normalizes over channels.
    """

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.gn = nn.GroupNorm(1, channels, eps=eps, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gn(x)


class ChannelMLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Conv2d(dim, hidden, kernel_size=1, bias=True)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Conv2d(hidden, dim, kernel_size=1, bias=True)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class SpiralTokenMix(nn.Module):
    """Token mixing along spiral ordering using depthwise Conv1d."""

    def __init__(self, dim: int, kernel_size: int = 3, drop: float = 0.0):
        super().__init__()
        padding = kernel_size // 2
        self.dwconv1d = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=dim,
            bias=True,
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        b, c, h, w = x.shape
        idx_cpu, inv_cpu = _spiral_indices(h, w)
        idx = idx_cpu.to(device=x.device)
        inv = inv_cpu.to(device=x.device)

        x_seq = x.view(b, c, h * w)  # row-major
        x_sp = x_seq.index_select(dim=2, index=idx)  # spiral order

        x_sp = self.dwconv1d(x_sp)
        x_sp = self.drop(x_sp)

        # Back to row-major order.
        x_seq = x_sp.index_select(dim=2, index=inv)
        x_out = x_seq.view(b, c, h, w)
        return x_out


class SpiralMLPBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
        token_kernel: int = 3,
        drop: float = 0.0,
    ):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.token_mix = SpiralTokenMix(dim, kernel_size=token_kernel, drop=drop)
        self.norm2 = LayerNorm2d(dim)
        self.mlp = ChannelMLP(dim, mlp_ratio=mlp_ratio, drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.token_mix(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


@dataclass
class SpiralMLPConfig:
    in_channels: int
    embed_dim: int = 128
    depth: int = 8
    mlp_ratio: float = 4.0
    drop: float = 0.0
    token_kernel: int = 3


class SpiralMLP(nn.Module):
    """A lightweight MLP-style backbone with spiral token mixing."""

    def __init__(self, cfg: SpiralMLPConfig):
        super().__init__()
        self.cfg = cfg
        self.stem = nn.Conv2d(cfg.in_channels, cfg.embed_dim, kernel_size=1, bias=True)

        blocks = []
        for _ in range(cfg.depth):
            blocks.append(
                SpiralMLPBlock(
                    dim=cfg.embed_dim,
                    mlp_ratio=cfg.mlp_ratio,
                    token_kernel=cfg.token_kernel,
                    drop=cfg.drop,
                )
            )
        self.blocks = nn.Sequential(*blocks)
        self.norm = LayerNorm2d(cfg.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x


class SpiralMLPPolicyValueHead(nn.Module):
    """Optional convenience head: produce (policy_logits, value) from features.

    This is intentionally generic: pass policy_size explicitly.
    """

    def __init__(self, dim: int, board_h: int, board_w: int, policy_size: int):
        super().__init__()
        self.board_h = board_h
        self.board_w = board_w
        self.policy_size = policy_size

        self.policy = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(dim * board_h * board_w, policy_size, bias=True),
        )
        self.value = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=True),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim, 1, bias=True),
            nn.Tanh(),
        )

    def forward(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        p = self.policy(feat)
        v = self.value(feat)
        return p, v


class SpiralMLPPolicyValueNetwork(nn.Module):
    """Backbone + generic policy/value heads.

    Usage example:
        net = SpiralMLPPolicyValueNetwork(
            in_channels=FEATURES,
            policy_size=POLICY_SIZE,
            board_h=9,
            board_w=9,
        )
    """

    def __init__(
        self,
        in_channels: int,
        policy_size: int,
        board_h: int = 9,
        board_w: int = 9,
        embed_dim: int = 128,
        depth: int = 8,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        token_kernel: int = 3,
    ):
        super().__init__()
        cfg = SpiralMLPConfig(
            in_channels=in_channels,
            embed_dim=embed_dim,
            depth=depth,
            mlp_ratio=mlp_ratio,
            drop=drop,
            token_kernel=token_kernel,
        )
        self.backbone = SpiralMLP(cfg)
        self.head = SpiralMLPPolicyValueHead(embed_dim, board_h, board_w, policy_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.backbone(x)
        return self.head(feat)
