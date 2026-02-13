import torch
import torch.nn as nn
import torch.nn.functional as F

from dlshogi.common import *


class Bias(nn.Module):
    def __init__(self, shape):
        super(Bias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(shape))

    def forward(self, x):
        return x + self.bias


# An ordinary implementation of Swish function (used for export when needed)
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


# Cache spiral indices per (H, W) on CPU; moved to device on demand.
_SPIRAL_CACHE = {}


def _spiral_indices(h: int, w: int):
    """Return (idx, inv_idx) for spiral order flatten indices.

    idx maps from row-major flatten -> spiral order.
    inv_idx is inverse mapping (spiral -> row-major).
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

    idx = torch.tensor([i * w + j for (i, j) in order], dtype=torch.long)
    inv = torch.empty_like(idx)
    inv[idx] = torch.arange(idx.numel(), dtype=torch.long)

    _SPIRAL_CACHE[key] = (idx, inv)
    return idx, inv


class SpiralTokenMix(nn.Module):
    """Token mixing along spiral ordering using depthwise Conv1d."""

    def __init__(self, channels: int, kernel_size: int = 3):
        super(SpiralTokenMix, self).__init__()
        padding = kernel_size // 2
        self.dwconv1d = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=channels,
            bias=False,
        )

    def forward(self, x):
        # x: (B, C, H, W)
        b, c, h, w = x.shape
        idx_cpu, inv_cpu = _spiral_indices(h, w)
        idx = idx_cpu.to(device=x.device)
        inv = inv_cpu.to(device=x.device)

        x_seq = x.view(b, c, h * w)  # row-major
        x_sp = x_seq.index_select(dim=2, index=idx)  # spiral order
        x_sp = self.dwconv1d(x_sp)
        x_seq = x_sp.index_select(dim=2, index=inv)
        return x_seq.view(b, c, h, w)


class SpiralMLPBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        activation=nn.ReLU(),
        mlp_ratio: float = 4.0,
        token_kernel: int = 3,
    ):
        super(SpiralMLPBlock, self).__init__()
        hidden = int(channels * mlp_ratio)

        self.norm1 = nn.BatchNorm2d(channels)
        self.token_mix = SpiralTokenMix(channels, kernel_size=token_kernel)

        self.norm2 = nn.BatchNorm2d(channels)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=False)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=False)
        self.act = activation

    def forward(self, x):
        x = x + self.token_mix(self.norm1(x))
        y = self.fc1(self.norm2(x))
        y = self.act(y)
        y = self.fc2(y)
        return x + y


class PolicyValueNetwork(nn.Module):
    """SpiralMLP-based policy/value network.

    Name examples (parsed by dlshogi/network/policy_value_network.py):
      - spiralmlp8x128
      - spiralmlp8x128_fcl256
      - spiralmlp8x128_swish

    Args follow existing dlshogi conventions:
      blocks   : depth
      channels : embedding dimension
      fcl      : value head hidden size
      activation: nn.ReLU() or nn.SiLU() etc.
    """

    def __init__(
        self,
        blocks: int,
        channels: int,
        activation=nn.ReLU(),
        fcl: int = 256,
        mlp_ratio: float = 4.0,
        token_kernel: int = 3,
    ):
        super(PolicyValueNetwork, self).__init__()

        # Input stem: same as resnet family (x1 board planes + x2 pieces-in-hand planes).
        self.l1_1_1 = nn.Conv2d(in_channels=FEATURES1_NUM, out_channels=channels, kernel_size=3, padding=1, bias=False)
        self.l1_1_2 = nn.Conv2d(in_channels=FEATURES1_NUM, out_channels=channels, kernel_size=1, padding=0, bias=False)
        self.l1_2 = nn.Conv2d(in_channels=FEATURES2_NUM, out_channels=channels, kernel_size=1, bias=False)
        self.norm1 = nn.BatchNorm2d(channels)
        self.act = activation

        # SpiralMLP blocks.
        self.blocks = nn.Sequential(
            *[
                SpiralMLPBlock(
                    channels=channels,
                    activation=activation,
                    mlp_ratio=mlp_ratio,
                    token_kernel=token_kernel,
                )
                for _ in range(blocks)
            ]
        )

        # policy network
        self.policy = nn.Conv2d(in_channels=channels, out_channels=MAX_MOVE_LABEL_NUM, kernel_size=1, bias=False)
        self.policy_bias = Bias(9 * 9 * MAX_MOVE_LABEL_NUM)

        # value network
        self.value_conv1 = nn.Conv2d(in_channels=channels, out_channels=MAX_MOVE_LABEL_NUM, kernel_size=1, bias=False)
        self.value_norm1 = nn.BatchNorm2d(MAX_MOVE_LABEL_NUM)
        self.value_fc1 = nn.Linear(9 * 9 * MAX_MOVE_LABEL_NUM, fcl)
        self.value_fc2 = nn.Linear(fcl, 1)

    def forward(self, x1, x2):
        u1_1_1 = self.l1_1_1(x1)
        u1_1_2 = self.l1_1_2(x1)
        u1_2 = self.l1_2(x2)
        h = self.act(self.norm1(u1_1_1 + u1_1_2 + u1_2))

        # SpiralMLP blocks
        h = self.blocks(h)

        # policy network
        h_policy = self.policy(h)
        h_policy = self.policy_bias(torch.flatten(h_policy, 1))

        # value network
        h_value = self.act(self.value_norm1(self.value_conv1(h)))
        h_value = self.act(self.value_fc1(torch.flatten(h_value, 1)))
        h_value = self.value_fc2(h_value)

        return h_policy, h_value

    def set_swish(self, memory_efficient=True):
        """Match resnet API: switch activation for training/export."""
        activation = nn.SiLU() if memory_efficient else Swish()
        for n, m in self.named_modules():
            if isinstance(m, PolicyValueNetwork) or isinstance(m, SpiralMLPBlock):
                m.act = activation
