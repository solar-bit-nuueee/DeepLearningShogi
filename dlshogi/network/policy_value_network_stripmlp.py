import torch
import torch.nn as nn
import torch.nn.functional as F

from dlshogi.common import FEATURES1_NUM, FEATURES2_NUM, MAX_MOVE_LABEL_NUM

# -----------------------------------------------------------------------------
# Strip-MLP (adapted for dlshogi 9x9)
# Based on: "Strip-MLP: Efficient Token Interaction for Vision MLP" (arXiv:2307.11458)
# - Strip MLP layer (cross-strip, width=3)
# - Strip Mixing Block: split channels -> CGSMM (global) + LSMM (local) then fuse
# - Channel Mixing Block: inverted bottleneck + GRN
#
# Differences vs. the paper:
# - No multi-stage downsampling (board is 9x9 fixed). We keep a single stage.
# - Policy/Value heads follow dlshogi conventions (policy logits are 9*9*MAX_MOVE_LABEL_NUM).
# - Input: x1 (FEATURES1_NUM,9,9), x2 (FEATURES2_NUM,9,9)
#   x2 is reduced by mean pooling to (B, FEATURES2_NUM) then projected and added to all tokens (A1).
# -----------------------------------------------------------------------------

class Bias(nn.Module):
    def __init__(self, shape: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(shape))

    def forward(self, x):
        return x + self.bias


class GRN(nn.Module):
    """Global Response Normalization (used in ConvNeXtV2; Strip-MLP uses GRN in channel mixing block)."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        # x: (B, H, W, C)
        gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)  # (B,1,1,C)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + self.eps)
        return x + self.gamma * (x * nx) + self.beta


class MLPBNGELU(nn.Module):
    """Paper's MLP block: FC -> BN -> GELU."""
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.bn = nn.BatchNorm1d(hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        # x: (B, N, C)
        b, n, c = x.shape
        x = self.fc1(x)                 # (B,N,H)
        x = self.bn(x.reshape(b * n, -1)).reshape(b, n, -1)
        x = self.act(x)
        x = self.fc2(x)
        return x


class StripMLPLayer(nn.Module):
    """Strip MLP layer with strip width=3 (cross-strip) as Eq.(4)(5) idea.

    We implement the essential operation for 2D grid (H,W,C):
      - Row mixing: for each column j, concat tokens from columns (j-1,j,j+1) then linear -> C
      - Col mixing: for each row i,    concat tokens from rows    (i-1,i,i+1) then linear -> C

    This is a faithful implementation of the core cross-strip formulation in the paper.
    """
    def __init__(self, dim: int, direction: str):
        super().__init__()
        assert direction in ("row", "col")
        self.direction = direction
        self.proj = nn.Linear(3 * dim, dim)

    def forward(self, x, H: int, W: int):
        # x: (B, N, C) with N=H*W
        b, n, c = x.shape
        x2d = x.view(b, H, W, c)

        if self.direction == "row":
            # mix along width (columns): use neighbors j-1, j, j+1
            left  = torch.roll(x2d, shifts=1, dims=2)
            mid   = x2d
            right = torch.roll(x2d, shifts=-1, dims=2)
            cat = torch.cat([left, mid, right], dim=-1)  # (B,H,W,3C)
        else:
            # mix along height (rows): use neighbors i-1, i, i+1
            up    = torch.roll(x2d, shifts=1, dims=1)
            mid   = x2d
            down  = torch.roll(x2d, shifts=-1, dims=1)
            cat = torch.cat([up, mid, down], dim=-1)      # (B,H,W,3C)

        out = self.proj(cat)  # (B,H,W,C)
        return out.view(b, n, c)


class CGSMM(nn.Module):
    """Cascade Group Strip Mixing Module.

    Paper:
      - Permute and split along channel into P patches
      - Apply Group Strip MLP (unshared across patches) in cascade: column then row
      - Restore then concatenate with input and apply Channel FC

    Here we keep a simplified but structurally aligned version:
      - Split channels into P groups, apply independent StripMLPLayer per group
      - Cascade: (col) then (row)
      - Concatenate original + mixed and fuse by Linear (Channel FC)

    This preserves the key idea: channel-wise specificity for token mixing and cascade row/col.
    """
    def __init__(self, dim: int, patches: int = 4):
        super().__init__()
        assert dim % patches == 0
        self.patches = patches
        self.group_dim = dim // patches

        self.col_mix = nn.ModuleList([StripMLPLayer(self.group_dim, "col") for _ in range(patches)])
        self.row_mix = nn.ModuleList([StripMLPLayer(self.group_dim, "row") for _ in range(patches)])

        self.channel_fc = nn.Linear(2 * dim, dim)

    def forward(self, x, H: int, W: int):
        # x: (B,N,C)
        b, n, c = x.shape
        xs = torch.split(x, self.group_dim, dim=-1)

        # column mixing per patch
        xs = [m(t, H, W) for m, t in zip(self.col_mix, xs)]
        x_col = torch.cat(xs, dim=-1)

        # row mixing per patch
        xs = torch.split(x_col, self.group_dim, dim=-1)
        xs = [m(t, H, W) for m, t in zip(self.row_mix, xs)]
        x_row = torch.cat(xs, dim=-1)

        # concat with shortcut and fuse
        out = self.channel_fc(torch.cat([x, x_row], dim=-1))
        return out


class LSMM(nn.Module):
    """Local Strip Mixing Module.

    Paper uses a small strip-MLP unit (strip width=3 and length=7) and re-weighting.
    For 9x9, we implement a local variant by mixing row+col with strip width=3 and
    a lightweight gating (softmax re-weight) across 3 branches.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.row = StripMLPLayer(dim, "row")
        self.col = StripMLPLayer(dim, "col")
        self.id  = nn.Identity()

        self.gate = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, 3)
        )

    def forward(self, x, H: int, W: int):
        # x: (B,N,C)
        b, n, c = x.shape
        a = self.row(x, H, W)
        b2 = self.col(x, H, W)
        c2 = self.id(x)

        # re-weight module: global pool then softmax over branches
        pooled = x.mean(dim=1)             # (B,C)
        w = F.softmax(self.gate(pooled), dim=-1)  # (B,3)
        w1, w2, w3 = w[:, 0].view(-1, 1, 1), w[:, 1].view(-1, 1, 1), w[:, 2].view(-1, 1, 1)

        return w1 * a + w2 * b2 + w3 * c2


class StripMixingBlock(nn.Module):
    """Strip Mixing Block (Fig.2b).

    Paper:
      X -> DWSC(3x3) -> MLP(FC+BN+GELU) -> split channels -> CGSMM + LSMM -> concat -> FC -> +X

    For dlshogi token layout, we implement:
      - Token->(B,H,W,C) -> depthwise conv
      - Back to tokens -> MLPBNGELU
      - Split channels -> CGSMM/LSMM -> concat -> FC -> residual
    """
    def __init__(self, dim: int, hidden: int, patches: int = 4):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.mlp = MLPBNGELU(dim, hidden)

        self.cgsmm = CGSMM(dim // 2, patches=max(1, patches // 2)) if dim >= 2 else nn.Identity()
        self.lsmm = LSMM(dim // 2) if dim >= 2 else nn.Identity()

        self.fuse = nn.Linear(dim, dim)

    def forward(self, x, H: int, W: int):
        # x: (B,N,C)
        b, n, c = x.shape
        x0 = x

        x2d = x.view(b, H, W, c).permute(0, 3, 1, 2)  # (B,C,H,W)
        x2d = self.dwconv(x2d)
        x = x2d.permute(0, 2, 3, 1).contiguous().view(b, n, c)

        x = self.mlp(x)

        # split channel
        x1, x2 = torch.chunk(x, 2, dim=-1)
        y1 = self.cgsmm(x1, H, W)
        y2 = self.lsmm(x2, H, W)
        y = torch.cat([y1, y2], dim=-1)

        y = self.fuse(y)
        return y + x0


class ChannelMixingBlock(nn.Module):
    """Channel Mixing Block (Fig.2c).

    Paper uses inverted bottleneck + GRN.
    We do LN -> Linear expand -> GELU -> GRN -> Linear project -> residual.
    """
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.grn = GRN(hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x, H: int, W: int):
        # x: (B,N,C)
        x0 = x
        x = self.ln(x)
        x = self.fc1(x)
        x = self.act(x)
        # GRN expects (B,H,W,C)
        b, n, c = x.shape
        x = self.grn(x.view(b, H, W, c)).view(b, n, c)
        x = self.fc2(x)
        return x + x0


class PolicyValueNetwork(nn.Module):
    """dlshogi Policy-Value network using Strip-MLP backbone.

    Config (initial):
      - embed dim d=128
      - channel hidden (mlp hidden) = 512
      - depth = 8 blocks (StripMixing+ChannelMixing)
      - strip width = 3 (fixed in StripMLPLayer)
      - x2: mean pool over 9x9 -> condition vector -> add to all tokens (A1)

    Outputs:
      - policy logits: (B, 9*9*MAX_MOVE_LABEL_NUM)
      - value logit: (B, 1)
    """
    def __init__(self, d: int = 128, hidden: int = 512, depth: int = 8, patches: int = 4, fcl: int = 256):
        super().__init__()
        self.H = 9
        self.W = 9
        self.N = 81

        # Token embedding (x1)
        self.embed_x1 = nn.Linear(FEATURES1_NUM, d)

        # Condition embedding (x2 mean pooled)
        self.embed_x2 = nn.Linear(FEATURES2_NUM, d)

        self.blocks = nn.ModuleList([])
        for _ in range(depth):
            self.blocks.append(StripMixingBlock(d, hidden, patches=patches))
            self.blocks.append(ChannelMixingBlock(d, hidden))

        # policy head: per-token logits then flatten like original
        self.policy_fc = nn.Linear(d, MAX_MOVE_LABEL_NUM)
        self.policy_bias = Bias(9 * 9 * MAX_MOVE_LABEL_NUM)

        # value head: pool then MLP
        self.value_fc1 = nn.Linear(d, fcl)
        self.value_fc2 = nn.Linear(fcl, 1)

        self.value_ln = nn.LayerNorm(d)

    def forward(self, x1, x2):
        # x1: (B,C1,9,9), x2: (B,C2,9,9)
        b = x1.shape[0]

        # tokens from x1
        x1t = x1.permute(0, 2, 3, 1).contiguous().view(b, self.N, FEATURES1_NUM)  # (B,81,C1)
        tok = self.embed_x1(x1t)  # (B,81,d)

        # x2 condition: mean pool over 9x9
        x2c = x2.mean(dim=(2, 3))  # (B,C2)
        cond = self.embed_x2(x2c).unsqueeze(1)  # (B,1,d)

        # A1: add once at input (close to dlshogi's first additive fusion)
        tok = tok + cond

        for blk in self.blocks:
            tok = blk(tok, self.H, self.W)

        # policy
        p = self.policy_fc(tok)                      # (B,81,labels)
        p = p.reshape(b, -1)                         # (B,81*labels)
        p = self.policy_bias(p)

        # value
        v = self.value_ln(tok)
        v = v.mean(dim=1)                            # (B,d)
        v = F.gelu(self.value_fc1(v))
        v = self.value_fc2(v)                        # (B,1)

        return p, v
