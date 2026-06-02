"""Vision Transformer — §2.

You implement: PatchEmbeddings, ViT.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbeddings(nn.Module):
    """Split an image into non-overlapping patches and project each to d_model.

    Implemented with a strided Conv2d whose kernel size and stride both equal
    `patch_size`.

    Args:
        img_size:   Input image side length (assumed square). Must be divisible
                    by patch_size.
        patch_size: Side length of each patch in pixels.
        d_model:    Output embedding dimension per patch.

    Forward:
        x: (B, 3, img_size, img_size) float tensor.
        returns: (B, num_patches, d_model) where num_patches = (img_size // patch_size) ** 2.
    """

    def __init__(self, img_size: int, patch_size: int, d_model: int) -> None:
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(3, d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.proj(x)          # (B, d_model, H//P, W//P)
        out = out.flatten(2)        # (B, d_model, N)
        return out.transpose(1, 2)  # (B, N, d_model)


# ---------------------------------------------------------------------------
# RoPE-aware attention building blocks
# (used when pos_encoding != "learned"; kept private to this module)
# ---------------------------------------------------------------------------

class _RoPEHead(nn.Module):
    """Single bidirectional attention head with RoPE applied to Q and K.

    The rope module handles both 1D and 2D cases via *rope_args:
      - RoPE1D: forward(x, positions)
      - RoPE2D: forward(x, x_coords, y_coords)
    """

    def __init__(self, d_model: int, head_dim: int, rope: nn.Module, dropout: float = 0.0) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.q_proj = nn.Linear(d_model, head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, head_dim, bias=False)
        self.rope = rope
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, *rope_args) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x)  # (B, T, head_dim)
        k = self.k_proj(x)
        v = self.v_proj(x)
        # RoPE expects (B, num_heads, T, head_dim); treat as 1 head here.
        q = self.rope(q.unsqueeze(1), *rope_args).squeeze(1)  # (B, T, head_dim)
        k = self.rope(k.unsqueeze(1), *rope_args).squeeze(1)
        attn = F.softmax((q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim), dim=-1)
        return self.dropout(attn) @ v


class _RoPEMHA(nn.Module):
    """Multi-head attention with RoPE."""

    def __init__(self, d_model: int, num_heads: int, rope: nn.Module, dropout: float = 0.0) -> None:
        super().__init__()
        head_dim = d_model // num_heads
        self.heads = nn.ModuleList([
            _RoPEHead(d_model, head_dim, rope, dropout) for _ in range(num_heads)
        ])
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, *rope_args) -> torch.Tensor:
        out = torch.cat([h(x, *rope_args) for h in self.heads], dim=-1)
        return self.dropout(self.out_proj(out))


class _RoPEBlock(nn.Module):
    """Pre-LayerNorm Transformer block with RoPE attention (encoder only)."""

    def __init__(self, d_model: int, num_heads: int, rope: nn.Module, dropout: float = 0.0) -> None:
        from basics.model import MLP
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = _RoPEMHA(d_model, num_heads, rope, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model=d_model, dropout=dropout)

    def forward(self, x: torch.Tensor, *rope_args) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), *rope_args)
        x = x + self.mlp(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# ViT
# ---------------------------------------------------------------------------

class ViT(nn.Module):
    """Vision Transformer.

    Pipeline:
      1. Patchify with `PatchEmbeddings`.
      2. Prepend a learnable [CLS] token.
      3. (learned) Add learned positional embedding.
         (rope1d) Apply 1D RoPE to Q,K inside each attention head.
         (rope2d) Apply 2D RoPE to Q,K using patch (x, y) grid indices.
      4. Pass the sequence through `num_blocks` Transformer Blocks.
      5. Apply a final LayerNorm.
      6. Return only the [CLS] slice — shape (B, d_model).

    For §5 (VLM), `return_all_tokens=True` returns (B, num_patches+1, d_model).

    For the extrapolation test (§6), learned PE is bilinearly interpolated from
    the training grid to the evaluation grid automatically in forward().

    Args:
        img_size, patch_size, d_model, num_heads, num_blocks, dropout,
        pos_encoding: one of "learned", "rope1d", "rope2d".
    """

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        d_model: int,
        num_heads: int,
        num_blocks: int,
        dropout: float = 0.1,
        pos_encoding: str = "learned",
    ) -> None:
        super().__init__()
        from basics.model import Block
        from basics.rope import RoPE1D, RoPE2D

        self.patch_embed = PatchEmbeddings(img_size, patch_size, d_model)
        self.num_patches = self.patch_embed.num_patches
        self.d_model = d_model
        self.img_size = img_size
        self.patch_size = patch_size
        self.pos_encoding = pos_encoding

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        head_dim = d_model // num_heads

        if pos_encoding == "learned":
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, d_model))
            self.blocks = nn.ModuleList([
                Block(d_model, num_heads, self.num_patches + 1, is_decoder=False, dropout=dropout)
                for _ in range(num_blocks)
            ])
        elif pos_encoding == "rope1d":
            # Precompute up to 1024 positions to support extrapolation.
            rope = RoPE1D(head_dim, max_seq_len=1024)
            self.blocks = nn.ModuleList([
                _RoPEBlock(d_model, num_heads, rope, dropout) for _ in range(num_blocks)
            ])
        elif pos_encoding == "rope2d":
            # grid_size=32 supports up to 32×32 patch grids (256×256 px at patch_size=8).
            rope = RoPE2D(head_dim, grid_size=32)
            self.blocks = nn.ModuleList([
                _RoPEBlock(d_model, num_heads, rope, dropout) for _ in range(num_blocks)
            ])
        else:
            raise ValueError(f"Unknown pos_encoding: {pos_encoding!r}")

        self.norm = nn.LayerNorm(d_model)

    def _get_pos_embed(self, T: int) -> torch.Tensor:
        """Return learned pos_embed, bilinearly interpolating if T != training size."""
        if self.pos_embed.shape[1] == T:
            return self.pos_embed
        cls_pe = self.pos_embed[:, :1, :]       # (1, 1, d)
        patch_pe = self.pos_embed[:, 1:, :]     # (1, N, d)
        N = patch_pe.shape[1]
        G = int(N ** 0.5)
        N_new = T - 1
        G_new = int(N_new ** 0.5)
        patch_2d = patch_pe.reshape(1, G, G, self.d_model).permute(0, 3, 1, 2).float()
        patch_2d = F.interpolate(patch_2d, size=(G_new, G_new), mode="bilinear", align_corners=False)
        patch_pe_new = patch_2d.permute(0, 2, 3, 1).reshape(1, N_new, self.d_model)
        return torch.cat([cls_pe, patch_pe_new.to(self.pos_embed.dtype)], dim=1)

    def _get_2d_coords(self, T: int, device: torch.device):
        """Return (x_coords, y_coords) for CLS + N patches (T tokens total).

        CLS gets (0, 0). Patch at row r, col c gets (x=c, y=r).
        """
        N = T - 1
        G = int(N ** 0.5)
        rows = torch.arange(G, device=device)
        cols = torch.arange(G, device=device)
        grid_r, grid_c = torch.meshgrid(rows, cols, indexing="ij")
        # x = column index (horizontal), y = row index (vertical)
        x_coords = torch.cat([torch.zeros(1, dtype=torch.long, device=device), grid_c.flatten()])
        y_coords = torch.cat([torch.zeros(1, dtype=torch.long, device=device), grid_r.flatten()])
        return x_coords, y_coords

    def forward(self, x: torch.Tensor, return_all_tokens: bool = False) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)                        # (B, N, d_model)
        cls = self.cls_token.expand(B, -1, -1)         # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1)                 # (B, N+1, d_model)
        T = x.shape[1]

        if self.pos_encoding == "learned":
            x = x + self._get_pos_embed(T)
            for block in self.blocks:
                x = block(x)
        elif self.pos_encoding == "rope1d":
            positions = torch.arange(T, device=x.device)
            for block in self.blocks:
                x = block(x, positions)
        elif self.pos_encoding == "rope2d":
            x_coords, y_coords = self._get_2d_coords(T, x.device)
            for block in self.blocks:
                x = block(x, x_coords, y_coords)

        x = self.norm(x)
        if return_all_tokens:
            return x
        return x[:, 0]                                 # (B, d_model)
