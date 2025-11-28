"""HOPE block components inspired by the Nested Learning paper."""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class RotaryPositionalEmbedding(nn.Module):
    """Applies rotary position embedding (ROPE) to query/key pairs."""

    def __init__(self, dim: int, theta: float = 10_000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        # x: (batch, seq, heads, head_dim)
        seq_len = x.size(1)
        freqs = torch.einsum("i , j -> i j", torch.arange(seq_len, device=x.device), self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)  # duplicate for sin/cos
        cos, sin = emb.cos(), emb.sin()
        x1, x2 = x[..., ::2], x[..., 1::2]
        rot_x = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
        return rot_x.flatten(-2)


class ChannelMixer(nn.Module):
    """Simple gated MLP used in place of Transformer FFN."""

    def __init__(self, hidden_size: int, expansion: int, dropout: float) -> None:
        super().__init__()
        inner = hidden_size * expansion
        self.fc1 = nn.Linear(hidden_size, inner)
        self.fc2 = nn.Linear(inner, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = F.silu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class FastAssociativeAttention(nn.Module):
    """Attention-like associative memory that updates every token."""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float, rope_theta: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryPositionalEmbedding(self.head_dim, theta=rope_theta)

    def forward(self, x: Tensor, cache: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        batch, seq, _ = x.shape
        qkv = self.qkv(x).view(batch, seq, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = self.rope(q)
        k = self.rope(k)

        if cache is not None:
            k = torch.cat([cache, k], dim=1)
            v = torch.cat([cache, v], dim=1)
        scores = torch.einsum("bthd,bshd->bhts", q, k) * self.scale
        attn = self.dropout(scores.softmax(dim=-1))
        out = torch.einsum("bhts,bshd->bthd", attn, v).reshape(batch, seq, -1)
        return self.out_proj(out), k.detach()


class DeepOptimizerMemory(nn.Module):
    """Associative momentum/preconditioner updated every step."""

    def __init__(self, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size)
        self.update = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: Tensor, grad_buffer: Optional[Tensor] = None) -> Tensor:
        precond = self.update(self.norm(hidden))
        if grad_buffer is not None:
            precond = precond + grad_buffer
        return self.dropout(precond)


class HOPEBlock(nn.Module):
    """HOPE block with nested fast and slow update paths."""

    def __init__(self, hidden_size: int, num_heads: int, ffn_mult: int, dropout: float, rope_theta: float) -> None:
        super().__init__()
        self.channel_mixer = ChannelMixer(hidden_size, ffn_mult, dropout)
        self.fast_memory = FastAssociativeAttention(hidden_size, num_heads, dropout, rope_theta)
        self.deep_optimizer = DeepOptimizerMemory(hidden_size, dropout)
        self.shortcut = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: Tensor, cache: Optional[Tensor] = None, grad_buffer: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        # Fast token-level memory update
        attn_out, new_cache = self.fast_memory(x, cache)

        # Channel mixing like HOPE's MLP branch
        mixed = self.channel_mixer(x + attn_out)

        # Deep optimizer style preconditioning (step-level memory)
        precond = self.deep_optimizer(mixed, grad_buffer)

        # Slow parameters integrate the nested signals
        updated = x + self.shortcut(mixed + precond)
        return updated, new_cache
