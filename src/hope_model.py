"""Full HOPE language model built from HOPE blocks."""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from .config import ModelConfig
from .hope_block import HOPEBlock


class HOPEModel(nn.Module):
    """Stack of HOPE blocks with token embedding and LM head."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList(
            [
                HOPEBlock(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_heads,
                    ffn_mult=config.ffn_mult,
                    dropout=config.dropout,
                    rope_theta=config.rope_theta,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.out_norm = nn.RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, tokens: Tensor, cache: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        x = self.embed(tokens)
        new_cache = cache
        grad_buffer: Optional[Tensor] = None
        for block in self.blocks:
            x, new_cache = block(x, cache=new_cache, grad_buffer=grad_buffer)
            grad_buffer = x.detach()
        x = self.out_norm(x)
        logits = self.lm_head(x)
        return logits, new_cache

    @torch.no_grad()
    def generate(self, prompt: Tensor, max_new_tokens: int) -> Tensor:
        seq = [prompt]
        cache: Optional[Tensor] = None
        for _ in range(max_new_tokens):
            tokens = torch.cat(seq, dim=1)
            logits, cache = self(tokens, cache=cache)
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            seq.append(next_token)
        return torch.cat(seq, dim=1)
