"""Minimal PyTorch-style sketch of a nested learning stack.

The goal is to mirror the paper's structure: a fast associative memory
(inner optimization problem) nested inside a slower task model (outer
optimization problem). The fast memory adapts on every batch, while the
outer model updates more slowly.

Compared with a Transformer block (which only updates weights through the
optimizer), this sketch:
- Adds a *fast* key/value memory that is updated inside the forward pass.
- Lets the slow path read the compressed representation produced by that
  fast memory before computing its own prediction.
- Tracks both fast- and slow-level losses so you can balance them during
  training, similar in spirit to Titans/HOPE where self-modification is
  exposed as an inner loop.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
from torch import Tensor, nn, optim
from torch.nn import functional as F


class FastAssociativeMemory(nn.Module):
    """A small key-value memory that learns at a higher frequency.

    This module represents the "fast" level in the nested stack. It uses
    gradient descent on its own loss (compressing key/value pairs) before
    the slower task model is updated.
    """

    def __init__(self, hidden_size: int, lr: float = 1e-2) -> None:
        super().__init__()
        self.lr = lr
        self.key_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.value_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden: Tensor) -> Tuple[Tensor, Tensor]:
        keys = self.key_proj(hidden)
        values = self.value_proj(hidden)
        return keys, values

    def fast_step(self, hidden: Tensor) -> Tensor:
        """Update memory weights by reconstructing the incoming activations."""
        keys, values = self(hidden)
        loss = F.mse_loss(values, hidden.detach())
        grads = torch.autograd.grad(loss, self.parameters(), retain_graph=False)
        with torch.no_grad():
            for param, grad in zip(self.parameters(), grads):
                param -= self.lr * grad
        return loss.detach()


class NestedLearningBlock(nn.Module):
    """Combines slow parameters with a fast memory update.

    The forward pass explicitly calls the fast-level update (similar to
    how Titans/HOPE allow self-modification), and then the slower head
    consumes the updated representation. This keeps the implementation
    close to a Transformer block while exposing the nested-learning idea.
    """

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.encoder = nn.Linear(input_size, hidden_size)
        self.fast_memory = FastAssociativeMemory(hidden_size)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        # (1) Encode inputs (analogous to a Transformer MLP/attention mix).
        hidden = torch.tanh(self.encoder(x))

        # (2) Fast level adapts on the current context flow (token-level).
        fast_loss = self.fast_memory.fast_step(hidden)

        # (3) Slow level consumes the fast-memory-compressed representation.
        keys, values = self.fast_memory(hidden)
        combined = hidden + 0.5 * (keys + values)
        output = self.head(combined)
        return output, fast_loss


@dataclass
class TrainingConfig:
    input_size: int = 8
    hidden_size: int = 32
    lr: float = 1e-3
    batches: int = 200


def toy_stream(config: TrainingConfig) -> Iterable[Tuple[Tensor, Tensor]]:
    rng = torch.Generator().manual_seed(7)
    for _ in range(config.batches):
        x = torch.randn(16, config.input_size, generator=rng)
        # A drifting target to force continual adaptation.
        drift = torch.randn(1, config.input_size, generator=rng)
        y = (x * drift).sum(dim=1, keepdim=True)
        yield x, y


def train_nested_learner(config: TrainingConfig = TrainingConfig()) -> None:
    model = NestedLearningBlock(config.input_size, config.hidden_size)
    outer_opt = optim.Adam(model.parameters(), lr=config.lr)

    for step, (x, y) in enumerate(toy_stream(config), start=1):
        outer_opt.zero_grad()
        pred, fast_loss = model(x)
        slow_loss = F.mse_loss(pred, y)

        # Weighted blend mirrors how HOPE balances fast and slow objectives.
        total_loss = slow_loss + 0.1 * fast_loss
        total_loss.backward()
        outer_opt.step()

        if step % 50 == 0:
            print(
                f"step={step:03d} slow_loss={slow_loss.item():.4f} "
                f"fast_loss={fast_loss.item():.4f}"
            )


if __name__ == "__main__":
    train_nested_learner()
