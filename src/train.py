"""Standalone training script mirroring kmccleary3301/nested_learning HOPE setup."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Tuple

import torch
from torch import Tensor, nn, optim
from torch.nn import functional as F

from .config import TrainingConfig
from .hope_model import HOPEModel


def cosine_warmup(step: int, cfg: TrainingConfig) -> float:
    if step < cfg.scheduler.warmup_steps:
        return cfg.optimizer.lr * step / max(1, cfg.scheduler.warmup_steps)
    progress = (step - cfg.scheduler.warmup_steps) / max(1, cfg.scheduler.total_steps - cfg.scheduler.warmup_steps)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return cfg.scheduler.min_lr + (cfg.optimizer.lr - cfg.scheduler.min_lr) * cosine


def make_optimizer(model: nn.Module, cfg: TrainingConfig) -> optim.Optimizer:
    if cfg.optimizer.name == "adamw":
        return optim.AdamW(
            model.parameters(),
            lr=cfg.optimizer.lr,
            betas=cfg.optimizer.betas,
            weight_decay=cfg.optimizer.weight_decay,
        )
    if cfg.optimizer.name == "lion":
        try:
            from lion_pytorch import Lion
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dep
            raise ModuleNotFoundError("Install lion-pytorch to use the Lion optimizer") from exc
        return Lion(model.parameters(), lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay)
    raise ValueError(f"Unsupported optimizer: {cfg.optimizer.name}")


def toy_token_stream(cfg: TrainingConfig) -> Iterable[Tuple[Tensor, Tensor]]:
    rng = torch.Generator().manual_seed(42)
    for _ in range(cfg.scheduler.total_steps):
        data = torch.randint(0, cfg.model.vocab_size, (cfg.batch_size, cfg.seq_len), generator=rng)
        target = data.roll(shifts=-1, dims=1)
        yield data, target


def loss_fn(logits: Tensor, targets: Tensor) -> Tensor:
    return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))


def train(cfg: TrainingConfig = TrainingConfig()) -> None:
    device = cfg.device if torch.cuda.is_available() and cfg.device == "cuda" else "cpu"
    model = HOPEModel(cfg.model).to(device)
    optimizer = make_optimizer(model, cfg)

    cache: Tensor | None = None
    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    for step, (tokens, targets) in enumerate(toy_token_stream(cfg), start=1):
        tokens = tokens.to(device)
        targets = targets.to(device)

        lr = cosine_warmup(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        logits, cache = model(tokens, cache=cache)
        loss = loss_fn(logits, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        if step % cfg.log_interval == 0:
            ppl = loss.exp().item()
            print(f"step={step:06d} loss={loss.item():.4f} ppl={ppl:.2f} lr={lr:.2e}")

        if step % cfg.checkpoint_interval == 0:
            ckpt_path = Path(cfg.checkpoint_dir) / f"hope_step_{step}.pt"
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, ckpt_path)

        if step >= cfg.scheduler.total_steps:
            break


if __name__ == "__main__":  # pragma: no cover
    train()
