"""Configuration helpers for HOPE/Nested Learning experiments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class OptimizerConfig:
    name: Literal["adamw", "lion"] = "adamw"
    lr: float = 3e-4
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.1


@dataclass
class SchedulerConfig:
    warmup_steps: int = 2000
    total_steps: int = 100_000
    min_lr: float = 3e-5


@dataclass
class ModelConfig:
    vocab_size: int = 32_000
    hidden_size: int = 1024
    ffn_mult: int = 4
    num_layers: int = 24
    num_heads: int = 8
    dropout: float = 0.1
    rope_theta: float = 10_000.0


@dataclass
class TrainingConfig:
    model: ModelConfig = ModelConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    scheduler: SchedulerConfig = SchedulerConfig()
    seq_len: int = 2048
    batch_size: int = 8
    grad_clip: float = 1.0
    device: str = "cuda"  # set to "cpu" automatically if CUDA unavailable
    log_interval: int = 50
    checkpoint_interval: int = 1000
    checkpoint_dir: str = "checkpoints"
