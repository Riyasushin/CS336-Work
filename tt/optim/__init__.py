"""Optimizers, losses, schedules. Stubs only — fill in during Week 2."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor


def get_adamw_cls() -> Any:
    """Return your AdamW optimizer class (the class itself, not an instance)."""
    raise NotImplementedError("tt.optim.get_adamw_cls (Week 2)")


def cross_entropy(inputs: Tensor, targets: Tensor) -> Tensor:
    raise NotImplementedError("tt.optim.cross_entropy (Week 2)")


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """In-place gradient clipping by combined L2 norm."""
    raise NotImplementedError("tt.optim.gradient_clipping (Week 2)")


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    raise NotImplementedError("tt.optim.get_lr_cosine_schedule (Week 2)")


__all__ = [
    "get_adamw_cls",
    "cross_entropy",
    "gradient_clipping",
    "get_lr_cosine_schedule",
]
