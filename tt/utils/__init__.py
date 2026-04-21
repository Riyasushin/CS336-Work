"""Training utilities: batching, checkpointing. Stubs only — Week 2."""

from __future__ import annotations

import os
from typing import IO, BinaryIO

import numpy.typing as npt
import torch


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    raise NotImplementedError("tt.utils.get_batch (Week 2)")


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    raise NotImplementedError("tt.utils.save_checkpoint (Week 2)")


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    raise NotImplementedError("tt.utils.load_checkpoint (Week 2)")


__all__ = ["get_batch", "save_checkpoint", "load_checkpoint"]
