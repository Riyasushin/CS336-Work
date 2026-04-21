"""Distributed training containers: DDP, FSDP, sharded optimizer.

Stubs only — fill in during Week 5/6. Use tt.parallel.comm for the
underlying collectives, or torch.distributed directly when you want
stream overlap / async handles that comm.py doesn't expose.
"""

from __future__ import annotations

import torch


class DDP(torch.nn.Module):
    """Distributed Data Parallel wrapper. Bucketed gradient all-reduce,
    overlapped with backward. Week 5 TODO.
    """

    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        raise NotImplementedError("tt.parallel.DDP.forward (Week 5)")

    def finish_gradient_synchronization(self) -> None:
        raise NotImplementedError(
            "tt.parallel.DDP.finish_gradient_synchronization (Week 5)"
        )


class FSDP(torch.nn.Module):
    """Fully Sharded Data Parallel wrapper. Week 5/6 TODO."""

    def __init__(self, module: torch.nn.Module, compute_dtype: torch.dtype | None = None):
        super().__init__()
        self.module = module
        self.compute_dtype = compute_dtype

    def forward(self, *args, **kwargs):
        raise NotImplementedError("tt.parallel.FSDP.forward (Week 5/6)")

    def finish_gradient_synchronization(self) -> None:
        raise NotImplementedError(
            "tt.parallel.FSDP.finish_gradient_synchronization (Week 5/6)"
        )

    def gather_full_params(self) -> dict[str, torch.Tensor]:
        raise NotImplementedError("tt.parallel.FSDP.gather_full_params (Week 5/6)")


class ShardedOptimizer(torch.optim.Optimizer):
    """ZeRO-Stage-1 style sharded optimizer. Week 5 TODO."""

    def __init__(self, params, optimizer_cls: type[torch.optim.Optimizer], **kwargs):
        self._init_args = (params, optimizer_cls, kwargs)
        # Intentionally do NOT call super().__init__ here — delegate to the
        # real implementation once you fill it in.
        raise NotImplementedError("tt.parallel.ShardedOptimizer (Week 5)")

    def step(self, closure=None):
        raise NotImplementedError("tt.parallel.ShardedOptimizer.step (Week 5)")


__all__ = ["DDP", "FSDP", "ShardedOptimizer"]
