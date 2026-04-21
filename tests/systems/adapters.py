"""Adapter layer for cs336_systems tests (Week 3/4/5/6).

Each ``get_*`` returns a class or module (not an instance). Each
``*_on_after_backward`` is a lifecycle callback that the test harness
invokes between backward and optimizer.step().

The distributed tests (DDP, FSDP) should be wired via tests/_spawn.py —
see tests/moe/test_moe.py for the pattern. Once you copy the assignment2
test files into ``tests/systems/``, adjust their ``mp.spawn`` / torchrun
scaffolding to call ``run_distributed`` from ``tests._spawn``.
"""

from __future__ import annotations

import torch


# ---------- Triton / FlashAttention ----------

def get_flashattention_autograd_function_pytorch() -> type:
    from tt.kernels import FlashAttentionPyTorch
    return FlashAttentionPyTorch


def get_flashattention_autograd_function_triton() -> type:
    from tt.kernels import FlashAttentionTriton
    return FlashAttentionTriton


# ---------- DDP ----------

def get_ddp(module: torch.nn.Module) -> torch.nn.Module:
    from tt.parallel import DDP
    return DDP(module)


def ddp_on_after_backward(ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    # Delegate to the container's hook, matching PA2 ``finish_gradient_synchronization``.
    ddp_model.finish_gradient_synchronization()


# ---------- FSDP ----------

def get_fsdp(
    module: torch.nn.Module, compute_dtype: torch.dtype | None = None
) -> torch.nn.Module:
    from tt.parallel import FSDP
    return FSDP(module, compute_dtype=compute_dtype)


def fsdp_on_after_backward(fsdp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    fsdp_model.finish_gradient_synchronization()


def fsdp_gather_full_params(fsdp_model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return fsdp_model.gather_full_params()


# ---------- Sharded optimizer (ZeRO-1) ----------

def get_sharded_optimizer(
    params, optimizer_cls: type[torch.optim.Optimizer], **kwargs
) -> torch.optim.Optimizer:
    from tt.parallel import ShardedOptimizer
    return ShardedOptimizer(params, optimizer_cls, **kwargs)
