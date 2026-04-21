"""Triton & fused kernels. Stubs only — fill in during Week 3/4.

Each class here is the return value of a ``get_*`` factory (see
cs336_systems/tests/adapters.py). Instantiate nothing here; the test
calls the class as a ``torch.autograd.Function`` subclass directly via
``ClassName.apply(...)``.
"""

from __future__ import annotations

import torch


class FlashAttentionPyTorch(torch.autograd.Function):
    """Pure-PyTorch FlashAttention-2. Fill in during Week 3."""

    @staticmethod
    def forward(ctx, *args, **kwargs):
        raise NotImplementedError("tt.kernels.FlashAttentionPyTorch.forward (Week 3)")

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError("tt.kernels.FlashAttentionPyTorch.backward (Week 4)")


class FlashAttentionTriton(torch.autograd.Function):
    """Triton-backed FlashAttention-2. Fill in during Week 3/4."""

    @staticmethod
    def forward(ctx, *args, **kwargs):
        raise NotImplementedError("tt.kernels.FlashAttentionTriton.forward (Week 3)")

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError("tt.kernels.FlashAttentionTriton.backward (Week 4)")


def fused_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Fused RMSNorm (Triton). Fill in during Week 3."""
    raise NotImplementedError("tt.kernels.fused_rmsnorm (Week 3)")


__all__ = [
    "FlashAttentionPyTorch",
    "FlashAttentionTriton",
    "fused_rmsnorm",
]
