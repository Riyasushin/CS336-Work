"""Fused RMSNorm / LayerNorm — forward + backward in one Triton pass.

对应 `tiny_training_basic.layers.RRMSNorm` 的优化版本：单 kernel 内完成
reduce、rsqrt、scale、(optional) shift，避免多次读写 HBM。
"""

from __future__ import annotations

import torch


class FusedRMSNorm(torch.autograd.Function):
    """Fused RMSNorm forward/backward（Triton）。"""

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5):
        raise NotImplementedError

    @staticmethod
    def backward(ctx, dY: torch.Tensor):
        raise NotImplementedError


class FusedLayerNorm(torch.autograd.Function):
    """Fused LayerNorm forward/backward（Triton）。"""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        eps: float = 1e-5,
    ):
        raise NotImplementedError

    @staticmethod
    def backward(ctx, dY: torch.Tensor):
        raise NotImplementedError
