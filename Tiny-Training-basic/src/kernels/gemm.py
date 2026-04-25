"""Fused GEMM kernel — matmul (+ bias) (+ activation / residual) in one pass.

来源：CSE234 PA2 Part 1（Triton matmul + ReLU + add 融合）。
计划 epilogue：none / relu / gelu / silu / bias / residual-add 的任意组合。
"""

from __future__ import annotations

import torch


def fused_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    bias: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    activation: str | None = None,
) -> torch.Tensor:
    """out = act(A @ B + bias) + residual，Triton 实现。

    Shapes:
        A: [M, K]  B: [K, N]
        bias:     [N]        (broadcast over M)
        residual: [M, N]
        return:   [M, N]
    """
    raise NotImplementedError
