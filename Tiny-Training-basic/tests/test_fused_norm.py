"""FusedRMSNorm correctness tests.

对照基准: `RRMSNorm` 当前的 PyTorch fp32 路径 (layers.py)。fused 版本必须在
forward 输出 + dx + dw 三件套上与基准 allclose, dtype 误差容忍按 fp32/fp16/bf16
分档放宽。

要求 CUDA: Triton kernel 不在 CPU 跑。无 GPU 直接 skip 整个文件。
"""

from __future__ import annotations

import pytest
import torch

CUDA_AVAILABLE = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not CUDA_AVAILABLE, reason="FusedRMSNorm 需要 CUDA")


def _ref_rmsnorm_fwd_bwd(
    x: torch.Tensor, weight: torch.Tensor, eps: float, dy: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """PyTorch 参考: 与 RRMSNorm.forward 完全等价 (内部 fp32, 输出 cast 回原 dtype)。"""
    in_dtype = x.dtype
    xf = x.detach().to(torch.float32).requires_grad_(True)
    wf = weight.detach().to(torch.float32).requires_grad_(True)
    rms = torch.sqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    y = (xf / rms) * wf
    # 用与 fused 版相同 dtype 的 dy 算 grad, 避免 dtype 不一致带来的 cast 噪声
    y_cast = y.to(in_dtype)
    y_cast.backward(dy)
    return y_cast.detach(), xf.grad.to(in_dtype), wf.grad.to(weight.dtype)


_TOLS = {
    torch.float32: dict(atol=1e-5, rtol=1e-5),
    torch.float16: dict(atol=2e-3, rtol=2e-3),
    torch.bfloat16: dict(atol=1e-2, rtol=1e-2),
}


@pytest.mark.parametrize("inplace_bwd", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "shape",
    [
        (2, 128, 128),    # tiny: BLOCK_N=128, num_warps=2
        (2, 512, 896),    # Qwen3-0.5B 的 d_model: BLOCK_N=1024, num_warps=4
        (4, 256, 4096),   # Llama-7B 的 d_model: BLOCK_N=4096, num_warps=8
    ],
)
def test_fused_rmsnorm_matches_pytorch(
    shape: tuple[int, int, int], dtype: torch.dtype, inplace_bwd: bool
) -> None:
    from tiny_training_basic.kernels.fused_norm import FusedRMSNorm

    torch.manual_seed(0)
    B, L, D = shape
    eps = 1e-5

    # randn / 5 压一下幅度, 防 fp16 sqrt 边界
    x_ref = (torch.randn(B, L, D, device="cuda", dtype=dtype) / 5).requires_grad_(False)
    w_ref = torch.randn(D, device="cuda", dtype=dtype) / 5
    dy = torch.randn(B, L, D, device="cuda", dtype=dtype) / 5

    # ---- ref ----
    # in-place bwd 会覆写 dy buffer, 所以 ref 用 dy 的副本算
    y_ref, dx_ref, dw_ref = _ref_rmsnorm_fwd_bwd(x_ref, w_ref, eps, dy.clone())

    # ---- fused ----
    x_f = x_ref.detach().clone().requires_grad_(True)
    w_f = w_ref.detach().clone().requires_grad_(True)
    y_f = FusedRMSNorm.apply(x_f, w_f, eps, inplace_bwd)
    y_f.backward(dy.clone())  # backward 可能 in-place 改 dy, 用副本
    dx_f, dw_f = x_f.grad, w_f.grad

    tol = _TOLS[dtype]
    torch.testing.assert_close(y_f, y_ref, **tol)
    torch.testing.assert_close(dx_f, dx_ref, **tol)
    # dw 跨 batch reduce, 误差比 elementwise 大一档 (尤其 fp16/bf16), 放宽一档
    dw_tol = {k: v * 5 for k, v in tol.items()}
    torch.testing.assert_close(dw_f, dw_ref, **dw_tol)


def test_rrmsnorm_use_fused_switch() -> None:
    """RRMSNorm(use_fused=True) 走 fused kernel, 输出与 use_fused=False 一致。"""
    from tiny_training_basic.layers import RRMSNorm

    torch.manual_seed(0)
    D = 896
    x = torch.randn(2, 128, D, device="cuda", dtype=torch.float32) / 5

    rn_torch = RRMSNorm(D, use_fused=False).cuda()
    rn_fused = RRMSNorm(D, use_fused=True).cuda()
    rn_fused.load_state_dict(rn_torch.state_dict())

    y_torch = rn_torch(x)
    y_fused = rn_fused(x)
    torch.testing.assert_close(y_fused, y_torch, atol=1e-5, rtol=1e-5)


def test_fused_rmsnorm_inplace_matches_pytorch() -> None:
    """fused_rmsnorm_inplace: y 覆写 x, 数值等价 RRMSNorm.forward。"""
    from tiny_training_basic.kernels.fused_norm import fused_rmsnorm_inplace

    torch.manual_seed(0)
    B, L, D = 2, 256, 896
    eps = 1e-5
    x = (torch.randn(B, L, D, device="cuda", dtype=torch.float32) / 5)
    weight = torch.randn(D, device="cuda", dtype=torch.float32) / 5

    # 参考: 跑非 in-place 路径
    y_ref, _, _ = _ref_rmsnorm_fwd_bwd(x, weight, eps, torch.zeros_like(x))

    # in-place: x 被覆盖, 返回值 = x
    x_inplace = x.clone()
    x_ptr_before = x_inplace.data_ptr()
    y = fused_rmsnorm_inplace(x_inplace, weight, eps)
    assert y.data_ptr() == x_ptr_before, "返回值应与 input 共享存储"
    assert x_inplace.data_ptr() == x_ptr_before
    torch.testing.assert_close(y, y_ref, atol=1e-5, rtol=1e-5)


def test_fused_rmsnorm_inplace_rejects_requires_grad() -> None:
    """requires_grad=True 时拒绝, 避免 autograd 用错。"""
    from tiny_training_basic.kernels.fused_norm import fused_rmsnorm_inplace

    x = torch.randn(2, 8, 128, device="cuda", requires_grad=True)
    w = torch.ones(128, device="cuda")
    with pytest.raises(AssertionError, match="autograd"):
        fused_rmsnorm_inplace(x, w, 1e-5)


def test_fused_rejects_oversize_d_model() -> None:
    """d_model > 8192 时显式 raise, 不静默掉。"""
    from tiny_training_basic.kernels.fused_norm import FusedRMSNorm

    x = torch.randn(2, 8, 16384, device="cuda", dtype=torch.float32)
    w = torch.ones(16384, device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match="超过单 block 上限"):
        FusedRMSNorm.apply(x, w, 1e-5)
