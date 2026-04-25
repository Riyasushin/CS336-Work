"""Fused RMSNorm forward + backward — 单 Triton kernel 完成 reduce / rsqrt / scale。

为什么要这份文件
----------------
对应 `tiny_training_basic.layers.RRMSNorm` 的优化版本。原 PyTorch 实装 (layers.py)
里 `(x.pow(2).mean(-1) + eps).sqrt()` → `x / rms * w` 至少触发 5 次 HBM round-trip
(pow / mean / add / sqrt / div+mul), 而 RMSNorm 是典型 memory-bound kernel ——
真正的算力开销远小于把张量从 HBM 搬到 SM 的开销, 所以"少读写一次 HBM"基本就直接
等比例反映在 wall-time 上。

fused 版本一个 program 处理一行 (last-dim 一行 d_model 个元素), 在 SRAM 内做完
所有 reduce / rsqrt / scale, 整行只读 1 次 写 1 次:
  fwd HBM 流量:  2 * tensor_bytes  (理论下界, weight 单行加载可视为 cache hit)
  bwd HBM 流量:  约 4 * tensor_bytes  (load x, dY 各一次; store dx 一次; dw 用
                                      atomic_add 写跨行 fp32 buffer 不算主流量)

这两个数除以 GPU HBM 带宽 (RTX 4060 ≈ 272 GB/s) 就是 wall-time 的物理下界,
bench_norm.py 会用 "实测耗时 / 这个下界" 反算带宽利用率。

设计选择 (硬编码, 不 autotune, 与 fa2.py 同精神)
-----------------------------------------------
1. **一行 = 一个 program** (`grid=(M,)`, M = ∏ leading_dims). 行内 reduce 由
   single-block 完成, 跨行无依赖, 天然并行。
2. **BLOCK_N = next_pow2(d_model)**, 单 block 加载整行。Qwen3-0.5B d_model=896,
   Llama-7B d_model=4096 都装得下。d_model > 8192 时降级到 PyTorch (这里直接
   raise, 让用户知道走错路)。
3. **fp32 accumulator**: 所有 reduce / rsqrt 都在 fp32, 与 RRMSNorm 的
   `x.to(float32)` 路径一致。输入/输出按调用方 dtype (fp16/bf16/fp32 全支持)。
4. **bwd 的 dw 用 fp32 atomic_add 跨行累加**: 这是 elementwise reduce 的标准
   做法。Ada/Ampere 上 fp32 atomic 是 native 单周期, partial 写完后再 cast 回
   weight 的 dtype。比起 launch 第二个 reduce kernel, fused 进同一个 bwd kernel
   省一次 dY/x 的 HBM 重读。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# 启发式 (硬编码查表, 不 autotune)
# ---------------------------------------------------------------------------


def _num_warps_for_block_n(BLOCK_N: int) -> int:
    """BLOCK_N 越大单 block 干的活越多, 多分点 warp 让 LD/ST 和 reduce 并发。

    经验值 (与 OpenAI Triton tutorials/05-layer-norm.py 大致对齐):
      BLOCK_N <= 256  -> 2 warp   (单行小, 多了反而 SM occupancy 浪费)
      BLOCK_N <= 1024 -> 4 warp
      BLOCK_N <= 4096 -> 8 warp
      BLOCK_N >  4096 -> 16 warp  (4060 上 16 warps/CTA = 512 threads, 仍 < 1024)
    """
    if BLOCK_N <= 256:
        return 2
    if BLOCK_N <= 1024:
        return 4
    if BLOCK_N <= 4096:
        return 8
    return 16


# ---------------------------------------------------------------------------
# Forward kernel
# ---------------------------------------------------------------------------


@triton.jit
def _rmsnorm_fwd_kernel(
    X,              # *T, (M, D)        输入
    Y,              # *T, (M, D)        输出 (与 X 同 dtype)
    W,              # *T, (D,)          weight
    Rstd,           # *fp32, (M,)       缓存给 backward 用 (省一次 reduce)
    stride_xm,      # X 行 stride (一般就是 D, 但允许非连续)
    stride_ym,      # Y 行 stride
    D,              # 列数 (≤ BLOCK_N)
    eps,            # 数值稳定项
    BLOCK_N: tl.constexpr,  # 编译期常量, next_pow2(D)
):
    """一行一个 program, 全部 reduce 在单 block 内完成。

    数学:
      rms  = sqrt(mean(x²) + eps)
      rstd = 1 / rms
      y    = x * rstd * w
    """
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < D

    # load 整行到 SRAM. mask 外用 0 填充, 不影响后续 sum(x²) 与 store
    x = tl.load(X + pid * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)

    # mean(x²): mask 外是 0, 平方还是 0, 直接 sum 再除 D 即可 (注意是除 D 不是 BLOCK_N)
    mean_sq = tl.sum(x * x, axis=0) / D
    rstd = 1.0 / tl.sqrt(mean_sq + eps)

    # 缓存 rstd 给 backward, 避免 bwd 再做一次 reduce (RMSNorm bwd 已经要 1 次新 reduce 了, 不重复)
    tl.store(Rstd + pid, rstd)

    y = x * rstd * w
    # store 时自动 cast 回 Y 的 dtype (Triton 行为)
    tl.store(Y + pid * stride_ym + cols, y, mask=mask)


# ---------------------------------------------------------------------------
# Backward kernel
# ---------------------------------------------------------------------------
#
# 数学推导 (列向量, 标量 L):
#   y_i = x_i * rstd * w_i,   rstd = 1/sqrt(mean(x²) + eps)
#   d(rstd)/d(x_j) = -rstd³ * x_j / D
#   ∂y_i/∂x_j = δ_ij * rstd * w_i  +  x_i * w_i * (-rstd³ * x_j / D)
#
#   dL/dx_j = Σ_i dY_i * ∂y_i/∂x_j
#           = dY_j * w_j * rstd  -  rstd³ * x_j / D * Σ_i (dY_i * w_i * x_i)
#           = rstd * [ dY_j * w_j  -  x_j * (rstd² / D) * Σ_i (dY_i * w_i * x_i) ]
#
#   令 c = (rstd² / D) * Σ_i (dY_i * w_i * x_i)    -- 标量 (per row)
#   则 dx_j = rstd * (dY_j * w_j - x_j * c)        -- elementwise + 1 个标量
#
#   dw_i = Σ_over_M (dY_i * x_i * rstd)            -- 跨 M reduce, 用 atomic_add
#
# bwd kernel 的 IO:
#   load:  x, dY, w, rstd            (前三个是 (M,D) 主流量, w/rstd 是小项)
#   store: dx                         ((M,D) 主流量)
#   atomic_add: dw_partial (fp32)    (跨行散乱写, 但只有 D 个目标, cache-friendly)


@triton.jit
def _rmsnorm_bwd_kernel(
    X,              # *T, (M, D)        forward 的输入
    DY,             # *T, (M, D)        从 autograd 传入的 grad output
    W,              # *T, (D,)
    Rstd,           # *fp32, (M,)       forward 缓存的 1/rms
    DX,             # *T, (M, D)        输出
    DW_partial,     # *fp32, (D,)       atomic_add 累加目标 (调用前 zero_)
    stride_xm,
    stride_dym,
    stride_dxm,
    D,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < D

    x = tl.load(X + pid * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(DY + pid * stride_dym + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(Rstd + pid)  # 标量

    # ---- dx ----
    dy_w = dy * w
    # c = rstd² * Σ(dy*w*x) / D  -- 标量, 在 SRAM 内 reduce
    c = rstd * rstd * tl.sum(dy_w * x, axis=0) / D
    dx = rstd * (dy_w - x * c)
    tl.store(DX + pid * stride_dxm + cols, dx, mask=mask)

    # ---- dw partial: 这一行对 dw 的贡献 = dY * x * rstd ----
    # atomic_add 到全局 fp32 buffer. fp32 atomic 在 sm_60+ 是 native, Ada 上单周期。
    # 不 mask 越界位 (它们的 dy*x 是 0, 加 0 是 no-op, 但 atomic 的 mask 仍要给, 否则
    # 会越界写到 cols >= D 的位置)
    dw_part = dy * x * rstd
    tl.atomic_add(DW_partial + cols, dw_part, mask=mask)


# ---------------------------------------------------------------------------
# autograd.Function wrapper
# ---------------------------------------------------------------------------


_MAX_BLOCK_N = 8192  # next_pow2 上限. d_model > 8192 走多 tile reduce, 这里不实装


class FusedRMSNorm(torch.autograd.Function):
    """Fused RMSNorm forward/backward (Triton)。

    与 `RRMSNorm.forward` 的数值约定完全一致:
      - 内部 fp32 计算
      - 输出 cast 回 input dtype
      - eps 加在 mean(x²) 后的 sqrt 内部

    限制:
      - 仅 CUDA (Triton kernel)
      - d_model ≤ 8192 (单 block 装下)
      - x.is_contiguous() 在 reshape(-1, D) 后；不是的话 wrapper 会 .contiguous() 一次
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        eps: float = 1e-5,
        inplace_bwd: bool = False,
    ) -> torch.Tensor:
        """
        inplace_bwd: backward 时把 dx 写到 dY 的 buffer, 省一份 (M, D) alloc。
                     trade-off: 实测 (4060) 速度比 alloc 路径慢 ~20-25%, 推测是
                     read+write 同一片 HBM 触发的 L2 write-allocate 与 atomic_add
                     写 dw_partial 在同一 SM 周期争用。**显存吃紧时开启**, 否则
                     默认 False 走 alloc 更快。
        """
        assert x.is_cuda, "FusedRMSNorm 需要 CUDA 张量 (Triton kernel)"
        assert weight.is_cuda and weight.dim() == 1
        D = x.shape[-1]
        assert weight.shape == (D,), f"weight shape {weight.shape} 不匹配 d_model={D}"

        # 把 leading dims 全部折叠成 M, kernel 视角是 (M, D) 二维
        orig_shape = x.shape
        x_2d = x.reshape(-1, D)
        # 非 contig 时 reshape 会自动 .contiguous() 触发拷贝, 这里强制一次保证 stride_xm = D
        x_2d = x_2d.contiguous()
        M = x_2d.shape[0]

        BLOCK_N = triton.next_power_of_2(D)
        if BLOCK_N > _MAX_BLOCK_N:
            raise ValueError(
                f"d_model={D} (BLOCK_N={BLOCK_N}) 超过单 block 上限 {_MAX_BLOCK_N}; "
                "请用 PyTorch 路径或实现多 tile reduce 版本"
            )

        y = torch.empty_like(x_2d)
        # rstd 必须 fp32: (1) bwd 直接用; (2) 跨 dtype 的 sqrt/div 精度敏感
        rstd = torch.empty(M, device=x.device, dtype=torch.float32)

        num_warps = _num_warps_for_block_n(BLOCK_N)
        _rmsnorm_fwd_kernel[(M,)](
            x_2d, y, weight, rstd,
            x_2d.stride(0), y.stride(0),
            D, eps,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )

        ctx.save_for_backward(x_2d, weight, rstd)
        ctx.BLOCK_N = BLOCK_N
        ctx.num_warps = num_warps
        ctx.orig_shape = orig_shape
        ctx.inplace_bwd = inplace_bwd
        return y.reshape(orig_shape)

    @staticmethod
    def backward(ctx, dY: torch.Tensor):
        x_2d, weight, rstd = ctx.saved_tensors
        orig_shape = ctx.orig_shape
        D = orig_shape[-1]
        M = x_2d.shape[0]

        dy_2d = dY.reshape(-1, D).contiguous()
        if ctx.inplace_bwd:
            # 复用 dy 的 buffer 写 dx, 省 (M, D) alloc。
            # autograd 协议下 grad_output 在 backward 返回后不再使用, 覆盖安全。
            # kernel 内 single-program-per-row, tl.load 一次性吸完 dy 到寄存器,
            # 之后 tl.store 回同位置无 RAW 冲突。
            dx = dy_2d
        else:
            # 默认: 单独 alloc dx, 让 read 和 write 不打架, 实测更快。
            dx = torch.empty_like(x_2d)

        # dw_partial 必须 fp32 (atomic_add 目标 + 跨 batch 累加防丢精度)。
        # 这是 D 个元素 (相对 M*D 主流量微不足道), 没法 in-place 复用别人的 buffer。
        dw_partial = torch.zeros(D, device=x_2d.device, dtype=torch.float32)

        _rmsnorm_bwd_kernel[(M,)](
            x_2d, dy_2d, weight, rstd, dx, dw_partial,
            x_2d.stride(0), dy_2d.stride(0), dx.stride(0),
            D,
            BLOCK_N=ctx.BLOCK_N,
            num_warps=ctx.num_warps,
        )

        dw = dw_partial.to(weight.dtype)
        return dx.reshape(orig_shape), dw, None, None


# ---------------------------------------------------------------------------
# Inference-only in-place 入口
# ---------------------------------------------------------------------------


def fused_rmsnorm_inplace(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """推理路径: y 直接覆写 x, 省一份 (M, D) output buffer + 不存 rstd。

    使用约束:
      - x.requires_grad 必须为 False (autograd 不能在 in-place 后反传 — 原 x 没了)
      - x 必须 contiguous (kernel 假设 stride 是 D, 且 in-place 没法处理非连续布局)

    返回的张量与 x 共享存储, 写返回值 == 写 x。
    """
    assert x.is_cuda and weight.is_cuda
    assert not x.requires_grad, "fused_rmsnorm_inplace 不能在 autograd 下用 (会丢 x)"
    assert x.is_contiguous(), "fused_rmsnorm_inplace 要求 x.is_contiguous()"
    D = x.shape[-1]
    assert weight.shape == (D,)

    BLOCK_N = triton.next_power_of_2(D)
    if BLOCK_N > _MAX_BLOCK_N:
        raise ValueError(f"d_model={D} 超过单 block 上限 {_MAX_BLOCK_N}")

    x_2d = x.view(-1, D)
    M = x_2d.shape[0]
    # rstd 仍要 alloc (kernel 签名要求), 但完事就丢, 不进 ctx
    rstd_dummy = torch.empty(M, device=x.device, dtype=torch.float32)
    num_warps = _num_warps_for_block_n(BLOCK_N)

    # in-place: Y 指向 X, kernel 内 load → SRAM 算 → store 回原位
    _rmsnorm_fwd_kernel[(M,)](
        x_2d, x_2d, weight, rstd_dummy,
        x_2d.stride(0), x_2d.stride(0),
        D, eps,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
    )
    return x


# ---------------------------------------------------------------------------
# LayerNorm 占位 (本轮先只做 RMSNorm; LN 的实装思路相同, 多一步 mean 与 bias)
# ---------------------------------------------------------------------------


class FusedLayerNorm(torch.autograd.Function):
    """Fused LayerNorm forward/backward (Triton). 待实装。"""

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
