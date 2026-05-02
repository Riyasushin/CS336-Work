"""RMSNorm bench: PyTorch 现状 vs torch.nn.functional.rms_norm vs FusedRMSNorm。

为什么要这个 bench
----------------
RMSNorm 是典型 memory-bound kernel: 算力开销 (mean(x²) + sqrt + mul) 远小于 HBM
读写, 所以"压测加速比"和"压测带宽利用率"是同一件事。三条实装的差距正好揭示
HBM 流量的累积差:
  - `RRMSNorm` (现 PyTorch 实装): pow / mean / add / sqrt / div+mul 五步, 中间结果
                                   反复落 HBM, 理论 HBM 流量 ≈ 6× tensor_bytes
  - `F.rms_norm` (PyTorch 内置融合): C++ 端融合一次写, 理论 ≈ 2× tensor_bytes
  - `FusedRMSNorm` (本 repo Triton): 同上 ≈ 2× tensor_bytes, 验证我们没漏什么

bench 同时打 fwd-only, fwd+bwd 两套, 因为 bwd 的 IO 模式不同 (多 dY 一次 load,
多 dx 一次 store, 加上 dw 跨 batch reduce), 加速比和 fwd 不会一致。

"用好整张卡"的判据
----------------
对 memory-bound kernel, "用好"的硬指标是 HBM 带宽利用率 = 实测带宽 / GPU 峰值带宽。
- RTX 4060 Laptop: 272 GB/s (此机器, GDDR6 128-bit @ 17 Gbps)
- 实践经验: 写得好的融合 kernel 在大 N 下能做到 70-90% 带宽利用率

bench 会按下面公式估算实测带宽:
  bytes_fwd  = 2 * M * D * dtype_size  (load x + store y, weight 视为 cache hit)
  bytes_bwd  = 4 * M * D * dtype_size  (load x, dY; store dx; dw 跨行散乱写, 量小)
  GB/s = (bytes / 1e9) / (time_s)

怎么运行
-------
    uv run python -m tiny_training_basic.kernels.bench_norm
    uv run python -m tiny_training_basic.kernels.bench_norm --dtype bf16
    uv run python -m tiny_training_basic.kernels.bench_norm --shapes '4096,4096;4096,8192'
"""

from __future__ import annotations

import argparse
import gc

import torch
import torch.nn.functional as F

from tiny_training_basic.kernels.fused_norm import FusedRMSNorm


# ---------------------------------------------------------------------------
# 三条实装的统一接口: (x, weight, eps) -> y, 保证可以塞进同一套 timing 框架
# ---------------------------------------------------------------------------


def _impl_rrmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """复刻 RRMSNorm.forward (use_fused=False) 的 fp32 路径。"""
    in_dtype = x.dtype
    xf = x.to(torch.float32)
    rms = torch.sqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return ((xf / rms) * weight).to(in_dtype)


def _impl_torch_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """PyTorch 内置 (C++ 融合) baseline。"""
    return F.rms_norm(x, normalized_shape=(x.shape[-1],), weight=weight, eps=eps)


def _impl_fused_alloc(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """默认: bwd 单独 alloc dx (更快, 多用一份显存)。"""
    return FusedRMSNorm.apply(x, weight, eps, False)


def _impl_fused_inplace(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """in-place bwd: dx 写到 dY buffer (更省显存, 实测速度损失 ~25%)。"""
    return FusedRMSNorm.apply(x, weight, eps, True)


_IMPLS = {
    "torch_naive": _impl_rrmsnorm,
    "torch_rmsnorm": _impl_torch_rms_norm,
    "triton_alloc": _impl_fused_alloc,
    "triton_inplace": _impl_fused_inplace,
}


# ---------------------------------------------------------------------------
# 带宽估算
# ---------------------------------------------------------------------------


def _bytes_fwd(M: int, D: int, dtype: torch.dtype) -> int:
    """fwd 主流量: load x + store y. weight (D,) 单次加载视为 L2 cache hit, 忽略。"""
    return 2 * M * D * torch.finfo(dtype).bits // 8


def _bytes_fwd_bwd(M: int, D: int, dtype: torch.dtype) -> int:
    """fwd+bwd 总流量:
       fwd:  load x, store y                  (2 MD)
       bwd:  load x, load dY, store dx        (3 MD)
       dw atomic_add: 散乱写 D 个 fp32 位置, M 次 → 量级 MD * 4/dtype_size, 但 cache 命中后不计 HBM
    保守估 5 MD * dtype_size.
    """
    return 5 * M * D * torch.finfo(dtype).bits // 8


# ---------------------------------------------------------------------------
# Bench runner
# ---------------------------------------------------------------------------


def _bench_one(
    fn,
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    dy: torch.Tensor | None,  # None → fwd-only
    warmup: int,
    rep: int,
) -> tuple[float, float]:
    """跑 (warmup + rep) 次, 返回 (min ms, max peak MB)。"""
    # warmup: 跳 Triton JIT compile + CUDA cache 冷启动
    for _ in range(warmup):
        x_ = x.detach().clone().requires_grad_(dy is not None)
        w_ = weight.detach().clone().requires_grad_(dy is not None)
        y = fn(x_, w_, eps)
        if dy is not None:
            y.backward(dy)
    torch.cuda.synchronize()

    times_ms: list[float] = []
    peaks_mb: list[float] = []
    for _ in range(rep):
        x_ = x.detach().clone().requires_grad_(dy is not None)
        w_ = weight.detach().clone().requires_grad_(dy is not None)

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        y = fn(x_, w_, eps)
        if dy is not None:
            y.backward(dy)
        end.record()
        torch.cuda.synchronize()

        times_ms.append(start.elapsed_time(end))
        peaks_mb.append(torch.cuda.max_memory_allocated() / (1024**2))
    return min(times_ms), max(peaks_mb)


def _correctness_check(x: torch.Tensor, weight: torch.Tensor, eps: float) -> None:
    """跑 bench 前先对一次, 防止数错被当成"加速"。"""
    y_ref = _impl_rrmsnorm(x, weight, eps)
    for name, fn in _IMPLS.items():
        if name == "torch_naive":
            continue
        # in-place 的 fwd 没区别 (in-place 只在 bwd 生效), 跑 fwd 即可
        y = fn(x, weight, eps)
        # bf16/fp16 用宽 atol; 这里只是 sanity, 不是单测
        atol = 5e-3 if x.dtype != torch.float32 else 1e-5
        torch.testing.assert_close(y, y_ref, atol=atol, rtol=atol)


def run(
    shapes: list[tuple[int, int]],  # (M, D)  -- M = B*L 折叠后
    dtype: torch.dtype,
    peak_bw_gbps: float,
    warmup: int,
    rep: int,
    eps: float,
) -> None:
    print(f"{'mode':<5} {'impl':<16} {'M':>6} {'D':>5}  {'time_ms':>9}  {'peak_MB':>9}  {'GB/s':>7}  {'%peak':>6}")
    print("-" * 80)

    for (M, D) in shapes:
        torch.manual_seed(0)
        x = torch.randn(M, D, device="cuda", dtype=dtype) / 5
        weight = torch.randn(D, device="cuda", dtype=dtype) / 5
        dy = torch.randn(M, D, device="cuda", dtype=dtype) / 5

        try:
            _correctness_check(x, weight, eps)
        except Exception as e:
            print(f"!! correctness check failed at M={M} D={D} dtype={dtype}: {e}")
            continue

        for mode_label, dy_arg, byte_fn in [
            ("fwd", None, _bytes_fwd),
            ("f+b", dy, _bytes_fwd_bwd),
        ]:
            for name, fn in _IMPLS.items():
                try:
                    t_ms, pk_mb = _bench_one(fn, x, weight, eps, dy_arg, warmup, rep)
                    bytes_ = byte_fn(M, D, dtype)
                    gbps = (bytes_ / 1e9) / (t_ms * 1e-3)
                    pct = gbps / peak_bw_gbps * 100
                    print(
                        f"{mode_label:<5} {name:<16} {M:>6} {D:>5}  "
                        f"{t_ms:>9.3f}  {pk_mb:>9.1f}  {gbps:>7.1f}  {pct:>5.1f}%"
                    )
                except torch.cuda.OutOfMemoryError:
                    print(f"{mode_label:<5} {name:<16} {M:>6} {D:>5}  {'OOM':>9}")
                    torch.cuda.empty_cache()
                except Exception as e:
                    print(f"{mode_label:<5} {name:<16} {M:>6} {D:>5}  ERR: {type(e).__name__}: {e}"[:90])
                    torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dtype", default="fp16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--rep", type=int, default=10)
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument(
        "--peak-bw",
        type=float,
        default=272.0,
        help="GPU 峰值 HBM 带宽 (GB/s). 默认 272 = RTX 4060 Laptop 实测值",
    )
    parser.add_argument(
        "--shapes",
        type=str,
        default=None,
        help="分号分隔的 (M,D) 元组. 不传用默认扫表",
    )
    args = parser.parse_args()

    assert torch.cuda.is_available(), "bench 需要 CUDA"
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]

    if args.shapes is not None:
        shapes = [tuple(int(x) for x in s.split(",")) for s in args.shapes.split(";")]
    else:
        # 默认: 两档 d_model (Qwen3-0.5B 896, Llama-7B 4096), M ∈ {1k, 4k, 16k, 64k}
        shapes = [
            (1024, 896), (4096, 896), (16384, 896), (65536, 896),
            (1024, 4096), (4096, 4096), (16384, 4096),
        ]

    print(
        f"dtype={args.dtype}  device={torch.cuda.get_device_name(0)}  "
        f"peak_bw={args.peak_bw} GB/s  warmup={args.warmup}  rep={args.rep}"
    )
    run(shapes, dtype=dtype, peak_bw_gbps=args.peak_bw, warmup=args.warmup, rep=args.rep, eps=args.eps)


if __name__ == "__main__":
    main()
