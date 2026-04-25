"""FA2 PyTorch vs Triton 实装的 benchmark (4D 输入工作量下)。

为什么要这个 bench
----------------
`fa2.py` 里有两份实装对比:
  - `FlashAttn2PyTorch`       纯 PyTorch 分块 (3D 接口), 学习正确性, 性能底
  - `FlashAttn2Triton4D`      Triton JIT kernel (4D native), 对齐 flash-attn 硬编码
接口形态不同, bench 里按它们各自最自然的方式调:
  - torch 版: 4D 输入 → 外层 for-loop 遍历 head, 每个 head 切出 3D (B, N, D) 跑 FA2PyTorch,
              结果拼回 4D。反映 "PyTorch 实装没有跨 head grid 并行" 这个真实代价。
  - triton 版: 4D 输入直接喂给 4D kernel, grid 沿 (B*H) 并行, 不做 for-loop head。

测的量:
  - 峰值 GPU 显存 (`max_memory_allocated`, fwd+bwd 整个调用期间)
  - 吞吐 (TFLOPS, fwd+bwd 总 FLOPs / GPU elapsed)

怎么运行
-------
uv 工作区根目录下:
    uv run python -m tiny_training_basic.kernels.bench
    uv run python -m tiny_training_basic.kernels.bench --causal --dtype fp16
    uv run python -m tiny_training_basic.kernels.bench --shapes '2,1024,8,64'
"""

from __future__ import annotations

import argparse
import gc

import torch

from tiny_training_basic.kernels.fa2 import (
    FlashAttn2PyTorch,
    FlashAttn2Triton4D,
)


# ---------------------------------------------------------------------------
# 4D 适配层 —— bench 专用
# ---------------------------------------------------------------------------


def _torch_fa2_4d(
    Q4: torch.Tensor,
    K4: torch.Tensor,
    V4: torch.Tensor,
    is_causal: bool,
) -> torch.Tensor:
    """For-loop head 跑 FlashAttn2PyTorch, 拼回 4D (B, H, N, D)。

    这是 PyTorch 实装的真实场景: 没有 head 并行, 只能串行循环。
    用 stack 让 autograd 能正确反传。
    """
    B, H, N, D = Q4.shape
    outputs = []
    for h in range(H):
        # 切 3D slice: (B, N, D). head 维在 dim=1 (torch 约定)
        q3 = Q4.select(1, h).contiguous()
        k3 = K4.select(1, h).contiguous()
        v3 = V4.select(1, h).contiguous()
        o3 = FlashAttn2PyTorch.apply(q3, k3, v3, is_causal)  # (B, N, D)
        outputs.append(o3)
    # 拼回 (B, H, N, D): stack 在 dim=1
    return torch.stack(outputs, dim=1)


def _triton_fa2_4d(
    Q4: torch.Tensor,
    K4: torch.Tensor,
    V4: torch.Tensor,
    is_causal: bool,
) -> torch.Tensor:
    """4D 输入直接调 Triton kernel, 不做 head 循环。"""
    return FlashAttn2Triton4D.apply(Q4, K4, V4, is_causal)


_IMPLS: dict[str, callable] = {
    "torch_FA2": _torch_fa2_4d,
    "triton_FA2": _triton_fa2_4d,
}


# ---------------------------------------------------------------------------
# FLOPs 估算
# ---------------------------------------------------------------------------


def _fa_flops(B: int, H: int, N: int, D: int, is_causal: bool) -> float:
    """fwd + bwd 的总 FLOPs 估算。

    fwd: Q K^T (2 BH N² D) + P V (2 BH N² D) = 4 BH N² D
    bwd: dV = P^T dO, dP = dO V^T, dS=..., dQ = dS K, dK = dS^T Q (4 matmul)
         + 重算 S ≈ 2 BH N² D, 总 ≈ 10 BH N² D
    合计 ≈ 14 BH N² D (non-causal); causal 近似为 half (实际略高, 这里不苛求)。
    """
    fwd = 4.0 * B * H * N * N * D
    bwd = 10.0 * B * H * N * N * D
    total = fwd + bwd
    if is_causal:
        total *= 0.5
    return total


# ---------------------------------------------------------------------------
# Bench runner
# ---------------------------------------------------------------------------


def _bench_one(
    fn: callable,
    Q4: torch.Tensor,
    K4: torch.Tensor,
    V4: torch.Tensor,
    dO4: torch.Tensor,
    is_causal: bool,
    warmup: int,
    rep: int,
) -> tuple[float, float]:
    """跑 (warmup + rep) 次 fwd+bwd, 返回 (min 耗时 ms, max 峰值显存 MB)。"""
    # warmup: 跳过 Triton JIT compile + CUDA cache 冷启动
    for _ in range(warmup):
        q = Q4.detach().clone().requires_grad_(True)
        k = K4.detach().clone().requires_grad_(True)
        v = V4.detach().clone().requires_grad_(True)
        o = fn(q, k, v, is_causal)
        o.backward(dO4)
    torch.cuda.synchronize()

    times_ms: list[float] = []
    peaks_mb: list[float] = []
    for _ in range(rep):
        q = Q4.detach().clone().requires_grad_(True)
        k = K4.detach().clone().requires_grad_(True)
        v = V4.detach().clone().requires_grad_(True)

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        o = fn(q, k, v, is_causal)
        o.backward(dO4)
        end.record()
        torch.cuda.synchronize()

        times_ms.append(start.elapsed_time(end))
        peaks_mb.append(torch.cuda.max_memory_allocated() / (1024**2))

    return min(times_ms), max(peaks_mb)


def run(
    shapes: list[tuple[int, int, int, int]],
    dtype: torch.dtype,
    is_causal: bool,
    warmup: int,
    rep: int,
) -> None:
    header = f"{'impl':<11} {'B':>2} {'H':>3} {'N':>5} {'D':>3}  {'time_ms':>9}  {'peak_MB':>9}  {'TFLOPS':>7}"
    print(header)
    print("-" * len(header))

    for (B, H, N, D) in shapes:
        torch.manual_seed(0)
        # randn / 10 压制量级, 避免 fp16 下 exp(scaled_S) overflow
        Q4 = torch.randn(B, H, N, D, device="cuda", dtype=dtype) / 10
        K4 = torch.randn(B, H, N, D, device="cuda", dtype=dtype) / 10
        V4 = torch.randn(B, H, N, D, device="cuda", dtype=dtype) / 10
        dO4 = torch.randn(B, H, N, D, device="cuda", dtype=dtype) / 10

        flops = _fa_flops(B, H, N, D, is_causal)

        for name, fn in _IMPLS.items():
            try:
                t_ms, pk_mb = _bench_one(fn, Q4, K4, V4, dO4, is_causal, warmup, rep)
                tflops = flops / (t_ms * 1e-3) / 1e12
                print(
                    f"{name:<11} {B:>2} {H:>3} {N:>5} {D:>3}  {t_ms:>9.2f}  {pk_mb:>9.1f}  {tflops:>7.2f}"
                )
            except torch.cuda.OutOfMemoryError:
                print(f"{name:<11} {B:>2} {H:>3} {N:>5} {D:>3}  {'OOM':>9}  {'-':>9}  {'-':>7}")
                torch.cuda.empty_cache()
            except Exception as e:
                msg = f"ERR: {type(e).__name__}: {e}"
                print(f"{name:<11} {B:>2} {H:>3} {N:>5} {D:>3}  {msg}"[:110])
                torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--causal", action="store_true", help="apply causal mask")
    parser.add_argument(
        "--dtype",
        default="fp32",
        choices=["fp32", "fp16", "bf16"],
        help="input dtype (fp32 默认; fp16/bf16 更接近真实训练)",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--rep", type=int, default=5)
    parser.add_argument(
        "--shapes",
        type=str,
        default=None,
        help="分号分隔的 (B,H,N,D) 元组, 比如 '2,8,512,64;2,8,1024,64'. 不传用默认",
    )
    args = parser.parse_args()

    assert torch.cuda.is_available(), "bench 需要 CUDA 来实测 torch/triton 的 GPU 性能"
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]

    if args.shapes is not None:
        shapes = [tuple(int(x) for x in s.split(",")) for s in args.shapes.split(";")]
    else:
        # 默认集: 固定 B=2, H=8, D=64 (Qwen3-0.5B 风格), 扫 N ∈ 小/中/大
        shapes = [
            (2, 8, 256, 64),
            (2, 8, 512, 64),
            (2, 8, 1024, 64),
            (2, 8, 2048, 64),
        ]

    print(
        f"dtype={args.dtype}  causal={args.causal}  warmup={args.warmup}  rep={args.rep}  "
        f"device={torch.cuda.get_device_name(0)}"
    )
    run(shapes, dtype=dtype, is_causal=args.causal, warmup=args.warmup, rep=args.rep)


if __name__ == "__main__":
    main()
