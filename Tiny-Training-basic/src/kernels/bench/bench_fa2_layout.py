"""Bench: BHSD-contig vs BSHD-physical 喂给 fa2 forward kernel 的实际差异。

要回答的具体问题:
  1. `.transpose(1,2).contiguous()` 单独多长时间? (这就是"省下来的那次 transpose")
  2. fa2 kernel 在 BHSD-contig (stride_qn=D) 下多长时间?
  3. fa2 kernel 在 BSHD-physical (stride_qn=H*D, 即从 QKV linear 直出 view) 下多长时间?
  4. 当前代码路径 (einops 出 view, 不 contig) 时, 等价于第 3 种, 把它也实测一遍对齐。

测的形状对齐 Qwen3-0.5B post-train 典型 workload (H_q=14, D=64).
"""

from __future__ import annotations

import torch

from tiny_training_basic.kernels.fa2 import _fa2_triton_forward_4d

# ---------------------------------------------------------------------------
# 工具: CUDA event 计时, 跑 N 次取中位数
# ---------------------------------------------------------------------------


def _bench(fn, *, warmup: int = 20, iters: int = 100) -> float:
    """返回单次 fn() 的中位数毫秒。"""
    # warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    # 为每次迭代做独立计时, 避免被批量 launch 的 graph 优化干扰
    times_ms = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times_ms.append(start.elapsed_time(end))
    times_ms.sort()
    return times_ms[len(times_ms) // 2]


# ---------------------------------------------------------------------------
# 三种 layout 下的 forward 调用
# ---------------------------------------------------------------------------


def make_inputs_bshd(B, L, H, D, dtype, device):
    """直接分配物理 (B, L, H, D)-contig 的 Q/K/V (= QKV linear 直出的形态)。"""
    qkv = torch.randn(3, B, L, H, D, device=device, dtype=dtype)
    Q_bshd = qkv[0]  # (B, L, H, D), contig
    K_bshd = qkv[1]
    V_bshd = qkv[2]
    return Q_bshd, K_bshd, V_bshd


def to_bhsd_view(t_bshd: torch.Tensor) -> torch.Tensor:
    """(B, L, H, D)-contig → (B, H, L, D) view (no copy). stride_qn = H*D."""
    return t_bshd.permute(0, 2, 1, 3)  # (B, H, L, D), 非 contig


def to_bhsd_contig(t_bshd: torch.Tensor) -> torch.Tensor:
    """(B, L, H, D)-contig → (B, H, L, D)-contig (real copy). stride_qn = D."""
    return t_bshd.permute(0, 2, 1, 3).contiguous()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


SHAPES = [
    # (B, L, H, D, label)
    (2, 2048, 14, 64, "Qwen3-0.5B-ish, B=2 L=2048"),
    (4, 1024, 14, 64, "B=4 L=1024"),
    (1, 4096, 14, 64, "long-seq, B=1 L=4096"),
    (8, 512,  14, 64, "small-seq large-batch, B=8 L=512"),
]


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    device = torch.device("cuda")
    dtype = torch.float16

    print(f"GPU: {torch.cuda.get_device_name(device)}, dtype={dtype}")
    print()

    header = (
        f"{'shape':<32} | {'transpose+contig (ms)':>22} | "
        f"{'kernel BSHD-physical (ms)':>26} | {'kernel BHSD-contig (ms)':>24} | "
        f"{'BHSD vs BSHD':>14}"
    )
    print(header)
    print("-" * len(header))

    for B, L, H, D, label in SHAPES:
        Q_bshd, K_bshd, V_bshd = make_inputs_bshd(B, L, H, D, dtype, device)

        # 1. 单独的 transpose+contiguous 时间 (Q+K+V 三次, 因为实际 attention 三个都要)
        def _transpose_contig():
            q = Q_bshd.permute(0, 2, 1, 3).contiguous()
            k = K_bshd.permute(0, 2, 1, 3).contiguous()
            v = V_bshd.permute(0, 2, 1, 3).contiguous()
            return q, k, v

        t_contig_ms = _bench(_transpose_contig)

        # 2. kernel 在 BSHD-physical (= 当前 einops 路径等价) 下的时间
        Q_view = to_bhsd_view(Q_bshd)
        K_view = to_bhsd_view(K_bshd)
        V_view = to_bhsd_view(V_bshd)
        # sanity: stride 应该是 (L*H*D, D, H*D, 1)
        assert Q_view.stride(-1) == 1
        assert Q_view.stride(-2) == H * D, f"expected stride_qn=H*D={H*D}, got {Q_view.stride(-2)}"
        assert Q_view.stride(-3) == D, f"expected stride_qh=D={D}, got {Q_view.stride(-3)}"

        def _fwd_bshd_physical():
            _fa2_triton_forward_4d(Q_view, K_view, V_view, is_causal=True)

        t_kernel_bshd_ms = _bench(_fwd_bshd_physical)

        # 3. kernel 在 BHSD-contig 下的时间
        Q_contig = to_bhsd_contig(Q_bshd)
        K_contig = to_bhsd_contig(K_bshd)
        V_contig = to_bhsd_contig(V_bshd)
        assert Q_contig.stride(-2) == D, f"expected stride_qn=D={D}, got {Q_contig.stride(-2)}"

        def _fwd_bhsd_contig():
            _fa2_triton_forward_4d(Q_contig, K_contig, V_contig, is_causal=True)

        t_kernel_bhsd_ms = _bench(_fwd_bhsd_contig)

        ratio = t_kernel_bhsd_ms / t_kernel_bshd_ms
        print(
            f"{label:<32} | {t_contig_ms:>22.4f} | "
            f"{t_kernel_bshd_ms:>26.4f} | {t_kernel_bhsd_ms:>24.4f} | "
            f"{ratio:>13.2f}x"
        )

    print()
    print("说明:")
    print("  - transpose+contig 这一列是 Q/K/V 三个 tensor 各做一次 .permute().contiguous() 的总耗时,")
    print("    代表 '走 BHSD-contig 路线时多付的一次 forward 转换'")
    print("  - kernel BSHD-physical = 输入是 (B, L, H, D)-contig 物理, 当作 (B, H, L, D) view 喂给")
    print("    kernel, 即 stride_qn = H*D。这个 = 当前 einops 路径, 也是新提议的 BSHD-direct 路径")
    print("  - kernel BHSD-contig = 多付一次 contig 后, 物理变成 (B, H, L, D)-contig, stride_qn = D")
    print("  - BHSD vs BSHD 列 = BHSD-contig kernel time / BSHD-physical kernel time")
    print("    >1 表示 contig 后 kernel 反而更慢 (memcpy 没赚回来), <1 表示 kernel 加速能补偿 memcpy")
    print()


if __name__ == "__main__":
    main()
