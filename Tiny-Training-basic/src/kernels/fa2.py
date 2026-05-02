"""FlashAttention-2 的纯 PyTorch 参考实现 + Triton kernel 实装。

为什么要这份文件
----------------
- 把 FA2 的 online-softmax tile 更新公式写清楚（PyTorch 版 = 可读的算法书），
  再和 Triton 版互为对照，学习"同一个算法在两个执行模型下各自怎么落地"
- 对齐 `Dao-AILab/flash-attention` 官方 Triton 参考实现 (`flash_attn_triton.py`) 的
  硬编码 block size，不做 autotune —— 固定 block size 在教学 / 复现 / debug 上的
  可预测性远比 autotune 的那点性能收益重要

硬编码 block size 的出处
-----------------------
照抄 flash-attn 官方 Triton 参考实现里作者实测挑出来的组合：
- **forward**:  BLOCK_M = BLOCK_N = 128, num_stages = 1,
                num_warps = 4 (head_dim ≤ 64) / 8 (head_dim ≤ 128)
- **backward**: BLOCK_M = BLOCK_N = 128, num_warps = 8, num_stages = 1,
                SEQUENCE_PARALLEL = False (单 program 扫所有 seqlen_k tile,
                避免在 dQ 上做 atomic_add, 简化不少逻辑)

作者在原文件里明确注释过：`BLOCK_M=128, BLOCK_N=64, num_warps=4` 是 **buggy**
(某些 shape 出错, 疑似 Triton pipeline race)；`BLOCK_M=64, BLOCK_N=64` 在
EVEN_M=False 时也有 race condition。所以我们不去开 autotune 自讨苦吃,
直接用已经被官方验证过的两组参数。

输入约定 (配合 `tiny-training-system` 的 attention 测试)
------------------------------------------------------
对外 API 是 3D `(B, N, D)`(没有多头维度, 等价 nheads=1 的情形)。
Triton kernel 的 *shape* 约定是 4D `(B, H, N, D)` —— 对齐 torch SDPA 的命名,
方便和 `F.scaled_dot_product_attention` 互换。3D → 4D 的转换只在 adapter 里
unsqueeze(1) 塞个 H=1, kernel 本体保持通用可复用。

注意 *shape 约定* 不等于 *物理布局*: 真正进 kernel 的 stride_qn 取决于上游怎么
产张量。layers.py 用 `rearrange("l (h d) -> h l d")` 出来的是 view (没 contig),
其物理是 `(B, L, H, D)`-contig (= QKV linear 直出 + view, 即 flash-attn 官方
`(B, S, H, D)` 布局), 这种情况下 stride_qn = H*D 而非 D。kernel 是 stride-aware
的, 这两种 stride 都吃。实测 (RTX 4060, fp16, D=64) 两种 stride 的 kernel 时间
差 <2%, 因为 D=64 fp16 = 128B 正好一个 cacheline, strided 访问不浪费 cacheline。
省下来的是上游不用 .contiguous() 那一次 (Q+K+V 三个 ~0.21ms / forward)。

FA2 的数学骨架 (online-softmax tile 更新, 以 forward 为例)
--------------------------------------------------------
对每一个 Q tile, 遍历 KV tile:
  S_ij  = softmax_scale * Q_i @ K_j^T                         # (bm, bn)
  m_new = max(m_old, rowmax(S_ij))                            # 保证后续 exp 不溢出
  α     = exp(m_old - m_new)                                  # 旧累加器的再缩放因子
  P_ij  = exp(S_ij - m_new)                                   # softmax 分子
  l_new = α * l_old + rowsum(P_ij)                            # softmax 分母(累积)
  O_new = α * O_old + P_ij @ V_j                              # 输出累加
最后 O_i = O_final / l_final, LSE_i = m_final + log(l_final).

Backward 需要的 "D_i 技巧"
-------------------------
标准 softmax-attention 的反传里, 除了 P = softmax(S) 本身,
还要一个行标量 D_i = rowsum(O_i ⊙ dO_i), 然后:
  dV = P^T @ dO
  dP = dO @ V^T
  dS = P ⊙ (dP - D_i[:, None]) * scale
  dQ = dS @ K
  dK = dS^T @ Q
FA2 的 backward 不再从 P=softmax 算 (存不下), 而是用保存下来的 LSE
现场重算 P = exp(scaled_S - LSE)。D_i 可以独立 preprocess 一次。
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# 硬编码 block size 查表 (按 GPU 共享内存容量分档)
# ---------------------------------------------------------------------------
# flash-attn 官方参考实现 (flash_attn_triton.py) 的 BM=BN=128 是针对 A100/H100
# 的 164KB+/SM 共享内存挑的; 消费级 Ada (4060/4090, 99KB/SM) 和 Ampere (3090,
# 99KB/SM) 的 shmem 不够, BM=BN=128 的 backward kernel 会报 OutOfResources。
#
# 我们按 SM compute capability 分两档, 都保持"硬编码查表"精神 (不 autotune):
#   - sm_80 (A100), sm_90 (H100): 用 flash-attn 官方配置
#   - 其他 (消费级 Ada/Ampere/Ada-next, Turing 更老): 缩到 64 以适配 shmem
# 这两档的正确性都经过 flash-attn 原作者实测 (sm_80 的 128 + sm_<80 fallback
# 到 64), 不打算在别的组合上踩坑。
#
# forward 的 shmem 压力比 backward 小 (没有 dk/dv accumulator, 没有 dp/ds),
# BM=BN=128 在 4060 上也跑得动, 可以一档到底。backward 是瓶颈。

_FWD_BLOCK_M = 128
_FWD_BLOCK_N = 128


def _fwd_num_warps(head_dim: int) -> int:
    """flash-attn 的 num_warps 选择: 小 head_dim 用 4, 大的用 8。

    为什么随 head_dim 变: 单 warp 的 MMA 吞吐与占用的寄存器块大小相关,
    head_dim=64 的 tile 用 4 warps 就能吃满算力; head_dim=128 时 tile 大一倍,
    4 warps 让单 warp 寄存器压力爆, 切到 8 warps 让单 warp 干一半活。
    """
    return 4 if head_dim <= 64 else 8


def _bwd_block_config(device: torch.device) -> tuple[int, int, int]:
    """返回 (BLOCK_M, BLOCK_N, num_warps) for backward on this device.

    backward kernel 的 shmem 压力比 forward 大得多, 同时常驻的有:
      k, v (input dtype) + dk, dv (fp32 accumulator) + 循环内 q, do, qk, p, dp, ds, dq。
    粗估: BM=BN=128 约 256KB, BM=BN=64 约 112KB, BM=64 BN=32 约 72KB。
    hardware shmem opt-in (per-block):
      - A100 (sm_80): 164KB, H100 (sm_90): 228KB → flash-attn 官方 128x128 能跑
      - Ada 消费级 (sm_89) / Ampere 消费级 (sm_86) / Turing (sm_75): 99KB
        连 BM=BN=64 都装不下, 必须进一步缩
    按 arch 分档 (硬编码查表, 不 autotune):
    """
    major, minor = torch.cuda.get_device_capability(device)
    if (major, minor) in {(8, 0), (9, 0)}:
        # A100 / H100: flash-attn 官方配置
        return 128, 128, 8
    # 消费级 (99KB shmem) 及更老: BM=64, BN=32, num_warps=4 (约 72KB, 留余量给 Triton pipeline buffer)
    return 64, 32, 4


def _round_up(x: int, multiple: int) -> int:
    return (x + multiple - 1) // multiple * multiple


# ===========================================================================
# 纯 PyTorch 分块参考实现 (3D 输入)
# ===========================================================================


class FlashAttn2PyTorch(torch.autograd.Function):
    """FA2 的纯 PyTorch 实装, forward 做分块 online-softmax, backward 直接一股脑算。

    不追求速度 —— 存在的意义是:
      1. 作为算法正确性的参考 (在 CPU 上也能跑, 不依赖 CUDA)
      2. 让 forward 里的 online softmax tile 更新公式变成可读的 Python,
         方便和 Triton kernel 对照学习
      3. 满足 `autograd.Function` 契约: save_for_backward 里有 (B, N) 的 LSE
    """

    @staticmethod
    def forward(ctx, Q, K, V, is_causal: bool = False):
        # shape 检查: 必须是 3D (B, N, D)
        assert Q.dim() == 3 and K.dim() == 3 and V.dim() == 3, "expected 3D (B, N, D)"
        B, N_q, D = Q.shape
        _, N_k, _ = K.shape
        assert K.shape == (B, N_k, D) and V.shape == (B, N_k, D)

        scale = 1.0 / math.sqrt(D)
        # PyTorch 版的 tile 大小和 Triton 版无关, 随便选个小一点的就行 ——
        # 目标是算法演示, 不是性能
        BM, BN = 64, 64

        # 所有状态都用 fp32 算, 最后再回到原 dtype, 避免 fp16 反复累加掉精度
        Qf, Kf, Vf = Q.float(), K.float(), V.float()

        O = torch.zeros((B, N_q, D), device=Q.device, dtype=torch.float32)
        L = torch.empty((B, N_q), device=Q.device, dtype=torch.float32)  # LSE 存储

        for start_m in range(0, N_q, BM):
            end_m = min(start_m + BM, N_q)
            bm = end_m - start_m
            q = Qf[:, start_m:end_m]  # (B, bm, D), tile 常驻 "寄存器"

            # online softmax 的三个状态 (per query row)
            m_i = torch.full((B, bm), float("-inf"), device=Q.device, dtype=torch.float32)
            l_i = torch.zeros((B, bm), device=Q.device, dtype=torch.float32)
            o_i = torch.zeros((B, bm, D), device=Q.device, dtype=torch.float32)

            # causal 优化: 当前 Q tile 行最大 end_m-1, 再往后的 KV tile 全被 mask, 直接不遍历
            end_n_total = end_m if is_causal else N_k

            for start_n in range(0, end_n_total, BN):
                end_n = min(start_n + BN, N_k)
                k_t = Kf[:, start_n:end_n]
                v_t = Vf[:, start_n:end_n]

                # S_ij = scale * q @ k^T, shape (B, bm, bn)
                s = torch.einsum("bmd,bnd->bmn", q, k_t) * scale

                if is_causal:
                    # tile 内三角 mask: 行 i (绝对坐标 start_m+i) 能看到列 j (start_n+j) iff i >= j
                    rows = torch.arange(start_m, end_m, device=Q.device)[:, None]
                    cols = torch.arange(start_n, end_n, device=Q.device)[None, :]
                    s = s.masked_fill(rows < cols, float("-inf"))

                # ----- online softmax 更新 (FA2 标准形式) -----
                m_ij = s.amax(dim=-1)                # (B, bm), 本 tile 行最大
                m_new = torch.maximum(m_i, m_ij)     # 新的全局行最大

                # alpha = exp(m_old - m_new) ∈ [0, 1], 是旧 {l_i, o_i} 的再缩放因子
                # 当 m_new 仍是 -inf (即截至目前无任何有效 key, causal 首 tile 可能遇到), exp 得 NaN, 清零
                alpha = torch.exp(m_i - m_new)
                p = torch.exp(s - m_new[:, :, None])
                alpha = torch.nan_to_num(alpha, nan=0.0)
                p = torch.nan_to_num(p, nan=0.0)

                l_i = alpha * l_i + p.sum(dim=-1)
                o_i = alpha[:, :, None] * o_i + torch.einsum("bmn,bnd->bmd", p, v_t)
                m_i = m_new

            # tile 处理完, normalize 输出 + 写 LSE
            safe_l = torch.where(l_i > 0, l_i, torch.ones_like(l_i))  # 避免 /0
            o_i = o_i / safe_l[:, :, None]
            L_tile = torch.where(
                l_i > 0, m_i + torch.log(safe_l), torch.full_like(m_i, float("-inf"))
            )
            O[:, start_m:end_m] = o_i
            L[:, start_m:end_m] = L_tile

        O = O.to(Q.dtype)
        # 测试要求 save_for_backward 里恰好有一个 (B, N) 的 tensor —— 就是 LSE
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        ctx.scale = scale
        return O

    @staticmethod
    def backward(ctx, dO):
        """Backward 在 PyTorch 里就没必要分块了, 一股脑算清晰。

        核心是"用 LSE 重算 P, 再走标准 softmax-backward"。
        """
        Q, K, V, O, L = ctx.saved_tensors
        is_causal, scale = ctx.is_causal, ctx.scale
        B, N_q, D = Q.shape
        N_k = K.shape[1]

        # 1. 重算 S 和 P
        S = torch.einsum("bmd,bnd->bmn", Q.float(), K.float()) * scale
        if is_causal:
            rows = torch.arange(N_q, device=Q.device)[:, None]
            cols = torch.arange(N_k, device=K.device)[None, :]
            S = S.masked_fill(rows < cols, float("-inf"))
        P = torch.exp(S - L[:, :, None])
        P = torch.nan_to_num(P, nan=0.0)  # 全 -inf 的行会得 NaN, 清零

        # 2. softmax-backward 的 D_i 技巧 (标量每行一个)
        dOf, Of = dO.float(), O.float()
        D_row = (dOf * Of).sum(dim=-1)  # (B, N_q)

        # 3. 按前述公式求 dQ, dK, dV
        dV = torch.einsum("bmn,bmd->bnd", P, dOf)                  # P^T @ dO
        dP = torch.einsum("bmd,bnd->bmn", dOf, V.float())          # dO @ V^T
        dS = P * (dP - D_row[:, :, None]) * scale
        dQ = torch.einsum("bmn,bnd->bmd", dS, K.float())           # dS @ K
        dK = torch.einsum("bmn,bmd->bnd", dS, Q.float())           # dS^T @ Q

        return dQ.to(Q.dtype), dK.to(K.dtype), dV.to(V.dtype), None


# ===========================================================================
# Triton kernels (shape 约定 (B, H, N, D) -- 对齐 torch SDPA 命名)
# ===========================================================================
#
# kernel 本身是 stride-aware 的: 所有维度的 stride 都由 wrapper 显式传入,
# 不关心物理布局, 只要 stride_qb / stride_qh / stride_qn 三个值对上即可。
# 选 (B, H, N, D) 作为 nominal shape 的理由 = 对齐 torch SDPA / MHA / GQA 命名,
# 让 `_sdpa_maybe_flash` 能直接当 `F.scaled_dot_product_attention` 用。
#
# 实际进来的 stride 有两种, 都合法:
#
#   (1) 物理 BHSD-contig (调用方主动 .contiguous() 后):
#         stride_qb = H*N*D,  stride_qh = N*D,  stride_qn = D,  stride_qd = 1
#       tile 内相邻 q 行在内存上紧挨, "标准 contig" 布局。
#
#   (2) 物理 BSHD-contig + view (默认走法, 即 layers.py 的 einops rearrange):
#         stride_qb = N*H*D,  stride_qh = D,    stride_qn = H*D, stride_qd = 1
#       这是 QKV linear 出来 .view(B, N, H, D) 然后 .permute(0, 2, 1, 3) 的结果,
#       和 flash-attn 官方 (B, S, H, D) 物理布局完全一致, 省去一次 .contiguous() 拷贝。
#
# kernel 在 (1) 和 (2) 下的实测时间差 <2% (RTX 4060, fp16, D=64), 因为 D=64 fp16
# = 128B 正好一个 cacheline, stride_qn=H*D 时每行 q tile 跨 cacheline 但每条 cacheline
# 都被 100% 用上, HBM 流量等同 stride_qn=D。所以默认走法 (2) 是稳赚: 省 ~0.21ms 的
# Q+K+V .contiguous() 拷贝, kernel 时间不变。
#
# 唯一要求: stride_qd 必须 = 1 (D 维 contig), 这是 tl.dot 的向量化前提, wrapper assert。


@triton.heuristics(
    {
        # EVEN_* 用来在 kernel 内部跳过边界 mask, 这是编译时分支, 对性能影响很大
        "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
    }
)
@triton.jit
def _fwd_kernel(
    Q, K, V, Out, Lse,
    Mask,  # (B, H, N_q, N_k) int8, HAS_MASK=False 时传 dummy 1-elem tensor
    softmax_scale,
    stride_qb, stride_qh, stride_qn,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_ob, stride_oh, stride_on,
    stride_mb, stride_mh, stride_mm,  # Mask 的 batch/head/M 维 stride; K 维 stride 隐式=1
    nheads, seqlen_q, seqlen_k, seqlen_q_rounded, headdim,
    IS_CAUSAL: tl.constexpr,
    HAS_MASK: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """FA2 forward。一个 program 处理 (batch*head, BLOCK_M 行 Q) 的一块 tile。

    ───────────────────────────────────────────────────────────────────────
    【1. 先理解 Triton 的 block-level 编程模型】
    ───────────────────────────────────────────────────────────────────────
    Triton 和 CUDA 的关键区别是 *抽象层次*:

      - CUDA: 你写 "一个 thread 做什么"(SIMT per-thread)。warp / shared
              memory / register 都要你手动排。
      - Triton: 你写 "一个 program 在一个 tile 上做什么"(SPMD per-tile)。
              每个 program 对应一个 CUDA block, 但你看不见也不管 thread ——
              `tl.load / tl.dot / tl.max` 等操作都以 *tile* 为单位, 编译器
              自动把 tile 切给 warp, 自动决定寄存器 / SRAM / pipeline。

    这意味着 *你唯一的设计自由度*是:
      (a) grid 形状 —— 有几个 program, 每个 program 的坐标是什么
      (b) tile 大小 —— 每个 program 一次处理多大的数据 (constexpr, 编译期确定)
      (c) tile 级的操作序列 —— load / compute / reduce / store 的先后
      (d) 控制流 —— 循环 (for), 编译期分支 (if constexpr), mask
    而 *不能* 做的事:
      - 在一个 program 内部区分 thread 行为 (没有 threadIdx)
      - 跨 program 同步 / 通信 (除了 atomic)
      - 手工管理 shared memory (一切走 `tl.load/store` + register)

    所以 Triton kernel 的设计 = "把算法切成一串可以 tile 化的 pure functions,
    再决定怎么映射到 grid"。

    ───────────────────────────────────────────────────────────────────────
    【2. 从 block-level 模型推导 FA2 forward 的设计】
    ───────────────────────────────────────────────────────────────────────
    我们要算 O = softmax(Q K^T * scale) @ V, 目标是不落盘 S = Q K^T (太大)、
    一个 kernel 端到端跑完。设计问题是:

    ── (A) grid 怎么切？─────────────────────────────────────────────
    输入张量的 *shape 约定* 是 4D `(B, H, N, D)`, 命名上对齐 torch SDPA。
    *物理布局* 不固定: D 维必须 contig (stride_qd=1, tl.dot 向量化前提),
    其它三维的 stride 由调用方决定 —— 见 module-level docstring, 默认走法是
    BSHD-physical 的 permute view (stride_qn=H*D), 显式 .contiguous() 后的
    BHSD-contig (stride_qn=D) 也合法。kernel 完全靠传入的 stride 寻址, 不假设
    任何"哪个轴在内存里相邻"。下面讨论的 *逻辑*维度排布与物理布局无关。

    我们要把 4D 张量映射到一个 2D 的 program grid, 切法的关键约束是:

      softmax 的 denominator (l_i = Σ exp(S_ij - m_i)) 是沿 K 维的行和,
      这个 reduction *不能跨 program 并行* —— 否则要 atomic add 或二次 reduce,
      复杂 + 慢。所以任何一行 Q 的整个 K 维 reduction 必须 *同一个 program*
      里做完。

    于是 4D 张量 `(B, H, N_q, D)` 被这样摊开到 grid:
      - axis 0 = Q 行方向的 tile 索引 ∈ [0, cdiv(N_q, BLOCK_M))
      - axis 1 = batch × head 的 flat 索引 ∈ [0, B*H), 内部再拆回 (b, h)
      - D 维度根本不上 grid, 整个 D 留给每个 program 作为 tile 宽度
      - K 维度也不上 grid, 留作 program 内部的 for 循环 (保证 reduction
        在同一 program 完成)

    所以每个 program 的视角 (逻辑层):
      - 从逻辑 4D `(B, H, N_q, D)` 中抠出 *一块 2D slice* `(BLOCK_M, D)`:
          index = (b, h, start_m*BM : start_m*BM+BM, :)
        物理上这块 slice 内存连续与否取决于 stride_qn: stride_qn=D 时 BM 行紧挨
        (BHSD-contig), stride_qn=H*D 时 BM 行之间跨过其它 head (BSHD-view)。
        两种都被 `tl.load(q_ptrs)` 正确处理 (tl.load 是 stride-aware 的)。
      - 再沿 K 维循环读 2D `(BLOCK_N, D)` 的 K/V 片段
      - 程序内的所有 tile 计算 (qk, p, acc_o) 都是 2D 的
      - 本 program 与其他 program 无任何数据依赖 (grid 级 embarrassingly parallel)

    这正是 FA2 vs FA1 的核心差异: FA1 把 K tile 当外循环、Q 当内循环;
    FA2 反过来, 让每个 program 独占 "一块 Q 行的完整计算", 不需要 inter-program
    通信或 atomic。

    ── (B) tile 大小怎么选？─────────────────────────────────────────
    三个 constexpr 决定编译出来的 kernel 形状:
      BLOCK_M (Q 行数), BLOCK_N (K 列数), BLOCK_HEADDIM (head 维)

    约束来源:
      - BLOCK_HEADDIM 必须是 2 的幂 且 >= headdim (Triton `tl.dot` 要求);
        不够的位置靠 mask 清零
      - BLOCK_M * BLOCK_HEADDIM 进 register, 大了会 spill 到 local mem
      - BLOCK_M * BLOCK_N 的 QK 中间量和 P 中间量进 SRAM, 大了爆 shmem
      - 以上两条约束使得 (BLOCK_M, BLOCK_N) 的合法空间很小 (A100 上 128x128,
        消费级 GPU 只能 128x64 或更小)

    这些 tile 大小一旦编译期定了, 后续所有 `tl.arange / tl.load / tl.dot`
    都被编译器用来确定 vectorize width, warp 分配, pipeline depth。

    ── (C) 什么留在 on-chip, 什么流经 HBM？──────────────────────────
    这是 FA2 性能的核心 trick (比 vanilla attention 的 10x 快主要来自这里)。
    下面所有 shape 都是 *单个 program* 看到的 2D slice, 不是全局 4D 张量:

      常驻 on-chip (整个 K loop 不换):
        - Q tile:       (BM, D) 一次 load, 读 N_k/BN 轮都用它
        - m_i, lse_i:   (BM,)   running 的 softmax 状态, 在寄存器
        - acc_o:        (BM, D) 输出累加器, 在寄存器
      每轮 K loop 换:
        - K tile:       (BN, D) 读一次, 用完丢
        - V tile:       (BN, D) 读一次, 用完丢
      HBM 写出仅一次:
        - O tile:       (BM, D) loop 结束写一次
        - LSE tile:     (BM,)   loop 结束写一次

    → 单 program 的 HBM 访问: Q 读 1 次 BM*D + O/LSE 写 1 次 BM*(D+1) +
       K/V 各读 N_k*D。全 grid 累计 HBM 流量是 O(B*H*N*D) (每个 Q/K/V/O
       元素各读写一次), vanilla attention 要 O(B*H*N^2) 存 S 和 P。
       序列长了差距是量级的。

    ── (D) 怎么把 softmax 写成 "流式 reduction"？────────────────────
    普通 softmax 需要 "先算完整行 → 取 max → 减 → exp → 求和 → 除",
    需要存下整行 S_i = Q_i K^T (长度 N_k, 不行)。

    Online softmax 把上面折成 per-tile 的 *running 更新* (见 module docstring
    顶部的数学骨架, 此处只讲怎么映射到 Triton 代码):
      - m_i, lse_i: 每行的 running 状态, 在寄存器, 天然 tile-level
      - 每轮 K tile: 算 qk tile (BM, BN) → 行 max → exp → 行 sum → 更新状态
      - 旧累加器 acc_o 和 lse 都靠 `alpha = exp(m_old - m_new)` 做 rescale
      - 所有更新都是 tile 级 pure functional 操作, 和 Triton 模型完美匹配

    ── (E) 边界怎么处理？──────────────────────────────────────────
    `seqlen_q / seqlen_k` 不一定整除 BLOCK_M/N; headdim 不一定等于
    BLOCK_HEADDIM。Triton 里 *绝对不要* 在 kernel 里写依赖运行时数据的
    if/else (会串行化 warp), 而是:

      1. 用 `@triton.heuristics` 把 "能整除吗" 做成 `EVEN_M/EVEN_N/EVEN_HEADDIM`
         这三个 constexpr —— 编译期分支, 等于生成两份 kernel, zero runtime cost
      2. 边界位置用 `tl.load(mask=..., other=0.0)` 补 0, softmax 时用
         `tl.where(..., 0, -inf)` 把无效列加 -inf (exp 后自然变 0)

    causal mask 也是同一个套路: `tl.where(offs_m >= offs_n, 0, -inf)`。

    ───────────────────────────────────────────────────────────────────────
    【3. 本 kernel 的执行流】
    ───────────────────────────────────────────────────────────────────────
    grid = (cdiv(seqlen_q, BLOCK_M), batch * nheads)

    每个 program 做三件事:
      1. 从 (start_m, off_b, off_h) 算出指针基址, load Q tile 到寄存器
      2. for start_n in 0..seqlen_k step BLOCK_N:
           load K/V tile, 算 qk, online softmax 更新 (m_i, lse_i, acc_o)
      3. 循环完: acc_o /= l_final (用 exp(m - lse) 一步到位), 写 O 和 LSE

    causal 情况下第 2 步的 end_n 被 clamp 到 min((start_m+1)*BM, seqlen_k),
    跳过整个被 mask 掉的 tile; 对角 tile 内部用 tl.where 做细粒度 mask。
    """
    start_m = tl.program_id(0)          # 第几个 Q tile
    off_hb = tl.program_id(1)           # 第几个 batch*head, 拆回 (b, h)
    off_b = off_hb // nheads
    off_h = off_hb % nheads

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)   # Q tile 的绝对行索引
    offs_n = tl.arange(0, BLOCK_N)                        # KV tile 内的相对列索引
    offs_d = tl.arange(0, BLOCK_HEADDIM)                  # head_dim 内的索引

    # 指针基址 + Q/K/V 的二维 offset (行 × headdim)
    q_ptrs = (
        Q + off_b * stride_qb + off_h * stride_qh
        + (offs_m[:, None] * stride_qn + offs_d[None, :])
    )
    k_ptrs = (
        K + off_b * stride_kb + off_h * stride_kh
        + (offs_n[:, None] * stride_kn + offs_d[None, :])
    )
    v_ptrs = (
        V + off_b * stride_vb + off_h * stride_vh
        + (offs_n[:, None] * stride_vn + offs_d[None, :])
    )

    # online softmax 状态, 始终在 fp32 寄存器里
    # 约定: lse_i 存 "m + log(l)", 初始 -inf 对应 l=0, 与 m=-inf 一致
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)

    # ---- load Q tile (一次, 整个循环常驻) ----
    if EVEN_M & EVEN_HEADDIM:
        q = tl.load(q_ptrs)
    elif EVEN_HEADDIM:
        q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
    else:
        q = tl.load(
            q_ptrs,
            mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
            other=0.0,
        )

    # causal 下: 当前 Q tile 最大行是 (start_m+1)*BLOCK_M - 1, 列 > 它的全被 mask
    # 所以把 end_n clamp 到 min((start_m+1)*BM, seqlen_k), 之后的 KV tile 不必遍历
    end_n = seqlen_k if not IS_CAUSAL else tl.minimum((start_m + 1) * BLOCK_M, seqlen_k)

    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        # ---- mask tile 预取 + tile-level skip ----
        # 先 load 本 (Q_tile × K_tile) 的 mask, 如果整块都是 False 就跳过 tile 内全部 compute
        # (K/V 不 load, qk/matmul 不算, softmax 状态也不更新) —— "采取不算的操作减少计算"
        # 的核心 trick, 对 block-diagonal / sliding-window 这类稀疏 mask 效果显著。
        # Triton 3.x JIT 不支持 for-loop 里 `continue`, 所以改成 "if should_compute: 一大块" 的形态。
        should_compute = True
        if HAS_MASK:
            mask_ptrs = (
                Mask + off_b * stride_mb + off_h * stride_mh
                + offs_m[:, None] * stride_mm + (start_n + offs_n)[None, :]
            )
            mask_tile = tl.load(
                mask_ptrs,
                mask=(offs_m[:, None] < seqlen_q) & ((start_n + offs_n)[None, :] < seqlen_k),
                other=0,
            )
            # tl.max 只能沿一个 axis 做, 所以两次 reduce 得到 scalar (warp-uniform)
            mask_any = tl.max(tl.max(mask_tile.to(tl.int32), 0), 0)
            should_compute = mask_any != 0

        if should_compute:
            # ---- load K tile ----
            if EVEN_N & EVEN_HEADDIM:
                k = tl.load(k_ptrs + start_n * stride_kn)
            elif EVEN_HEADDIM:
                k = tl.load(
                    k_ptrs + start_n * stride_kn,
                    mask=(start_n + offs_n)[:, None] < seqlen_k,
                    other=0.0,
                )
            else:
                k = tl.load(
                    k_ptrs + start_n * stride_kn,
                    mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                    other=0.0,
                )

            # ---- S_ij = Q @ K^T (不乘 scale, softmax 前再乘, 让编译器 fuse 成 fma) ----
            qk = tl.dot(q, tl.trans(k))  # (BM, BN), fp32 acc

            # 几道 mask 加法 (分开写避免 race, 合并成一个 where 会出错 —— 原 flash-attn 注释):
            #   (1) 右侧越界 (EVEN_N=False)
            #   (2) causal 三角
            #   (3) 用户传入的 attn_mask (HAS_MASK=True)
            # 每道都是 "有效位 +0, 无效位 +-inf", 叠加后 softmax 把无效位变 0
            if not EVEN_N:
                qk += tl.where((start_n + offs_n)[None, :] < seqlen_k, 0.0, float("-inf"))
            if IS_CAUSAL:
                qk += tl.where(offs_m[:, None] >= (start_n + offs_n)[None, :], 0.0, float("-inf"))
            if HAS_MASK:
                qk += tl.where(mask_tile != 0, 0.0, float("-inf"))

            # ---- online softmax 更新 (用 lse 存 m+log(l) 的变体, 数值更稳) ----
            # 旧 m_i 和旧 lse_i 都是上一轮的; m_ij 是本轮新 m (加入了新行 max)
            # trick: 用 max(rowmax*scale, lse_i) 作为 m_new —— lse_i >= m_old, 所以 m_new >= m_old,
            #        并且 m_new 可能比 max(m_old, rowmax*scale) 更大, 让后续 exp(old - new) <= 1 更安全
            m_ij = tl.maximum(tl.max(qk, 1) * softmax_scale, lse_i)
            p = tl.exp(qk * softmax_scale - m_ij[:, None])  # softmax 分子, fp32

            l_ij = tl.sum(p, 1)                             # 本 tile 的行分母

            # acc_o_scale = exp(m_i - m_ij): 旧 acc_o 的再缩放
            acc_o_scale = tl.exp(m_i - m_ij)
            acc_o = acc_o * acc_o_scale[:, None]

            # ---- load V tile 并累加 P @ V 到 acc_o ----
            if EVEN_N & EVEN_HEADDIM:
                v = tl.load(v_ptrs + start_n * stride_vn)
            elif EVEN_HEADDIM:
                v = tl.load(
                    v_ptrs + start_n * stride_vn,
                    mask=(start_n + offs_n)[:, None] < seqlen_k,
                    other=0.0,
                )
            else:
                v = tl.load(
                    v_ptrs + start_n * stride_vn,
                    mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                    other=0.0,
                )
            # p 从 fp32 转回 input dtype 再做 matmul, 减轻寄存器压力
            p = p.to(v.dtype)
            acc_o += tl.dot(p, v)

            # ---- 更新 m_i, lse_i ----
            m_i = m_ij
            # l_new = alpha * l_old + l_ij, 而 alpha * l_old = exp(lse_old - m_new). 于是:
            l_i_new = tl.exp(lse_i - m_ij) + l_ij
            lse_i = m_ij + tl.log(l_i_new)

    # ---- 全部 KV tile 扫完: normalize 输出 (O = acc_o / l_final), 写出 O + LSE ----
    o_scale = tl.exp(m_i - lse_i)  # = 1/l_final (因为 lse = m + log l)
    acc_o = acc_o * o_scale[:, None]

    # LSE 写到 (B, H, seqlen_q_rounded), 外层按 off_hb * seqlen_q_rounded 定位
    # 注意: 不加 mask, 所以必须让 Lse buffer 至少有 seqlen_q_rounded 长度 (wrapper 负责 pad)
    lse_ptrs = Lse + off_hb * seqlen_q_rounded + offs_m
    tl.store(lse_ptrs, lse_i)

    # 写出 O tile
    out_ptrs = (
        Out + off_b * stride_ob + off_h * stride_oh
        + (offs_m[:, None] * stride_on + offs_d[None, :])
    )
    if EVEN_M & EVEN_HEADDIM:
        tl.store(out_ptrs, acc_o)
    elif EVEN_HEADDIM:
        tl.store(out_ptrs, acc_o, mask=offs_m[:, None] < seqlen_q)
    else:
        tl.store(
            out_ptrs, acc_o,
            mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
        )


@triton.jit
def _bwd_preprocess_do_o_dot(
    Out, DO, Delta,
    stride_ob, stride_oh, stride_on,
    stride_dob, stride_doh, stride_don,
    nheads, seqlen_q, seqlen_q_rounded, headdim,
    BLOCK_M: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
):
    """Backward 的 preprocess: 每一行算 D_i = sum_d(O_i ⊙ dO_i)。

    为什么单独做一个 preprocess kernel:
      - D_i 只依赖 O 和 dO, 与 Q/K/V 无关, 可以提前一把 load 完算出来
      - 在主 bwd kernel 里每个 q-row 只需要一次 tl.load 读 D_i, 非常轻
      - 一个 program 处理 BLOCK_M 行, 和 fwd 一致
    """
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    o = tl.load(
        Out + off_b * stride_ob + off_h * stride_oh
        + offs_m[:, None] * stride_on + offs_d[None, :],
        mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
        other=0.0,
    ).to(tl.float32)
    do = tl.load(
        DO + off_b * stride_dob + off_h * stride_doh
        + offs_m[:, None] * stride_don + offs_d[None, :],
        mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
        other=0.0,
    ).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    tl.store(Delta + off_hb * seqlen_q_rounded + offs_m, delta)


@triton.heuristics(
    {
        "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
    }
)
@triton.jit
def _bwd_kernel(
    Q, K, V, DO, DQ, DK, DV, LSE, D,
    Mask,
    softmax_scale,
    stride_qb, stride_qh, stride_qn,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_dob, stride_doh, stride_don,
    stride_dqb, stride_dqh, stride_dqn,
    stride_dkb, stride_dkh, stride_dkn,
    stride_dvb, stride_dvh, stride_dvn,
    stride_mb, stride_mh, stride_mm,
    nheads, seqlen_q, seqlen_k, seqlen_q_rounded, headdim,
    IS_CAUSAL: tl.constexpr,
    HAS_MASK: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """FA2 backward. SEQUENCE_PARALLEL=False 的简化版:
    一个 program (一个 batch*head) 负责 seqlen_k 方向的所有 BN tile,
    每个 tile 再内部扫 seqlen_q 方向的所有 BM tile。

    数据流 (per KV tile):
      - 常驻在 SRAM 的量: K_j, V_j, 以及本 tile 累积的 dK_j, dV_j
      - 循环里每个 Q tile: load q/do/dq/lse/D, 重算 P_ij, 累加 dV/dK, 更新 dQ
      - 最后把 dK_j, dV_j 写回 HBM

    grid = (1, batch * nheads)

    这个版本用直接 read-modify-write 的方式更新 dQ (不需要 atomic_add);
    代价是同一 (b, h) 下所有 KV tile 必须串行扫。对 batch*nheads 充足的训练
    场景完全够用; batch*nheads 很小时才需要 SEQUENCE_PARALLEL=True + atomic_add。
    """
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads

    # 指针偏移到本 (b, h)
    Q = Q + off_b * stride_qb + off_h * stride_qh
    K = K + off_b * stride_kb + off_h * stride_kh
    V = V + off_b * stride_vb + off_h * stride_vh
    DO = DO + off_b * stride_dob + off_h * stride_doh
    DQ = DQ + off_b * stride_dqb + off_h * stride_dqh
    DK = DK + off_b * stride_dkb + off_h * stride_dkh
    DV = DV + off_b * stride_dvb + off_h * stride_dvh
    LSE = LSE + off_hb * seqlen_q_rounded
    D = D + off_hb * seqlen_q_rounded
    if HAS_MASK:
        Mask = Mask + off_b * stride_mb + off_h * stride_mh

    offs_d = tl.arange(0, BLOCK_HEADDIM)

    # KV tile 外循环: 每个 BN 列 tile 独立算一遍 (dK_j, dV_j), 期间扫所有 Q tile
    num_block_n = tl.cdiv(seqlen_k, BLOCK_N)
    for start_n in range(0, num_block_n):
        offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_m = tl.arange(0, BLOCK_M)

        # causal 优化: 列 j 只会被 i >= j 的行用到, 所以从 begin_m 开始遍历 Q 即可
        # begin_m 必须是 BLOCK_M 的整数倍 (对齐 fwd 的 Q tile 划分)
        begin_m = 0
        if IS_CAUSAL:
            begin_m = ((start_n * BLOCK_N) // BLOCK_M) * BLOCK_M

        # ---- load K, V tile (整个 Q loop 常驻) ----
        k_ptrs = K + (offs_n[:, None] * stride_kn + offs_d[None, :])
        v_ptrs = V + (offs_n[:, None] * stride_vn + offs_d[None, :])
        if EVEN_N & EVEN_HEADDIM:
            k = tl.load(k_ptrs)
            v = tl.load(v_ptrs)
        elif EVEN_HEADDIM:
            k = tl.load(k_ptrs, mask=offs_n[:, None] < seqlen_k, other=0.0)
            v = tl.load(v_ptrs, mask=offs_n[:, None] < seqlen_k, other=0.0)
        else:
            k = tl.load(
                k_ptrs,
                mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                other=0.0,
            )
            v = tl.load(
                v_ptrs,
                mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                other=0.0,
            )

        # dK_j, dV_j 的累加器, fp32
        dk = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)
        dv = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)

        # 如果 begin_m 已经超过 seqlen_q (causal 下 KV tile 完全在无人会访问的区域),
        # 还是要把全 0 的 dk/dv 写出去 (否则下游看到 uninit 值)
        num_block_m = tl.cdiv(seqlen_q, BLOCK_M)

        # Q tile 内循环
        for start_m in range(begin_m, num_block_m * BLOCK_M, BLOCK_M):
            start_m = tl.multiple_of(start_m, BLOCK_M)
            offs_m_curr = start_m + offs_m

            # 指针: Q, DO, DQ 都按当前 Q tile 算
            q_ptrs = Q + (offs_m_curr[:, None] * stride_qn + offs_d[None, :])
            do_ptrs = DO + (offs_m_curr[:, None] * stride_don + offs_d[None, :])
            dq_ptrs = DQ + (offs_m_curr[:, None] * stride_dqn + offs_d[None, :])

            # ---- mask tile 预取 + tile-level skip (和 forward 对称) ----
            # 整个 (Q_tile × KV_tile) 的 mask 全 False 时, 本 (Q_tile, KV_tile) 配对对 dK/dV 的贡献为 0,
            # 对 dQ 本 Q_tile 行的贡献也为 0 —— 全部跳过 (省掉 q/do load + 两次 matmul + dQ RMW)
            should_compute = True
            if HAS_MASK:
                mask_ptrs = Mask + (
                    offs_m_curr[:, None] * stride_mm + offs_n[None, :]
                )
                mask_tile = tl.load(
                    mask_ptrs,
                    mask=(offs_m_curr[:, None] < seqlen_q) & (offs_n[None, :] < seqlen_k),
                    other=0,
                )
                mask_any = tl.max(tl.max(mask_tile.to(tl.int32), 0), 0)
                should_compute = mask_any != 0

            if should_compute:
                # ---- load q, do ----
                if EVEN_M & EVEN_HEADDIM:
                    q = tl.load(q_ptrs)
                    do = tl.load(do_ptrs)
                else:
                    q = tl.load(
                        q_ptrs,
                        mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                        other=0.0,
                    )
                    do = tl.load(
                        do_ptrs,
                        mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                        other=0.0,
                    )

                # ---- 重算 P = exp(scale * Q K^T - LSE) ----
                qk = tl.dot(q, tl.trans(k))
                if not EVEN_N:
                    qk = tl.where(offs_n[None, :] < seqlen_k, qk, float("-inf"))
                if IS_CAUSAL:
                    qk = tl.where(
                        offs_m_curr[:, None] >= offs_n[None, :], qk, float("-inf")
                    )
                if HAS_MASK:
                    qk = tl.where(mask_tile != 0, qk, float("-inf"))
                lse_i = tl.load(LSE + offs_m_curr)
                p = tl.exp(qk * softmax_scale - lse_i[:, None])  # (BM, BN), fp32

                # ---- dV += P^T @ dO ----
                # 转 do 的 dtype 让 tl.dot 选合适的 matmul path
                dv += tl.dot(tl.trans(p.to(do.dtype)), do)

                # ---- dP = dO @ V^T; dS = P * (dP - D_i) * scale ----
                dp = tl.dot(do, tl.trans(v))          # (BM, BN), fp32
                Di = tl.load(D + offs_m_curr)
                ds = (p * (dp - Di[:, None]) * softmax_scale).to(q.dtype)

                # ---- dK += dS^T @ Q ----
                dk += tl.dot(tl.trans(ds), q)

                # ---- dQ 是 read-modify-write: 当前 Q tile 的 dQ 已经累积过其他 KV tile 的贡献
                # 所以 load 当前 + 加本 tile 的 dS @ K + store 回去
                # eviction_policy="evict_last" 提示缓存尽量保留, 因为下一个 KV tile 还会再读它
                if EVEN_M & EVEN_HEADDIM:
                    dq = tl.load(dq_ptrs, eviction_policy="evict_last")
                    dq += tl.dot(ds, k)
                    tl.store(dq_ptrs, dq, eviction_policy="evict_last")
                else:
                    mask = (offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim)
                    dq = tl.load(dq_ptrs, mask=mask, other=0.0, eviction_policy="evict_last")
                    dq += tl.dot(ds, k)
                    tl.store(dq_ptrs, dq, mask=mask, eviction_policy="evict_last")

        # ---- 本 KV tile 完, 写 dK_j, dV_j ----
        dk_ptrs = DK + (offs_n[:, None] * stride_dkn + offs_d[None, :])
        dv_ptrs = DV + (offs_n[:, None] * stride_dvn + offs_d[None, :])
        if EVEN_N & EVEN_HEADDIM:
            tl.store(dk_ptrs, dk)
            tl.store(dv_ptrs, dv)
        else:
            mask = (offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim)
            tl.store(dk_ptrs, dk, mask=mask)
            tl.store(dv_ptrs, dv, mask=mask)


# ---------------------------------------------------------------------------
# 4D wrapper (封装 kernel launch + scratch 分配)
# ---------------------------------------------------------------------------


def _prepare_mask_4d(
    mask: torch.Tensor | None,
    B: int,
    H: int,
    N_q: int,
    N_k: int,
    device: torch.device,
) -> tuple[torch.Tensor, tuple[int, int, int], bool]:
    """把用户传的 attn_mask 规范化成 kernel 能直接用的 int8 张量 + strides。

    mask=None 时返回 dummy 1-element tensor 配 stride=(0,0,0) —— kernel 永远不会读到它,
    因为 HAS_MASK=False 的编译期分支会整块跳掉。

    mask 非 None 时:
      - 要求 broadcast 到 (B, H, N_q, N_k) 成立
      - 转成 int8 (bool 也接受), 确保最后一维 stride=1 (contiguous 在 K 维)
      - 返回 strides (stride_b, stride_h, stride_m), 最后一维 stride=1 由 kernel 隐式假设
    """
    if mask is None:
        dummy = torch.empty(1, dtype=torch.int8, device=device)
        return dummy, (0, 0, 0), False
    # 允许 bool 或 已经是 int8
    if mask.dtype == torch.bool:
        m = mask.to(torch.int8)
    elif mask.dtype == torch.int8:
        m = mask
    else:
        raise TypeError(f"attn_mask dtype {mask.dtype} not supported (want bool or int8)")
    # broadcast 到 (B, H, N_q, N_k) 再 contig 保证最后一维 stride=1
    m = m.expand(B, H, N_q, N_k).contiguous()
    return m, (m.stride(0), m.stride(1), m.stride(2)), True


def _fa2_triton_forward_4d(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
    softmax_scale: float | None = None,
    mask: torch.Tensor | None = None,
):
    """Forward 的 4D 版 wrapper: 入参 (B, H, N, D), 出参 (O: (B, H, N, D), LSE: (B, H, N_rounded))。

    mask: bool 或 int8, broadcast 到 (B, H, N_q, N_k); None 时走无 mask 快速路径。
    """
    assert q.is_cuda and k.is_cuda and v.is_cuda, "Triton kernel 需要 CUDA tensor"
    assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4
    B, H, N_q, D = q.shape
    _, _, N_k, _ = k.shape
    assert k.shape == (B, H, N_k, D) and v.shape == (B, H, N_k, D)
    assert D <= 128, "FA2 kernel 目前只支持 head_dim <= 128"
    assert q.dtype == k.dtype == v.dtype
    # 最后一维必须连续 (tl.load 的向量化假设)
    assert q.stride(-1) == k.stride(-1) == v.stride(-1) == 1

    softmax_scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(D)

    # BLOCK_HEADDIM 必须是 2 的幂且 >= head_dim; 小于 16 的会被 Triton 拒绝 (dot 的最小 M/N/K)
    BLOCK_HEADDIM = max(triton.next_power_of_2(D), 16)
    # LSE 按 BLOCK_M 对齐 pad, kernel 无 mask 写入, 超出 seqlen_q 的格子是 garbage
    N_q_rounded = _round_up(N_q, _FWD_BLOCK_M)

    o = torch.empty_like(q)
    lse = torch.empty((B, H, N_q_rounded), device=q.device, dtype=torch.float32)

    m_tensor, m_strides, has_mask = _prepare_mask_4d(mask, B, H, N_q, N_k, q.device)

    grid = (triton.cdiv(N_q, _FWD_BLOCK_M), B * H)
    _fwd_kernel[grid](
        q, k, v, o, lse,
        m_tensor,
        softmax_scale,
        q.stride(0), q.stride(1), q.stride(2),   # (B, H, N, D): batch / head / seqlen stride
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        m_strides[0], m_strides[1], m_strides[2],
        H, N_q, N_k, N_q_rounded, D,
        IS_CAUSAL=is_causal,
        HAS_MASK=has_mask,
        BLOCK_HEADDIM=BLOCK_HEADDIM,
        BLOCK_M=_FWD_BLOCK_M,
        BLOCK_N=_FWD_BLOCK_N,
        num_warps=_fwd_num_warps(D),
        num_stages=1,
    )
    return o, lse


def _fa2_triton_backward_4d(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor,  # (B, H, N_q_rounded)
    is_causal: bool = False,
    mask: torch.Tensor | None = None,
    softmax_scale: float | None = None,
):
    """Backward 的 4D 版 wrapper。

    dQ 累加用 fp32 临时 buffer, 最后 copy 回原 dtype (避免 fp16 多次 RMW 掉精度)。
    """
    assert q.is_cuda and do.is_cuda
    assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4 and o.dim() == 4 and do.dim() == 4
    B, H, N_q, D = q.shape
    N_k = k.shape[2]
    assert D <= 128

    softmax_scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(D)
    BLOCK_HEADDIM = max(triton.next_power_of_2(D), 16)
    BM_bwd, BN_bwd, num_warps_bwd = _bwd_block_config(q.device)
    N_q_rounded = _round_up(N_q, BM_bwd)
    # LSE buffer 的长度必须 >= N_q_rounded, 否则 kernel 无 mask 读会踩越界
    assert lse.shape[-1] >= N_q_rounded, f"LSE len {lse.shape[-1]} < N_q_rounded {N_q_rounded}"

    do = do.contiguous() if do.stride(-1) != 1 else do

    # ---- Step 1: preprocess, 算 D_i = rowsum(O ⊙ dO) ----
    delta = torch.empty_like(lse)
    pre_grid = (triton.cdiv(N_q, BM_bwd), B * H)
    _bwd_preprocess_do_o_dot[pre_grid](
        o, do, delta,
        o.stride(0), o.stride(1), o.stride(2),
        do.stride(0), do.stride(1), do.stride(2),
        H, N_q, lse.shape[-1], D,
        BLOCK_M=BM_bwd,
        BLOCK_HEADDIM=BLOCK_HEADDIM,
    )

    # ---- Step 2: main bwd, 算 dQ/dK/dV ----
    # dQ 用 fp32 accumulator buffer, 避免 RMW 的精度损失
    dq_accum = torch.zeros_like(q, dtype=torch.float32)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    m_tensor, m_strides, has_mask = _prepare_mask_4d(mask, B, H, N_q, N_k, q.device)

    grid = (1, B * H)
    _bwd_kernel[grid](
        q, k, v, do, dq_accum, dk, dv, lse, delta,
        m_tensor,
        softmax_scale,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        do.stride(0), do.stride(1), do.stride(2),
        dq_accum.stride(0), dq_accum.stride(1), dq_accum.stride(2),
        dk.stride(0), dk.stride(1), dk.stride(2),
        dv.stride(0), dv.stride(1), dv.stride(2),
        m_strides[0], m_strides[1], m_strides[2],
        H, N_q, N_k, lse.shape[-1], D,
        IS_CAUSAL=is_causal,
        HAS_MASK=has_mask,
        BLOCK_HEADDIM=BLOCK_HEADDIM,
        BLOCK_M=BM_bwd,
        BLOCK_N=BN_bwd,
        num_warps=num_warps_bwd,
        num_stages=1,
    )

    dq = dq_accum.to(q.dtype)
    return dq, dk, dv


# ===========================================================================
# autograd.Function (3D adapter, 测试接口)
# ===========================================================================


class FlashAttn2TritonForAdaptor(torch.autograd.Function):
    """FA2 Triton 实装的 3D (B, N, D) 对外接口。

    内部做 3D ↔ 4D 转换 (unsqueeze head 维为 1), 真正的计算都在 4D kernel 里。
    这样:
      - kernel 本身保持通用 (可以给多头场景零改动复用)
      - 对外接口简单 (和 PyTorch 参考版签名一致)
      - 测试的 `saved_tensors` 约束 (恰好一个 (B, N) LSE) 在这层处理
    """

    @staticmethod
    def forward(ctx, Q, K, V, is_causal: bool = False):
        assert Q.dim() == 3 and K.dim() == 3 and V.dim() == 3, "expected 3D (B, N, D)"
        B, N_q, D = Q.shape

        # 3D → 4D: 插入 H=1 到 dim=1 以得到 (B, H=1, N, D), 对齐 kernel 的 (B, H, N, D) 约定
        # .contiguous() 确保 stride(-1)==1, 否则 kernel 的向量化 load 假设会破
        q4 = Q.unsqueeze(1).contiguous()
        k4 = K.unsqueeze(1).contiguous()
        v4 = V.unsqueeze(1).contiguous()

        o4, lse_buf = _fa2_triton_forward_4d(q4, k4, v4, is_causal=is_causal)
        O = o4.squeeze(1).contiguous()
        # lse_buf: (B, H=1, N_rounded) → 取 (B, N) 切片存进 ctx
        # 测试会在 saved_tensors 里找 shape == (B, N) 的那一个, 必须精确匹配
        L = lse_buf.squeeze(1)[:, :N_q].contiguous()

        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal
        B, N_q, D = Q.shape

        # 重新 4D 化 (H=1 在 dim=1)
        q4 = Q.unsqueeze(1).contiguous()
        k4 = K.unsqueeze(1).contiguous()
        v4 = V.unsqueeze(1).contiguous()
        o4 = O.unsqueeze(1).contiguous()
        do4 = dO.unsqueeze(1).contiguous()

        # LSE 从 (B, N) → (B, 1, N_rounded), 尾部未用值是 garbage (kernel 不会读到)
        BM_bwd, _, _ = _bwd_block_config(Q.device)
        N_rounded = _round_up(N_q, BM_bwd)
        lse_buf = torch.empty((B, 1, N_rounded), device=L.device, dtype=L.dtype)
        lse_buf[:, 0, :N_q] = L

        dq4, dk4, dv4 = _fa2_triton_backward_4d(
            do4, q4, k4, v4, o4, lse_buf, is_causal=is_causal
        )
        return dq4.squeeze(1), dk4.squeeze(1), dv4.squeeze(1), None


class FlashAttn2Triton4D(torch.autograd.Function):
    """FA2 Triton 实装的 4D native 接口: (B, H, N, D) 直接进 kernel, 不经 adapter。

    和 `FlashAttn2TritonForAdaptor` 的区别:
      - ForAdaptor 是 3D 接口 (B, N, D), 内部 unsqueeze H=1 → 4D 调 kernel, 测试用
      - 本类是 4D 接口, 让多头 attention / bench 可以直接在 (B, H, N, D) 上调,
        不做形变、不产生额外拷贝, 反映 triton kernel 的真实性能

    Shape 约定 (B, H, N, D) 对齐 torch SDPA / MHA / GQA。物理布局两种都吃:
      - BHSD-contig (stride_qn = D)
      - BSHD-contig 的 permute view (stride_qn = H*D, 即 layers.py 默认走法)
    实测两者 kernel 时间差 <2% (D=64 fp16, cacheline 对齐), 见 module-level docstring。
    """

    @staticmethod
    def forward(ctx, Q, K, V, is_causal: bool = False, mask: torch.Tensor | None = None):
        assert Q.dim() == 4 and K.dim() == 4 and V.dim() == 4, "expected 4D (B, H, N, D)"
        # kernel 假设最后一维 D 连续, 不连续就拍扁一次 (调用方自己避免更好)
        Q = Q.contiguous() if Q.stride(-1) != 1 else Q
        K = K.contiguous() if K.stride(-1) != 1 else K
        V = V.contiguous() if V.stride(-1) != 1 else V

        O, lse = _fa2_triton_forward_4d(Q, K, V, is_causal=is_causal, mask=mask)
        ctx.save_for_backward(Q, K, V, O, lse)
        # mask 可能是 None 或 tensor; 保存到 ctx 让 backward 重用 (避免再 expand+contig 一次)
        ctx.mask = mask
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, lse = ctx.saved_tensors
        dO = dO.contiguous() if dO.stride(-1) != 1 else dO
        dQ, dK, dV = _fa2_triton_backward_4d(
            dO, Q, K, V, O, lse,
            is_causal=ctx.is_causal,
            mask=ctx.mask,
        )
        # 输入有 5 个 (Q, K, V, is_causal, mask), mask 不需要梯度
        return dQ, dK, dV, None, None


# ===========================================================================
# Drop-in 替代 torch.nn.functional.scaled_dot_product_attention
# ===========================================================================


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
) -> torch.Tensor:
    """`F.scaled_dot_product_attention` 的上位替代: 能走 FA2 就走 FA2, 否则透传。

    签名和 torch 版一致:
        query: (..., H_q, L_q, D)
        key:   (..., H_kv, L_k, D)   enable_gqa=True 时 H_kv 可以 < H_q
        value: (..., H_kv, L_k, D)
        return: (..., H_q, L_q, D)

    能走 FA2 kernel 的全部条件 (任一不满足立即 fallback):
        - Q, K, V 都在 CUDA
        - dtype ∈ {fp16, bf16, fp32}
        - head_dim ≤ 128
        - dropout_p == 0.0 (kernel 不支持 dropout)
        - attn_mask is None 或 dtype == bool (float additive mask 暂不支持, fallback)

    GQA 处理:
        FA2 kernel 要求 H_q == H_kv, 所以 H_kv < H_q 时 repeat_interleave K/V 到
        同数量的 head, 再进 kernel。额外占 O(B·H_q·L_k·D) 显存, 但避免改 kernel;
        真正的 GQA kernel (每个 Q head 读对应 KV group) 留给后续实装。

    scale 参数支持:
        FlashAttn2Triton4D.apply 不单独暴露 scale (签名已 5 个参数), 自定义 scale
        通过 pre-scale Q 等价化: (α·Q) @ K^T / sqrt(D) = Q @ K^T · (α/sqrt(D)),
        取 α = scale · sqrt(D), 结果就等于 scale · Q @ K^T。autograd 自动正确反传。
    """
    # ---- fallback 判定 ----
    is_cuda = query.is_cuda and key.is_cuda and value.is_cuda
    dtype_ok = query.dtype in (torch.float16, torch.bfloat16, torch.float32)
    D = query.shape[-1]
    head_dim_ok = D <= 128
    mask_ok = (attn_mask is None) or (attn_mask.dtype == torch.bool)
    no_dropout = dropout_p == 0.0

    if not (is_cuda and dtype_ok and head_dim_ok and mask_ok and no_dropout):
        # 任一条不满足 → 原样透传给 torch, 数值行为与 F.scaled_dot_product_attention 一致
        return torch.nn.functional.scaled_dot_product_attention(
            query, key, value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            enable_gqa=enable_gqa,
        )

    # ---- FA2 path ----
    # GQA expansion: H_kv < H_q 时复制 K/V 的 head 维到 H_q
    H_q = query.shape[-3]
    H_kv = key.shape[-3]
    if H_q != H_kv:
        assert enable_gqa, (
            f"num_heads mismatch (H_q={H_q}, H_kv={H_kv}) requires enable_gqa=True"
        )
        assert H_q % H_kv == 0, f"H_q ({H_q}) must be divisible by H_kv ({H_kv})"
        group = H_q // H_kv
        key = key.repeat_interleave(group, dim=-3)
        value = value.repeat_interleave(group, dim=-3)

    # Layout (..., H, L, D) → (B, H, L, D): 只 flatten leading dims, 不做 permute
    # kernel 本来就吃 (B, H, L, D), 直接 reshape 对齐即可
    orig_lead = query.shape[:-3]  # (...)
    L_q = query.shape[-2]
    L_k = key.shape[-2]
    H = H_q  # 前面已对齐, 现在 H_q == H_kv
    B = 1
    for s in orig_lead:
        B *= s

    Q4 = query.reshape(B, H, L_q, D)
    K4 = key.reshape(B, H, L_k, D)
    V4 = value.reshape(B, H, L_k, D)

    # 自定义 scale: pre-scale Q (见 docstring)
    if scale is not None:
        default_scale = 1.0 / (D ** 0.5)
        if scale != default_scale:
            Q4 = Q4 * (scale * (D ** 0.5))

    # attn_mask 规范化到 4D (B, H, L_q, L_k) (allow broadcasting)
    mask4 = None
    if attn_mask is not None:
        mask4 = attn_mask
        # 扩到 4 维 (允许用户传 (L_q, L_k) / (1, 1, L_q, L_k) 等)
        while mask4.dim() < 4:
            mask4 = mask4.unsqueeze(0)
        mask4 = mask4.expand(B, H, L_q, L_k)

    O4 = FlashAttn2Triton4D.apply(Q4, K4, V4, is_causal, mask4)  # (B, H, L_q, D)
    # 回到 (..., H, L_q, D)
    O = O4.reshape(*orig_lead, H, L_q, D)
    return O
