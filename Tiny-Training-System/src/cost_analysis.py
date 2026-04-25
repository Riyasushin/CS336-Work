"""PA3 Part 2: 训练成本核算 + Scaling-law 求最优 N, D。

包含三个对外接口（按 PA3 PDF 要求的签名）：
    * ``model_training_cost_analysis_llama(config_path)``
    * ``model_training_cost_analysis_deepseek(config_path)``
    * ``get_optimal_N_D_from_cost(budget_usd)``

—— 设计与计量约定 ——
* 1 次 (M, K) × (K, N) matmul 计 ``2·M·N·K`` FLOPs（乘加各算 1）。
* 训练阶段总计算量按 Kaplan / Chinchilla 经典近似 ``C ≈ 6 N D``：前向 2ND，
  反向 4ND（输入 grad + 权重 grad）。
* 题面说 "fixed fp16"，本模块按 8 bytes/param（权重 2 + 梯度 2 + Adam m,v 各 2）
  的口径估算训练 state；activations 一律 fp16。
* 题面要求 "checkpoint rematerialization at each transformer boundary"，
  解读为：每个 layer 边界保存 1 份 ``(B, S, H)`` 残差，重做该层 forward 时
  其余 intermediates 才同时存活。所以 peak ≈
    全模型 (W + g + opt) + L · (B,S,H) boundary + 1 层 working activations。
* FLOPs 的 ``per-layer forward`` 默认按 ``B=1, S=max_position_embeddings``
  评估，便于和 PDF 范例比对。

—— Scaling law (PDF 给定) ——
    L(N, D) = 406.4 / N^0.34 + 410.7 / D^0.29 + 1.69
约束 ``6 N D = C``，Lagrange 解析解：
    α A · D^β = β B · N^α   ⇒   D = ((βB)/(αA))^(1/β) · N^(α/β)
代入约束闭式解 N，再回推 D。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Tuple

# ============================================================================
# 通用常量
# ============================================================================
FP16 = 2  # bytes
FP32 = 4
FLOPS_PER_TOKEN_PARAM = 6.0  # forward 2 + backward 4

# "fixed fp16" 训练口径：W (fp16) + g (fp16) + Adam m,v (fp16)
WEIGHT_BYTES = FP16
GRAD_BYTES = FP16
OPT_BYTES_PER_PARAM = 2 * FP16  # m + v

# Scaling law 系数
ALPHA = 0.34
BETA = 0.29
A_COEF = 406.4
B_COEF = 410.7
L0 = 1.69

# GPU 选项（题面 2.2 给定）
MFU = 0.40
GPU_SPECS = {
    # name: (price USD/hour, FP16 peak TFLOPS)
    "A100": {"price_per_hour": 4.0, "tflops_fp16_peak": 312.0},
    "V100": {"price_per_hour": 2.5, "tflops_fp16_peak": 125.0},
    "T4":   {"price_per_hour": 1.0, "tflops_fp16_peak": 65.0},
}


def _load_config(path) -> dict:
    return json.loads(Path(path).read_text())


# ============================================================================
# Part 2.1: Llama 类（dense, MHA/GQA + SwiGLU + RMSNorm + RoPE）
# ============================================================================

def _llama_layer_params(H: int, I: int, h: int, h_kv: int) -> int:
    """单层参数：4 个投影 + 3 个 SwiGLU 矩阵 + 2 个 RMSNorm scale。

    GQA 下 K/V 投影是 ``H × (h_kv·d)``，不是 ``H × H``。
    Llama-7B 是 MHA（h_kv == h），Llama-2/3 起部分尺寸用 GQA。
    """
    d = H // h
    proj_q = H * H
    proj_k = H * (h_kv * d)
    proj_v = H * (h_kv * d)
    proj_o = H * H
    mlp = 3 * H * I  # gate / up / down
    norm = 2 * H  # input_layernorm + post_attention_layernorm
    return proj_q + proj_k + proj_v + proj_o + mlp + norm


def _llama_total_params(cfg: dict) -> int:
    H = cfg["hidden_size"]
    I = cfg["intermediate_size"]
    L = cfg["num_hidden_layers"]
    V = cfg["vocab_size"]
    h = cfg["num_attention_heads"]
    h_kv = cfg.get("num_key_value_heads", h)
    tied = cfg.get("tie_word_embeddings", False)

    embed = V * H
    lm_head = 0 if tied else V * H
    final_norm = H
    layers = L * _llama_layer_params(H, I, h, h_kv)
    # Llama 用 RoPE，无 learned positional embedding（题面要求列出，这里显式 = 0）
    pos_embed = 0
    return embed + pos_embed + lm_head + final_norm + layers


def _llama_layer_fwd_flops(cfg: dict, B: int, S: int) -> int:
    H = cfg["hidden_size"]
    I = cfg["intermediate_size"]
    h = cfg["num_attention_heads"]
    h_kv = cfg.get("num_key_value_heads", h)
    d = H // h

    # 投影 (Q, K, V, O)
    proj = 0
    proj += 2 * B * S * H * H                # Q
    proj += 2 * B * S * H * (h_kv * d)       # K
    proj += 2 * B * S * H * (h_kv * d)       # V
    proj += 2 * B * S * H * H                # O

    # 标准 attention (non-flash)：
    #   QK^T: (B, h, S, d) × (B, h, d, S) -> 2·B·h·S²·d
    #   softmax × V: (B, h, S, S) × (B, h, S, d) -> 2·B·h·S²·d
    # GQA 下计算阶段 K/V 仍要 broadcast 到 h 路（参数省，计算不省）
    attn = 2 * (2 * B * h * S * S * d)

    # SwiGLU MLP: gate, up, down 各 1 次 matmul = 6·B·S·H·I
    mlp = 6 * B * S * H * I
    return proj + attn + mlp


def _llama_layer_peak_activation_bytes(cfg: dict, B: int, S: int) -> int:
    """单层 forward 期峰值 activation 字节（fp16，不开 FlashAttention）。

    数清楚同时存活的 intermediates：
      - 输入残差（边界 save）
      - Q, K, V 投影输出
      - attention scores + softmax_out
      - attention output (h·d 合并)
      - mlp gate / up / silu(gate)*up
      - mlp 输出
    """
    H = cfg["hidden_size"]
    I = cfg["intermediate_size"]
    h = cfg["num_attention_heads"]
    h_kv = cfg.get("num_key_value_heads", h)
    d = H // h

    saved_input = B * S * H
    qkv = B * S * H + 2 * B * S * (h_kv * d)
    attn_scores = 2 * B * h * S * S  # scores + softmax_out
    attn_out = B * S * H
    mlp_inter = 3 * B * S * I  # gate / up / silu(gate)*up
    mlp_out = B * S * H

    elements = saved_input + qkv + attn_scores + attn_out + mlp_inter + mlp_out
    return FP16 * elements


def model_training_cost_analysis_llama(model_config_path) -> Tuple[int, float, float]:
    """Llama 类模型成本核算。

    Returns:
        (total_params, fwd_TFLOPs_per_layer, peak_memory_GB)
    """
    cfg = _load_config(model_config_path)
    B = 1
    S = cfg.get("max_position_embeddings") or cfg.get("max_sequence_length")

    P_total = _llama_total_params(cfg)
    L = cfg["num_hidden_layers"]
    H = cfg["hidden_size"]

    fwd_flops_layer = _llama_layer_fwd_flops(cfg, B, S)
    peak_act_layer = _llama_layer_peak_activation_bytes(cfg, B, S)

    model_state = (WEIGHT_BYTES + GRAD_BYTES + OPT_BYTES_PER_PARAM) * P_total
    boundary_saves = FP16 * L * B * S * H
    peak_bytes = model_state + boundary_saves + peak_act_layer

    return P_total, fwd_flops_layer / 1e12, peak_bytes / (1024 ** 3)


# ============================================================================
# Part 2.3: DeepSeek-V3（MLA + MoE FFN，前 first_k_dense_replace 层是 dense）
# ============================================================================

def _mla_layer_params(cfg: dict) -> int:
    """MLA (Multi-head Latent Attention) 参数：

    q_a_proj   : H -> q_lora_rank
    q_b_proj   : q_lora_rank -> h · (qk_nope + qk_rope)
    kv_a_proj_with_mqa : H -> (kv_lora_rank + qk_rope)   # 含 decoupled key-rope
    kv_b_proj  : kv_lora_rank -> h · (qk_nope + v_head)  # W_UK || W_UV concat
    o_proj     : h · v_head -> H
    + q_a_layernorm (q_lora_rank), kv_a_layernorm (kv_lora_rank)
    """
    H = cfg["hidden_size"]
    h = cfg["num_attention_heads"]
    qk_nope = cfg["qk_nope_head_dim"]
    qk_rope = cfg["qk_rope_head_dim"]
    v_head = cfg["v_head_dim"]
    q_lora = cfg["q_lora_rank"]
    kv_lora = cfg["kv_lora_rank"]

    q_a = H * q_lora
    q_b = q_lora * h * (qk_nope + qk_rope)
    kv_a = H * (kv_lora + qk_rope)
    kv_b = kv_lora * h * (qk_nope + v_head)
    o = (h * v_head) * H
    norms = q_lora + kv_lora
    return q_a + q_b + kv_a + kv_b + o + norms


def _mla_layer_fwd_flops(cfg: dict, B: int, S: int) -> int:
    H = cfg["hidden_size"]
    h = cfg["num_attention_heads"]
    qk_nope = cfg["qk_nope_head_dim"]
    qk_rope = cfg["qk_rope_head_dim"]
    v_head = cfg["v_head_dim"]
    q_lora = cfg["q_lora_rank"]
    kv_lora = cfg["kv_lora_rank"]

    flops = 0
    flops += 2 * B * S * H * q_lora                          # q_a_proj
    flops += 2 * B * S * q_lora * h * (qk_nope + qk_rope)    # q_b_proj
    flops += 2 * B * S * H * (kv_lora + qk_rope)             # kv_a_proj
    flops += 2 * B * S * kv_lora * h * (qk_nope + v_head)    # kv_b_proj

    # 标准 attention（题目对 MLA absorb-kernel 不做要求；按朴素展开算）
    qk_dim = qk_nope + qk_rope
    flops += 2 * B * h * S * S * qk_dim     # QK^T
    flops += 2 * B * h * S * S * v_head     # softmax × V
    flops += 2 * B * S * h * v_head * H     # o_proj
    return flops


def _moe_block_params(cfg: dict) -> Tuple[int, int]:
    """MoE FFN 块参数：(总参数, 每 token 激活的参数)。

    routed_experts: ``n_routed`` 个 SwiGLU expert，每个 ``3·H·I_moe``。
    shared_experts: ``n_shared`` 个常驻 SwiGLU expert（每 token 必走）。
    router/gate   : ``H × n_routed``。
    """
    H = cfg["hidden_size"]
    I_moe = cfg["moe_intermediate_size"]
    n_routed = cfg["n_routed_experts"]
    n_shared = cfg["n_shared_experts"]
    k = cfg["num_experts_per_tok"]

    expert_params = 3 * H * I_moe
    routed = n_routed * expert_params
    shared = n_shared * expert_params
    router = H * n_routed

    total = router + routed + shared
    activated = router + (k + n_shared) * expert_params
    return total, activated


def _moe_block_fwd_flops(cfg: dict, B: int, S: int) -> int:
    """MoE FFN 单层 forward FLOPs：路由 + 激活的 (k + n_shared) 个 expert。"""
    H = cfg["hidden_size"]
    I_moe = cfg["moe_intermediate_size"]
    n_routed = cfg["n_routed_experts"]
    n_shared = cfg["n_shared_experts"]
    k = cfg["num_experts_per_tok"]

    tokens = B * S
    router = 2 * tokens * H * n_routed
    expert_flops_per_token = 6 * H * I_moe   # SwiGLU: gate + up + down
    activated_per_token = k + n_shared
    expert = activated_per_token * tokens * expert_flops_per_token
    return router + expert


def model_training_cost_analysis_deepseek(model_config_path) -> Tuple[int, float, float]:
    """DeepSeek-V3 类（MLA + MoE）模型成本核算。

    返回的 ``total_params`` 是真正占显存的总参数（含未激活的 routed expert）。
    返回的 ``flops_per_layer`` 取一层 MoE 层的 forward FLOPs（更具代表性；
    前 ``first_k_dense_replace`` 层是 dense MLP，不在此口径里）。
    """
    cfg = _load_config(model_config_path)
    H = cfg["hidden_size"]
    L = cfg["num_hidden_layers"]
    L_dense = cfg["first_k_dense_replace"]
    L_moe = L - L_dense
    V = cfg["vocab_size"]
    I_dense = cfg["intermediate_size"]
    I_moe = cfg["moe_intermediate_size"]
    n_mtp = cfg.get("num_nextn_predict_layers", 0)

    embed = V * H
    lm_head = 0 if cfg.get("tie_word_embeddings", False) else V * H

    mla = _mla_layer_params(cfg)
    dense_mlp = 3 * H * I_dense
    norms_per_layer = 2 * H
    moe_total, moe_activated = _moe_block_params(cfg)

    dense_layer = mla + dense_mlp + norms_per_layer
    moe_layer = mla + moe_total + norms_per_layer

    total = embed + lm_head + H + L_dense * dense_layer + L_moe * moe_layer
    # MTP 层结构上 ≈ 1 个 transformer 层 + 一个额外 head（这里保守按 1 个 MoE 层估）
    if n_mtp:
        total += n_mtp * moe_layer

    # 单 MoE 层 forward FLOPs（用 4K 训练窗口；DeepSeek-V3 训练用 4K-8K）
    B, S = 1, 4096
    layer_flops = _mla_layer_fwd_flops(cfg, B, S) + _moe_block_fwd_flops(cfg, B, S)

    # 显存估算（fp16 口径；DeepSeek 实际用 fp8+bf16，这里按题目约定 fp16）
    # Note: 只有激活的 expert 在反向时产生 grad / 更新 opt state；
    # 但 forward 静态显存仍按全部参数核算（routed expert 权重始终在卡上）。
    P_total = total
    model_state = (WEIGHT_BYTES + GRAD_BYTES + OPT_BYTES_PER_PARAM) * P_total
    boundary_saves = FP16 * L * B * S * H
    # 单 MoE 层 working activations 的粗估：
    #   - MLA 内 Q/K/V/scores 几份 SBH + B·h·S²
    #   - MoE 内 (k + n_shared) 个 expert 的 gate / up / silu(gate)*up = 3·B·S·I_moe
    h = cfg["num_attention_heads"]
    activated_experts = cfg["num_experts_per_tok"] + cfg["n_shared_experts"]
    working = FP16 * (
        6 * B * S * H                            # MLA 投影/输出 ~ 6 份 SBH 上界
        + 2 * B * h * S * S                      # attention scores + softmax_out
        + 3 * B * S * I_moe * activated_experts  # 激活 expert 的 mlp intermediate
    )
    peak_bytes = model_state + boundary_saves + working

    return P_total, layer_flops / 1e12, peak_bytes / (1024 ** 3)


def deepseek_activated_params_per_token(cfg_or_path) -> int:
    """每 token 实际参与计算的参数数（DeepSeek-V3 文档的 "activated params"）。

    用作辅助分析（不在题面三件套里），便于检查 ~37B 这个数字。
    """
    cfg = cfg_or_path if isinstance(cfg_or_path, dict) else _load_config(cfg_or_path)
    H = cfg["hidden_size"]
    L = cfg["num_hidden_layers"]
    L_dense = cfg["first_k_dense_replace"]
    L_moe = L - L_dense
    V = cfg["vocab_size"]
    I_dense = cfg["intermediate_size"]

    embed = V * H
    lm_head = 0 if cfg.get("tie_word_embeddings", False) else V * H

    mla = _mla_layer_params(cfg)
    dense_layer = mla + 3 * H * I_dense + 2 * H
    _, moe_activated = _moe_block_params(cfg)
    moe_layer_active = mla + moe_activated + 2 * H

    return embed + lm_head + H + L_dense * dense_layer + L_moe * moe_layer_active


# ============================================================================
# Part 2.2: Scaling law optimization
# ============================================================================

def _select_best_gpu() -> str:
    """挑 effective TFLOPS / 美元/小时 最高的 GPU。"""
    best, best_eff = None, -1.0
    for name, s in GPU_SPECS.items():
        eff = s["tflops_fp16_peak"] * MFU / s["price_per_hour"]
        if eff > best_eff:
            best, best_eff = name, eff
    return best


def _budget_to_total_flops(budget_usd: float, gpu: str) -> float:
    s = GPU_SPECS[gpu]
    hours = budget_usd / s["price_per_hour"]
    seconds = hours * 3600.0
    effective_tflops = s["tflops_fp16_peak"] * MFU
    return seconds * effective_tflops * 1e12


def get_optimal_N_D_from_cost(cost_budget) -> Tuple[float, float, float, str]:
    """求 ``min L(N,D)  s.t.  6 N D = C`` 的最优 (N, D, C, GPU)。

    解析推导：
        ∂/∂N: -α A / N^(α+1) = μ · D
        ∂/∂D: -β B / D^(β+1) = μ · N
        相除得 α A · D^β = β B · N^α
        ⇒ D = ((βB)/(αA))^(1/β) · N^(α/β)
        代入 6 N D = C 得 N^(1 + α/β) = C / (6 · ((βB)/(αA))^(1/β))。
    """
    gpu = _select_best_gpu()
    C = _budget_to_total_flops(cost_budget, gpu)

    coef = (BETA * B_COEF) / (ALPHA * A_COEF)         # ((βB)/(αA))
    D_coef = coef ** (1.0 / BETA)                     # D / N^(α/β)
    exp_ratio = ALPHA / BETA                          # α/β ≈ 1.1724

    N = (C / (FLOPS_PER_TOKEN_PARAM * D_coef)) ** (1.0 / (1.0 + exp_ratio))
    D = D_coef * N ** exp_ratio
    return N, D, C, gpu


def scaling_law_loss(N: float, D: float) -> float:
    """对外暴露 PDF 给的 scaling law，方便单元测试与曲线绘制。"""
    return A_COEF / (N ** ALPHA) + B_COEF / (D ** BETA) + L0


# ============================================================================
# CLI（与 PA3 PDF 命令行兼容）
# ============================================================================

def _format_human(x: float) -> str:
    for unit, scale in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(x) >= scale:
            return f"{x / scale:.3f}{unit}"
    return f"{x:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Model training cost analysis (PA3 Part 2)")
    parser.add_argument("--model_config", type=str, default=None, help="path to model config JSON")
    parser.add_argument("--training_budget", type=float, default=None, help="training budget in USD")
    args = parser.parse_args()

    if args.model_config:
        path = Path(args.model_config)
        name = path.name.lower()
        if "deepseek" in name:
            n, t, m = model_training_cost_analysis_deepseek(str(path))
        elif "llama" in name or "my_model" in name:
            n, t, m = model_training_cost_analysis_llama(str(path))
        else:
            raise SystemExit(f"Unknown model config (filename must contain 'llama'/'deepseek'/'my_model'): {path}")
        print(f"Number of parameters: {n}  ({_format_human(n)})")
        print(f"Number of TFLOPs: {t}")
        print(f"Peak memory cost: {m} GBs")

    if args.training_budget:
        N, D, C, gpu = get_optimal_N_D_from_cost(args.training_budget)
        print(f"best_gpu: {gpu}")
        print(f"training_budget_flops: {C}")
        print(f"Optimal N: {N}  ({_format_human(N)})")
        print(f"Optimal D: {D}  ({_format_human(D)} tokens)")


if __name__ == "__main__":
    main()
