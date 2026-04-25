"""PA3 Part 2 cost-analysis 单元测试。

校验目标：
  * Llama-7B 总参数 ≈ 已发布的 6.74B
  * Llama-7B 单层 forward FLOPs 与 ``2·P_layer·S`` 在合理倍数内一致
  * Llama-7B fp16 训练显存量级合理（>40 GB；不带激活也得 ~50 GB）
  * DeepSeek-V3 总参数 ≈ 671B（含 1 个 MTP 层时偏大一些是预期）
  * DeepSeek-V3 每 token 激活 ≈ 37B（与官方文档一致）
  * Scaling-law 求解：A100 胜出；总 FLOPs ≈ 5.6e23；约束 6ND=C 严格成立
  * 自选模型 my_model_config.json 落在 N* 附近且 6ND 不超过预算
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tiny_training_system.cost_analysis import (
    deepseek_activated_params_per_token,
    get_optimal_N_D_from_cost,
    model_training_cost_analysis_deepseek,
    model_training_cost_analysis_llama,
    scaling_law_loss,
)

CONFIG_DIR = Path(__file__).parent.parent / "configs"
LLAMA_CONFIG = CONFIG_DIR / "llama_7b_config.json"
DEEPSEEK_CONFIG = CONFIG_DIR / "deepseek_v3_config.json"
MY_CONFIG = CONFIG_DIR / "my_model_config.json"


# ---------------- Llama-7B (Part 2.1) ----------------

def test_llama7b_params_around_6_74B():
    n, _, _ = model_training_cost_analysis_llama(LLAMA_CONFIG)
    assert 6.7e9 < n < 6.8e9, f"Llama-7B 期望 ~6.74B params, 实际 {n / 1e9:.3f}B"


def test_llama7b_layer_flops_consistent_with_2PS():
    """单层 forward FLOPs 应与 ``2·P_layer·S`` 量级一致。"""
    cfg = json.loads(LLAMA_CONFIG.read_text())
    H, I, S = cfg["hidden_size"], cfg["intermediate_size"], cfg["max_position_embeddings"]
    P_layer = 4 * H * H + 3 * H * I + 2 * H
    expected_TF = 2 * P_layer * S / 1e12
    _, tflops, _ = model_training_cost_analysis_llama(LLAMA_CONFIG)
    # attention scores S² 项让实际略高，1.0× ~ 1.5× 都算正常
    assert 0.9 * expected_TF <= tflops <= 1.6 * expected_TF


def test_llama7b_peak_memory_in_fp16_training_range():
    """fp16 + grad + Adam(m,v fp16) = 8 bytes/param ≈ 50 GB；活动激活再加几 GB。"""
    _, _, gb = model_training_cost_analysis_llama(LLAMA_CONFIG)
    assert 45.0 < gb < 80.0, f"Llama-7B fp16 训练峰值期望 50-60 GB 量级, 实际 {gb:.2f} GB"


# ---------------- DeepSeek-V3 (Part 2.3 bonus) ----------------

def test_deepseek_total_params_around_671B():
    """总参数应在 670B-690B 区间（含 1 个 MTP 层时上限稍高，与官方 671B 数字接近）。"""
    n, _, _ = model_training_cost_analysis_deepseek(DEEPSEEK_CONFIG)
    assert 6.5e11 < n < 7.0e11, f"DeepSeek-V3 期望 ~671B params, 实际 {n / 1e9:.1f}B"


def test_deepseek_activated_params_per_token_around_37B():
    """官方文档：每 token 激活约 37B。"""
    activated = deepseek_activated_params_per_token(DEEPSEEK_CONFIG)
    assert 36.0e9 < activated < 39.0e9, f"激活参数期望 ~37B, 实际 {activated / 1e9:.2f}B"


def test_deepseek_active_fraction_below_6_percent():
    """MoE 的关键卖点：激活/总参数比 < 6%（37B / 671B ≈ 5.5%）。"""
    total, _, _ = model_training_cost_analysis_deepseek(DEEPSEEK_CONFIG)
    active = deepseek_activated_params_per_token(DEEPSEEK_CONFIG)
    assert active / total < 0.06


# ---------------- Scaling law (Part 2.2) ----------------

def test_best_gpu_is_a100():
    _, _, _, gpu = get_optimal_N_D_from_cost(5_000_000)
    # A100 effective: 312 × 0.4 / 4 = 31.2 TFLOPS/$
    # T4   effective: 65 × 0.4 / 1  = 26   TFLOPS/$
    # V100 effective: 125 × 0.4 / 2.5 = 20 TFLOPS/$
    assert gpu == "A100"


def test_total_budget_flops_around_5_6e23():
    _, _, C, _ = get_optimal_N_D_from_cost(5_000_000)
    expected = (5_000_000 / 4.0) * 3600 * 312e12 * 0.4
    assert math.isclose(C, expected, rel_tol=1e-9)
    assert 5.5e23 < C < 5.7e23


def test_optimal_satisfies_6ND_equals_C():
    N, D, C, _ = get_optimal_N_D_from_cost(5_000_000)
    assert math.isclose(6 * N * D, C, rel_tol=1e-6)


def test_optimal_N_in_tens_of_billions():
    N, D, _, _ = get_optimal_N_D_from_cost(5_000_000)
    # 该 scaling law 下最优 N ≈ 47B；放宽到 30-80B
    assert 3e10 < N < 8e10
    # D 在 ~1-3 万亿 tokens 区间
    assert 1e12 < D < 3e12


def test_optimal_loss_is_local_minimum():
    """检查 (N*, D*) 在约束面上确实是局部极小：沿约束面扰动 5% 后 loss 应升高。"""
    N, D, C, _ = get_optimal_N_D_from_cost(5_000_000)
    L_star = scaling_law_loss(N, D)
    for r in (0.95, 1.05):
        N2 = N * r
        D2 = C / (6 * N2)  # 沿约束面扰动
        assert scaling_law_loss(N2, D2) > L_star


def test_my_model_within_budget():
    """自选模型不应超出 5M USD 预算下的总训练 FLOPs。"""
    cfg = json.loads(MY_CONFIG.read_text())
    n, _, _ = model_training_cost_analysis_llama(MY_CONFIG)
    D = cfg["_planned_training_tokens"]
    _, _, C, _ = get_optimal_N_D_from_cost(5_000_000)
    used = 6 * n * D
    assert used <= 1.02 * C, f"自选模型总 FLOPs {used:.2e} 超过预算 {C:.2e}"


def test_my_model_size_near_optimum():
    """自选模型 N 应在最优 N* 的 0.5x ~ 1.5x 区间内。"""
    n, _, _ = model_training_cost_analysis_llama(MY_CONFIG)
    N_star, _, _, _ = get_optimal_N_D_from_cost(5_000_000)
    assert 0.5 * N_star <= n <= 1.5 * N_star
