"""DPO per-instance loss（supplement PDF §5.3 Problem `dpo_loss`）.

DPO formula (PDF §5 Eq. 3):
    ℓ_DPO = -log σ( β·[log π_θ(y_w|x) - log π_ref(y_w|x)] - β·[log π_θ(y_l|x) - log π_ref(y_l|x)] )

PDF hint：直接算 unconditional log-prob log π(x ⊕ y)，prompt 段在差值里相消：
    log π(y_w|x) - log π(y_l|x) = log π(x ⊕ y_w) - log π(x ⊕ y_l)
    （因为 log π(x) 在 chosen 和 rejected 上是同一个常数）

PDF 实装要求：
    1. 用 Alpaca template（与 SFT 同款）拼接 prompt + response
    2. response 末尾**加 EOS** token
    3. lm 和 lm_ref 可能在不同 device（loss 返回到 lm 的 device）
    4. 不对 lm_ref 求梯度（frozen reference）
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .data import ALPACA_TEMPLATE


def _unconditional_log_prob(model: PreTrainedModel, ids: torch.Tensor) -> torch.Tensor:
    """计算 log π(x_1, x_2, ..., x_T) = Σ_{t≥1} log π(x_t | x_<t)。

    Args:
        ids: (1, T) token ids
    Returns:
        scalar tensor (在 model 的 device 上)
    """
    # forward 拿 logits (1, T, V)
    logits = model(ids).logits
    # cast fp32 让 log_softmax 在大 vocab 上数值稳定（HF cross_entropy 同款）
    if logits.dtype in (torch.bfloat16, torch.float16):
        logits = logits.float()
    # log p(x_t | x_<t) 只在位置 1..T-1 上有；用 logits[:-1] 预测 ids[1:]
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)        # (1, T-1, V)
    labels = ids[:, 1:]                                          # (1, T-1)
    token_log_p = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # (1, T-1)
    return token_log_p.sum()


def compute_per_instance_dpo_loss(
    lm: PreTrainedModel,
    lm_ref: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> torch.Tensor:
    """单实例 DPO loss。

    PDF §5.3 deliverable：返回 tensor in lm.device，可 backward。
    """
    eos = tokenizer.eos_token or "<|endoftext|>"

    # 1) 拼接 Alpaca 模板 + EOS
    text_chosen = ALPACA_TEMPLATE.format(prompt=prompt, response=response_chosen) + eos
    text_rejected = ALPACA_TEMPLATE.format(prompt=prompt, response=response_rejected) + eos

    # 2) tokenize（gpt2 tokenizer 不会自动加 BOS；其它 tokenizer 通过
    #    add_special_tokens=False 保证我们自己控制特殊 token）
    ids_chosen = tokenizer.encode(text_chosen, add_special_tokens=False, return_tensors="pt")
    ids_rejected = tokenizer.encode(text_rejected, add_special_tokens=False, return_tensors="pt")

    # 3) 找两个 model 的 device（PDF 说可能不同）
    lm_dev = next(lm.parameters()).device
    ref_dev = next(lm_ref.parameters()).device

    # 4) 跑 4 次 forward：lm × {chosen, rejected} + ref × {chosen, rejected}
    # lm 这两个跑带梯度（训练时要 backward）；ref 用 inference_mode 省显存
    log_p_lm_chosen = _unconditional_log_prob(lm, ids_chosen.to(lm_dev))
    log_p_lm_rejected = _unconditional_log_prob(lm, ids_rejected.to(lm_dev))
    with torch.inference_mode():
        log_p_ref_chosen = _unconditional_log_prob(lm_ref, ids_chosen.to(ref_dev))
        log_p_ref_rejected = _unconditional_log_prob(lm_ref, ids_rejected.to(ref_dev))

    # 5) DPO loss（先把 ref 的 scalar 搬到 lm.device 再算）
    log_p_ref_chosen = log_p_ref_chosen.to(lm_dev)
    log_p_ref_rejected = log_p_ref_rejected.to(lm_dev)

    # diff_chosen   = log π_θ(y_w|x) - log π_ref(y_w|x)
    # diff_rejected = log π_θ(y_l|x) - log π_ref(y_l|x)
    # （prompt 项 log π(x) 在 (lm 的) chosen 和 rejected 上相同 → 相减消掉；
    #  log π_ref(x) 同理；所以 unconditional log π(x⊕y) 直接代入也对）
    diff_chosen = log_p_lm_chosen - log_p_ref_chosen
    diff_rejected = log_p_lm_rejected - log_p_ref_rejected

    # logits = β (diff_chosen - diff_rejected)
    # loss = -log σ(logits) = softplus(-logits)（数值稳定形式）
    logits_dpo = beta * (diff_chosen - diff_rejected)
    loss = -F.logsigmoid(logits_dpo)
    return loss
