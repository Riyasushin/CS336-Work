import torch


def rcross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    数值稳定 cross-entropy（A1 §5.2 eq.17）。

        CE_i = -log softmax(x_i)[t_i]
             = -x_i[t_i] + logsumexp(x_i)

    稳定写法（用 max 先平移，防 exp 溢出）:
        m      = max(x, dim=-1, keepdim=True)
        shift  = x - m
        log_Z  = log(sum(exp(shift), dim=-1))                    # = logsumexp - m
        CE     = log_Z - shift[target]                           # = logsumexp(x) - x[target]

    两处 m 相抵消，所以最终表达式里根本不用加回 m——只要用 shifted 的 target 项和 log_Z 相减即可。

    Shape:
        logits:  (..., vocab)
        targets: (...)       int
        return:  scalar（batch 上平均）
    """
    m = logits.amax(dim=-1, keepdim=True)
    shifted = logits - m
    log_z = shifted.exp().sum(dim=-1).log()                             # (...)
    target_shifted = shifted.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    nll = log_z - target_shifted
    return nll.mean()
