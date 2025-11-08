from typing import Any, Dict
import torch
from cs336_basics.my_utils import softmax

from collections.abc import Callable, Iterable 
from typing import Optional, Tuple
import torch
import math

def cross_entropy(in_logits: torch.Tensor, targets: torch.Tensor):
    """
    Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    # 检查1：inputs必须是二维张量（batch_size, vocab_size）
    assert in_logits.ndim == 2, f"inputs must be 2D (batch_size, vocab_size), got {in_logits.ndim}D"
    # 检查2：targets必须是一维张量（batch_size,）
    assert targets.ndim == 1, f"targets must be 1D (batch_size,), got {targets.ndim}D"
    # 检查3：batch_size必须一致
    batch_size = in_logits.size(0)
    assert targets.size(0) == batch_size, \
        f"batch size mismatch: inputs has {batch_size}, targets has {targets.size(0)}"
    # 检查4：inputs必须是浮点型（logits是连续值）
    assert torch.is_floating_point(in_logits), f"inputs must be floating-point tensor, got {in_logits.dtype}"
    # 检查5：targets必须是整数型（类别索引是离散整数）
    assert not torch.is_floating_point(targets), f"targets must be integer tensor, got {targets.dtype}"
    # 检查6：类别索引必须在有效范围 [0, vocab_size-1] 内
    vocab_size = in_logits.size(1)
    assert (targets >= 0).all() and (targets < vocab_size).all(), \
        f"targets must be in [0, {vocab_size-1}], but found out-of-range values"

    # 对每个样本的logits，减去其最大值（保持数值稳定）
    max_val = in_logits.max(dim=-1, keepdim=True).values
    in_logits = in_logits - max_val
    
    # 计算exp(stable_logits)并求和（分母部分）
    in_exp = torch.exp(in_logits)
    sum_exp = torch.sum(in_exp, dim=1, keepdim=True)
    
    # 提取目标类的logits，并计算log(softmax)
    target_logits = in_logits.gather(1, targets.unsqueeze(1))  # 用gather定位目标类logits
    log_softmax = target_logits - torch.log(sum_exp)
    
    # 计算交叉熵损失并取batch平均
    return -log_softmax.mean()
    

class SGDOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {
            "lr": lr,
        }
        super().__init__(params, defaults=defaults)
    
    def step(self, closure: Optional[Callable] = None):
        """
        执行一次优化步骤
        """
        loss = None if closure is None else closure()
        
        # 对每一个参数组进行梯度下降
        for group in self.param_groups:
            lr = group["lr"] # 同一个参数组有相同的学习率
            for param in group["params"]:
                if param.grad is None:
                    continue
                
                state = self.state[param] # 读取之前状态
                t = state.get("t", 0) # 读取当前迭代次数
                grad = param.grad.data # 读取梯度
                param.data -= lr / math.sqrt(t + 1) * grad # 根据梯度和迭代次数更新参数，lr会随着代数增加逐渐衰减
                state["t"] = t + 1 # 迭代次数+1
        
        return loss

class RJAdamW(torch.optim.Optimizer):
    def __init__(self, params: Iterable[torch.Tensor] | Iterable[Dict[str, Any]] | Iterable[Tuple[str, torch.Tensor]], lr, betas =(0.9, 0.999), weight_decay=0.01, eps=1e-8) -> None:
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        
        defaults: Dict[str, Any] = {
            "lr": lr,
            "betas": betas,
            "weight_decay": weight_decay,
            "eps":eps,
        }
        super().__init__(params, defaults)
    
    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    # 初始化状态
                    state['m'] = torch.zeros_like(p.data) # 一阶动量
                    state['v'] = torch.zeros_like(p.data) # 二阶动量
                t = state.get("t", 1) # Get iteration number from the state, or initial value 1
                
                grad = p.grad.data
                
                # 从标准答案抄的
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients")
                
                state['m'] = beta1 * state['m'] + (1 - beta1) * grad
                
                state['v'] = beta2 * state['v'] + (1 - beta2) * (grad ** 2)
                
                alpha_t = lr * math.sqrt(1 - (beta2 ** t)) / (1 - (beta1 ** t))
                
                p.data -= alpha_t * state['m'] / (torch.sqrt(state['v']) + eps)
                
                p.data -= lr * weight_decay * p.data
                
                state['t'] = t + 1
        return loss
        


def lr_cosine_schedule_with_warmup(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    if it < warmup_iters:
        # warm up
        return it / warmup_iters * max_learning_rate
    elif it < cosine_cycle_iters:
        # cosine annealing
        cos_percent = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        return min_learning_rate + 1 / 2.0 * (1 + math.cos( cos_percent * math.pi )) * (max_learning_rate - min_learning_rate)
    else:
        # post-annealing
        return min_learning_rate


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_norm: float, norm_type: float = 2.0, eps: float = 1e-6) -> None:
    """
    Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.
    梯度裁剪，防止梯度爆炸
    
    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    
    # compute total_norm
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            total_norm += torch.sum(p.grad ** norm_type)
    total_norm = total_norm ** 0.5
    
    # clip grads with norm
    clip_coef = max_norm / (total_norm + eps) 
    if clip_coef < 1.0:
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)
    
        

