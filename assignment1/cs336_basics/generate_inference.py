from cs336_basics.my_module import *
from cs336_basics.train_bpe import BPE_Tokenizer
from cs336_basics.my_utils import softmax
import torch






def apply_top_p(probs, top_p):
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # 找到累积和超过 top_p 的第一个索引
    # 技巧：我们想保留 (cumulative_probs <= top_p) 的元素，但要额外保留一个
    # torch.nonzero(cumulative_probs > top_p) 给出第一个超过的索引
    # .squeeze() 确保即使只有一个元素也是标量
    # .item() 返回该索引
    
    # 获取第一个超过 top_p 的元素的索引
    mask = cumulative_probs > top_p
    # 确保至少保留概率最大的一个 token
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = False
    
    # 将要过滤的元素的概率设为0
    sorted_probs[mask] = 0.0

    # 重新归一化（防止舍入误差导致的非零和）
    sorted_probs.div_(sorted_probs.sum(dim=-1, keepdim=True))

    # 还原到原始词汇表顺序
    # 这是一个反向操作，需要使用原索引 sorted_indices 来还原
    # 如果您使用张量操作，则需要使用 torch.scatter_
    
    # 简单的方法：创建一个新的零张量，然后用 sorted_probs 填充
    new_probs = torch.zeros_like(probs)
    new_probs.scatter_(-1, sorted_indices, sorted_probs)
    
    return new_probs

def decode_batch(prompts: list[str], model: torch.nn.Module, tokenizer: BPE_Tokenizer, endoftext_token_id: int,
                 max_tokens: int = 512, temperature: float = 1.0, top_p: float = 1.0, device='cpu'):
    
    batch_size = len(prompts)
    
    # 1. 编码所有 prompts 并 padding 到同长度
    token_ids_list = [torch.tensor(tokenizer.encode(p), dtype=torch.long) for p in prompts]
    max_len = max(len(ids) for ids in token_ids_list)
    input_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
    for i, ids in enumerate(token_ids_list):
        input_ids[i, :len(ids)] = ids
    input_ids = input_ids.to(device)
    
    # 2. 保存每个序列的生成 token
    generated_tokens = [[] for _ in range(batch_size)]
    finished_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
    
    for _ in range(max_tokens):
        # 3. 获取模型输出
        logits = model(input_ids)  # 输出  [batch_size, vocab_size]
        
        # 4. temperature
        scaled_logits = logits / temperature
        probs = softmax(scaled_logits, dim=-1)
        
        # 5. top-p
        if top_p < 1.0:
            probs = torch.stack([apply_top_p(p, top_p) for p in probs])
        
        # 6. 采样下一个 token
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
        
        # 7. 更新 finished_mask
        finished_mask |= (next_tokens == endoftext_token_id)
        next_tokens = next_tokens.masked_fill(finished_mask, 0)  # 已完成的序列填 0
        
        # 8. 保存生成 token
        for i in range(batch_size):
            if not finished_mask[i]:
                generated_tokens[i].append(next_tokens[i].item())
        
        # 9. 拼接到 input_ids
        input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)
        
        # 10. 全部完成则提前结束
        if finished_mask.all():
            break
    
    return [tokenizer.decode(per_set) for per_set in generated_tokens]
