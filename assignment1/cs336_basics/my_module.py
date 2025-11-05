import torch
import torch.nn as nn
import math
from einops import rearrange, repeat, einsum
from cs336_basics.my_utils import softmax

class RJLinear(nn.Module):
    
    def __init__(self, in_features: int, out_features: int, device: torch.device|None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        kwargs = {'device': device, 'dtype': dtype}
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), **kwargs)
        )
        
        # linear model的初始化，在3.4.1见到的，为 正态分布
        # 初始化逻辑：截断正态分布 N(μ=0, σ²=2/(d_in+d_out))，截断范围[-3σ, 3σ]
        variance = 2.0 / (self.in_features + self.out_features)
        std_dev = math.sqrt(variance)
        
        # 使用 torch.nn.init.trunc_normal_ 初始化权重
        # 截断范围为 [-3σ, 3σ]
        torch.nn.init.trunc_normal_(
            tensor=self.weight, 
            mean=0.0, 
            std=std_dev, 
            a= -3.0 * std_dev,  # 截断下限
            b= 3.0 * std_dev   # 截断上限
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T
    
class RJEmbedding(nn.Module):
    '''
    maps integer token IDsinto a vector space of dimensiond_model
    '''
    def __init__(self,num_embeddings: int, embedding_dim: int, device: torch.device | None=None, dtype: torch.dtype | None=None) -> None:
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        kwargs = {'device': device, 'dtype': dtype}
        self.weight = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), **kwargs)
        )
        
        torch.nn.init.trunc_normal_(
            tensor=self.weight, 
            mean=0.0, 
            std=1, 
            a= -3.0,  # 截断下限
            b= 3.0,   # 截断上限
        )
        
        
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        '''
        Lookup the embedding vectorsfor the given token IDs.
        实现 Embedding 层的前向传播。
        Args:
            x: shape (...) - batch_size * sequence_length 的输入 token ids
        Returns:
            shape (... D) - batch_size * sequence_length * embedding_dim 的嵌入向量
        '''
        
        # 检查输入type
        if token_ids.dtype not in (torch.long, torch.int32, torch.int64):
            raise TypeError(f"[Embedding] Expected torch.long dtype for input tensor, got {token_ids.dtype}")
        
        # 检查token_ids 范围合法性
        if torch.any(token_ids < 0) or torch.any(token_ids >= self.num_embeddings):
            raise ValueError(f"[Embedding] Token ids must be in range [0, {self.num_embeddings})")
        
        # 算
        return self.weight[token_ids]

#########################################################################################################################################################################################



class RJRMSnorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype|None = None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps

        kwargs = {'device': device, 'dtype': dtype}
        self.g = nn.Parameter(torch.ones(d_model, **kwargs))
         
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Process an input tensor of shape(batch_size, sequence_length, d_model)
        and return a tensor of the same shape
        '''
        # Remember to upcast your input totorch.float32before performing the normalization
        # (andlater downcast to the original dtype)
        in_dtype = x.dtype
        x = x.to(torch.float32)
        #  在特征维上算 rms
        rms = torch.sqrt(self.eps + torch.mean(x*x, dim=-1, keepdim=True))
        result = rearrange(self.g, 'd -> 1 1 d') * (x / rms)
        return result.to(in_dtype)


class RJSwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype|None = None):
        super().__init__()
        kwargs = {'device': device, 'dtype': dtype}
        self.d_model = d_model
        self.d_ff = d_ff
        self.W1 = nn.Parameter(
            torch.empty((d_ff, d_model), **kwargs)
        )
        self.W2 = nn.Parameter(
            torch.empty((d_model, d_ff), **kwargs)
        )
        self.W3 = nn.Parameter(
            torch.empty((d_ff, d_model), **kwargs)
        )
        
        # 使用与 Linear 层相同的初始化策略
        for weight in [self.W1, self.W2, self.W3]:
            variance = 2.0 / (d_model + d_ff)
            std = math.sqrt(variance)
            torch.nn.init.trunc_normal_(
                weight,
                mean=0.0,
                std=std,
                a=-3.0 * std,
                b=3.0 * std
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        实现 SwiGLU 前向传播: W₂(SiLU(W₁x) ⊙ W₃x)
        
        Args:
            x: Float[Tensor, "... d_model"] - 输入特征
            
        Returns:
            Float[Tensor, "... d_model"] - 输出特征
        """
        x1 = x @ self.W1.T  # (..., d_ff)
        x3 = x @ self.W3.T  # (..., d_ff)
        
        silu_x1 = silu(x1)  # (..., d_ff)
        
        gate = silu_x1 * x3  # (..., d_ff)
        
        return gate @ self.W2.T  # (..., d_model)

class RJRoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        '''
        Args:
            d_k: query and key dimension
            max_seq_len: Maximun sequence length that will be inputted
        '''
        super().__init__()
        assert d_k % 2 == 0, "d_k must be even"
        self.theta = float(theta)
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        pair_count = d_k // 2
        k = torch.arange(pair_count, dtype=torch.float32, device=device) # [0, 1, 2, ..., d_k // 2 - 1]
        inv_freq = self.theta ** ((-2 * k) / d_k) # handout上是错的
        
        positions = torch.arange(max_seq_len, dtype=torch.float32, device=device) # (max_seq_len, 1)
        angles = positions[:, None] * inv_freq[None, :] # (max_seq_len, pair_count)
        cos_cache = torch.cos(angles)
        sin_cache = torch.sin(angles)
        
        # 定义缓冲区时，显式标注类型为 torch.Tensor。 处理报错用
        self.inv_freq: torch.Tensor
        self.cos_cache: torch.Tensor
        self.sin_cache: torch.Tensor
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("cos_cache", cos_cache, persistent=False) # TODO why
        self.register_buffer("sin_cache", sin_cache, persistent=False)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            x: (..., seq_len, d_k)
            token_positions: (..., seq_len)
        Return:
            x after RoPE
        '''
        if x.shape[-1] != self.d_k:
            raise ValueError(f'Expected last dimension of x to be {self.d_k}, got {x.shape[-1]}')
        
        # decide whether to use cache: token_positions are integer indexes within [0, max_seq_len)
        # ensure token_positions is on same device as caches for min/max checks
        # XXX 为什么这一步？？
        try:
            tp_min = token_positions.min()
            tp_max = token_positions.max()
        except RuntimeError:
            # if token_positions on different device, move to x.device for checks (non-destructive copy)
            token_positions = token_positions.to(x.device)
            tp_min = token_positions.min()
            tp_max = token_positions.max()
            
        # try using cache when positions are integer indices within [0, max_seq_len)
        use_cache = (not torch.is_floating_point(token_positions) and 
                    tp_min >= 0 and 
                    tp_max < self.max_seq_len)

        if use_cache:
            # 使用预计算的缓存
            pos_indices = token_positions.long().to(self.cos_cache.device)
            cos_val = self.cos_cache[pos_indices]  # (..., seq_len, pair_count)
            sin_val = self.sin_cache[pos_indices]  # (..., seq_len, pair_count)
        else:
            inv = self.inv_freq.to(x.device)
            
            angles = token_positions.float().unsqueeze(-1) * (inv.unsqueeze(0).unsqueeze(0))
            cos_val = torch.cos(angles)
            sin_val = torch.sin(angles)
        
        x_rotated = self._apply_rotation_einsum(x, cos_val, sin_val)
            
        return x_rotated
    def _apply_rotation_einsum(self, x: torch.Tensor, cos_val: torch.Tensor, sin_val:  torch.Tensor) -> torch.Tensor:
        
        # 拆成 成对的奇偶维度, 神奇的einops
        x_pairs = rearrange(x, '... s (p c) -> ... s p c', c=2)
        
        # 使用einsum应用旋转矩阵
        # 旋转矩阵: [cos, -sin; sin, cos]
        rotated_pairs = torch.stack([
            x_pairs[..., 0] * cos_val - x_pairs[..., 1] * sin_val,  # even'
            x_pairs[..., 0] * sin_val + x_pairs[..., 1] * cos_val   # odd'
        ], dim=-1)
        
        
        result = rearrange(rotated_pairs, '... s p c -> ... s (p c)')
        return result

def silu(in_features: torch.Tensor) -> torch.Tensor:
    if in_features.dtype not in {torch.float, torch.float32, torch.float16, torch.float64}:
        raise ValueError(f"[SiLU] Tokens is at type: {in_features.dtype}")
    return in_features * torch.sigmoid(in_features)

def scaled_dot_product_attention( Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    '''
     Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Float[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    
    softmax(Q^T \times K) \times V
    '''
    if Q.shape[-1] != K.shape[-1]:
        raise ValueError(f"Q and K must have same last dimension. Got Q: {Q.shape}, K: {K.shape}")
    
    if K.shape[-2] != V.shape[-2]:
        raise ValueError(f"K and V must have same sequence length. Got K: {K.shape}, V: {V.shape}")
    
    if mask is not None:
        if Q.shape[-2] != mask.shape[-2] or K.shape[-2] != mask.shape[-1]:
            raise ValueError(f"Mask shape {mask.shape} doesn't match Q {Q.shape} and K {K.shape}")
    
    d_k = Q.shape[-1]
    
    QK_score = einsum(Q, K, '... queries d_k, ... keys d_k -> ... queries keys') # 注意 省略号之后要有空格
    attention_score = QK_score / (d_k ** 0.5)
    if mask is not None:
        attention_score = attention_score.masked_fill(~mask, float('-inf')) # 正确实现mask !
    attention_weights = softmax(attention_score, dim=-1)
    output = einsum(attention_weights, V, '... queries k, ... k d_v -> ... queries d_v')
    return output
    
class RJCausalMultiHeadSelfAttention(nn.Module):
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    
    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    def __init__(self,d_model: int,num_heads: int, positional_encoder: RJRoPE|None = None, device: torch.device | None = None, dtype: torch.dtype|None = None) :
        '''
        Args:
            d_model / d_in (int): Dimensionality of the feedforward input and output.
            num_heads (int): Number of heads to use in multi-headed attention.
            max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection

        '''
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度
        self.d_v = self.d_k
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"
        
        # 生成上三角掩码，diagonal=1表示对角线以上的元素为True（未来位置）
        
        kwargs = {'device': device, 'dtype': dtype}
        
        self.q_proj = RJLinear(self.d_model, self.num_heads * self.d_k, **kwargs)
        self.k_proj = RJLinear(self.d_model, self.num_heads * self.d_k, **kwargs)
        self.v_proj = RJLinear(self.d_model, self.num_heads * self.d_v, **kwargs)
        
        self.output_proj = RJLinear(self.num_heads * self.d_v, self.d_model, **kwargs)
        
        self.rope = positional_encoder # rope
        
        
        

    def forward(self, in_features: torch.Tensor, token_positions: torch.Tensor):
        '''
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.
        '''
        *b, sequence_length, d_model = in_features.size()
        if d_model != self.d_model:
            raise ValueError(f'[RJCausalMultiHeadSelfAttention] in_features.size()={in_features.size()}, where d_model({d_model}) != self.d_model({self.d_model})')
        
        Q = self.q_proj(in_features)
        K = self.k_proj(in_features)
        V = self.v_proj(in_features)
        
        # multi-headX
        # Python 集合 {Q, K, V} 是无序的，每次迭代顺序可能不同。这意味着 Q,K,V 的赋值顺序是随机的
        # Q, K, V = (
        #     rearrange(X, '... s (h dk) -> ... h s dk', h=self.num_heads, dk=self.d_k) 
        #     for X in {Q, K, V}
        # )
        Q = rearrange(Q, '... s (h dk) -> ... h s dk', h=self.num_heads)
        K = rearrange(K, '... s (h dk) -> ... h s dk', h=self.num_heads)
        V = rearrange(V, '... s (h dk) -> ... h s dk', h=self.num_heads)
        
        if token_positions is None:
            # token_positions = rearrange(
            #     torch.arange(sequence_length, device=in_features.device),'seq -> b... seq',
            #     b=[1] * len(b)
            # )
            seq = torch.arange(sequence_length, device=in_features.device)
            token_positions = seq.view(*([1] * len(b)), sequence_length)
            token_positions = token_positions.expand(*b, sequence_length)
        
        # Duplicate token positions for each head
        token_positions = rearrange(token_positions, "... seq -> ... 1 seq")
        if self.rope:
            Q = self.rope(x=Q, token_positions=token_positions) # (... h s d_k)
            K = self.rope(x=K, token_positions=token_positions)
            
        # construct causal mask of shape (..., 1, seq_len, seq_len) so it broadcasts over heads
        base_mask = torch.tril(torch.ones((sequence_length, sequence_length), dtype=torch.bool, device=in_features.device))
        # expand to batch dims + head dim = len(b) + 1 leading ones
        expand_shape = (*([1] * (len(b) + 1)), sequence_length, sequence_length)
        mask = base_mask.view(*expand_shape)  # will broadcast to (..., h, seq_len, seq_len)

        attn_output = scaled_dot_product_attention(Q, K, V, mask)
        attn_output = rearrange(attn_output, "... heads seq d_v -> ... seq (heads d_v)").contiguous()
        
        output = self.output_proj(attn_output)
        return output

class RJTransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, theta: float, max_seq_len: int,device: torch.device | None = None, dtype: torch.dtype|None = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0, "d_model % num_heads should be 0"
        self.d_k = d_model // num_heads
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype
        self.max_seq_len = max_seq_len
        kwargs = {'device': device, 'dtype': dtype}

        # 两个 RMS 是不同的！
        self.rms_norm_mha = RJRMSnorm(d_model=self.d_model, device=self.device, dtype=self.dtype)
        self.rms_norm_ffn = RJRMSnorm(d_model=self.d_model, device=self.device, dtype=self.dtype)
        self.rope = RJRoPE(theta=theta, d_k=self.d_k, max_seq_len=self.max_seq_len, device=self.device)
        self.mha = RJCausalMultiHeadSelfAttention(d_model=self.d_model, num_heads=self.num_heads, positional_encoder=self.rope, **kwargs)
        
        self.ffn = RJSwiGLU(d_model=self.d_model, d_ff=self.d_ff, **kwargs)
        
    def forward(self, in_features: torch.Tensor, token_positions: torch.Tensor):
        s1 = in_features + self.mha(self.rms_norm_mha(in_features), token_positions)
        s2 = self.ffn(self.rms_norm_ffn(s1)) + s1
        return s2
        
class RJTransformerLM(nn.Module):
    def __init__(self, vocab_size: int, context_length:int, num_layers: int, num_heads:int,d_ff: int, d_model:int, rope_theta: float, device: torch.device | None = None, dtype: torch.dtype|None = None) -> None:
        super().__init__()
        kwargs = {"device": device, "dtype": dtype}
        self.token_embedding = RJEmbedding(num_embeddings=vocab_size, embedding_dim=d_model, **kwargs)
        assert d_model % num_heads == 0, "d_model % num_heads should be 0" 
        self.rope = RJRoPE(theta=rope_theta, d_k=(d_model // num_heads), max_seq_len=context_length, device=device)
        
        self.blocks = nn.ModuleList([
            RJTransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                theta=rope_theta,
                max_seq_len=context_length,
                **kwargs
            ) for _ in range(num_layers)
        ])
        
        self.norm = RJRMSnorm(d_model=d_model, device=device, dtype=dtype)
        
        self.output_embedding = RJLinear(in_features=d_model, out_features=vocab_size)
    
    def forward(self, token_ids: torch.Tensor):
        x = self.token_embedding(token_ids)
        
        for block in self.blocks:
            x = block(x, None) # positions ?
        
        x = self.norm(x)
        logits = self.output_embedding(x)
        return logits

        
        
        