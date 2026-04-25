import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# 上位替代: 能走 FA2 Triton kernel 就走 FA2 (CUDA + 支持 dtype + bool mask), 否则透传 F.sdpa
# 这样 CPU / unsupported shape / float attn_mask 等场景行为不变, 只是在 GPU 训练时自动加速
from tiny_training_basic.kernels.fa2 import scaled_dot_product_attention as _sdpa_maybe_flash


def rsilu(x: torch.Tensor) -> torch.Tensor:
    """
    SiLU / Swish activation。

    A1 §3.4 eq.5 (p.21):
        SiLU(x) = x · σ(x) = x / (1 + e^(-x))

    Shape: 任意 shape，element-wise。
    """
    return x * torch.sigmoid(x)


def rsoftmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    数值稳定 softmax (A1 §3.4.4 eq.10, p.23)。

        softmax(x)_i = exp(x_i) / Σ exp(x_j)

    稳定技巧：先减去指定维度的 max 再 exp，防 exp(大数)=inf。
    softmax 对平移不变，所以 exp(x - max) 归一化后和 exp(x) 归一化结果等价。
    """
    x_shifted = x - x.amax(dim=dim, keepdim=True)
    exp_x = torch.exp(x_shifted)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


def rscaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Scaled dot-product attention (A1 §3.4.4 eq.11, p.24)。

        Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

    Shapes:
        Q:    [..., queries, d_k]
        K:    [..., keys,    d_k]
        V:    [..., keys,    d_v]
        mask: [..., queries, keys]  Bool，可选
        return: [..., queries, d_v]

    Mask 约定（PDF §3.4.4）：True=关注，False=不关注。
    在 softmax 前把 False 位置的 score 置为 -inf，softmax 后这些位置概率为 0。
    """
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    attn = rsoftmax(scores, dim=-1)
    return attn @ V


class RLinear(nn.Module):
    """
    不带 bias 的 Linear。前缀 R 用于和 torch.nn.Linear 区分。

    __init__(in_features, out_features, device=None, dtype=None)
    Parameter: self.weight, shape [out_features, in_features]（无 bias）
    Init (A1 §3.3.1 p.17):
        σ² = 2 / (in_features + out_features)
        nn.init.trunc_normal_(W, mean=0, std=sqrt(σ²), a=-3σ, b=3σ)
    Forward: x [..., in_features] -> x @ W.T -> [..., out_features]
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        std = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T


class REmbedding(nn.Module):
    """
    查表式 Embedding。前缀 R 用于和 torch.nn.Embedding 区分。

    __init__(vocab_size, d_model, device=None, dtype=None)
    Parameter: self.weight, shape [vocab_size, d_model]
    Init (A1 §3.3.1 p.17):
        σ² = 1
        nn.init.trunc_normal_(W, mean=0, std=1, a=-3, b=3)
    Forward: token_ids [...] (int) -> self.weight[token_ids] -> [..., d_model]
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.weight = nn.Parameter(
            torch.empty(vocab_size, d_model, device=device, dtype=dtype)
        )
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


class RSwiGLU(nn.Module):
    """
    SwiGLU 前馈网络，无 bias。

    A1 §3.4 eq.7 (p.21) column-vector 形式：
        FFN(x) = SwiGLU(x, W1, W2, W3) = W2 · (SiLU(W1·x) ⊙ (W3·x))
    Row-vector / batched 实现：
        out = (rsilu(x @ W1.T) * (x @ W3.T)) @ W2.T

    __init__(d_model, d_ff, device=None, dtype=None)
        W1 ∈ [d_ff, d_model], W2 ∈ [d_model, d_ff], W3 ∈ [d_ff, d_model]
        三个 W 按 RLinear 的 trunc_normal_ 规则初始化（A1 §3.3.1）。
        经验值 d_ff ≈ 8/3 * d_model，对齐到 64 的倍数——这是训练配置，
        layer 本身不做约束。

    Sub-modules（复用 RLinear，state_dict 会有 w{1,2,3}.weight）:
        self.w1: RLinear(d_model, d_ff)
        self.w2: RLinear(d_ff,    d_model)
        self.w3: RLinear(d_model, d_ff)

    Forward:
        x: [..., d_model] -> [..., d_model]
        先 assert x.shape[-1] == d_model。
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = RLinear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = RLinear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = RLinear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.d_model, (
            f"expected last dim = d_model ({self.d_model}), got {x.shape[-1]}"
        )
        return self.w2(rsilu(self.w1(x)) * self.w3(x))


class RRMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization。前缀 R 用于和 torch.nn.RMSNorm 区分。

    A1 §3.4.1 eq.4 (p.19–20):
        RMSNorm(a_i) = a_i / RMS(a) · g_i
        RMS(a) = sqrt( (1/d_model) · Σ_{i=1}^{d_model} a_i²  +  ε )

    __init__(d_model, eps=1e-5, device=None, dtype=None)
    Parameter: self.weight, shape [d_model]
    Init (A1 §3.3.1): 全 1
    eps 默认 1e-5（PDF 建议值）。

    Forward: x [..., d_model] -> [..., d_model]
        5 步（你自己写）:
          1. 记 in_dtype = x.dtype, x 先 upcast 到 float32（PDF 要求，防 x² 溢出）
          2. 在最后一维求 mean(x²), 加 eps, 开方 -> rms, shape [..., 1]
          3. x / rms
          4. * self.weight  (broadcast [d_model] 到 [..., d_model])
          5. 结果 cast 回 in_dtype
    """

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_type = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x = ((x / rms) * self.weight).to(in_type)
        return x


class RRoPE(nn.Module):
    """
    Rotary Position Embedding，half-rotated 配对（Qwen / Llama / HF 约定）。

    配对：维度 i 与 i + d_k/2 组成一对（d_k 必须偶数）。
    频率：θ_k = base^(-2k/d_k),  k = 0 .. d_k/2 - 1。
    位置 m 处的旋转角 = m · θ_k。

    旋转写法（与 transformers.models.llama.apply_rotary_pos_emb 一致）：
        rotate_half(x) = concat(-x[..., d/2:], x[..., :d/2])
        y = x * cos + rotate_half(x) * sin
    其中 cos/sin 的最后一维把 [cos(mθ_k)] 复制两份拼成 d_k 长度，
    正好对前半和后半都乘同一组 cos，sin 则分别作用于 rotate_half 的两段。

    预计算 cos/sin 到 max_seq_len，注册为 non-persistent buffer；
    forward 时用 token_positions 索引，再按 x 的 rank 自动升维广播，
    支持 x 形如 (..., seq, d_k) 或 (..., n_heads, seq, d_k)。

    __init__(d_k, theta, max_seq_len, device=None)
    forward(x, token_positions) -> same shape as x
    """

    def __init__(
        self,
        d_k: int,
        theta: float,
        max_seq_len: int,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        assert d_k % 2 == 0, f"d_k must be even, got {d_k}"
        self.d_k = d_k
        self.theta = theta
        self.max_seq_len = max_seq_len

        half = torch.arange(0, d_k, 2, device=device, dtype=torch.float32)
        freqs = 1.0 / (theta ** (half / d_k))                           # [d_k/2]
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        angles = positions[:, None] * freqs[None, :]                    # [max_seq_len, d_k/2]
        cos = torch.cat([torch.cos(angles), torch.cos(angles)], dim=-1) # [max_seq_len, d_k]
        sin = torch.cat([torch.sin(angles), torch.sin(angles)], dim=-1)
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor,
    ) -> torch.Tensor:
        cos = self.cos_cached[token_positions]  # (..., seq, d_k)
        sin = self.sin_cached[token_positions]
        # x 可能比 positions 多一个 head 维 (..., H, seq, d_k)，在 -3 位插入 1 来广播
        while cos.dim() < x.dim():
            cos = cos.unsqueeze(-3)
            sin = sin.unsqueeze(-3)
        x1, x2 = x[..., : self.d_k // 2], x[..., self.d_k // 2 :]
        rotate_half = torch.cat([-x2, x1], dim=-1)
        return (x * cos + rotate_half * sin).to(x.dtype)


class RGroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention（GQA），可选 RoPE，causal self-attention。

    形状流水：
        x: (..., L, d_model)
            │ W_Q (n_heads    · d_head, d_model)
            │ W_K (n_kv_heads · d_head, d_model)
            │ W_V (n_kv_heads · d_head, d_model)
            ▼
        Q: (..., L, n_heads    · d_head)  ─rearrange─> (..., n_heads,    L, d_head)
        K: (..., L, n_kv_heads · d_head)  ─rearrange─> (..., n_kv_heads, L, d_head)
        V:                           同 K
            │
            │ RoPE 作用在 Q/K 的 (L, d_head) 上；V 不动
            │ SDPA(is_causal=True, enable_gqa=True) — KV 按 n_heads/n_kv_heads 组复制
            ▼
        attn: (..., n_heads, L, d_head)  ─rearrange─> (..., L, d_model)
            │ W_O (d_model, d_model)
            ▼
        out: (..., L, d_model)

    约束：
        d_model % num_heads == 0, num_heads % num_kv_heads == 0
        d_head = d_model // num_heads  (K/V 头也用同一个 d_head)
        n_kv_heads == n_heads 时退化为 MHA；n_kv_heads == 1 时是 MQA。

    Sub-modules（state_dict 的 key）:
        q_proj.weight      [n_heads    · d_head, d_model]
        k_proj.weight      [n_kv_heads · d_head, d_model]
        v_proj.weight      [n_kv_heads · d_head, d_model]
        output_proj.weight [d_model, d_model]
        rope (可选, non-persistent buffer, 不进 state_dict)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int,
        max_seq_len: int | None = None,
        theta: float | None = None,
        use_rope: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        assert num_heads % num_kv_heads == 0, (
            f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        )
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.d_head = d_model // num_heads
        self.use_rope = use_rope

        self.q_proj = RLinear(d_model, num_heads * self.d_head, device=device, dtype=dtype)
        self.k_proj = RLinear(d_model, num_kv_heads * self.d_head, device=device, dtype=dtype)
        self.v_proj = RLinear(d_model, num_kv_heads * self.d_head, device=device, dtype=dtype)
        self.output_proj = RLinear(d_model, d_model, device=device, dtype=dtype)

        if use_rope:
            assert max_seq_len is not None and theta is not None, (
                "use_rope=True requires max_seq_len and theta"
            )
            self.rope: RRoPE | None = RRoPE(
                d_k=self.d_head, theta=theta, max_seq_len=max_seq_len, device=device
            )
        else:
            self.rope = None

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        Q = rearrange(self.q_proj(x), "... l (h d) -> ... h l d", h=self.num_heads)
        K = rearrange(self.k_proj(x), "... l (h d) -> ... h l d", h=self.num_kv_heads)
        V = rearrange(self.v_proj(x), "... l (h d) -> ... h l d", h=self.num_kv_heads)

        if self.use_rope:
            if token_positions is None:
                token_positions = torch.arange(x.shape[-2], device=x.device)
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        attn = _sdpa_maybe_flash(Q, K, V, is_causal=True, enable_gqa=True)
        attn = rearrange(attn, "... h l d -> ... l (h d)")
        return self.output_proj(attn)


class RMultiHeadAttention(RGroupedQueryAttention):
    """
    Multi-Head Attention = GQA 在 num_kv_heads == num_heads 的退化情况。
    继承 RGroupedQueryAttention 并固定 num_kv_heads = num_heads。
    state_dict 的 key 与父类一致。
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int | None = None,
        theta: float | None = None,
        use_rope: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            d_model=d_model,
            num_heads=num_heads,
            num_kv_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            use_rope=use_rope,
            device=device,
            dtype=dtype,
        )

