import torch
import torch.nn as nn

from tiny_training_basic.layers import (
    REmbedding,
    RLinear,
    RMultiHeadAttention,
    RRMSNorm,
    RSwiGLU,
)


class RTransformerBlock(nn.Module):
    """
    Pre-norm Transformer block（A1 §3.5）：

        y   = x + attn(ln1(x))
        out = y + ffn(ln2(y))

    组件：
        ln1:  RRMSNorm(d_model)
        attn: RMultiHeadAttention(d_model, num_heads) + RoPE on (L, d_head)
        ln2:  RRMSNorm(d_model)
        ffn:  RSwiGLU(d_model, d_ff)

    state_dict 的 key（与 adapter.run_transformer_block 一致）：
        ln1.weight
        attn.{q_proj, k_proj, v_proj, output_proj}.weight
        ln2.weight
        ffn.{w1, w2, w3}.weight

    forward(x, token_positions=None):
        x: (..., L, d_model)
        token_positions 默认 arange(L)，由 attn 内部处理。
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.ln1 = RRMSNorm(d_model, device=device, dtype=dtype)
        self.attn = RMultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            use_rope=True,
            device=device,
            dtype=dtype,
        )
        self.ln2 = RRMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = RSwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), token_positions=token_positions)
        x = x + self.ffn(self.ln2(x))
        return x


class RTransformerLM(nn.Module):
    """
    Transformer LM（A1 §3.6）：

        h0        = token_embeddings(ids)
        h_{i+1}   = block_i(h_i)                 for i in [0, num_layers)
        h_final   = ln_final(h_{num_layers})
        logits    = lm_head(h_final)             # unnormalized

    state_dict 的 key（与 adapter.run_transformer_lm 一致）：
        token_embeddings.weight
        layers.{i}.ln1.weight
        layers.{i}.attn.{q_proj, k_proj, v_proj, output_proj}.weight
        layers.{i}.ln2.weight
        layers.{i}.ffn.{w1, w2, w3}.weight
        ln_final.weight
        lm_head.weight

    Forward:
        in_indices: (B, L) int  ->  logits: (B, L, vocab_size)
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.token_embeddings = REmbedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                RTransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=rope_theta,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RRMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = RLinear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, in_indices: torch.Tensor) -> torch.Tensor:
        x = self.token_embeddings(in_indices)
        for block in self.layers:
            x = block(x)
        x = self.ln_final(x)
        return self.lm_head(x)
