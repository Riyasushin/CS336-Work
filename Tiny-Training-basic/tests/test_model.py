import numpy
import torch
import torch.nn.functional as F
from einops import rearrange

from .adapters import (
    run_embedding,
    run_grouped_query_self_attention,
    run_grouped_query_self_attention_with_rope,
    run_linear,
    run_multihead_self_attention,
    run_multihead_self_attention_with_rope,
    run_rmsnorm,
    run_rope,
    run_scaled_dot_product_attention,
    run_silu,
    run_swiglu,
    run_transformer_block,
    run_transformer_lm,
)


def _apply_rope_half_rotated(x: torch.Tensor, theta: float, positions: torch.Tensor) -> torch.Tensor:
    """Reference RoPE with half-rotated pairing (dim i pairs with i+d/2), Qwen/Llama/HF style.

    Complex number k = x[k] + i*x[k+d/2] for k in [0, d/2), rotated by m*theta_k with
    theta_k = 1 / theta^(2k/d). Matches `transformers.models.llama.modeling_llama.apply_rotary_pos_emb`.

    x: (..., seq_len, d_head); positions: (..., seq_len).
    """
    d = x.shape[-1]
    assert d % 2 == 0
    freqs = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float32) / d))
    angles = positions.unsqueeze(-1).float() * freqs  # (..., seq_len, d/2)
    cos = torch.cat([torch.cos(angles), torch.cos(angles)], dim=-1)  # (..., seq_len, d)
    sin = torch.cat([torch.sin(angles), torch.sin(angles)], dim=-1)
    # broadcast onto x's head dim: x is (..., H, L, d); cos/sin are (..., L, d).
    while cos.dim() < x.dim():
        cos = cos.unsqueeze(-3)
        sin = sin.unsqueeze(-3)
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    rotate_half = torch.cat([-x2, x1], dim=-1)
    return (x * cos + rotate_half * sin).to(x.dtype)


def _gqa_reference(
    in_features: torch.Tensor,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    o_proj_weight: torch.Tensor,
    n_heads: int,
    n_kv_heads: int,
    *,
    rope: bool = False,
    theta: float | None = None,
    positions: torch.Tensor | None = None,
) -> torch.Tensor:
    Q = F.linear(in_features, q_proj_weight)
    K = F.linear(in_features, k_proj_weight)
    V = F.linear(in_features, v_proj_weight)
    Q = rearrange(Q, "... l (h d) -> ... h l d", h=n_heads)
    K = rearrange(K, "... l (h d) -> ... h l d", h=n_kv_heads)
    V = rearrange(V, "... l (h d) -> ... h l d", h=n_kv_heads)
    if rope:
        assert theta is not None and positions is not None
        Q = _apply_rope_half_rotated(Q, theta, positions)
        K = _apply_rope_half_rotated(K, theta, positions)
    attn = F.scaled_dot_product_attention(Q, K, V, is_causal=True, enable_gqa=True)
    attn = rearrange(attn, "... h l d -> ... l (h d)")
    return F.linear(attn, o_proj_weight)


def test_linear(numpy_snapshot, ts_state_dict, in_embeddings, d_model, d_ff):
    w1_weight = ts_state_dict[0]["layers.0.ffn.w1.weight"]
    output = run_linear(
        d_in=d_model,
        d_out=d_ff,
        weights=w1_weight,
        in_features=in_embeddings,
    )
    numpy_snapshot.assert_match(output)


def test_embedding(numpy_snapshot, ts_state_dict, in_indices, vocab_size, d_model):
    embedding_weight = ts_state_dict[0]["token_embeddings.weight"]
    output = run_embedding(
        vocab_size=vocab_size,
        d_model=d_model,
        weights=embedding_weight,
        token_ids=in_indices,
    )
    numpy_snapshot.assert_match(output)


def test_swiglu(numpy_snapshot, ts_state_dict, in_embeddings, d_model, d_ff):
    w1_weight, w2_weight, w3_weight = [ts_state_dict[0][f"layers.0.ffn.{k}.weight"] for k in ["w1", "w2", "w3"]]

    actual_output = run_swiglu(
        d_model=d_model,
        d_ff=d_ff,
        w1_weight=w1_weight,
        w2_weight=w2_weight,
        w3_weight=w3_weight,
        in_features=in_embeddings,
    )
    numpy_snapshot.assert_match(actual_output, atol=1e-5)


def test_scaled_dot_product_attention(numpy_snapshot, q, k, v, mask):
    actual_output = run_scaled_dot_product_attention(Q=q, K=k, V=v, mask=mask)
    numpy_snapshot.assert_match(
        actual_output,
        atol=1e-5,
    )


def test_4d_scaled_dot_product_attention(numpy_snapshot, q, k, v, mask):
    # Shape: (batch_size, num_heads, seq_len, d_k)
    q, k, v = (rearrange(x, "(batch head) seq d -> batch head seq d", head=2) for x in (q, k, v))
    mask = rearrange(mask, "(batch head) query key -> batch head query key", head=2)

    actual_output = run_scaled_dot_product_attention(Q=q, K=k, V=v, mask=mask)
    numpy_snapshot.assert_match(
        actual_output,
        atol=1e-5,
    )


def test_multihead_self_attention(numpy_snapshot, in_embeddings, d_model, n_heads, ts_state_dict):
    d, _ = ts_state_dict
    q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight = [
        d[f"layers.0.attn.{k}_proj.weight"] for k in ["q", "k", "v", "output"]
    ]
    actual_output = run_multihead_self_attention(
        d_model=d_model,
        num_heads=n_heads,
        q_proj_weight=q_proj_weight,
        k_proj_weight=k_proj_weight,
        v_proj_weight=v_proj_weight,
        o_proj_weight=o_proj_weight,
        in_features=in_embeddings,
    )
    numpy_snapshot.assert_match(actual_output, atol=1e-5)


def test_multihead_self_attention_with_rope(
    in_embeddings, d_model, n_heads, ts_state_dict, n_keys, theta, pos_ids
):
    d, _ = ts_state_dict
    q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight = [
        d[f"layers.0.attn.{k}_proj.weight"] for k in ["q", "k", "v", "output"]
    ]
    pos_ids = rearrange(pos_ids, "seq -> 1 seq")
    actual = run_multihead_self_attention_with_rope(
        d_model=d_model,
        num_heads=n_heads,
        max_seq_len=n_keys,
        theta=theta,
        q_proj_weight=q_proj_weight,
        k_proj_weight=k_proj_weight,
        v_proj_weight=v_proj_weight,
        o_proj_weight=o_proj_weight,
        in_features=in_embeddings,
        token_positions=pos_ids,
    )
    expected = _gqa_reference(
        in_embeddings, q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight,
        n_heads=n_heads, n_kv_heads=n_heads,
        rope=True, theta=theta, positions=pos_ids,
    )
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)


def test_grouped_query_self_attention(in_embeddings, d_model, n_heads, n_kv_heads, d_head):
    torch.manual_seed(42)
    q_proj_weight = torch.randn(n_heads * d_head, d_model)
    k_proj_weight = torch.randn(n_kv_heads * d_head, d_model)
    v_proj_weight = torch.randn(n_kv_heads * d_head, d_model)
    o_proj_weight = torch.randn(d_model, n_heads * d_head)

    actual = run_grouped_query_self_attention(
        d_model=d_model,
        num_heads=n_heads,
        num_kv_heads=n_kv_heads,
        q_proj_weight=q_proj_weight,
        k_proj_weight=k_proj_weight,
        v_proj_weight=v_proj_weight,
        o_proj_weight=o_proj_weight,
        in_features=in_embeddings,
    )
    expected = _gqa_reference(
        in_embeddings, q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight,
        n_heads=n_heads, n_kv_heads=n_kv_heads,
    )
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)


def test_grouped_query_self_attention_with_rope(
    in_embeddings, d_model, n_heads, n_kv_heads, d_head, n_queries, theta, pos_ids
):
    torch.manual_seed(43)
    q_proj_weight = torch.randn(n_heads * d_head, d_model)
    k_proj_weight = torch.randn(n_kv_heads * d_head, d_model)
    v_proj_weight = torch.randn(n_kv_heads * d_head, d_model)
    o_proj_weight = torch.randn(d_model, n_heads * d_head)
    positions = rearrange(pos_ids, "seq -> 1 seq")

    actual = run_grouped_query_self_attention_with_rope(
        d_model=d_model,
        num_heads=n_heads,
        num_kv_heads=n_kv_heads,
        max_seq_len=n_queries,
        theta=theta,
        q_proj_weight=q_proj_weight,
        k_proj_weight=k_proj_weight,
        v_proj_weight=v_proj_weight,
        o_proj_weight=o_proj_weight,
        in_features=in_embeddings,
        token_positions=positions,
    )
    expected = _gqa_reference(
        in_embeddings, q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight,
        n_heads=n_heads, n_kv_heads=n_kv_heads,
        rope=True, theta=theta, positions=positions,
    )
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)


# NOTE: transformer_block / transformer_lm snapshots below were generated with
# classic interleaved RoPE. After switching run_rope to half-rotated (Qwen/Llama) convention,
# these snapshots are stale and must be regenerated once the adapters are implemented.
def test_transformer_lm(
    numpy_snapshot, vocab_size, n_keys, d_model, n_layers, n_heads, d_ff, theta, ts_state_dict, in_indices
):
    state_dict, _ = ts_state_dict

    actual_output = run_transformer_lm(
        vocab_size=vocab_size,
        context_length=n_keys,
        d_model=d_model,
        num_layers=n_layers,
        num_heads=n_heads,
        d_ff=d_ff,
        rope_theta=theta,
        weights=state_dict,
        in_indices=in_indices,
    )
    numpy_snapshot.assert_match(actual_output, atol=1e-4, rtol=1e-2)


def test_transformer_lm_truncated_input(
    numpy_snapshot, vocab_size, n_keys, d_model, n_layers, n_heads, d_ff, theta, ts_state_dict, in_indices
):
    in_indices_truncated = in_indices[..., : in_indices.shape[-1] // 2]
    truncated_actual_output = run_transformer_lm(
        vocab_size=vocab_size,
        context_length=n_keys,
        d_model=d_model,
        num_layers=n_layers,
        num_heads=n_heads,
        d_ff=d_ff,
        rope_theta=theta,
        weights=ts_state_dict[0],
        in_indices=in_indices_truncated,
    )

    numpy_snapshot.assert_match(
        truncated_actual_output,
        atol=1e-4,
    )


def test_transformer_block(numpy_snapshot, ts_state_dict, in_embeddings, d_model, n_heads, d_ff, n_keys, theta):
    block_weights = {k.replace("layers.0.", ""): v for k, v in ts_state_dict[0].items() if "layers.0." in k}

    actual_output = run_transformer_block(
        d_model=d_model,
        num_heads=n_heads,
        d_ff=d_ff,
        max_seq_len=n_keys,
        theta=theta,
        weights=block_weights,
        in_features=in_embeddings,
    )
    numpy_snapshot.assert_match(
        actual_output,
        atol=1e-4,
    )


def test_rmsnorm(numpy_snapshot, ts_state_dict, in_embeddings):
    state_dict, _ = ts_state_dict
    reference_weights = state_dict["layers.1.ln1.weight"]
    d_model = reference_weights.shape[0]

    actual_output = run_rmsnorm(d_model=d_model, eps=1e-5, weights=reference_weights, in_features=in_embeddings)

    numpy_snapshot.assert_match(actual_output, atol=1e-4)


def test_rope(in_embeddings, d_model, theta, n_queries, pos_ids):
    actual = run_rope(
        d_model, theta=theta, max_seq_len=n_queries, in_query_or_key=in_embeddings, token_positions=pos_ids
    )
    expected = _apply_rope_half_rotated(in_embeddings, theta, pos_ids)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)


def test_silu_matches_pytorch():
    x = torch.tensor(
        [
            [0.2352, 0.9259, 0.5189, 0.4725, 0.9730],
            [0.7581, 0.9692, 0.2129, 0.9345, 0.0149],
        ]
    )
    expected_output = F.silu(x)
    actual_output = run_silu(x)
    numpy.testing.assert_allclose(actual_output.detach().numpy(), expected_output.detach().numpy(), atol=1e-5)
