"""Adapter layer for cs336_basics tests (Week 1/2).

Contract: every function here matches the signature expected by the
imported test files (copied from ``assignment1-basics/tests/``). Each one
constructs a ``tt.*`` object, installs reference weights onto it, and
returns the output of a forward pass.

When you implement a ``tt.*`` class, the corresponding adapter here
should start working with **no changes to this file**. If you prefer a
different internal API (e.g. nn.Parameter instead of plain attribute),
adjust only the 1-2 lines below that assign weights onto ``layer``.

Convention: we copy reference weights using ``.data.copy_()`` when the
attribute is an ``nn.Parameter``, else plain attribute assignment. Helper
``_set_weight`` below handles both.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
from torch import Tensor


def _set_weight(layer: Any, name: str, value: Tensor) -> None:
    """Install a reference weight onto `layer.<name>`.

    Works whether your implementation stores weights as a plain attribute
    or as an ``nn.Parameter``. Users of the adapter layer don't need to
    know which convention tt uses — pick either and it'll work.
    """
    attr = getattr(layer, name, None)
    if isinstance(attr, torch.nn.Parameter):
        with torch.no_grad():
            attr.data.copy_(value)
    else:
        setattr(layer, name, value)


# ---------- tt.nn wrappers ----------

def run_linear(d_in, d_out, weights, in_features):
    from tt.nn import Linear
    layer = Linear(d_in, d_out)
    _set_weight(layer, "weight", weights)
    return layer(in_features)


def run_embedding(vocab_size, d_model, weights, token_ids):
    from tt.nn import Embedding
    layer = Embedding(vocab_size, d_model)
    _set_weight(layer, "weight", weights)
    return layer(token_ids)


def run_rmsnorm(d_model, eps, weights, in_features):
    from tt.nn import RMSNorm
    layer = RMSNorm(d_model, eps=eps)
    _set_weight(layer, "weight", weights)
    return layer(in_features)


def run_swiglu(d_model, d_ff, w1_weight, w2_weight, w3_weight, in_features):
    from tt.nn import SwiGLU
    layer = SwiGLU(d_model, d_ff)
    _set_weight(layer, "w1_weight", w1_weight)
    _set_weight(layer, "w2_weight", w2_weight)
    _set_weight(layer, "w3_weight", w3_weight)
    return layer(in_features)


def run_scaled_dot_product_attention(Q, K, V, mask=None):
    from tt.nn import scaled_dot_product_attention
    return scaled_dot_product_attention(Q, K, V, mask)


def run_multihead_self_attention(
    d_model, num_heads, q_proj_weight, k_proj_weight,
    v_proj_weight, o_proj_weight, in_features,
):
    from tt.nn import MultiHeadSelfAttention
    layer = MultiHeadSelfAttention(d_model, num_heads)
    _set_weight(layer, "q_proj_weight", q_proj_weight)
    _set_weight(layer, "k_proj_weight", k_proj_weight)
    _set_weight(layer, "v_proj_weight", v_proj_weight)
    _set_weight(layer, "o_proj_weight", o_proj_weight)
    return layer(in_features)


def run_multihead_self_attention_with_rope(
    d_model, num_heads, max_seq_len, theta,
    q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight,
    in_features, token_positions=None,
):
    from tt.nn import MultiHeadSelfAttentionWithRoPE
    layer = MultiHeadSelfAttentionWithRoPE(d_model, num_heads, max_seq_len, theta)
    _set_weight(layer, "q_proj_weight", q_proj_weight)
    _set_weight(layer, "k_proj_weight", k_proj_weight)
    _set_weight(layer, "v_proj_weight", v_proj_weight)
    _set_weight(layer, "o_proj_weight", o_proj_weight)
    return layer(in_features, token_positions)


def run_rope(d_k, theta, max_seq_len, in_query_or_key, token_positions):
    from tt.nn import RoPE
    layer = RoPE(d_k, theta, max_seq_len)
    return layer(in_query_or_key, token_positions)


def run_transformer_block(
    d_model, num_heads, d_ff, max_seq_len, theta, weights, in_features
):
    from tt.nn import TransformerBlock
    block = TransformerBlock(d_model, num_heads, d_ff, max_seq_len, theta)
    block.load_weights(weights)
    return block(in_features)


def run_transformer_lm(
    vocab_size, context_length, d_model, num_layers, num_heads,
    d_ff, rope_theta, weights, in_indices,
):
    from tt.models import TransformerLM
    model = TransformerLM(
        vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta
    )
    model.load_weights(weights)
    return model(in_indices)


def run_silu(in_features):
    from tt.nn import silu
    return silu(in_features)


def run_softmax(in_features, dim):
    from tt.nn import softmax
    return softmax(in_features, dim)


# ---------- tt.optim / tt.utils wrappers ----------

def run_cross_entropy(inputs, targets):
    from tt.optim import cross_entropy
    return cross_entropy(inputs, targets)


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    from tt.optim import gradient_clipping
    return gradient_clipping(parameters, max_l2_norm)


def get_adamw_cls():
    from tt.optim import get_adamw_cls as _inner
    return _inner()


def run_get_lr_cosine_schedule(
    it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters
):
    from tt.optim import get_lr_cosine_schedule
    return get_lr_cosine_schedule(
        it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters
    )


def run_get_batch(dataset: npt.NDArray, batch_size: int, context_length: int, device: str):
    from tt.utils import get_batch
    return get_batch(dataset, batch_size, context_length, device)


def run_save_checkpoint(
    model, optimizer, iteration, out: str | os.PathLike | BinaryIO | IO[bytes]
):
    from tt.utils import save_checkpoint
    return save_checkpoint(model, optimizer, iteration, out)


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes], model, optimizer
) -> int:
    from tt.utils import load_checkpoint
    return load_checkpoint(src, model, optimizer)


# ---------- tt.data wrappers ----------

def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
):
    from tt.data import BPETokenizer
    return BPETokenizer(vocab, merges, special_tokens)


def run_train_bpe(input_path, vocab_size, special_tokens, **kwargs):
    from tt.data import train_bpe
    return train_bpe(input_path, vocab_size, special_tokens, **kwargs)
