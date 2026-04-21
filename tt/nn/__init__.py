"""Neural network building blocks. Stubs only — fill in during Week 1/2.

Every class has the minimal public surface the adapter layer needs:
a constructor that stores hyperparameters plus a ``.weight`` attribute (or
equivalent) that the adapter can overwrite with reference weights, and
``__call__`` that raises ``NotImplementedError`` until you implement it.

Replace each stub with an ``nn.Module`` (or plain class) of your choice.
As long as ``__call__`` returns the correct tensor, the adapters layer
routes cs336_basics test signatures into your implementation unchanged.
"""

from __future__ import annotations

import torch
from torch import Tensor


_NOT_IMPL = "tt.nn.{name} not implemented (Week 1/2 TODO)"


class Linear:
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        self.in_features = in_features
        self.out_features = out_features
        self.weight: Tensor | None = None

    def __call__(self, x: Tensor) -> Tensor:
        raise NotImplementedError(_NOT_IMPL.format(name="Linear"))


class Embedding:
    def __init__(self, vocab_size: int, d_model: int, device=None, dtype=None):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.weight: Tensor | None = None

    def __call__(self, token_ids: Tensor) -> Tensor:
        raise NotImplementedError(_NOT_IMPL.format(name="Embedding"))


class RMSNorm:
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        self.d_model = d_model
        self.eps = eps
        self.weight: Tensor | None = None

    def __call__(self, x: Tensor) -> Tensor:
        raise NotImplementedError(_NOT_IMPL.format(name="RMSNorm"))


class SwiGLU:
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1_weight: Tensor | None = None
        self.w2_weight: Tensor | None = None
        self.w3_weight: Tensor | None = None

    def __call__(self, x: Tensor) -> Tensor:
        raise NotImplementedError(_NOT_IMPL.format(name="SwiGLU"))


class RoPE:
    def __init__(self, d_k: int, theta: float, max_seq_len: int, device=None):
        self.d_k = d_k
        self.theta = theta
        self.max_seq_len = max_seq_len

    def __call__(self, x: Tensor, token_positions: Tensor) -> Tensor:
        raise NotImplementedError(_NOT_IMPL.format(name="RoPE"))


class MultiHeadSelfAttention:
    def __init__(self, d_model: int, num_heads: int):
        self.d_model = d_model
        self.num_heads = num_heads
        self.q_proj_weight: Tensor | None = None
        self.k_proj_weight: Tensor | None = None
        self.v_proj_weight: Tensor | None = None
        self.o_proj_weight: Tensor | None = None

    def __call__(self, x: Tensor) -> Tensor:
        raise NotImplementedError(_NOT_IMPL.format(name="MultiHeadSelfAttention"))


class MultiHeadSelfAttentionWithRoPE:
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, theta: float):
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.q_proj_weight: Tensor | None = None
        self.k_proj_weight: Tensor | None = None
        self.v_proj_weight: Tensor | None = None
        self.o_proj_weight: Tensor | None = None

    def __call__(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        raise NotImplementedError(
            _NOT_IMPL.format(name="MultiHeadSelfAttentionWithRoPE")
        )


class TransformerBlock:
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
    ):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta

    def load_weights(self, weights: dict[str, Tensor]) -> None:
        """The adapter layer will call this to install reference weights.
        You decide the internal shape (nn.Module state_dict, attribute
        assignment, whatever); just make sure this method accepts the key
        set documented in assignment1-basics/tests/adapters.py::run_transformer_block.
        """
        raise NotImplementedError(_NOT_IMPL.format(name="TransformerBlock.load_weights"))

    def __call__(self, x: Tensor) -> Tensor:
        raise NotImplementedError(_NOT_IMPL.format(name="TransformerBlock"))


def scaled_dot_product_attention(
    Q: Tensor, K: Tensor, V: Tensor, mask: Tensor | None = None
) -> Tensor:
    raise NotImplementedError(_NOT_IMPL.format(name="scaled_dot_product_attention"))


def silu(x: Tensor) -> Tensor:
    raise NotImplementedError(_NOT_IMPL.format(name="silu"))


def softmax(x: Tensor, dim: int) -> Tensor:
    raise NotImplementedError(_NOT_IMPL.format(name="softmax"))


__all__ = [
    "Linear",
    "Embedding",
    "RMSNorm",
    "SwiGLU",
    "RoPE",
    "MultiHeadSelfAttention",
    "MultiHeadSelfAttentionWithRoPE",
    "TransformerBlock",
    "scaled_dot_product_attention",
    "silu",
    "softmax",
]
