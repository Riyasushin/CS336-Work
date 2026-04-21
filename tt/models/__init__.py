"""Top-level models. Stubs only — fill in during Week 2."""

from __future__ import annotations

from torch import Tensor


class TransformerLM:
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
    ):
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta

    def load_weights(self, weights: dict[str, Tensor]) -> None:
        """The adapter layer calls this to install reference weights. See
        assignment1-basics/tests/adapters.py::run_transformer_lm for the
        expected key set."""
        raise NotImplementedError("tt.models.TransformerLM.load_weights (Week 2)")

    def __call__(self, in_indices: Tensor) -> Tensor:
        raise NotImplementedError("tt.models.TransformerLM (Week 2)")


__all__ = ["TransformerLM"]
