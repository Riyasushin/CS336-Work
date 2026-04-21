"""Data: tokenizers, BPE, data loading. Stubs only — Week 1/2."""

from __future__ import annotations

import os
from typing import Any


class BPETokenizer:
    """BPE tokenizer. Week 1 TODO. The adapter's get_tokenizer factory
    routes here; your __init__ must accept (vocab, merges, special_tokens)."""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError("tt.data.BPETokenizer.encode (Week 1)")

    def decode(self, ids: list[int]) -> str:
        raise NotImplementedError("tt.data.BPETokenizer.decode (Week 1)")


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs: Any,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    raise NotImplementedError("tt.data.train_bpe (Week 1)")


__all__ = ["BPETokenizer", "train_bpe"]
