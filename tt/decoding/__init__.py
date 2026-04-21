"""Decoding: speculative decoding. Stubs only — Week 7.

PA3 Part 3 is a HuggingFace notebook; your task is to port the speculative
decoding loop (target model + draft model + verification) into a proper
Python module that accepts any tt.models.TransformerLM as target/draft.
"""

from __future__ import annotations

from typing import Any

import torch


def speculative_decode(
    target_model: Any,
    draft_model: Any,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    num_draft_tokens: int = 4,
) -> torch.Tensor:
    raise NotImplementedError("tt.decoding.speculative_decode (Week 7)")


__all__ = ["speculative_decode"]
