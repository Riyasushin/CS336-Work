"""Ported from cse234-w25-PA/pa3/part1/test_moe.py.

Design notes:
- ``test_simple_moe`` runs single-process; ``SimpleMoE`` is fully translated
  (no student TODO), so this test passes out of the box and acts as a smoke
  test for the Router / Expert / rng wiring.
- ``test_{tp,ep}_moe_world_size_*`` spawn N worker processes via
  ``tests._spawn.run_distributed`` (gloo backend, CPU-only).

Assertions (each TP/EP worker):
  1. Shape matches (BATCH_SIZE, OUTPUT_DIM).
  2. Output is finite.
  3. Output is not all-zero — a sentinel that catches the stub state, where
     ``ShardedLinear.__call__`` / ``MoE_TP.forward`` / ``MoE_EP.forward``
     return ``torch.zeros(...)`` directly. Once you implement any of these,
     the router's random gates × random expert weights produce nonzero
     output with overwhelming probability.
  4. Cross-rank consistency — TP and EP are both designed so every rank
     sees the SAME final output after combine (the router is replicated,
     intra-expert all_reduce / all_to_all merge contributions). We assert
     ``allreduce(out) == world_size * out`` within fp32 tolerance.

Assertions (1)+(2) alone were not enough: a zeros tensor passes both, so
unimplemented stubs tested green. Adding (3) flips the default state to
RED until you write code; (4) additionally catches a subtle class of bugs
where one rank produces wrong values.

We do NOT compare distributed output to ``SimpleMoE`` numerically:
``ShardedLinear`` and ``Linear`` consume numpy RNG streams at different
rates (``(in, out/ws)`` + ``out/ws`` bias vs ``(in, out)`` weight + zeros
bias), so ``MoE_TP`` and ``SimpleMoE`` produce different weights for the
same seed. That's PA3's design, not a bug in our port.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.distributed as dist

from tests._spawn import run_distributed
from tt.moe import MoE_EP, MoE_TP, SimpleMoE
from tt.moe.rng import register_rng

# --- shared config (matches PA3's test_moe.py) ------------------------------

BATCH_SIZE = 10
FEATURE_DIM = 10
HIDDEN_DIM = 10
OUTPUT_DIM = 10


def _make_input(batch_size: int, feature_dim: int) -> torch.Tensor:
    """Deterministic input tensor (numpy-seeded, then converted to torch).
    Identical on every rank for distributed tests."""
    np.random.seed(0)
    X = np.random.randn(batch_size, feature_dim).astype(np.float32)
    return torch.from_numpy(X)


def _assert_nonzero(out: torch.Tensor, rank: int, where: str) -> None:
    """Sentinel: stubs return zeros. Once you implement the TODO, the
    router's softmax × random expert weights produce nonzero output."""
    if torch.all(out == 0):
        raise AssertionError(
            f"[rank {rank}] {where} returned an all-zero tensor. "
            f"This is the stub state — implement the corresponding "
            f"TODO in tt/moe/ and re-run."
        )


def _assert_cross_rank_consistent(out: torch.Tensor, rank: int, where: str) -> None:
    """Both MoE_TP and MoE_EP produce identical output on every rank
    (router is replicated; sharding is hidden behind the forward pass).
    Verify by checking allreduce(out) == world_size * out."""
    world_size = dist.get_world_size()
    summed = out.clone().contiguous()
    dist.all_reduce(summed)
    expected = out * world_size
    if not torch.allclose(summed, expected, atol=1e-5, rtol=1e-5):
        max_diff = (summed - expected).abs().max().item()
        raise AssertionError(
            f"[rank {rank}] {where}: outputs differ across ranks "
            f"(max |allreduce(out) − world_size*out| = {max_diff:.3e}). "
            f"Every rank should produce the identical combined output."
        )


# --- simple: single-process --------------------------------------------------

def test_simple_moe():
    """SimpleMoE is fully implemented. This test passes today and guards
    against regressions in Router / Expert / rng wiring."""
    num_experts, topk = 10, 10

    # Match PA3's test_simple_moe: register per-rank RNG even though
    # SimpleMoE only uses the 'expert' scope.
    register_rng("expert_with_rank", np.random.RandomState(0 + 100))

    model = SimpleMoE(
        input_dim=FEATURE_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
        num_experts=num_experts,
        topk=topk,
    )
    X = _make_input(BATCH_SIZE, FEATURE_DIM)
    out = model(X)

    assert out.shape == (BATCH_SIZE, OUTPUT_DIM), (
        f"expected {(BATCH_SIZE, OUTPUT_DIM)}, got {tuple(out.shape)}"
    )
    assert torch.isfinite(out).all(), "non-finite output"
    assert not torch.all(out == 0), (
        "SimpleMoE output is all zeros — Router/Expert are the reference "
        "implementation; check tt/moe/layers.py for regressions."
    )


# --- TP: distributed --------------------------------------------------------

def _tp_worker(rank: int, world_size: int) -> None:
    # Per-rank RNG registration keeps API parity with EP; MoE_TP itself
    # uses only the 'expert' scope (replicated).
    register_rng("expert_with_rank", np.random.RandomState(rank + 100))

    model = MoE_TP(
        input_dim=FEATURE_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
        num_experts=world_size,
        topk=world_size,
    )
    X = _make_input(BATCH_SIZE, FEATURE_DIM)
    out = model(X)

    assert out.shape == (BATCH_SIZE, OUTPUT_DIM), (
        f"[rank {rank}] expected {(BATCH_SIZE, OUTPUT_DIM)}, "
        f"got {tuple(out.shape)}"
    )
    assert torch.isfinite(out).all(), f"[rank {rank}] non-finite output"
    _assert_nonzero(out, rank, "MoE_TP.forward")
    _assert_cross_rank_consistent(out, rank, "MoE_TP")


def test_tp_moe_world_size_2():
    run_distributed(_tp_worker, world_size=2)


def test_tp_moe_world_size_4():
    run_distributed(_tp_worker, world_size=4)


# --- EP: distributed --------------------------------------------------------

def _ep_worker(rank: int, world_size: int) -> None:
    # MoE_EP uses 'expert_with_rank' — each rank owns a DIFFERENT expert.
    register_rng("expert_with_rank", np.random.RandomState(rank + 100))

    model = MoE_EP(
        input_dim=FEATURE_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
        num_experts=world_size,
        topk=world_size,
    )
    X = _make_input(BATCH_SIZE, FEATURE_DIM)
    out = model(X)

    assert out.shape == (BATCH_SIZE, OUTPUT_DIM), (
        f"[rank {rank}] expected {(BATCH_SIZE, OUTPUT_DIM)}, "
        f"got {tuple(out.shape)}"
    )
    assert torch.isfinite(out).all(), f"[rank {rank}] non-finite output"
    _assert_nonzero(out, rank, "MoE_EP.forward")
    _assert_cross_rank_consistent(out, rank, "MoE_EP")


def test_ep_moe_world_size_2():
    run_distributed(_ep_worker, world_size=2)


def test_ep_moe_world_size_4():
    run_distributed(_ep_worker, world_size=4)
