"""SimpleMoE / MoE_TP / MoE_EP ported from cse234-w25-PA/pa3/part1/moe.py.

Fully translated:
    - SimpleMoE: single-process reference, complete.

Preserved as student TODOs (filled in by you during Week 7):
    - MoE_TP.forward: TP-style MoE using ShardedLinear internals.
    - MoE_EP.forward: EP-style MoE using all-to-all dispatch/combine.
"""

from __future__ import annotations

import torch

from tt.moe.layers import Expert, Router, ShardedExpert
from tt.moe.rng import rng_context
from tt.parallel import comm


class SimpleMoE:
    """Single-process reference MoE. Routes each token to its top-k experts
    and returns a gate-weighted combination of their outputs."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_experts: int,
        topk: int = 1,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.topk = min(topk, num_experts)

        with rng_context("router"):
            self.router = Router(input_dim, num_experts)
        with rng_context("expert"):
            self.experts = [
                Expert(input_dim, hidden_dim, output_dim)
                for _ in range(num_experts)
            ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        indices, gates = self.router(x, self.topk)

        outputs = torch.zeros(
            (batch_size, self.output_dim), dtype=x.dtype, device=x.device
        )
        for k in range(self.topk):
            for i in range(batch_size):
                expert_idx = int(indices[i, k].item())
                gate = gates[i, k]
                item = x[i : i + 1]                      # (1, input_dim)
                expert_output = self.experts[expert_idx](item)
                outputs[i] = outputs[i] + gate * expert_output[0]
        return outputs

    def __call__(self, x):
        return self.forward(x)


class MoE_TP:
    """Tensor-parallel MoE. Every rank holds a slice of every expert's
    weights via ShardedExpert. The router is replicated across ranks.

    Intra-expert all-reduce is handled inside ShardedLinear, so once you
    implement ShardedLinear.__call__ correctly, this forward is essentially
    a gated sum over experts — no additional collective needed here.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_experts: int,
        topk: int = 1,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.topk = min(topk, num_experts)
        self.rank = comm.get_rank()
        self.world_size = comm.get_world_size()

        with rng_context("router"):
            self.router = Router(input_dim, num_experts)
        with rng_context("expert"):
            self.experts = [
                ShardedExpert(input_dim, hidden_dim, output_dim)
                for _ in range(num_experts)
            ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        outputs = torch.zeros(
            (batch_size, self.output_dim), dtype=x.dtype, device=x.device
        )

        # Step 1 (provided): replicated routing.
        indices, gates = self.router(x, self.topk)

        # TODO(student): Step 2 — TP-style forward pass.
        # The router is deterministic and replicated, so `indices` and `gates`
        # are identical across ranks. For each (i, k) pair you can call
        # self.experts[indices[i, k]](x[i:i+1]) which internally does its own
        # all-reduce inside ShardedLinear and returns the full output on every
        # rank. Accumulate into `outputs[i]` weighted by gates[i, k].
        #
        # Once you want to optimize, batch tokens by expert to avoid the
        # inner Python loop (same trick as SimpleMoE's optional optimization).

        return outputs

    def __call__(self, x):
        return self.forward(x)


class MoE_EP:
    """Expert-parallel MoE. Each rank owns exactly one whole expert.

    Convention: ``num_experts == world_size`` and rank ``r`` owns expert ``r``.
    The router is replicated. Forward uses all-to-all twice: once to dispatch
    tokens to the rank owning their chosen expert, once to combine the
    per-expert outputs back to the originating rank.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_experts: int,
        topk: int = 1,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.topk = min(topk, num_experts)
        self.rank = comm.get_rank()
        self.world_size = comm.get_world_size()

        assert num_experts == self.world_size, (
            f"MoE_EP expects num_experts ({num_experts}) == world_size "
            f"({self.world_size})"
        )

        with rng_context("router"):
            self.router = Router(input_dim, self.num_experts)
        with rng_context("expert_with_rank"):
            self.expert = Expert(input_dim, hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        outputs = torch.zeros(
            (batch_size, self.output_dim), dtype=x.dtype, device=x.device
        )

        # Step 1 (provided): replicated routing.
        indices, gates = self.router(x, self.topk)

        # TODO(student): Steps 2-4 — EP-style dispatch/compute/combine.
        #
        # 2. Build the send buffer: for each local token i and each k, send
        #    a copy of x[i] to rank indices[i, k]. Compute per-rank send
        #    counts (length == world_size). Permute the (token, k) pairs so
        #    items destined for the same rank are contiguous.
        #
        # 3. Dispatch with all_to_all:
        #        recv = comm.all_to_all_single(
        #            send_buf, input_split_sizes=send_counts
        #        )
        #    Now `recv` contains all tokens routed to THIS rank's expert
        #    (possibly duplicated tokens from multiple k values).
        #    Run expert_out = self.expert(recv).
        #
        # 4. Combine with a second all_to_all:
        #    The output_split_sizes of the combine call equals the
        #    input_split_sizes of the dispatch (by symmetry). Un-permute and
        #    scatter results back to the originating (i, k) slot, multiply
        #    by gates[i, k], and sum into `outputs[i]`.
        #
        # Gotcha: when computing send_counts, use:
        #   torch.bincount(indices.flatten(), minlength=self.world_size)
        # to avoid a Python-level loop. But remember to permute x accordingly
        # using torch.argsort(indices.flatten()).

        return outputs

    def __call__(self, x):
        return self.forward(x)
