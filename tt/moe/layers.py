"""MoE building blocks ported from cse234-w25-PA/pa3/part1/moe.py.

Translation decisions:
- Plain classes (not nn.Module) to match PA3's shape. Swap to
  nn.Module / nn.Parameter later if you want autograd through training.
- Weights are drawn from the numpy RandomState in tt.moe.rng so that
  reproducibility semantics (which RNG is active in which scope) match PA3,
  then wrapped as torch tensors at init time.
- Distributed primitives come from tt.parallel.comm (torch.distributed under
  the hood); no mpi4py.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tt.moe.rng import get_rng, rng_context
from tt.parallel import comm

_DEFAULT_DTYPE = torch.float32


def _as_tensor(array, dtype=_DEFAULT_DTYPE) -> torch.Tensor:
    return torch.from_numpy(array).to(dtype)


class Linear:
    """y = x @ W + b. No nn.Parameter — matches PA3's functional style."""

    def __init__(self, in_features: int, out_features: int, dtype=_DEFAULT_DTYPE):
        w = get_rng().randn(in_features, out_features) * 0.01
        self.weight = _as_tensor(w, dtype)
        self.bias = torch.zeros(out_features, dtype=dtype)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight + self.bias


class Expert:
    """Hidden-layer MLP with ReLU."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        with rng_context("expert"):
            self.fc1 = Linear(input_dim, hidden_dim)
            self.fc2 = Linear(hidden_dim, output_dim)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.fc1(x))
        return self.fc2(hidden)


class Router:
    """Top-k softmax router: returns (indices, gates) of shape (B, K)."""

    def __init__(self, input_dim: int, num_experts: int):
        self.linear = Linear(input_dim, num_experts)

    def __call__(self, x: torch.Tensor, topk: int = 1):
        logits = self.linear(x)                                 # (B, E)
        probs = F.softmax(logits, dim=-1)                       # (B, E)

        # top-k by probability; torch.topk returns (values, indices)
        gates, indices = torch.topk(probs, k=topk, dim=-1)      # (B, K), (B, K)

        # renormalize gates so each row sums to 1
        gates = gates / gates.sum(dim=-1, keepdim=True)
        return indices, gates


class ShardedLinear:
    """Column-sharded linear. Rank r owns ``W[:, r*L : (r+1)*L]`` where
    ``L = out_features / world_size``.

    Bias follows PA3's choice (random init, not zeros). The ``__call__``
    body is left for you to implement.
    """

    def __init__(self, in_features: int, out_features: int, dtype=_DEFAULT_DTYPE):
        self.rank = comm.get_rank()
        self.world_size = comm.get_world_size()

        assert out_features % self.world_size == 0, (
            f"Output features ({out_features}) must be evenly divisible by "
            f"world size ({self.world_size})"
        )

        self.out_features_global = out_features
        self.local_out_features = out_features // self.world_size
        self.output_offset = self.rank * self.local_out_features

        w = get_rng().randn(in_features, self.local_out_features) * 0.01
        self.weight = _as_tensor(w, dtype)
        b = get_rng().randn(self.local_out_features)
        self.bias = _as_tensor(b, dtype)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Handle empty batch (MoE_EP may pass zero rows to some ranks).
        if x.shape[0] == 0:
            return torch.zeros(
                (0, self.out_features_global),
                dtype=x.dtype,
                device=x.device,
            )

        # TODO(student): produce the full (batch, out_features_global) output.
        # Two equivalent strategies; pick one:
        #   (A) Compute local partial = x @ self.weight + self.bias
        #       then comm.all_gather(partial, dim=1) to concat along features.
        #   (B) Write partial into a (batch, out_features_global) zero tensor
        #       at columns [self.output_offset : self.output_offset + L],
        #       then comm.all_reduce(result) — since off-slice entries are
        #       zero on every rank, the sum equals the true concatenation.
        # The PA3 hint image illustrates strategy (B).
        result = torch.zeros(
            (x.shape[0], self.out_features_global),
            dtype=x.dtype,
            device=x.device,
        )
        return result


class ShardedExpert:
    """Expert whose Linear layers are column-sharded across ranks (TP-style)."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        with rng_context("expert"):
            self.fc1 = ShardedLinear(input_dim, hidden_dim)
            self.fc2 = ShardedLinear(hidden_dim, output_dim)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.fc1(x))
        return self.fc2(hidden)
