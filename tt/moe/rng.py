"""Named numpy RNG scopes, copied verbatim from cse234-w25-PA/pa3/part1/rng.py.

Kept as numpy (not torch.Generator) so weight init is bit-reproducible
against PA3's reference. The torch Linear classes pull numpy arrays from
these RNGs and wrap them with ``torch.from_numpy``.

Registered defaults:
    "expert"            — shared across ranks, used by replicated experts.
    "router"            — shared across ranks, used by the router.
    "testing"           — reserved for test seeds.
    "expert_with_rank"  — intentionally unregistered; MoE_EP tests must
                          register this per rank (e.g. seed = rank + 100)
                          so each rank owns a different expert.
"""

from contextlib import contextmanager

import numpy as np

_registered_rngs: dict[str, np.random.RandomState] = {}
current_rng: np.random.RandomState = np.random.RandomState(0)


def register_rng(name: str, rng: np.random.RandomState | None = None) -> None:
    """Register a named RNG. Re-registering replaces the previous one."""
    global _registered_rngs
    if rng is None:
        rng = np.random.RandomState(0)
    _registered_rngs[name] = rng


def get_rng() -> np.random.RandomState:
    """Return the currently-active RNG (default or scoped)."""
    return current_rng


@contextmanager
def rng_context(name: str):
    """Temporarily switch the active RNG to the named one."""
    global current_rng
    prev = current_rng
    if name not in _registered_rngs:
        register_rng(name)
    current_rng = _registered_rngs[name]
    try:
        yield
    finally:
        current_rng = prev


register_rng("expert")
register_rng("router")
register_rng("testing")
