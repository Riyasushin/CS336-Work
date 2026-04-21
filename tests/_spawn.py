"""Distributed-test harness: run a target function on N spawned processes.

Uses ``torch.multiprocessing.spawn`` + gloo backend so tests run on CPU and
don't need NCCL/CUDA or ``torchrun``. Reusable by future DDP / TP / FSDP tests.

Usage:
    from tests._spawn import run_distributed

    def _worker(rank: int, world_size: int) -> None:
        # inside this function, torch.distributed is initialized
        ...
        assert ...

    def test_something():
        run_distributed(_worker, world_size=4)

Constraints:
    - ``target`` must be a module-level function (picklable), not a lambda or
      nested function. This is a ``mp.spawn`` requirement, not ours.
    - Worker assertions become pytest failures with full tracebacks.
"""

from __future__ import annotations

import importlib
import os
import socket
import sys
import traceback
from typing import Callable

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp


def _find_free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _worker(
    rank: int,
    world_size: int,
    backend: str,
    master_port: int,
    target_qualname: str,
    sys_paths: list[str],
    err_queue,
) -> None:
    """Spawned-process entry point. Lives at module level for picklability."""
    # Spawned interpreter starts with a fresh sys.path — inject the parent's
    # paths so 'tt.*' and 'tests.*' are importable.
    for p in sys_paths:
        if p and p not in sys.path:
            sys.path.insert(0, p)

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    try:
        dist.init_process_group(
            backend=backend, rank=rank, world_size=world_size
        )
        try:
            module_path, func_name = target_qualname.split(":")
            module = importlib.import_module(module_path)
            target = getattr(module, func_name)
            target(rank, world_size)
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()
    except BaseException:
        err_queue.put((rank, traceback.format_exc()))
        raise


def run_distributed(
    target: Callable[[int, int], None],
    world_size: int,
    backend: str = "gloo",
) -> None:
    """Spawn ``world_size`` processes, each running ``target(rank, world_size)``.

    Raises ``pytest.fail`` with the concatenated worker tracebacks if any
    worker fails (assertion or exception).
    """
    if not hasattr(target, "__module__") or not hasattr(target, "__name__"):
        pytest.fail(
            f"run_distributed target must be a named module-level function, "
            f"got {target!r}"
        )
    target_qualname = f"{target.__module__}:{target.__name__}"

    ctx = mp.get_context("spawn")
    err_queue = ctx.Queue()
    master_port = _find_free_port()
    sys_paths = list(sys.path)

    procs = []
    for rank in range(world_size):
        p = ctx.Process(
            target=_worker,
            args=(
                rank,
                world_size,
                backend,
                master_port,
                target_qualname,
                sys_paths,
                err_queue,
            ),
        )
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

    errors = []
    while not err_queue.empty():
        errors.append(err_queue.get())
    if errors:
        msg = "\n".join(
            f"===== rank {r} =====\n{tb}" for r, tb in sorted(errors)
        )
        pytest.fail(f"distributed worker(s) raised:\n{msg}")

    for i, p in enumerate(procs):
        if p.exitcode != 0:
            pytest.fail(f"worker rank {i} exited with code {p.exitcode}")
