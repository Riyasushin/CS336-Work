"""Functional wrappers over torch.distributed.

These mirror the primitives used by `cse234-w25-PA/pa3/part1/mpiwrapper.py`
(MPI.COMM_WORLD). When torch.distributed is not initialized, every function
degrades to a single-rank no-op so the same call sites work both under
`torch.multiprocessing.spawn` and in plain pytest.

Primitive mapping from PA3:
    mpi.get_rank()              ->  get_rank()
    mpi.get_size()              ->  get_world_size()
    mpi.bcast(x, root=0)        ->  broadcast(x, src=0)
    mpi.allreduce(x)            ->  all_reduce(x)
    mpi.allgather(x)            ->  all_gather(x)
    mpi.alltoall(list_of_bufs)  ->  all_to_all_single(tensor, splits)
    mpi.barrier()               ->  barrier()
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist


def is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_initialized() else 1


def barrier() -> None:
    if is_initialized():
        dist.barrier()


def broadcast(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    if is_initialized() and get_world_size() > 1:
        dist.broadcast(tensor, src=src)
    return tensor


def all_reduce(
    tensor: torch.Tensor,
    op: "dist.ReduceOp" = dist.ReduceOp.SUM,
) -> torch.Tensor:
    """In-place all-reduce; returns the same tensor for convenience."""
    if is_initialized() and get_world_size() > 1:
        dist.all_reduce(tensor, op=op)
    return tensor


def all_gather(tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """All-gather tensors of identical shape, concatenated along ``dim``.

    Requires every rank to pass the same shape. For ragged shapes see
    ``all_gather_object`` in torch.distributed.
    """
    world_size = get_world_size()
    if world_size == 1:
        return tensor.clone()
    gathered = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor.contiguous())
    return torch.cat(gathered, dim=dim)


def all_to_all_single(
    input: torch.Tensor,
    input_split_sizes: Optional[list[int]] = None,
    output_split_sizes: Optional[list[int]] = None,
) -> torch.Tensor:
    """All-to-all over dim 0.

    PA3's MPI call takes ``alltoall(list_of_arrays)`` where element ``i``
    goes to rank ``i``. The torch equivalent is a single contiguous tensor
    with per-rank split sizes.

    Args:
        input: Contiguous tensor of shape (sum(input_split_sizes), *).
        input_split_sizes: Rows of ``input`` sent to each rank. If None,
            ``input`` is split evenly (must be divisible by world_size).
        output_split_sizes: Rows received from each rank. If None, sizes
            are exchanged first via a small size-tensor all_to_all.

    Returns:
        Tensor of shape (sum(output_split_sizes), *) on each rank.
    """
    world_size = get_world_size()
    if world_size == 1:
        return input.clone()

    if input_split_sizes is None:
        assert input.shape[0] % world_size == 0, (
            f"input rows ({input.shape[0]}) not divisible by "
            f"world_size ({world_size})"
        )
        input_split_sizes = [input.shape[0] // world_size] * world_size

    if output_split_sizes is None:
        send_sizes = torch.tensor(
            input_split_sizes, dtype=torch.long, device=input.device
        )
        recv_sizes = torch.empty_like(send_sizes)
        dist.all_to_all_single(recv_sizes, send_sizes)
        output_split_sizes = recv_sizes.tolist()

    output = torch.empty(
        (sum(output_split_sizes),) + tuple(input.shape[1:]),
        dtype=input.dtype,
        device=input.device,
    )
    dist.all_to_all_single(
        output,
        input.contiguous(),
        output_split_sizes,
        input_split_sizes,
    )
    return output
