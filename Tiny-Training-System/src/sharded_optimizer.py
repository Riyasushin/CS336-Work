"""ShardedOptimizer — ZeRO-1 简化版 optimizer state sharding。

设计来自 A2 §6：
- 每 rank 只为 ~1/world_size 的 params 维护 optimizer state（AdamW 的 exp_avg / exp_avg_sq 等）
- backward 后每 rank 仍持有完整梯度（由外层 DDP all-reduce 保证一致）
- step 里本 rank 的 inner optimizer 只更新自己 own 的 params
- 最后"每 rank 广播自己的 shard"让所有 rank 的权重重新同步

和正统 ZeRO-1 的差异：
- 梯度仍然 all-reduce（不做 reduce-scatter），因此 grad 显存不减少、只有 optimizer state 显存省下
- 通信量总体相近（AR + rank-broadcasts ≈ RS + AG）

Sync 策略：**per-rank flat buffer + world_size 次 broadcast**。
相比 per-param broadcast（共 P 次 call），这里把 comm call 数降到 world_size，
打包/拆包的思路和 DDPBucketed 完全一致——每次 broadcast 的 payload 是一整个 rank 的 shard。
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

import torch
import torch.distributed as dist
import torch.nn as nn


class ShardedOptimizer(torch.optim.Optimizer):
    """Wrap 任意 `torch.optim.Optimizer` 做 optimizer state sharding。

    Public interface 遵循 A2 §6：
    - `__init__(params, optimizer_cls, **kwargs)`
    - `step(closure=None, **kwargs)`
    - `add_param_group(param_group)`

    继承 `torch.optim.Optimizer` 是为了白嫖 `zero_grad()`、LR scheduler、state_dict 等接口。
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter] | Iterable[dict[str, Any]],
        optimizer_cls: type[torch.optim.Optimizer],
        **kwargs: Any,
    ) -> None:
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = kwargs

        # —— owner 分配表（所有 rank 必须算出完全一致的映射） ——
        # id(param) -> owner rank。用 id 是因为 nn.Parameter 不一定 hashable-safe；
        # tied weights 是同一个 Python 对象，id 天然去重、只被分配一次。
        self._param_to_rank: dict[int, int] = {}
        # 每 rank 当前累计 numel，greedy 分配用。各 rank 独立维护同一份状态
        # （因为输入 params 的顺序和每个 param 的 numel 各 rank 一致）。
        self._rank_load: list[int] = [0] * self.world_size

        # —— inner optimizer 和 sync buffer ——
        # 必须等 super().__init__ 跑完（它会回调 add_param_group 填充 self.param_groups）
        # 才能知道本 rank 要 own 哪些 params。先占位 None。
        self._inner: torch.optim.Optimizer | None = None
        # 每 rank 的 owned param 列表 + 预分配 flat buffer（用于 sync 时打包/拆包）
        self._rank_params: list[list[nn.Parameter]] = [[] for _ in range(self.world_size)]
        self._rank_flat: list[torch.Tensor | None] = [None] * self.world_size

        # 触发 super().__init__ → 回调 self.add_param_group → 填 self.param_groups + 分配 owner
        # defaults={} 因为真正的超参全交给 inner 持有；外层只要 param_groups 的结构方便遍历广播。
        super().__init__(params, defaults={})

        # super 返回后 param_groups 已齐，构造 inner + sync buffer
        self._build_inner_and_buffers()

    # -----------------------------------------------------------------
    # 重建 inner + flat buffer —— 初次构造和 add_param_group 都会走这里
    # -----------------------------------------------------------------
    def _build_inner_and_buffers(self) -> None:
        # (1) 按 owner 把 self.param_groups 里的 params 分桶到 self._rank_params
        self._rank_params = [[] for _ in range(self.world_size)]
        for g in self.param_groups:
            for p in g["params"]:
                self._rank_params[self._param_to_rank[id(p)]].append(p)

        # (2) 为每个 rank 的 shard 预分配一块连续 flat buffer。
        # 注意：假设同一 rank 的 shard 内 dtype/device 一致（ToyModel 全 fp32 满足）。
        # 混合精度场景要按 dtype 拆多块——先留 TODO。
        self._rank_flat = []
        for r in range(self.world_size):
            params = self._rank_params[r]
            if not params:
                # 某 rank 没分到任何 param（例如 world_size > num_params）
                self._rank_flat.append(None)
                continue
            total = sum(p.numel() for p in params)
            self._rank_flat.append(
                torch.empty(total, dtype=params[0].dtype, device=params[0].device)
            )

        # (3) 构造本 rank 的 inner optimizer——param_groups 内只留 owner==self.rank 的那些
        local_groups: list[dict[str, Any]] = []
        for g in self.param_groups:
            local = [p for p in g["params"] if self._param_to_rank[id(p)] == self.rank]
            if not local:
                continue
            # 把 group 里的超参（lr/betas/...）原样继承过来，只替换 params 列表
            local_groups.append(
                {**{k: v for k, v in g.items() if k != "params"}, "params": local}
            )

        # 本 rank 没分到任何 param 时 inner 设为 None；step 里会跳过
        self._inner = (
            self.optimizer_cls(local_groups, **self.optimizer_kwargs) if local_groups else None
        )

    # -----------------------------------------------------------------
    # super().__init__ 会对每个 param_group 调这个；训练中也可能被用户调
    # -----------------------------------------------------------------
    def add_param_group(self, param_group: dict[str, Any]) -> None:
        # greedy by size：把新 param 塞给当前累计 numel 最少的 rank。
        # ties 时 min() 取最小 index，保证各 rank 决策一致。
        for p in param_group["params"]:
            owner = min(range(self.world_size), key=lambda r: self._rank_load[r])
            self._param_to_rank[id(p)] = owner
            self._rank_load[owner] += p.numel()

        # 外层 param_groups 保留完整 param list（step 里要遍历全部 param 做广播）
        super().add_param_group(param_group)

        # 如果是 super().__init__ 回调进来的：self._inner 还是 None，这里什么都不做，
        # 等 __init__ 末尾统一 _build_inner_and_buffers()。
        # 如果是训练途中新增：inner 已存在，这里要重建（注意会丢失 inner 的 optimizer state；
        # 真要保留需要手工迁移 self.state，暂不处理）。
        if self._inner is not None:
            self._build_inner_and_buffers()

    # -----------------------------------------------------------------
    # step：inner.step（只动 owned params） + 每 rank 广播自己的 shard
    # -----------------------------------------------------------------
    def step(self, closure: Callable[[], float] | None = None, **kwargs: Any):
        # 本 rank 只算自己 own 的那部分更新；其他 rank 的 params 此刻是旧值
        loss = self._inner.step(closure, **kwargs) if self._inner is not None else None

        # Sync 阶段：按 rank 顺序，每个 rank 把自己更新过的 shard 打进 flat buffer
        # 并 broadcast，其他 rank 收到后拆回对应 params。
        # comm call 数 = world_size（每次 src 不同的 broadcast）。
        for r in range(self.world_size):
            flat = self._rank_flat[r]
            if flat is None:
                continue
            params = self._rank_params[r]

            if r == self.rank:
                # 自己是 owner：把刚 inner.step 更新过的 params 打进 flat buffer
                offset = 0
                for p in params:
                    n = p.numel()
                    flat[offset : offset + n].copy_(p.data.view(-1))
                    offset += n

            # 所有 rank 都要参与这个 collective，src=r 表示这次 broadcast 由 rank r 发出
            dist.broadcast(flat, src=r)

            if r != self.rank:
                # 自己不是 owner：把刚收到的 flat 拆回各 param 的 .data
                offset = 0
                for p in params:
                    n = p.numel()
                    p.data.copy_(flat[offset : offset + n].view_as(p.data))
                    offset += n

        return loss
