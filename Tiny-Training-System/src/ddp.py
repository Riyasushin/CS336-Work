"""DDP — distributed data parallel wrappers.

两个容器：
- `DDP`：per-parameter async all-reduce（naive overlap）。comm call 数 = 需要梯度的参数数。
- `DDPBucketed`：按 bucket_size_mb 分桶，一个桶 ready 就 fire 一次异步 all-reduce。
  comm call 数 ≈ ceil(总字节 / bucket 字节)，同时保留与反向的 overlap。

两者都在 __init__ 里 rank-0 broadcast 所有参数和 buffer，保证起点一致。
`finish_gradient_synchronization()` 等所有 handle，再除以 world_size。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn


class DDP(nn.Module):
    """Per-parameter overlapped all-reduce DDP container.

    最简 overlap 版本：每个需要梯度的 param 一个 all-reduce，grad 就绪立即发 async comm。
    comm call 数 = 需梯度 param 数；小 tensor 多时被启动开销拖累
    """

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()
        # 反向时 hook 会把 (handle, param) 填进来，finish_... 里按顺序 wait。
        self._handles: list[tuple[dist.Work, nn.Parameter]] = []

        # 从 rank 0 广播所有参数与 buffer，保证各 rank 起点一致（含 requires_grad=False 的）。
        with torch.no_grad():
            for p in self.module.parameters():
                dist.broadcast(p.data, src=0)
            for b in self.module.buffers():
                dist.broadcast(b.data, src=0)

        # 每个需要梯度的 param 挂 post_accumulate_grad_hook。
        # tied weights 是同一个 Parameter 对象，parameters() 已按 id 去重，只挂一次。
        for p in self.module.parameters():
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(self._make_hook())

    def _make_hook(self):
        def hook(param: nn.Parameter) -> None:
            # async_op=True 立刻返回 Work 句柄，传输在后台 stream/线程上跑，
            # 与反向的后续层 overlap。除以 world_size 留到 finish_... 一次性做，
            # 避免在 hook 里等 wait()（那会毁掉 overlap）。
            handle = dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM, async_op=True)
            self._handles.append((handle, param))

        return hook

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        # optimizer.step() 前调用：等完所有 handle 再把 SUM→AVG。
        # overlap 理想时，这里大多数 wait() 立刻返回。
        for handle, param in self._handles:
            handle.wait()
            param.grad.data.div_(self.world_size)
        self._handles.clear()


@dataclass
class _Bucket:
    """一个 all-reduce 桶。

    - params: 这个桶覆盖的参数列表（所有 grad 就绪后一起 reduce）
    - flat:   预分配的连续 buffer，所有 grad 会被 copy 进这块、作为一次 all-reduce 的 payload
    - ready:  已就绪的 grad 计数，等于 len(params) 时表示可以发 comm
    - handle: 异步 all-reduce 的 Work 对象；None 表示当前还没发或已经被 wait 掉
    """

    params: list[nn.Parameter]
    flat: torch.Tensor
    ready: int = 0
    handle: dist.Work | None = None


class DDPBucketed(nn.Module):
    """Bucketed async all-reduce DDP container.

    设计要点：
    1. **倒序分桶**：PyTorch 反向是从 loss 回溯到输入，最后一层的 param 最先拿到 grad。
       倒序遍历 `parameters()` 装桶，就让"最早就绪的 param"落在 index 小的桶里，
       这些桶最早 fire all-reduce，与后面还没算完的 backward 层 overlap 最充分。
    2. **按字节上限 + dtype 封桶**：参数累计字节数超过阈值、或 dtype 发生切换时封桶。
       dtype 切换要新开桶，因为一个 flat buffer 只能持一种 dtype。
    3. **ready 计数**：每个 param 的 `post_accumulate_grad_hook` 里对桶的 `ready` 加 1，
       满员那一刻把桶里所有 grad 拷进 flat buffer，发一次 `all_reduce(async_op=True)`。
    4. **finish_gradient_synchronization**：等所有还活着的 handle，把 flat 除以 world_size
       得到真正的"平均梯度"，再 copy 回各 param 的 `.grad`。
    """

    def __init__(self, module: nn.Module, bucket_size_mb: float = 25.0) -> None:
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()
        # 桶的字节上限。25MB 是 PyTorch 原生 DDP 的默认——对典型的 transformer 是
        # "启动开销可忽略 & 仍有多个桶可 overlap" 的甜点区。
        self._bucket_bytes = int(bucket_size_mb * 1024 * 1024)

        # —— Step 1: 从 rank 0 广播所有参数/buffer，保证所有 rank 起点一致 ——
        # requires_grad=False 的 Parameter 和 BatchNorm running stats 之类的 buffer
        # 同样要广播，否则各 rank 会从随机初始化跑开。
        with torch.no_grad():
            for p in self.module.parameters():
                dist.broadcast(p.data, src=0)
            for b in self.module.buffers():
                dist.broadcast(b.data, src=0)

        # —— Step 2: 构桶 ——
        # `.parameters()` 的顺序是 module 注册顺序（从 first→last 层）。
        # 反向顺序是 last→first，所以这里 `[::-1]` 让最先就绪的 param 进最前排的桶。
        grad_params = [p for p in self.module.parameters() if p.requires_grad][::-1]
        self._buckets: list[_Bucket] = []
        # id(param) -> 所在桶的 index。用 id 是因为 nn.Parameter 不 hashable-safe，
        # 且 tied weights 是同一个 Python 对象，用 id 天然去重。
        self._param_to_bucket: dict[int, int] = {}

        cur: list[nn.Parameter] = []  # 当前正在装的桶
        cur_bytes = 0
        for p in grad_params:
            p_bytes = p.numel() * p.element_size()
            # 封桶条件：累计字节会超阈值，或者 dtype 和当前桶不同（必须换 flat buffer）。
            # `cur` 非空时才封，避免单个超大 param 被强制拆——它自成一桶即可。
            if cur and (cur_bytes + p_bytes > self._bucket_bytes or p.dtype != cur[0].dtype):
                self._seal_bucket(cur)
                cur, cur_bytes = [], 0
            cur.append(p)
            cur_bytes += p_bytes
        if cur:
            self._seal_bucket(cur)

        # —— Step 3: 给每个需要梯度的 param 挂 hook ——
        # hook 在该 param 的 grad 完成累积后触发。tied weights 虽然在 forward 里
        # 被用多次，但只对应一个 AccumulateGrad 节点，hook 一次 backward 只触发一次。
        for p in grad_params:
            p.register_post_accumulate_grad_hook(self._make_hook())

    def _seal_bucket(self, params: list[nn.Parameter]) -> None:
        """给定一组参数，预分配 flat buffer 并登记进 self._buckets。"""
        total = sum(p.numel() for p in params)
        # flat buffer 的 dtype/device 跟随桶内首个 param（此时桶内 dtype 已保证一致）。
        flat = torch.zeros(total, dtype=params[0].dtype, device=params[0].device)
        idx = len(self._buckets)
        self._buckets.append(_Bucket(params=list(params), flat=flat))
        for p in params:
            self._param_to_bucket[id(p)] = idx

    def _make_hook(self):
        """闭包生成 hook；每次 backward 时 param 的 grad 就绪后调用。"""

        def hook(param: nn.Parameter) -> None:
            bucket = self._buckets[self._param_to_bucket[id(param)]]
            bucket.ready += 1
            # 还没满员就只计数，继续等同桶其他 param 就绪。
            if bucket.ready < len(bucket.params):
                return

            # 满员：把桶内每个 grad flatten 后按顺序拷入 flat buffer。
            # 这里的拷贝是本桶的"打包"成本——换来后面一次性 all-reduce 省掉 N-1 次启动开销。
            offset = 0
            for p in bucket.params:
                n = p.numel()
                bucket.flat[offset : offset + n].copy_(p.grad.data.view(-1))
                offset += n
            # async_op=True：立刻返回 handle，实际传输在后台 stream/线程上跑，
            # 与 backward 剩余部分 overlap。
            bucket.handle = dist.all_reduce(bucket.flat, op=dist.ReduceOp.SUM, async_op=True)

        return hook

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        """optimizer.step() 前调用：等所有 handle，把 summed grad / world_size 写回 .grad。"""
        for bucket in self._buckets:
            if bucket.handle is not None:
                # wait() 是 per-handle 的阻塞点。如果 overlap 得好，此处大多立即返回。
                bucket.handle.wait()
                # SUM → 平均：除以 world_size 得到各 rank 等效的"全局平均梯度"。
                bucket.flat.div_(self.world_size)
                # 把 flat buffer 拆回每个 param 的 .grad。view_as 恢复原 shape，
                # copy_ 写入（而非替换）是为了保留原 .grad 的 storage/identity，
                # 让 optimizer 里可能持有的 grad 引用继续有效。
                offset = 0
                for p in bucket.params:
                    n = p.numel()
                    p.grad.data.copy_(bucket.flat[offset : offset + n].view_as(p.grad))
                    offset += n
                bucket.handle = None
            # 为下一次 backward 把计数归零。
            bucket.ready = 0
