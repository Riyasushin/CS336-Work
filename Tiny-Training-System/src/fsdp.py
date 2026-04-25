"""FSDP — fully-sharded data parallel wrapper (A2 §7 简化版).

实装要点：
- 只对 `RLinear.weight` / `REmbedding.weight` 沿 dim 0 做分片（这两类权重是大头）；
  其余参数（`RRMSNorm.weight` 等）replicated，grad 通过 `all_reduce` 取均值同步。
- `dim0` 不被 world_size 整除时在尾部 pad 零，记住 `dim0` 供 gather 时截回去。
- mixed precision（`compute_dtype` 不为 None）：forward/backward 的 compute 在低精度，
  master 权重和 optimizer 状态保持 fp32；grad 最后 cast 回 fp32 再 reduce-scatter。
- comm 原语：
    * 权重同步用 `all_gather`（每 rank 一份 shard → 拼成 full）
    * 梯度同步用 `all_reduce(SUM) + /world_size`（逻辑上 == reduce-scatter，
      gloo 的 reduce_scatter 支持受限，此方式在 gloo/NCCL 上都 OK）。
- 所有 comm 用同步接口（gloo CPU 后端），`finish_gradient_synchronization()` 是 no-op。
  想做 overlap 时再改异步。

和 DDP / DDPBucketed 的区别：
- DDP 每 rank 持完整权重 + 完整 grad + 完整 optimizer state
- FSDP 每 rank 只持 shard 大小的权重 + shard 大小的 grad；optimizer 跑在 shard 上
  forward/backward 需要的 full 权重靠临时 all-gather 凑出来、算完即释放
"""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn


class FSDP(nn.Module):
    """Fully-sharded data parallel container."""

    def __init__(
        self,
        module: nn.Module,
        compute_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.module = module
        self.compute_dtype = compute_dtype
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        # —— Step 1: 从 rank 0 广播所有参数/buffer，保证各 rank 初值一致 ——
        with torch.no_grad():
            for p in self.module.parameters():
                dist.broadcast(p.data, src=0)
            for b in self.module.buffers():
                dist.broadcast(b.data, src=0)

        # —— Step 2: 找出要 shard 的 module，把它们的 weight 替换成 shard ——
        # 每个条目记 {mod, full_shape, shard_size, dim0, pad}
        self._sharded: dict[str, dict[str, Any]] = {}
        # id(sharded_weight) -> mod_name，用于 gather_full_params / grad hook 去重
        self._sharded_param_id_to_name: dict[int, str] = {}

        # 延迟 import 避免循环
        from tiny_training_basic.layers import REmbedding, RLinear

        for name, mod in list(self.module.named_modules()):
            if isinstance(mod, (RLinear, REmbedding)):
                self._install_shard(name, mod)

        # —— Step 3: 给 replicated 参数挂 grad all-reduce hook ——
        # 注意：sharded 参数的 grad 在另一条路径（_make_sharded_grad_hook）处理
        for p in self.module.parameters():
            if id(p) in self._sharded_param_id_to_name:
                continue
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(self._replicated_grad_hook)

    # -----------------------------------------------------------------
    # 把一个 module 的 weight 切成 shard + 挂好 4 个 hook
    # -----------------------------------------------------------------
    def _install_shard(self, mod_name: str, mod: nn.Module) -> None:
        w = mod.weight
        dim0 = w.shape[0]
        # ceil-div：保证能整除 world_size。不能整除时尾部 pad 零，
        # 截回原尺寸靠 info['dim0']。
        shard_size = (dim0 + self.world_size - 1) // self.world_size
        pad = shard_size * self.world_size - dim0

        with torch.no_grad():
            if pad > 0:
                padded = torch.cat(
                    [w.data, torch.zeros(pad, *w.shape[1:], dtype=w.dtype, device=w.device)],
                    dim=0,
                )
            else:
                padded = w.data
            # 本 rank 只留自己那一段 shard（clone 是为了脱离原 storage，以免替换后还挂着 full）
            shard = padded[
                self.rank * shard_size : (self.rank + 1) * shard_size
            ].clone().contiguous()
            new_param = nn.Parameter(shard, requires_grad=True)

        # 替换 Parameter。identity 变了，所以 hook 必须在这之后注册；
        # optimizer 也要在 FSDP wrap 之后构造（测试里确实是这个顺序）。
        mod.weight = new_param

        self._sharded[mod_name] = {
            "mod": mod,
            "full_shape": (dim0, *w.shape[1:]),
            "shard_size": shard_size,
            "dim0": dim0,
            "pad": pad,
        }
        self._sharded_param_id_to_name[id(mod.weight)] = mod_name

        # 注册 4 个 hook（闭包捕获 mod_name）
        mod.register_forward_pre_hook(self._make_fwd_pre(mod_name))
        mod.register_forward_hook(self._make_fwd_post(mod_name))
        mod.register_full_backward_pre_hook(self._make_bwd_pre(mod_name))
        mod.weight.register_post_accumulate_grad_hook(self._make_sharded_grad_hook(mod_name))

    # -----------------------------------------------------------------
    # 所有 sharded module 的 all-gather：把本 rank 的 shard 拼成 full fp32
    # -----------------------------------------------------------------
    def _all_gather_full(self, mod_name: str) -> torch.Tensor:
        info = self._sharded[mod_name]
        mod = info["mod"]
        shard = mod.weight.data.contiguous()
        # `dist.all_gather` 要求每个元素同 shape——我们靠 pad 保证了
        shards = [torch.empty_like(shard) for _ in range(self.world_size)]
        dist.all_gather(shards, shard)
        full_padded = torch.cat(shards, dim=0)
        if info["pad"] > 0:
            return full_padded[: info["dim0"]].contiguous()
        return full_padded

    # -----------------------------------------------------------------
    # Forward hook 对
    # -----------------------------------------------------------------
    def _make_fwd_pre(self, mod_name: str):
        """forward 开始前：all-gather → full → cast → swap 进 weight.data。"""

        def hook(m: nn.Module, inp):
            full_fp32 = self._all_gather_full(mod_name)
            # 存 shard（fp32）引用，forward 完要复位回来
            m._fsdp_saved_data = m.weight.data
            if self.compute_dtype is not None:
                # cast 到低精度——为了和 baseline 的 mixed-precision hook 语义对齐
                m.weight.data = full_fp32.to(self.compute_dtype)
            else:
                m.weight.data = full_fp32

        return hook

    def _make_fwd_post(self, mod_name: str):
        """forward 结束后：复位 weight.data 为 shard(fp32) + 清 grad。

        清 grad 对齐 baseline 的 mixed-precision hook 写法：forward 里万一有 grad 被累进
        （罕见，但可能），这里先清空，确保 backward 累计的起点为 None。
        """

        def hook(m: nn.Module, inp, out):
            m.weight.data = m._fsdp_saved_data
            del m._fsdp_saved_data
            m.weight.grad = None

        return hook

    # -----------------------------------------------------------------
    # Backward-pre hook：本 module 的 backward 即将开始，需要 full weight
    # -----------------------------------------------------------------
    def _make_bwd_pre(self, mod_name: str):
        """再 all-gather 一次（forward 结束后 data 已复位为 shard）→ full → swap。

        为什么要重新 gather 而不保留 forward 的 full：
        - 想要 FSDP 省显存的本意就是 forward 完立刻释放 full，不长期占用。
        - 换来的是一次额外 all-gather 开销（典型权衡）。
        """

        def hook(m: nn.Module, grad_output):
            full_fp32 = self._all_gather_full(mod_name)
            m._fsdp_saved_data_bwd = m.weight.data  # fp32 shard
            if self.compute_dtype is not None:
                m.weight.data = full_fp32.to(self.compute_dtype)
            else:
                m.weight.data = full_fp32
            # 保证 grad 从 None 起步；如果不清，下面 AccumulateGrad 会尝试往 shard-shape
            # 的旧 grad 上 add full-shape 新 grad，shape mismatch。
            m.weight.grad = None

        return hook

    # -----------------------------------------------------------------
    # post_accumulate_grad_hook：grad 累完了，reduce-scatter + 复位
    # -----------------------------------------------------------------
    def _make_sharded_grad_hook(self, mod_name: str):
        """此时 weight.grad 是 full-shape、在 compute_dtype（或 fp32）上。

        要做的事：
        1) grad cast 到 fp32（master 精度）
        2) pad 到整除 world_size
        3) all_reduce(SUM) + /world_size ← 逻辑 reduce-scatter：先全局求和再切自己那段
        4) 切出本 rank 的 shard，写回 weight.grad
        5) weight.data 复位为 fp32 shard（optimizer 看到 shard-shape 的 data 和 grad）
        """

        def hook(param: nn.Parameter):
            info = self._sharded[mod_name]
            mod = info["mod"]

            full_grad = param.grad.to(torch.float32)
            if info["pad"] > 0:
                padded = torch.cat(
                    [
                        full_grad,
                        torch.zeros(
                            info["pad"],
                            *full_grad.shape[1:],
                            dtype=torch.float32,
                            device=full_grad.device,
                        ),
                    ],
                    dim=0,
                )
            else:
                padded = full_grad.contiguous()

            # all_reduce(SUM) + /world_size 是标准 ZeRO-3 语义。
            # 注意：fp32 严格按位相等做不到——non-parallel 是 sum_{total_N} 单次归约，
            # FSDP 是 sum_{local_N per rank} 再跨 rank SUM 再 /world_size，浮点加法
            # 不满足结合律，两条路径差 1 ulp 是必然。test_fsdp_correctness[fp32]
            # 用 torch.equal 严格相等，因此偶发 failure 是 fp32 精度下限不是 bug。
            dist.all_reduce(padded, op=dist.ReduceOp.SUM)
            padded.div_(self.world_size)

            shard_grad = padded[
                self.rank * info["shard_size"] : (self.rank + 1) * info["shard_size"]
            ].clone().contiguous()

            # weight.data 先复位（必须在 set grad 之前：grad 和 data shape 要一致）
            param.data = mod._fsdp_saved_data_bwd
            del mod._fsdp_saved_data_bwd
            param.grad = shard_grad

        return hook

    # -----------------------------------------------------------------
    # 非 sharded 参数（如 RRMSNorm.weight）：grad 直接 all_reduce 取均值
    # -----------------------------------------------------------------
    def _replicated_grad_hook(self, param: nn.Parameter) -> None:
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data.div_(self.world_size)

    # -----------------------------------------------------------------
    # public API
    # -----------------------------------------------------------------
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        # gloo 所有 comm 都是同步的，没有 async handle 要等；留接口以便以后切异步。
        pass

    def gather_full_params(self) -> dict[str, torch.Tensor]:
        """把所有 sharded 参数 all-gather 回 full shape，返回 `name -> full_tensor`。

        replicated 参数直接复制一份 .data（避免外部改动影响内部）。
        """
        result: dict[str, torch.Tensor] = {}
        for name, p in self.module.named_parameters():
            mod_name = self._sharded_param_id_to_name.get(id(p))
            if mod_name is not None:
                result[name] = self._all_gather_full(mod_name)
            else:
                result[name] = p.data.clone()
        return result
