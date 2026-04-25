"""SFT 数据 packing（supplement PDF §3.2.1）。

PackedSFTDataset:
    把 (prompt, response) 对按 Alpaca 模板拼接成 doc string，全 doc 拼接成单
    token 流（doc 间用 <|end_of_text|> 分隔），按 seq_length 切定长 chunk。
    每 chunk row 提供 input_ids / labels（labels 是 input_ids 内 shift 1）。

    与"每个 prompt+response 一行 + padding"（主 PDF §4.3 用法）的区别：
    - packing 没有 padding 浪费，GPU 利用率高
    - 但 chunk 边界会跨文档：模型在边界 token 上学的是"前一个 doc 末 + 后一
      个 doc 初"的 transition，在 mask 不区分 prompt/response 时也包含
      cross-doc loss。supplement PDF 默认这样跑，是 instruction tuning 的
      标准做法。

iterate_batches:
    一个 DataLoader 工厂；返回的对象有 __len__ + __iter__，每次产生一个 batch
    of {input_ids: (B, L), labels: (B, L)}。
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase


# Alpaca 模板（supplement PDF §3.2.1 给的原版）
ALPACA_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{prompt}\n\n"
    "### Response:\n{response}"
)


class PackedSFTDataset(Dataset):
    """把 jsonl 文档全 packing 成单 token 流，按 seq_length 切等长 chunk。

    Args:
        tokenizer: HF tokenizer
        dataset_path: jsonl 路径（每行 {"prompt", "response"}）
        seq_length: 每 chunk 的 token 数
        shuffle: 是否 doc 级 shuffle 后再 packing（**注意**：是 doc 级，不是
            chunk 级；shuffle=True 让 packing 后的 chunk 边界完全不同 → tests
            用 shuffle=True/False 对比验证）

    返回（per item）：
        {"input_ids": (seq_length,), "labels": (seq_length,)}，dtype=torch.long
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        dataset_path: os.PathLike,
        seq_length: int,
        shuffle: bool,
    ):
        # 1) 读 jsonl 全量
        rows = []
        with Path(dataset_path).open() as f:
            for line in f:
                rows.append(json.loads(line))

        # 2) doc 级 shuffle（在 packing 前；shuffle 在文档边界处生效，让 chunk
        #    内容随机打散）
        if shuffle:
            rng = random.Random(0)  # 测试要求 shuffle 与 unshuffled 不同；用固定 seed 让结果可复现
            rng.shuffle(rows)

        # 3) tokenize 每个 doc，doc 间用 eos_token_id（<|end_of_text|>）分隔
        # PDF 明确："Llama 3.1 8B base uses the <|end_of_text|> token" → 用 eos
        eot_id = tokenizer.eos_token_id
        if eot_id is None:
            # Qwen 等没单独 EOS 的 tokenizer 兜底
            eot_id = tokenizer.convert_tokens_to_ids("<|end_of_text|>")

        # snapshot 实际结构：每个 doc 都以 BOS 起始，doc 间用 EOS 分隔，
        # 最后一个 doc 后面**不**追加 EOS（共 N 个 BOS + N-1 个 EOS）：
        #   [BOS + alpaca(d0)] EOS [BOS + alpaca(d1)] EOS ... [BOS + alpaca(d_{N-1})]
        # 相当于：每 doc 加上 add_special_tokens=True 自动 prepend BOS；
        # 文档之间手动插入 EOS。
        all_tokens: list[int] = []
        for i, doc in enumerate(rows):
            if i > 0:
                # doc 间分隔符；放在每个非首 doc 的 BOS 之前
                all_tokens.append(eot_id)
            text = ALPACA_TEMPLATE.format(prompt=doc["prompt"], response=doc["response"])
            # add_special_tokens=True 让 tokenizer 自动加 BOS 在每个 doc 前
            toks = tokenizer.encode(text, add_special_tokens=True)
            all_tokens.extend(toks)

        self.tokens = torch.tensor(all_tokens, dtype=torch.long)
        self.seq_length = seq_length
        # n_chunks = (N - 1) // L —— 最后 chunk 的 labels 需要 1 个 extra token
        # PDF page 9 的 example：tokens=[0..10] (N=11), L=4 → n_chunks = 10//4 = 2 ✓
        self.n_chunks = max(0, (len(self.tokens) - 1) // seq_length)

    def __len__(self) -> int:
        return self.n_chunks

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        # 越界必须抛 IndexError —— 否则 Python 的 "fallback iteration via
        # __getitem__"（无 __iter__ 时 zip(ds, ...) 会调用 ds[0], ds[1], ...）
        # 会无限循环，因为越界 slice 返回空 tensor 而不报错。
        if i < 0 or i >= self.n_chunks:
            raise IndexError(i)
        s = i * self.seq_length
        e = s + self.seq_length
        # input_ids = tokens[s:e]；labels = tokens[s+1:e+1]（shift by 1）
        # .clone() 避免 in-place 操作影响共享底层 storage
        return {
            "input_ids": self.tokens[s:e].clone(),
            "labels": self.tokens[s + 1 : e + 1].clone(),
        }


def iterate_batches(dataset: Dataset, batch_size: int, shuffle: bool):
    """返回一个 DataLoader（默认 collate 把 dict-of-tensor 堆成 dict-of-batched）。

    PDF §3.2.1 (b) hint：'You may find torch.utils.data.DataLoader to be useful.'
    DataLoader 默认 collate_fn 对 {"input_ids": tensor, "labels": tensor} stack
    成 {"input_ids": (B, L), "labels": (B, L)}，dtype 保持 long。

    支持：
        - len(loader) = ceil(len(dataset) / batch_size)（drop_last=False 默认）
        - iter(loader) 产生 batch_size 个样本一组的 dict batch
        - 最后一个 batch 可能不足 batch_size（保留，符合 PDF 测试期望）
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=0,
    )
