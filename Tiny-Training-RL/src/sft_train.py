"""SFT 训练循环和评测共用的工具集。

把原本写在 scripts/train_sft.py 里的"非原语层"工具搬上来，让 §5 EI、
§7 GRPO、§9 leaderboard 三个训练脚本都能 import 复用：

    数据：_qa_to_sft / load_sft_jsonl / SFTDataset / collate_pad
    vLLM：init_vllm / load_policy_into_vllm_instance / evaluate_policy_vllm
    调度：cosine_lr / set_lr
    通用：infinite_iter / set_seed

§4.2 的 6 个原语仍在 src/sft.py。本模块只放"组装层"代码，不放算法本身 ——
算法（loss 公式、masked normalize 选择等）严格留在 src/sft.py 和 src/grpo.py。
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Iterable
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase


# =============================================================================
# 数据：jsonl → SFT batch tensor
# =============================================================================

def _qa_to_sft(row: dict, prompt_template: str) -> dict:
    """{"question","answer"} 或 {"problem","answer"} → {"prompt","response"}。

    GSM8K answer 的 "rationale\\n#### final" 格式被 rsplit("####", 1) 分开：
        - rsplit 从右往左切，确保即使 rationale 段误带 "####"（罕见），
          切出来的最后一段仍是真正的 final answer
        - maxsplit=1：只切一次

    response 前面留一个空格 " {rationale} </think> ..."：r1_zero prompt 末尾
    是 "Assistant: <think>"（无尾空格），response 第一字符如果直接接 rationale
    会让 tokenizer 把 ">" 和 rationale 第一个字粘成同一个 token，造成 train
    时和 infer 时的 token 边界不一致。
    """
    q = row.get("question") or row["problem"]
    a = row["answer"]
    if "####" in a:
        rationale, final = a.rsplit("####", 1)
        rationale = rationale.strip()
        final = final.strip()
    else:
        rationale, final = "", a.strip()
    prompt = prompt_template.format(question=q)
    response = f" {rationale} </think> <answer> {final} </answer>"
    return {"prompt": prompt, "response": response}


def load_sft_jsonl(path: Path, prompt_template: str | None = None) -> list[dict]:
    """读 jsonl；自动识别 schema 并统一成 {"prompt","response"}。

    支持的输入：
        - {"prompt", "response"}    # PDF 原生 sft.jsonl，已拼好 r1_zero 模板
        - {"question", "answer"}    # GSM8K
        - {"problem", "answer"}     # MATH-style raw
    """
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            obj = json.loads(line)
            if "prompt" in obj and "response" in obj:
                rows.append({"prompt": obj["prompt"], "response": obj["response"]})
            elif ("question" in obj or "problem" in obj) and "answer" in obj:
                assert prompt_template is not None, "GSM8K/MATH 风格需要 prompt_template"
                rows.append(_qa_to_sft(obj, prompt_template))
            else:
                raise ValueError(f"unknown row schema: {list(obj.keys())}")
    return rows


class SFTDataset(Dataset):
    """惰性 tokenize 的 SFT Dataset；DataLoader 通过 collate_pad 做 batch padding。

    EI 训练里 dataset 在每个 EI step 会被替换（filtered rollouts），构造代价
    必须低 —— 直接传 list[dict] 进来，不做预 tokenize。
    """

    def __init__(self, rows: list[dict], tokenizer: PreTrainedTokenizerBase, max_seq_len: int = 512):
        self.rows = rows
        self.tok = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        # 调原语 #1 —— src/sft.py::tokenize_prompt_and_output
        from .sft import tokenize_prompt_and_output
        row = self.rows[idx]
        out = tokenize_prompt_and_output([row["prompt"]], [row["response"]], self.tok)
        d = {k: v.squeeze(0) for k, v in out.items()}
        # 截断超长样例（裁尾会丢 </answer>，这是 SFT 已知 bias）
        if d["input_ids"].shape[0] > self.max_seq_len:
            d = {k: v[: self.max_seq_len] for k, v in d.items()}
        return d


def collate_pad(batch: list[dict], pad_id: int) -> dict[str, torch.Tensor]:
    """batch 内右 pad 到最长。

    pad 策略：
        - input_ids / labels    ← pad_id（pad 位置 mask=0，不参与 loss）
        - response_mask         ← 0
    """
    max_len = max(b["input_ids"].shape[0] for b in batch)
    out = {"input_ids": [], "labels": [], "response_mask": []}
    for b in batch:
        pad = max_len - b["input_ids"].shape[0]
        out["input_ids"].append(F.pad(b["input_ids"], (0, pad), value=pad_id))
        out["labels"].append(F.pad(b["labels"], (0, pad), value=pad_id))
        out["response_mask"].append(F.pad(b["response_mask"], (0, pad), value=0))
    return {k: torch.stack(v, dim=0) for k, v in out.items()}


# =============================================================================
# vLLM 封装（PDF §4.3 starter code 同款）
# =============================================================================

def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    """在指定 device 上起 vLLM LLM。

    monkey-patch 两个内部断言（详见 docs/sft_training.md）：
        1. torch.distributed.get_world_size → 1：让 vLLM 走单卡路径
        2. Worker._assert_memory_footprint_increased_during_profiling → no-op：
           profiling 阶段断言假阳性兜底
    """
    from vllm import LLM
    from vllm.model_executor import set_random_seed as vllm_set_random_seed

    vllm_set_random_seed(seed)
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )


def load_policy_into_vllm_instance(policy: PreTrainedModel, llm) -> None:
    """把当前 policy state_dict 灌进 vLLM。

    每次推理（rollout / eval）前必须调；否则 vLLM 跑的是 init 时的旧 ckpt。
    出处：PDF §4.3 starter，TRL grpo_trainer.py L670。
    """
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def evaluate_policy_vllm(
    llm,
    val_rows: list[dict],
    prompt_template: str,
    reward_fn,
    sampling_params,
    log_n_samples: int = 8,
) -> tuple[dict, list[dict]]:
    """val 集生成 + reward 评分。

    返回：
        - metrics: {n, accuracy, format_rate}
        - samples: 前 log_n_samples 条原始样本，给 log_generations 用
    """
    prompts = [prompt_template.format(question=r.get("question") or r.get("problem")) for r in val_rows]
    gts: list[str] = []
    for r in val_rows:
        a = r.get("answer", "")
        gts.append(a.rsplit("####", 1)[-1].strip() if "####" in a else a.strip())

    outs = llm.generate(prompts, sampling_params)
    n = len(prompts)
    n_correct = n_format = 0
    samples = []
    for i, (prompt, gt, out) in enumerate(zip(prompts, gts, outs)):
        text = out.outputs[0].text
        r = reward_fn(text, gt)
        n_correct += int(r["answer_reward"] == 1.0)
        n_format += int(r["format_reward"] == 1.0)
        if i < log_n_samples:
            samples.append({"prompt": prompt, "generation": text, "ground_truth": gt, "reward": r})

    metrics = {
        "n": n,
        "accuracy": n_correct / n,
        "format_rate": n_format / n,
    }
    return metrics, samples


# =============================================================================
# LR schedule 与小工具
# =============================================================================

def cosine_lr(step: int, warmup_steps: int, total_steps: int, peak_lr: float, min_lr_ratio: float = 0.1) -> float:
    """linear warmup 到 peak_lr，再 cosine decay 到 peak_lr × min_lr_ratio。

    PDF §4.3 推荐：3% warmup + cosine decay。min_lr_ratio=0.1 是 Qwen / DeepSeek
    系列经验值，让训练后期还能微调而不卡 lr=0。
    """
    if step < warmup_steps:
        return peak_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, progress)
    coeff = 0.5 * (1 + math.cos(math.pi * progress))
    return peak_lr * (min_lr_ratio + (1 - min_lr_ratio) * coeff)


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for pg in optimizer.param_groups:
        pg["lr"] = lr


def infinite_iter(loader: DataLoader) -> Iterable:
    """把 DataLoader 包成无限循环（多 epoch）。

    SFT 默认单 epoch（PDF §4.3），但 max_train_examples=128 时一个 epoch
    只有 64 step，需要重复多 epoch 才到 max_steps。EI 同理：每个 EI step
    的 filtered dataset 也可能很小，需要多遍过一遍。
    """
    while True:
        for batch in loader:
            yield batch


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
