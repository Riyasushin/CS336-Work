"""SFT training loop for §4.3 (sft_experiment).

PDF §4.3 任务（一句话总结）
==========================
用 Algorithm 1 把 base 模型 fine-tune 到能产出 r1_zero 模板的解题轨迹：
    pretrain ckpt --(SFT)--> 学会 <think>...</think> <answer>...</answer> 模板。

§3 zero-shot 已经验证 base 模型在 r1_zero prompt 下 80% 不合规（accuracy 1.56%）；
SFT 是修复"模板 mismatch"的第一步。具体跑法 PDF 要：
    - 2 GPU：policy 一卡训 + vLLM 一卡 in-the-loop 评测
    - 数据：/data/a5-alignment/MATH/sft.jsonl（每行 {"prompt", "response"}）
    - 模型：/data/a5-alignment/models/Qwen2.5-Math-1.5B
    - 训练：单 epoch、context 512、effective batch 32、lr 2e-5、cosine decay、3% warmup
    - 评测：每 N 步在 MATH validation 上跑 1024 题
    - 扫数据规模 {128, 256, 512, 1024, full}，让 full 能 ≥15% acc
    - 过滤后再跑（只留答对的 SFT 样例）

为什么本机不跑
--------------
单卡 RTX 4060 8GB；1.5B bf16 + grad + activation + optimizer state 远超
8GB。脚本本身是 PDF 的 deliverable，结构尽可能对齐 starter code，方便
搬到 2×80GB 集群直接跑。

数据兜底
--------
/data/a5-alignment/MATH/sft.jsonl 在本机不可得；GSM8K train (7473 题) 在
/data/CS336-use/assignment5-alignment/gsm8k/train.jsonl，schema 是
{"question", "answer"}。`load_sft_jsonl` 自动检测：
    - {"prompt","response"}    → 直接用
    - {"question","answer"}    → 用 r1_zero template + 把 "<rationale> #### <final>"
                                 转成 "<think>{rationale}</think><answer>{final}</answer>"
    - {"problem","answer"}     → 同上（MATH raw 风格）

调用六个 SFT 原语
-----------------
（这些都在 src/sft.py 里实装了；这里只是组装；调用关系一图：）

    Dataset.__getitem__
        └─► tokenize_prompt_and_output  ← Problem #1
    train_loop.microbatch
        ├─► get_response_log_probs       ← Problem #3 (内部用 #2 compute_entropy)
        └─► sft_microbatch_train_step    ← Problem #5 (内部用 #4 masked_normalize)
    eval_loop
        ├─► get_response_log_probs (return_token_entropy=True)  ← #2 + #3
        └─► log_generations              ← Problem #6

CLI
---
    uv run python scripts/train_sft.py \\
        --model /data/a5-alignment/models/Qwen2.5-Math-1.5B \\
        --train-data /data/a5-alignment/MATH/sft.jsonl \\
        --val-data /data/a5-alignment/MATH/validation.jsonl \\
        --output-dir /data/runs/sft \\
        --learning-rate 2e-5 --train-batch-size 32 --micro-batch-size 2 \\
        --max-train-examples -1 --max-steps 2000 --eval-every 100 \\
        --policy-device cuda:0 --vllm-device cuda:1 \\
        --wandb-project tiny-training-rl

§4.3 (1) 数据规模扫描：改 --max-train-examples ∈ {128,256,512,1024,-1}。
§4.3 (2) filtered-SFT：先 scripts/filter_sft.py 把 sft.jsonl 过滤成 only-correct，
         再 --train-data 喂过滤后的文件。
"""

from __future__ import annotations

import argparse
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
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


# =============================================================================
# 1. 数据：从 jsonl 到 batch tensor
# =============================================================================
#
# 数据流：
#   sft.jsonl ──load_sft_jsonl──► [{"prompt", "response"}, ...]
#                                              │
#                                              ▼
#                                       SFTDataset (lazy tokenize)
#                                              │  __getitem__ 调原语 #1
#                                              ▼
#                                       collate_pad（batch 内右 pad）
#                                              │
#                                              ▼
#                              {input_ids, labels, response_mask} 形状 (B, T)
#
# 设计取舍：
#   惰性 tokenize 而非预 tokenize 整个数据集 —— sft.jsonl 量级 ~10K-100K，
#   可以预 tokenize；但脚本要兼容更大数据集（比如 EI 后期生成的 rollout
#   train set），统一用 lazy 路径，第一 epoch 慢一点但内存友好。
# =============================================================================

def _gsm8k_to_sft(row: dict, prompt_template: str) -> dict:
    """{"question","answer"} → {"prompt","response"}。

    GSM8K 的 answer 字段是 "rationale steps\\n#### final" 的格式（"####" 是
    GSM8K 数据集自己的 sentinel）。SFT 里我们要教模型生成 r1_zero 模板下
    完整的 "<think> rationale </think> <answer> final </answer>"，所以：
        - prompt   = r1_zero(question)             （让模型看到 "<think>" 起始）
        - response = " rationale </think> <answer> final </answer>"
                                  ↑ 注意第一个 <think> 已经在 prompt 末尾了，
                                    response 不能再开一次

    response 前面留一个空格，是因为 prompt 末尾 "Assistant: <think>" 之后没
    空格；如果 response 直接接 rationale 字符，分词时会把 think 的尖括号和
    rationale 第一个字粘到一起，造成 token 边界飘移（少数 case 会让
    train→infer 分布不一致）。空一格是最简单的对齐方式。
    """
    q = row["question"]
    a = row["answer"]
    if "####" in a:
        # 用 rsplit("####", 1) 而不是 split：
        #   GSM8K 的 "####" 是数据集自己规定的最终答案 sentinel，**约定**
        #   出现在整个 answer 字符串的末尾、紧贴最终数字。但 rationale 段
        #   理论上可能也写出 "####"（比如 latex 里的 align 行；GSM8K 极
        #   罕见，但 MATH-flavor 数据可能有）。rsplit 从右往左切，强保证
        #   "最后一个 ####" 才是 sentinel —— 这样即便 rationale 误带 ####
        #   也只有最后一个（真 sentinel）参与切分。
        # maxsplit=1：只切一次，保证 final 一定是单段、不会再被进一步切碎。
        rationale, final = a.rsplit("####", 1)
        rationale = rationale.strip()
        final = final.strip()
    else:
        # MATH 风格的 raw answer 没 "####"：把整个 a 当 final，rationale 留空。
        # 这种情形 SFT 学到的只有 "→ <answer>final</answer>"，think 段是空的；
        # 训练时 token 数量短一些。
        rationale, final = "", a.strip()
    prompt = prompt_template.format(question=q)
    response = f" {rationale} </think> <answer> {final} </answer>"
    return {"prompt": prompt, "response": response}


def load_sft_jsonl(path: Path, prompt_template: str | None = None) -> list[dict]:
    """读 jsonl 数据集；自动识别三种 schema 并统一成 {"prompt","response"}。"""
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            obj = json.loads(line)
            if "prompt" in obj and "response" in obj:
                # PDF 原生 sft.jsonl —— 已经按 r1_zero 模板拼好了
                rows.append({"prompt": obj["prompt"], "response": obj["response"]})
            elif "question" in obj and "answer" in obj:
                # GSM8K 兜底
                assert prompt_template is not None, "GSM8K 风格需要 prompt_template"
                rows.append(_gsm8k_to_sft(obj, prompt_template))
            elif "problem" in obj and "answer" in obj:
                # MATH-style raw（仍保留以备将来），把 problem→question 后转
                rows.append(_gsm8k_to_sft({"question": obj["problem"], "answer": obj["answer"]}, prompt_template))
            else:
                raise ValueError(f"unknown row schema: {list(obj.keys())}")
    return rows


class SFTDataset(Dataset):
    """惰性 tokenize 的 SFT Dataset。

    每条样例独立 tokenize 后返回三件套（input_ids / labels / response_mask），
    长度不一；DataLoader 通过 `collate_pad` 把 batch 内 pad 到等长。

    为什么不在 __init__ 里预 tokenize：
        sft.jsonl 在 GRPO 后续 EI rollout 阶段会被频繁覆写（每轮迭代生成新
        rollout）；惰性路径让脚本在不同数据规模 / 多次重跑下都不需要预热。
        代价：第一个 epoch tokenize 是 CPU 串行的，但 num_workers≥2 就能
        和 GPU forward 重叠掩掉。
    """

    def __init__(self, rows: list[dict], tokenizer: PreTrainedTokenizerBase, max_seq_len: int):
        self.rows = rows
        self.tok = tokenizer
        self.max_seq_len = max_seq_len  # 超过的样例会被裁掉尾部 —— 见 __getitem__

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        # 用我们 §4.2 实装的原语 #1（src/sft.py::tokenize_prompt_and_output）
        from tiny_training_rl.sft import tokenize_prompt_and_output
        row = self.rows[idx]
        out = tokenize_prompt_and_output([row["prompt"]], [row["response"]], self.tok)

        # 单条样例 squeeze batch 维（DataLoader 会沿 dim=0 重新堆回）
        d = {k: v.squeeze(0) for k, v in out.items()}

        # 截断超长样例：sft 数据里偶有长 rationale，超过 max_seq_len 就裁尾。
        # 注意：截尾会丢失 response 末尾的 "</answer>" 段，让模型看不到完整
        # 终止信号 —— 这是 SFT 的已知 bias 来源；不过实践上 max_seq_len=512
        # 对 GSM8K 几乎没影响，对 MATH 长题需要调到 1024+。
        if d["input_ids"].shape[0] > self.max_seq_len:
            d = {k: v[: self.max_seq_len] for k, v in d.items()}
        return d


def collate_pad(batch: list[dict], pad_id: int) -> dict[str, torch.Tensor]:
    """把不同长度的 (input_ids, labels, response_mask) 三元组 pad 到 batch 内最长。

    pad 选项：
        - input_ids   ← pad_id    （pad 位置不参与 loss 因为 response_mask=0）
        - labels      ← pad_id    （同上）
        - response_mask ← 0        （pad 位置 mask=0，sft loss 跳过）

    为什么用 batch 内最长而不是脚本全局最长（如 max_seq_len）：
        全 batch pad 到 max_seq_len 大部分 token 是 padding，浪费 FLOPs。
        batch 内 pad 让 sequence 长度浮动 —— 短 batch 跑得快，长 batch 慢
        但少；总训练时长更短。
        代价：每个 step 的 token 总数不固定，吞吐 metric 抖动；用 wandb 看
        平均就行。
    """
    max_len = max(b["input_ids"].shape[0] for b in batch)
    out = {"input_ids": [], "labels": [], "response_mask": []}
    for b in batch:
        L = b["input_ids"].shape[0]
        pad = max_len - L
        out["input_ids"].append(F.pad(b["input_ids"], (0, pad), value=pad_id))
        out["labels"].append(F.pad(b["labels"], (0, pad), value=pad_id))
        out["response_mask"].append(F.pad(b["response_mask"], (0, pad), value=0))
    return {k: torch.stack(v, dim=0) for k, v in out.items()}


# =============================================================================
# 2. vLLM eval helper（PDF §4.3 starter code，保留 monkey-patch 的原因解释）
# =============================================================================

def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    """在指定 device 上起一个 vLLM LLM 实例，用作 in-the-loop validation 推理引擎。

    两个 monkey-patch 是 PDF 给的 starter code（出处 TRL grpo_trainer.py L22759），
    解释如下：

    patch #1: torch.distributed.get_world_size → 1
        vLLM 启动时若发现 torch.distributed 已初始化（policy 用了 DDP/FSDP），
        会按多卡分布式初始化路径走，但我们想让 vLLM 独占一张卡（不分布式），
        强行返回 world_size=1 让 vLLM 走单卡路径。

    patch #2: Worker._assert_memory_footprint_increased_during_profiling → no-op
        vLLM 在 startup 时跑一遍 dummy forward 测显存峰值，断言显存确实增长
        了（防止 lazy-init bug）；某些 GPU + 某些 vLLM 版本组合下这个断言
        会假阳性触发；patch 掉是稳妥的兜底。

    参数：
        model_id: HF repo 或本地路径
        device:   "cuda:1" 之类，必须和 policy 不同卡
        gpu_memory_utilization: vLLM 占 device 显存比例；默认 0.85，剩余给
                  KV cache 流和 OS 反应空间。policy 那卡训练时显存吃紧，
                  这里如果两卡总显存大可以提到 0.90；小卡降到 0.70。
    """
    from vllm import LLM
    from vllm.model_executor import set_random_seed as vllm_set_random_seed

    vllm_set_random_seed(seed)
    # TODO patch 是在做什么
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
    """把当前 policy 的 state_dict 灌进 vLLM 实例的 model runner。

    每次 eval 之前必须调一次 —— 否则 vLLM 跑的还是 init_vllm 时刻的旧权重，
    eval acc 会卡在 step 0 的水平不动。

    实现细节（抄自 PDF §4.3 starter，TRL grpo_trainer.py L670）：
        vLLM 0.7 内部模型对象的访问路径是 llm.llm_engine.model_executor.
        driver_worker.model_runner.model；它有个 load_weights(items)
        方法接受 (name, tensor) 迭代器。

    注意：load_weights 直接拷 weight 不做 dtype 转换 —— policy 是 bf16，
    vLLM 也是 bf16，吻合；如果 policy 改成 fp32，要先 .to(bf16)。
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
    """在 vLLM 上跑 val 集生成 + reward 评分。

    返回：
        - metrics: {n, accuracy, format_rate, avg_response_len_*}
        - samples: 前 log_n_samples 条原始 (prompt, generation, gt, reward) tuple，
                   后续喂给 log_generations(原语 #6) 做更详细的统计。
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
    rewards_for_log = []
    gens_for_log = []
    for i, (prompt, gt, out) in enumerate(zip(prompts, gts, outs)):
        text = out.outputs[0].text
        r = reward_fn(text, gt)
        n_correct += int(r["answer_reward"] == 1.0)
        n_format += int(r["format_reward"] == 1.0)
        if i < log_n_samples:
            samples.append({"prompt": prompt, "generation": text, "ground_truth": gt, "reward": r})
            rewards_for_log.append(r)
            gens_for_log.append(text)

    metrics = {
        "n": n,
        "accuracy": n_correct / n,
        "format_rate": n_format / n,
    }
    return metrics, samples


# =============================================================================
# 3. LR schedule —— linear warmup + cosine decay
# =============================================================================

def cosine_lr(step: int, warmup_steps: int, total_steps: int, peak_lr: float, min_lr_ratio: float = 0.1) -> float:
    """linear warmup 到 peak_lr，再 cosine decay 到 peak_lr × min_lr_ratio。

    PDF §4.3 推荐：linear warmup（总训练步数 3%）+ cosine decay。min_lr_ratio
    保留小学习率（10%）让训练后期还能继续微调，但避免 lr=0 卡住。
    """
    if step < warmup_steps:
        # warmup：(step+1) / warmup —— +1 让第 0 步 lr 不是 0
        return peak_lr * (step + 1) / max(1, warmup_steps)
    # decay：cosine 从 peak 到 peak × min_lr_ratio
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, progress)
    coeff = 0.5 * (1 + math.cos(math.pi * progress))
    return peak_lr * (min_lr_ratio + (1 - min_lr_ratio) * coeff)


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for pg in optimizer.param_groups:
        pg["lr"] = lr


# =============================================================================
# 4. 主训练循环
# =============================================================================
#
#  for step in 1..max_steps:
#      for _ in 1..grad_accum:                     ← microbatch 累计梯度
#          fwd: get_response_log_probs(...)        ← 原语 #3 (含 #2 compute_entropy)
#          bwd: sft_microbatch_train_step(...)     ← 原语 #5 (含 #4 masked_normalize)
#      grad_clip → set_lr(cosine) → optimizer.step()
#      if step % eval_every == 0:                  ← in-the-loop validation
#          load_policy_into_vllm_instance(...)
#          metrics, samples = evaluate_policy_vllm(...)
#          log_generations(samples → wandb)        ← 原语 #6
#      if step % save_every == 0:
#          policy.save_pretrained(...)
# =============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--train-data", required=True)
    p.add_argument("--val-data", default=None)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--prompt-name", default="r1_zero")
    p.add_argument("--max-train-examples", type=int, default=-1, help="-1 = all；§4.3(1) 用 {128,256,512,1024,-1}")
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--learning-rate", type=float, default=2e-5, help="PDF §4.3 推荐 2e-5；扫数据时可联调")
    p.add_argument("--warmup-ratio", type=float, default=0.03, help="PDF §4.3 推荐 3%% (linear warmup of total steps)")
    p.add_argument("--train-batch-size", type=int, default=32, help="effective batch（grad accum 后）")
    p.add_argument("--micro-batch-size", type=int, default=2, help="单 GPU 显存能塞下的 batch")
    p.add_argument("--max-seq-len", type=int, default=512, help="PDF §4.3 推荐 512；MATH 长题 → 1024")
    p.add_argument("--grad-clip", type=float, default=1.0, help="PDF §4.3 推荐 1.0")
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--eval-num", type=int, default=128, help="val 子集大小；PDF 要 ≥1024 才稳，本地 demo 128")
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--policy-device", default="cuda:0")
    p.add_argument("--vllm-device", default="cuda:1")
    p.add_argument("--vllm-gpu-mem", type=float, default=0.85)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--no-grad-checkpoint", action="store_true",
                   help="默认开 gradient_checkpointing 省显存；80GB 卡 + 1.5B 不开也行")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ----- 4.1 加载 tokenizer + policy -----
    # bf16：1.5B 在 fp32 下 ~6GB 模型权重 + 2× 6GB grad/activation + 优化器
    # state（AdamW 2× param 量）几乎要 30GB；bf16 减半。flash-attention-2
    # 进一步把长 seq 的 attention 内存从 O(T^2) 降到 O(T) —— 1024 seq 时
    # 收益最明显。
    print(f"[load] policy on {args.policy_device}")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        # Qwen 没单独 pad token，用 eos 兜底；padding 位置反正 mask=0 不参与 loss
        tok.pad_token_id = tok.eos_token_id
    try:
        policy = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
        )
    except Exception as e:
        print(f"  flash-attn-2 unavailable ({e}); fallback to sdpa")
        policy = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    policy = policy.to(args.policy_device)
    if not args.no_grad_checkpoint:
        # gradient checkpointing：用 1× 重新前向换 ~30% 显存。1.5B + bf16 +
        # batch=2 + seq=512 在 80GB 卡上不开也能跑，但开了能 batch=4 提吞吐。
        policy.gradient_checkpointing_enable()

    # AdamW 配 PDF §4.3 starter（betas=(0.9,0.95)，weight_decay=0）—— 这套
    # 是 Qwen / DeepSeek 系列大模型 SFT 默认参数，跟 §7 GRPO 优化器同款。
    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.0
    )

    # ----- 4.2 数据 -----
    from tiny_training_rl.prompts import load as load_prompt
    prompt_template = load_prompt(args.prompt_name)
    train_rows = load_sft_jsonl(Path(args.train_data), prompt_template)
    if args.max_train_examples > 0:
        train_rows = train_rows[: args.max_train_examples]
    print(f"[data] train rows: {len(train_rows)}")

    train_ds = SFTDataset(train_rows, tok, max_seq_len=args.max_seq_len)
    grad_accum = max(1, args.train_batch_size // args.micro_batch_size)
    print(f"[data] grad_accum={grad_accum} (eff bs={args.train_batch_size}, micro bs={args.micro_batch_size})")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.micro_batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_pad(b, tok.pad_token_id),
        num_workers=2,
        drop_last=True,
        # pin_memory：CPU→GPU 拷贝更快；grad_checkpointing 时收益更大
        pin_memory=True,
    )

    # ----- 4.3 vLLM eval -----
    val_rows: list[dict] = []
    llm = None
    sampling_params = None
    reward_fn = None
    if args.val_data and Path(args.val_data).exists():
        print(f"[load] vllm on {args.vllm_device}")
        llm = init_vllm(args.model, args.vllm_device, args.seed, args.vllm_gpu_mem)
        from vllm import SamplingParams
        from tiny_training_rl import grader
        reward_fn = grader.r1_zero_reward_fn
        sampling_params = SamplingParams(
            temperature=1.0, top_p=1.0, max_tokens=1024,
            # stop=["</answer>"]：让 vLLM 在第二个答案 tag 处自己停，
            # 避免后面 token 被采到日志里干扰 reward 判断（与 §3 同款）
            stop=["</answer>"],
            include_stop_str_in_output=True,
        )
        with Path(args.val_data).open() as f:
            for i, line in enumerate(f):
                if i >= args.eval_num:
                    break
                val_rows.append(json.loads(line))
        print(f"[data] val rows: {len(val_rows)}")

    # ----- 4.4 wandb -----
    wandb = None
    if args.wandb_project:
        try:
            import wandb as wb
            wb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
            # PDF §4.3 starter：分 train/* 与 eval/* 两套 step axis，让 wandb
            # 能把训练曲线和评测曲线对齐到统一 x 轴而不互相 squash
            wb.define_metric("train_step")
            wb.define_metric("eval_step")
            wb.define_metric("train/*", step_metric="train_step")
            wb.define_metric("eval/*", step_metric="eval_step")
            wandb = wb
        except Exception as e:
            print(f"  wandb unavailable: {e}")

    # ----- 4.5 主循环 -----
    from tiny_training_rl.sft import (
        compute_entropy,                # 原语 #2
        get_response_log_probs,         # 原语 #3
        sft_microbatch_train_step,      # 原语 #5（内部用 #4）
        log_generations,                # 原语 #6
    )

    total_steps = args.max_steps
    warmup_steps = int(total_steps * args.warmup_ratio)
    step = 0
    accumulated_loss = 0.0
    accumulated_entropy = 0.0
    t0 = time.perf_counter()

    train_iter = _infinite(train_loader)
    while step < total_steps:
        # --- (a) microbatch loop：grad_accum 次前向 + backward 累积梯度 ---
        for _ in range(grad_accum):
            batch = next(train_iter)
            input_ids = batch["input_ids"].to(args.policy_device, non_blocking=True)
            labels = batch["labels"].to(args.policy_device, non_blocking=True)
            mask = batch["response_mask"].to(args.policy_device, non_blocking=True)

            # 原语 #3：一次 forward 同时拿 log_probs（loss 用）和 token_entropy
            # （metric 用）。return_token_entropy=True 在前向后多一次
            # logsumexp + 加权和（compute_entropy 内部），代价小。
            out = get_response_log_probs(policy, input_ids, labels, return_token_entropy=True)
            log_probs = out["log_probs"]
            tok_entropy = out["token_entropy"]

            # 原语 #5：在 microbatch 内部直接 backward()，让计算图当场释放
            # （否则 grad_accum 个 microbatch 的图同时驻留显存，1.5B + bf16 +
            # G=16 就上 30+ GB）。loss 已经做了 / grad_accum 缩放，等价于
            # 一个大 batch 的 mean。
            loss, _ = sft_microbatch_train_step(
                log_probs, mask.float(), grad_accum, normalize_constant=1.0
            )
            accumulated_loss += loss.item()

            # 训练熵：只在 response mask 区域取均值（用 .float() 让 fp32 精度，
            # 避免 bf16 上加和时丢精度）。这个量对 §7 GRPO 训练监控很重要 ——
            # 模型从 base → SFT 后 entropy 一般会显著下降；反之 GRPO 早期
            # entropy 反弹是 sign of "policy 在探索未学过的解"。
            mask_f = mask.float()
            n_resp = mask_f.sum().clamp(min=1)
            mean_entropy = (tok_entropy * mask_f).sum() / n_resp
            accumulated_entropy += mean_entropy.item()

        # --- (b) gradient clip + LR step + optimizer.step ---
        # PDF §4.3 推荐 grad_clip=1.0：bf16 训练偶有 spike，clip 防止单步爆掉
        gnorm = torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
        lr = cosine_lr(step, warmup_steps, total_steps, args.learning_rate)
        set_lr(optimizer, lr)
        optimizer.step()
        # set_to_none=True 比 zero_grad() 快：直接把 .grad 设 None 而不是
        # 写零；下一次 backward 再分配。
        optimizer.zero_grad(set_to_none=True)
        step += 1

        # --- (c) 训练 metric log ---
        if step % 10 == 0 or step == 1:
            elapsed = time.perf_counter() - t0
            log = {
                "train/loss": accumulated_loss,
                "train/lr": lr,
                "train/grad_norm": float(gnorm),
                "train/avg_token_entropy": accumulated_entropy / grad_accum,
                "train/elapsed_sec": elapsed,
                "train_step": step,
            }
            print(
                f"step {step:5d}/{total_steps} "
                f"loss={accumulated_loss:.4f} lr={lr:.2e} "
                f"gnorm={float(gnorm):.3f} entropy={accumulated_entropy/grad_accum:.3f}"
            )
            if wandb:
                wandb.log(log)
        accumulated_loss = 0.0
        accumulated_entropy = 0.0

        # --- (d) 周期性 vLLM eval ---
        if llm and step % args.eval_every == 0:
            print(f"[eval] step {step} ...")
            policy.eval()
            with torch.inference_mode():
                # 必须先把当前 policy 灌进 vLLM 实例 —— 否则跑的是旧 ckpt
                load_policy_into_vllm_instance(policy, llm)
                metrics, samples = evaluate_policy_vllm(
                    llm, val_rows, prompt_template, reward_fn, sampling_params,
                    log_n_samples=8,
                )
            policy.train()

            # 原语 #6：log_generations 把 8 条样本汇总成详细字典
            # （含 prompt/gen/reward/avg length by correctness）；这里把
            # rewards 和 generations 重新打散喂进去
            samples_for_log = log_generations(
                prompts=[s["prompt"] for s in samples],
                generations=[s["generation"] for s in samples],
                ground_truths=[s["ground_truth"] for s in samples],
                rewards=[s["reward"] for s in samples],
                token_entropies=None,  # eval 不算 entropy（vLLM 不返回）
            )
            print(
                f"  acc={metrics['accuracy']:.3%} format={metrics['format_rate']:.3%} "
                f"avg_len={samples_for_log['avg_response_len']:.1f} "
                f"avg_len_correct={samples_for_log['avg_response_len_correct']:.1f}"
            )
            if wandb:
                wandb.log({
                    "eval/accuracy": metrics["accuracy"],
                    "eval/format_rate": metrics["format_rate"],
                    "eval/avg_response_len": samples_for_log["avg_response_len"],
                    "eval/avg_response_len_correct": samples_for_log["avg_response_len_correct"],
                    "eval/avg_response_len_incorrect": samples_for_log["avg_response_len_incorrect"],
                    "eval_step": step,
                })
                # 上 8 条样本作为 wandb Table，方便人眼审查 in-the-loop generation
                wandb.log({
                    "eval/samples": wandb.Table(
                        columns=["prompt", "generation", "ground_truth", "reward"],
                        data=[
                            [s["prompt"], s["generation"], s["ground_truth"], s["reward"]["reward"]]
                            for s in samples
                        ],
                    ),
                    "eval_step": step,
                })

        # --- (e) ckpt save ---
        if step % args.save_every == 0 or step == total_steps:
            ckpt_dir = out_dir / f"step_{step}"
            ckpt_dir.mkdir(exist_ok=True)
            # save_pretrained 写 model.safetensors + config.json + tokenizer
            # （tokenizer 也存：让 ckpt 自包含，eval 时一个目录就够）
            policy.save_pretrained(ckpt_dir)
            tok.save_pretrained(ckpt_dir)
            print(f"[save] {ckpt_dir}")

    if wandb:
        wandb.finish()


def _infinite(loader: DataLoader) -> Iterable:
    """把 DataLoader 包成无限循环（多 epoch）。

    SFT 默认单 epoch（PDF §4.3 推荐），但当 max_train_examples=128 时一个
    epoch 只有 64 step（grad_accum=2），需要重复多 epoch 才到 max_steps；
    `_infinite` 让训练循环不必关心 epoch 边界。
    """
    while True:
        for batch in loader:
            yield batch


if __name__ == "__main__":
    main()
