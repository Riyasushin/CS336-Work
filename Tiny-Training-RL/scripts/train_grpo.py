"""GRPO training loop（PDF §7.2 Problem `grpo_train_loop`）.

PDF §7.2 任务一句话总结
=======================
组装 §6 的 policy gradient 数学 + §7.2 的 6 个原语，实装 Algorithm 3：

    Algorithm 3 (GRPO):
        π ← π_init
        for grpo_step in 1..n_grpo_steps:
            D_b ← sample n_prompts questions from D
            π_old ← π                                       # 冻结 rollout 生成器
            for q in D_b:
                o^(1..G) ~ π_old(·|q)                       # vLLM n=G
            r^(i) = R(q, o^(i))                             # r1_zero_reward_fn
            A^(i) = (r - mean_g) / (std_g + ε)              # group-normalized
            for inner_step in 1..n_train_steps_per_rollout_batch:
                update π via GRPO objective                 # naive PG / GRPO-Clip
        return π

GRPO vs §5 EI 的关键区别
========================
- EI **筛**正确 rollout 当 SFT 数据；GRPO **保留全部** rollout，用 advantage 加权
- EI 用 `-log p` (NLL) 反传；GRPO 用 `-A · log p` 或 `-min(r·A, clip(r)·A)` 反传
- GRPO 利用了"答错 (A<0) 也提供学习信号"——EI 直接扔掉
- 数学上：EI 是 verified-reward + 阈值筛选的退化 RL；GRPO 是连续 advantage 的 RL

GRPO vs §4.3 SFT
================
- SFT 数据来自 R1 蒸馏；GRPO 数据是 policy 自己跑出来的（rollout）
- SFT 每条样例 reward 隐含为 1（trust GT trace）；GRPO 每条 rollout 各有 advantage
- SFT 一遍数据 = 一个 epoch；GRPO 一批 rollout 可跑 1 (on-policy) 或 K (off-policy) 内层 epoch

PDF §7.2 starter hyperparams（已填进 default）
=================================================
    n_grpo_steps                = 200
    learning_rate               = 1e-5
    advantage_eps               = 1e-6
    rollout_batch_size          = 256          # 一次 rollout 总条数
    group_size                  = 8            # G
    sampling_temperature        = 1.0
    sampling_min_tokens         = 4            # 防 vLLM 空串 NaN
    sampling_max_tokens         = 1024
    epochs_per_rollout_batch    = 1            # on-policy
    train_batch_size            = 256          # = rollout_batch on-policy
    gradient_accumulation_steps = 128          # micro = 256/128 = 2
    loss_type                   = reinforce_with_baseline
    use_std_normalization       = True         # group std normalization
    AdamW(lr, betas=(0.9, 0.95), wd=0.0)

Sanity asserts（PDF starter 给的，照抄）：
    train_batch_size % gradient_accumulation_steps == 0
    micro_train_batch_size = train_batch_size // gradient_accumulation_steps
    rollout_batch_size % group_size == 0
    n_prompts_per_rollout_batch = rollout_batch_size // group_size
    train_batch_size >= group_size

CLI（部署到 2 卡集群）
======================
    uv run python scripts/train_grpo.py \\
        --model /data/a5-alignment/models/Qwen2.5-Math-1.5B \\
        --questions-data /data/a5-alignment/MATH/train.jsonl \\
        --val-data /data/a5-alignment/MATH/validation.jsonl \\
        --output-dir /data/runs/grpo_default \\
        --policy-device cuda:0 --vllm-device cuda:1 \\
        --wandb-project tiny-training-rl

§8 实验：改 --loss-type / --use-std-normalization / --epochs-per-rollout-batch /
        --learning-rate 等开关跑不同 ablation。
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Iterable, Literal

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from tiny_training_rl.sft_train import (
    cosine_lr,
    init_vllm,
    load_policy_into_vllm_instance,
    set_lr,
    set_seed,
)


# =============================================================================
# 1. 数据：rollout → tokenized batch with old_log_probs + advantage
# =============================================================================
#
# 与 §4.3 SFT 数据流的差别：每条样例除了 (input_ids, labels, response_mask)
# 还要带：
#   - advantage: 标量，per-rollout
#   - raw_reward: 标量，per-rollout（监控用）
#   - old_log_probs: (T,) 张量，与 labels 同长度
#
# old_log_probs 是 GRPO-Clip 必需的（per-token IS ratio = exp(policy - old)）。
# PDF §7.2 提示：off-policy 多 epoch 时，old_log_probs **只在 rollout 后算
# 一次**，inner loop 里复用，不能每个 epoch 重算（那样就不是 IS 了）。
# =============================================================================

class GRPORolloutDataset(Dataset):
    """每个 row 是 dict: {input_ids, labels, response_mask, advantage, old_log_probs}。

    构造时已 tokenize + 计算好 old_log_probs（在 prepare_rollout_batch 里）。
    """

    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


def collate_grpo_pad(batch: list[dict], pad_id: int) -> dict[str, torch.Tensor]:
    """batch 内 pad input_ids/labels/response_mask/old_log_probs；advantage 直接 stack。

    pad 选项：
        - input_ids / labels        ← pad_id（pad 位置 mask=0 不参与 loss）
        - response_mask             ← 0
        - old_log_probs             ← 0.0（pad 位置 mask=0，loss 计算被屏蔽）
        - advantage                 ← stack 成 (B, 1)
        - raw_reward                ← stack 成 (B, 1)
    """
    max_len = max(b["input_ids"].shape[0] for b in batch)
    keys_to_pad = ("input_ids", "labels", "response_mask", "old_log_probs")
    out: dict[str, list] = {k: [] for k in keys_to_pad}
    advantages = []
    raw_rewards = []
    for b in batch:
        L = b["input_ids"].shape[0]
        pad = max_len - L
        out["input_ids"].append(F.pad(b["input_ids"], (0, pad), value=pad_id))
        out["labels"].append(F.pad(b["labels"], (0, pad), value=pad_id))
        out["response_mask"].append(F.pad(b["response_mask"], (0, pad), value=0))
        out["old_log_probs"].append(F.pad(b["old_log_probs"], (0, pad), value=0.0))
        advantages.append(b["advantage"])
        raw_rewards.append(b["raw_reward"])
    stacked = {k: torch.stack(v, dim=0) for k, v in out.items()}
    stacked["advantages"] = torch.tensor(advantages, dtype=torch.float32).unsqueeze(-1)  # (B, 1)
    stacked["raw_rewards"] = torch.tensor(raw_rewards, dtype=torch.float32).unsqueeze(-1)
    return stacked


def prepare_rollout_batch(
    sft_rows: list[dict],
    advantages: torch.Tensor,
    raw_rewards: torch.Tensor,
    tokenizer,
    policy,
    device: str,
    max_seq_len: int,
    micro_batch_size: int,
) -> list[dict]:
    """对每条 rollout 做 tokenize + 算 old_log_probs，返回 GRPORolloutDataset 用的 row list。

    步骤：
        1) 单条 tokenize（用 §4.2 原语 #1），裁尾到 max_seq_len
        2) 临时 batch + pad，把 policy forward 跑一遍**inference_mode**得 old_log_probs
        3) 把 old_log_probs（去除 pad 部分）填回每条 row

    关键：步骤 2 的 forward **不**带 grad（torch.inference_mode）—— 这是 PDF
    §7.2 off-policy 实装的核心，否则计算图会一直驻留。
    """
    from tiny_training_rl.sft import tokenize_prompt_and_output, get_response_log_probs

    # ---- step 1: tokenize 每条 ----
    raw_rows = []
    for i, row in enumerate(sft_rows):
        out = tokenize_prompt_and_output([row["prompt"]], [row["response"]], tokenizer)
        d = {k: v.squeeze(0) for k, v in out.items()}
        if d["input_ids"].shape[0] > max_seq_len:
            d = {k: v[:max_seq_len] for k, v in d.items()}
        d["advantage"] = advantages[i].item()
        d["raw_reward"] = raw_rewards[i].item()
        d["len"] = d["input_ids"].shape[0]   # 记录真实长度，pad 后用来切回
        raw_rows.append(d)

    # ---- step 2: 跑 forward 得 old_log_probs ----
    # 为了显存安全分 micro 跑；每个 micro 内部 pad，得到 (B, T_max_in_micro) 的
    # log_probs，然后切回每条的真实长度填回 row
    pad_id = tokenizer.pad_token_id
    policy.eval()
    with torch.inference_mode():
        for start in range(0, len(raw_rows), micro_batch_size):
            chunk = raw_rows[start : start + micro_batch_size]
            max_len = max(r["len"] for r in chunk)
            input_ids = torch.stack([
                F.pad(r["input_ids"], (0, max_len - r["len"]), value=pad_id) for r in chunk
            ]).to(device)
            labels = torch.stack([
                F.pad(r["labels"], (0, max_len - r["len"]), value=pad_id) for r in chunk
            ]).to(device)
            out = get_response_log_probs(policy, input_ids, labels, return_token_entropy=False)
            log_probs = out["log_probs"].cpu()         # (B, T_max_in_micro), fp32

            for j, r in enumerate(chunk):
                # 切回真实长度（去掉 pad）
                r["old_log_probs"] = log_probs[j, : r["len"]].contiguous()
                del r["len"]                            # 不再需要
    policy.train()
    return raw_rows


# =============================================================================
# 2. Rollout 阶段（与 §5 EI 类似但不筛掉错误 rollout）
# =============================================================================

def rollout_all(
    llm,
    questions: list[dict],
    G: int,
    prompt_template: str,
    sampling_params,
) -> list[dict]:
    """每题跑 G 个 rollout，**全部保留**（不像 EI 筛 reward=1）。

    返回扁平 list，长度 = n_prompts × G，顺序：q0_rollout0..q0_rolloutG-1, q1_..., ...
    重要：顺序必须与 group_size 切片对齐，让 compute_group_normalized_rewards 能
    按 reshape((n_prompts, G)) 找到正确的同 group。
    """
    prompts = [prompt_template.format(question=q["question"]) for q in questions]
    outs = llm.generate(prompts, sampling_params)

    rollouts = []
    for q, out in zip(questions, outs):
        prompt_full = prompt_template.format(question=q["question"])
        # vLLM n=G 模式下，out.outputs 是长 G 的 list[CompletionOutput]
        for sample in out.outputs:
            rollouts.append({
                "prompt": prompt_full,
                "response": sample.text,
                "ground_truth": q["ground_truth"],
            })
    return rollouts


def load_questions_jsonl(path: Path) -> list[dict]:
    """读问题集，与 train_ei.py 同款 schema 检测。"""
    rows = []
    with path.open() as f:
        for line in f:
            obj = json.loads(line)
            q = obj.get("question") or obj.get("problem") or obj.get("prompt")
            a = obj.get("answer", obj.get("response", ""))
            gt = a.rsplit("####", 1)[-1].strip() if "####" in a else a.strip()
            rows.append({"question": q, "ground_truth": gt})
    return rows


# =============================================================================
# 3. Inner training loop —— 一个 rollout batch 内 K 个 epoch × n_microbatch
# =============================================================================

def run_grpo_inner_loop(
    policy,
    optimizer,
    dataset: GRPORolloutDataset,
    pad_id: int,
    args,
    global_step: int,
    total_steps: int,
    wandb=None,
) -> tuple[int, dict]:
    """在 rollout_batch 上跑 epochs_per_rollout × n_microbatches_per_rollout 步训练。

    on-policy 默认（epochs=1, train_batch=rollout_batch）：内层只跑一遍数据。
    off-policy（epochs=K, train_batch<rollout_batch）：内层跑 K 遍，每遍 shuffle。

    重要：old_log_probs 在 prepare_rollout_batch 里已经计算并存进 dataset row，
    inner loop 不再重算 —— 这就是 §7.2 off-policy 的 IS 修正生效条件。
    """
    from tiny_training_rl.sft import get_response_log_probs
    from tiny_training_rl.grpo import grpo_microbatch_train_step

    grad_accum = max(1, args.train_batch_size // args.micro_train_batch_size)
    n_inner_optimizer_steps = (
        args.epochs_per_rollout_batch
        * (len(dataset) // args.train_batch_size)
    )
    if n_inner_optimizer_steps == 0:
        # rollout_batch 比 train_batch 还小 —— 默认 hyperparam 下不会触发，但兜底
        return global_step, {"skipped": True}

    sum_loss = 0.0
    sum_clip_frac = 0.0
    sum_entropy = 0.0
    n_micro = 0

    train_loader = DataLoader(
        dataset,
        batch_size=args.micro_train_batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_grpo_pad(b, pad_id),
        num_workers=0,           # rollout dataset 已经在内存里，多 worker 反而开销
        drop_last=True,
        pin_memory=True,
    )

    for inner_step in range(n_inner_optimizer_steps):
        # 一个 optimizer step = grad_accum 个 microbatch
        for _ in range(grad_accum):
            try:
                batch = next(train_iter)
            except (NameError, StopIteration):
                train_iter = iter(train_loader)
                batch = next(train_iter)

            input_ids = batch["input_ids"].to(args.policy_device, non_blocking=True)
            labels = batch["labels"].to(args.policy_device, non_blocking=True)
            mask = batch["response_mask"].to(args.policy_device, non_blocking=True).float()
            old_log_probs = batch["old_log_probs"].to(args.policy_device, non_blocking=True)
            advantages = batch["advantages"].to(args.policy_device, non_blocking=True)
            raw_rewards = batch["raw_rewards"].to(args.policy_device, non_blocking=True)

            # 当前 policy 的 log_probs（带 grad）
            out = get_response_log_probs(policy, input_ids, labels, return_token_entropy=True)
            policy_log_probs = out["log_probs"]
            tok_entropy = out["token_entropy"]

            # GRPO microbatch fwd+bwd（原语 #6）—— 内部按 loss_type 分发
            # 注意：原语会自己 backward()，并按 grad_accum 缩放
            # length_norm_method 决定沿 seq 的聚合：
            #   "mean" → masked_mean (默认 None)；"fixed" → masked_normalize(K=max_seq_len)
            length_const = args.max_seq_len if args.length_norm_method == "fixed" else None
            loss, mb_metadata = grpo_microbatch_train_step(
                policy_log_probs=policy_log_probs,
                response_mask=mask,
                gradient_accumulation_steps=grad_accum,
                loss_type=args.loss_type,
                raw_rewards=raw_rewards,
                advantages=advantages,
                old_log_probs=old_log_probs,
                cliprange=args.cliprange,
                length_normalize_constant=length_const,
            )
            sum_loss += loss.item()
            n_resp = mask.sum().clamp(min=1)
            sum_entropy += ((tok_entropy * mask).sum() / n_resp).item()
            if "clip_fraction" in mb_metadata:
                sum_clip_frac += mb_metadata["clip_fraction"].item()
            n_micro += 1

        # optimizer step + grad clip + lr schedule
        gnorm = torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
        warmup = int(total_steps * args.warmup_ratio)
        lr = cosine_lr(global_step, warmup, total_steps, args.learning_rate)
        set_lr(optimizer, lr)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1

        if wandb and (global_step % 5 == 0):
            wandb.log({
                "train/loss": loss.item() * grad_accum,
                "train/lr": lr,
                "train/grad_norm": float(gnorm),
                "train_step": global_step,
            })

    metrics = {
        "n_micro": n_micro,
        "avg_loss": sum_loss / max(1, n_micro),
        "avg_token_entropy": sum_entropy / max(1, n_micro),
        "avg_clip_fraction": sum_clip_frac / max(1, n_micro) if args.loss_type == "grpo_clip" else 0.0,
    }
    return global_step, metrics


# =============================================================================
# 4. 主循环
# =============================================================================

def main():
    p = argparse.ArgumentParser()

    # 模型 / 数据 / IO
    p.add_argument("--model", required=True)
    p.add_argument("--questions-data", required=True)
    p.add_argument("--val-data", default=None)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--prompt-name", default="r1_zero")

    # PDF starter hyperparams
    p.add_argument("--n-grpo-steps", type=int, default=200)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--warmup-ratio", type=float, default=0.0,
                   help="GRPO 一般不 warmup（policy 已经被 SFT 过），保留接口")
    p.add_argument("--advantage-eps", type=float, default=1e-6)
    p.add_argument("--rollout-batch-size", type=int, default=256, help="单次 rollout 总条数 = n_prompts × G")
    p.add_argument("--group-size", type=int, default=8, help="G")
    p.add_argument("--epochs-per-rollout-batch", type=int, default=1, help="K = on-policy 设 1，off-policy 设 >1")
    p.add_argument("--train-batch-size", type=int, default=256, help="effective batch；on-policy = rollout_batch_size")
    p.add_argument("--gradient-accumulation-steps", type=int, default=128)
    p.add_argument("--micro-train-batch-size", type=int, default=None,
                   help="不传则按 train_batch_size / grad_accum 自动算")
    p.add_argument("--loss-type",
                   choices=["no_baseline", "reinforce_with_baseline", "grpo_clip", "grpo_no_clip"],
                   default="reinforce_with_baseline")
    # BooleanOptionalAction：让 --use-std-normalization / --no-use-std-normalization 都可用
    # （§8 grpo_group_standard_deviation 需要从 CLI 关掉 std normalization）
    p.add_argument("--use-std-normalization", action=argparse.BooleanOptionalAction, default=True,
                   help="GRPO 默认 True；§8 grpo_group_standard_deviation 用 --no-use-std-normalization 做 ablation")
    p.add_argument("--cliprange", type=float, default=0.2, help="GRPO-Clip ε")
    # §8 grpo_length_normalization：在 masked_mean (dim=-1) 与 masked_normalize (常数 K) 间切换
    p.add_argument("--length-norm-method", choices=["mean", "fixed"], default="mean",
                   help="mean = masked_mean (GRPO default)；fixed = masked_normalize(K=max-seq-len) Dr.GRPO 长度公平")
    # §8 grpo_prompt_ablation：换 reward 函数（r1_zero 严格匹配 vs question_only 宽松）
    p.add_argument("--reward-fn-name", choices=["r1_zero", "question_only"], default="r1_zero",
                   help="r1_zero 要求 </think> <answer> 模板；question_only 只要能 parse 出答案")

    # 采样
    p.add_argument("--sampling-temperature", type=float, default=1.0)
    p.add_argument("--sampling-top-p", type=float, default=1.0)
    p.add_argument("--sampling-min-tokens", type=int, default=4, help="防 vLLM 空串 NaN")
    p.add_argument("--sampling-max-tokens", type=int, default=1024)

    # 训练通用
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--eval-num", type=int, default=1024,
                   help="PDF §7.2 推荐 ≥1024 才稳；本地 demo 可降到 128")
    p.add_argument("--save-every", type=int, default=50)

    # device / wandb / etc
    p.add_argument("--policy-device", default="cuda:0")
    p.add_argument("--vllm-device", default="cuda:1")
    p.add_argument("--vllm-gpu-mem", type=float, default=0.85)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--no-grad-checkpoint", action="store_true")
    args = p.parse_args()

    # ---- 4.0 sanity asserts (PDF starter) ----
    assert args.train_batch_size % args.gradient_accumulation_steps == 0, \
        "train_batch_size must be divisible by gradient_accumulation_steps"
    if args.micro_train_batch_size is None:
        args.micro_train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    assert args.rollout_batch_size % args.group_size == 0, \
        "rollout_batch_size must be divisible by group_size"
    n_prompts_per_rollout_batch = args.rollout_batch_size // args.group_size
    assert args.train_batch_size >= args.group_size, \
        "train_batch_size must be >= group_size"
    n_microbatches_per_rollout_batch = args.rollout_batch_size // args.micro_train_batch_size
    print(f"[config] n_prompts/rollout={n_prompts_per_rollout_batch}, "
          f"micro_bs={args.micro_train_batch_size}, "
          f"n_microbatches/rollout={n_microbatches_per_rollout_batch}, "
          f"epochs/rollout={args.epochs_per_rollout_batch}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
    set_seed(args.seed)

    # ---- 4.1 加载 policy + tokenizer + vllm ----
    print(f"[load] policy on {args.policy_device}")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
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
        policy.gradient_checkpointing_enable()

    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.0
    )

    print(f"[load] vllm on {args.vllm_device}")
    llm = init_vllm(args.model, args.vllm_device, args.seed, args.vllm_gpu_mem)

    # ---- 4.2 数据 ----
    from tiny_training_rl.prompts import load as load_prompt
    from tiny_training_rl import grader
    prompt_template = load_prompt(args.prompt_name)
    # 选 reward 函数（§8 grpo_prompt_ablation 用 question_only 宽松判定）
    reward_fn = {
        "r1_zero": grader.r1_zero_reward_fn,
        "question_only": grader.question_only_reward_fn,
    }[args.reward_fn_name]

    questions = load_questions_jsonl(Path(args.questions_data))
    print(f"[data] question pool: {len(questions)} rows")

    val_rows = []
    if args.val_data and Path(args.val_data).exists():
        with Path(args.val_data).open() as f:
            for i, line in enumerate(f):
                if i >= args.eval_num:
                    break
                val_rows.append(json.loads(line))
        print(f"[data] val rows: {len(val_rows)}")

    from vllm import SamplingParams
    rollout_params = SamplingParams(
        temperature=args.sampling_temperature,
        top_p=args.sampling_top_p,
        max_tokens=args.sampling_max_tokens,
        min_tokens=args.sampling_min_tokens,
        n=args.group_size,
        stop=["</answer>"],
        include_stop_str_in_output=True,
        seed=args.seed,
    )
    val_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=1024,
        stop=["</answer>"], include_stop_str_in_output=True,
    )

    # ---- 4.3 wandb ----
    wandb = None
    if args.wandb_project:
        try:
            import wandb as wb
            wb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
            wb.define_metric("train_step")
            wb.define_metric("eval_step")
            wb.define_metric("grpo_step")
            wb.define_metric("train/*", step_metric="train_step")
            wb.define_metric("eval/*", step_metric="eval_step")
            wb.define_metric("rollout/*", step_metric="grpo_step")
            wandb = wb
        except Exception as e:
            print(f"  wandb unavailable: {e}")

    # ---- 4.4 GRPO 主循环 ----
    from tiny_training_rl.grpo import compute_group_normalized_rewards
    from tiny_training_rl.sft_train import evaluate_policy_vllm

    # cosine LR 全程 schedule（n_grpo_steps × inner_steps_per_grpo）
    inner_per_grpo = max(
        1,
        args.epochs_per_rollout_batch * (args.rollout_batch_size // args.train_batch_size),
    )
    rough_total_steps = args.n_grpo_steps * inner_per_grpo

    rng = torch.Generator().manual_seed(args.seed)
    global_step = 0
    t_start = time.perf_counter()

    for grpo_step in range(1, args.n_grpo_steps + 1):
        step_t0 = time.perf_counter()
        print(f"\n=== GRPO step {grpo_step}/{args.n_grpo_steps} ===")

        # ---- (a) sample n_prompts questions ----
        idx = torch.randint(0, len(questions), (n_prompts_per_rollout_batch,), generator=rng).tolist()
        D_b = [questions[i] for i in idx]

        # ---- (b) rollout：n=G，π_old ← π，把 policy 灌进 vllm ----
        policy.eval()
        with torch.inference_mode():
            load_policy_into_vllm_instance(policy, llm)
            rollouts = rollout_all(llm, D_b, args.group_size, prompt_template, rollout_params)
        policy.train()
        rollout_t = time.perf_counter() - step_t0
        # rollouts 长度 = n_prompts × G，顺序：q0_r0..q0_rG-1, q1_..., ...

        # ---- (c) 计算 reward + group-normalized advantage（原语 #1）----
        repeated_gts = [r["ground_truth"] for r in rollouts]
        rollout_responses = [r["response"] for r in rollouts]
        advantages, raw_rewards, reward_meta = compute_group_normalized_rewards(
            reward_fn=reward_fn,
            rollout_responses=rollout_responses,
            repeated_ground_truths=repeated_gts,
            group_size=args.group_size,
            advantage_eps=args.advantage_eps,
            normalize_by_std=args.use_std_normalization,
        )
        format_rate = sum(reward_fn(r, gt)["format_reward"] for r, gt in zip(rollout_responses, repeated_gts)) / len(rollout_responses)
        print(f"  [rollout] mean_r={reward_meta['reward_mean']:.3f} format={format_rate:.3%} ({rollout_t:.1f}s)")
        if wandb:
            wandb.log({
                "rollout/reward_mean": reward_meta["reward_mean"],
                "rollout/reward_std": reward_meta["reward_std"],
                "rollout/format_rate": format_rate,
                "rollout/wall_time_sec": rollout_t,
                "grpo_step": grpo_step,
            })

        # ---- (d) 计算 old_log_probs（一次！inner loop 复用）----
        sft_rows = [{"prompt": r["prompt"], "response": r["response"]} for r in rollouts]
        rollout_rows = prepare_rollout_batch(
            sft_rows=sft_rows,
            advantages=advantages,
            raw_rewards=raw_rewards,
            tokenizer=tok,
            policy=policy,
            device=args.policy_device,
            max_seq_len=args.max_seq_len,
            micro_batch_size=args.micro_train_batch_size,
        )

        # ---- (e) inner training loop ----
        sft_t0 = time.perf_counter()
        rollout_ds = GRPORolloutDataset(rollout_rows)
        global_step, inner_metrics = run_grpo_inner_loop(
            policy=policy,
            optimizer=optimizer,
            dataset=rollout_ds,
            pad_id=tok.pad_token_id,
            args=args,
            global_step=global_step,
            total_steps=rough_total_steps,
            wandb=wandb,
        )
        inner_t = time.perf_counter() - sft_t0
        print(f"  [inner] global_step={global_step} avg_loss={inner_metrics['avg_loss']:.4f} "
              f"avg_entropy={inner_metrics['avg_token_entropy']:.3f} "
              f"clip_frac={inner_metrics['avg_clip_fraction']:.3%} ({inner_t:.1f}s)")

        # ---- (f) eval ----
        if val_rows and grpo_step % args.eval_every == 0:
            policy.eval()
            with torch.inference_mode():
                load_policy_into_vllm_instance(policy, llm)
                metrics, _ = evaluate_policy_vllm(
                    llm, val_rows, prompt_template, reward_fn, val_params,
                )
            policy.train()
            print(f"  [eval] acc={metrics['accuracy']:.3%} format={metrics['format_rate']:.3%}")
            if wandb:
                wandb.log({
                    "eval/accuracy": metrics["accuracy"],
                    "eval/format_rate": metrics["format_rate"],
                    "eval_step": grpo_step,
                })

        # ---- (g) ckpt ----
        if grpo_step % args.save_every == 0 or grpo_step == args.n_grpo_steps:
            ckpt_dir = out_dir / f"step_{grpo_step}"
            ckpt_dir.mkdir(exist_ok=True)
            policy.save_pretrained(ckpt_dir)
            tok.save_pretrained(ckpt_dir)
            print(f"  [save] {ckpt_dir}")

    total_t = time.perf_counter() - t_start
    print(f"\n[done] total time: {total_t:.1f}s, global_steps: {global_step}")
    if wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
