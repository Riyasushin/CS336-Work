"""SFT training with PackedSFTDataset（supplement PDF §3.2.2 Problem `sft_script`）.

与主 PDF §4.3 train_sft.py 的差异（关键改动）
=============================================
1. **数据**: 用 PackedSFTDataset（packed token 流，固定长度 chunk）替代 SFTDataset
   （单 doc + padding）
2. **Loss**: 全 token 参与 cross-entropy；不需要 response_mask
3. **eval**: in-loop validation 切到 supplement §2 的 4 个评测脚本（MMLU /
   GSM8K / AlpacaEval / SST），由用户在外部跑（脚本只写 ckpt）

PDF §3.2.2 任务（Deliverable）:
    A complete training script that supports:
    - Configurable hyperparams
    - Gradient accumulation for larger effective batch
    - Periodic train/val logging（wandb）

PDF §3.3 推荐 hyperparams（已填默认）:
    n_train_examples = 1 epoch（PDF: "single epoch using a context length of 512"）
    batch_size = 32  (effective; via grad accum)
    micro_batch_size = 2 (per-GPU squeeze)
    learning_rate = 2e-5
    cosine LR + 3% warmup
    grad_clip = 1.0

CLI:
    uv run python scripts/train_sft_packed.py \\
        --model /data/a5-alignment/models/Llama-3.1-8B \\
        --train-data /data/a5-alignment/safety_augmented_ultrachat_200k_single_turn/train.jsonl.gz \\
        --output-dir /data/runs/sft_packed_8b \\
        --seq-length 512 \\
        --train-batch-size 32 --micro-batch-size 2 \\
        --learning-rate 2e-5 \\
        --max-steps 5000

注意 supplement PDF 的 sft.jsonl 是 .gz 压缩的；本脚本通过 `gzip` 读
（也支持非压缩 .jsonl）。本机不可得 200K 数据集，使用 `Tiny-Training-RL/
tests/fixtures/sft_sample.jsonl`（5 doc）做 demo 烟测。

为什么本机不跑
==============
PDF 用 Llama 3.1 8B Base（4-5GB bf16），加 grad/activation/AdamW state ~30GB+，
本机 8GB 远不够。脚本只写不跑；上 2×80GB H100 跑。
"""

from __future__ import annotations

import argparse
import gzip
import json
import time
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from tiny_training_rl.data import PackedSFTDataset, iterate_batches
from tiny_training_rl.sft_train import (
    cosine_lr,
    infinite_iter,
    set_lr,
    set_seed,
)


def _maybe_gunzip(path: Path) -> Path:
    """如果输入是 .gz，临时解压到内存 / 临时文件。

    supplement PDF sft.jsonl.gz 在 Together cluster 上是压缩的；
    PackedSFTDataset 的 __init__ 直接 open(path) 读，所以 .gz 要先解压。
    实务上数据集大（200K, 几百 MB），不要解压到内存；解压到 /tmp 文件。
    """
    if path.suffix == ".gz":
        out = Path("/tmp") / path.with_suffix("").name
        if not out.exists():
            print(f"[gunzip] {path} → {out}")
            with gzip.open(path, "rb") as fin, out.open("wb") as fout:
                fout.write(fin.read())
        return out
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--train-data", required=True, help="sft.jsonl 或 .jsonl.gz")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seq-length", type=int, default=512,
                   help="PDF §3.2.2 推荐 512；MATH 长题需 1024+")
    p.add_argument("--max-steps", type=int, default=5000)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--warmup-ratio", type=float, default=0.03,
                   help="PDF §3.2.2 推荐 3%% (linear warmup)")
    p.add_argument("--train-batch-size", type=int, default=32)
    p.add_argument("--micro-batch-size", type=int, default=2)
    p.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True,
                   help="doc 级 shuffle（packing 前）")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--policy-device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--no-grad-checkpoint", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
    set_seed(args.seed)

    # ----- 加载 tokenizer + policy -----
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

    # ----- 数据 -----
    train_path = _maybe_gunzip(Path(args.train_data))
    print(f"[data] loading {train_path}")
    train_ds = PackedSFTDataset(tok, train_path, args.seq_length, shuffle=args.shuffle)
    print(f"[data] packed: {len(train_ds)} chunks of length {args.seq_length}")

    grad_accum = max(1, args.train_batch_size // args.micro_batch_size)
    print(f"[config] grad_accum={grad_accum}, eff_bs={args.train_batch_size}, micro_bs={args.micro_batch_size}")

    # PackedSFTDataset 已经返回等长 chunk，DataLoader 默认 collate 直接 stack
    train_loader = iterate_batches(train_ds, args.micro_batch_size, shuffle=True)

    # ----- wandb -----
    wandb = None
    if args.wandb_project:
        try:
            import wandb as wb
            wb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
            wb.define_metric("train_step")
            wb.define_metric("train/*", step_metric="train_step")
            wandb = wb
        except Exception as e:
            print(f"  wandb unavailable: {e}")

    # ----- 主循环 -----
    total_steps = args.max_steps
    warmup_steps = int(total_steps * args.warmup_ratio)
    step = 0
    accum_loss = 0.0
    t0 = time.perf_counter()

    train_iter = infinite_iter(train_loader)
    while step < total_steps:
        # microbatch loop
        for _ in range(grad_accum):
            batch = next(train_iter)
            input_ids = batch["input_ids"].to(args.policy_device, non_blocking=True)
            labels = batch["labels"].to(args.policy_device, non_blocking=True)

            # Packed dataset 全 token 都参与 loss（无 response_mask）；用
            # cross_entropy 直接算 next-token NLL。
            # logits forward；HF model 输出 (B, T, V)
            out = policy(input_ids)
            logits = out.logits
            # logits 转 fp32 让 cross_entropy 在 fp32 下算（HF 标准做法）
            if logits.dtype in (torch.bfloat16, torch.float16):
                logits = logits.float()
            # cross_entropy 期望 (N, V) + (N,)；reshape
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=tok.pad_token_id,
            )
            loss = loss / grad_accum
            loss.backward()
            accum_loss += loss.item()

        # optimizer step
        gnorm = torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
        lr = cosine_lr(step, warmup_steps, total_steps, args.learning_rate)
        set_lr(optimizer, lr)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1

        if step % 10 == 0 or step == 1:
            elapsed = time.perf_counter() - t0
            print(f"step {step:5d}/{total_steps} loss={accum_loss:.4f} lr={lr:.2e} "
                  f"gnorm={float(gnorm):.3f} elapsed={elapsed:.0f}s")
            if wandb:
                wandb.log({
                    "train/loss": accum_loss,
                    "train/lr": lr,
                    "train/grad_norm": float(gnorm),
                    "train_step": step,
                })
        accum_loss = 0.0

        # save
        if step % args.save_every == 0 or step == total_steps:
            ckpt_dir = out_dir / f"step_{step}"
            ckpt_dir.mkdir(exist_ok=True)
            policy.save_pretrained(ckpt_dir)
            tok.save_pretrained(ckpt_dir)
            print(f"[save] {ckpt_dir}")

    if wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
