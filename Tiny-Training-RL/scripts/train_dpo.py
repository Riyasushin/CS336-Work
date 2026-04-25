"""DPO training loop（supplement PDF §5.4 Problem `dpo_training`）.

PDF §5.4 任务:
    1. 双 LM 设置：policy 一卡 + ref 一卡（都加载 instruction-tuned 同一 ckpt）
    2. 200 条作为 val set（PDF 说"separate out a small number of examples,
       e.g. 200, as a validation set"）
    3. 训练 1 epoch over HH，跟踪每 step loss + val classification accuracy
       （chosen log-prob > rejected log-prob 比例）
    4. 保存最高 val accuracy 的 ckpt
    5. PDF 推荐: batch_size=64 (effective via grad accum), β=0.1, lr=1e-6,
       optimizer = **RMSprop**（不用 AdamW）

Deliverable:
    - 训练脚本 + val accuracy 学习曲线截图
    - DPO ckpt（用于后续 §5.4(2-4) AlpacaEval/SST/MMLU/GSM8K 评测）

CLI:
    uv run python scripts/train_dpo.py \\
        --instruction-tuned-ckpt /data/runs/sft_llama8b/step_6250 \\
        --hh-data /data/runs/hh_combined.jsonl \\
        --output-dir /data/runs/dpo_llama8b \\
        --policy-device cuda:0 --ref-device cuda:1 \\
        --beta 0.1 --learning-rate 1e-6 \\
        --train-batch-size 64 --max-steps 1000 \\
        --wandb-project tiny-training-rl

为什么本机不跑
==============
2 卡 LM（policy + ref）= 2 × 16GB Llama 8B bf16 + grad/optim state；本机
8GB 远不够。脚本只写不跑；上 2×80GB 集群跑。
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tiny_training_rl.dpo import compute_per_instance_dpo_loss, _unconditional_log_prob
from tiny_training_rl.data import ALPACA_TEMPLATE
from tiny_training_rl.sft_train import set_seed


def load_hh_combined(path: Path, val_size: int, seed: int) -> tuple[list[dict], list[dict]]:
    """读 #26 输出的 hh_combined.jsonl，按 seed shuffle 后切 train/val。"""
    rows = []
    with path.open() as f:
        for line in f:
            rows.append(json.loads(line))
    rng = random.Random(seed)
    rng.shuffle(rows)
    val = rows[:val_size]
    train = rows[val_size:]
    return train, val


def evaluate_dpo_classification_acc(
    lm,
    lm_ref,
    tokenizer,
    val_rows: list[dict],
    beta: float,
) -> dict:
    """val implicit reward classification accuracy。

    DPO 的 implicit reward = β·(log π_θ - log π_ref)。判定：
        - 对每条 val 样例，分别算 r_chosen 和 r_rejected
        - 正确分类：r_chosen > r_rejected
    返回 {accuracy, n}
    """
    eos = tokenizer.eos_token or "<|endoftext|>"
    lm_dev = next(lm.parameters()).device
    ref_dev = next(lm_ref.parameters()).device

    n_correct = 0
    lm.eval()
    lm_ref.eval()
    with torch.inference_mode():
        for r in val_rows:
            t_chosen = ALPACA_TEMPLATE.format(prompt=r["instruction"], response=r["chosen"]) + eos
            t_rejected = ALPACA_TEMPLATE.format(prompt=r["instruction"], response=r["rejected"]) + eos
            ids_chosen = tokenizer.encode(t_chosen, return_tensors="pt", add_special_tokens=False)
            ids_rejected = tokenizer.encode(t_rejected, return_tensors="pt", add_special_tokens=False)

            log_p_lm_c = _unconditional_log_prob(lm, ids_chosen.to(lm_dev))
            log_p_lm_r = _unconditional_log_prob(lm, ids_rejected.to(lm_dev))
            log_p_ref_c = _unconditional_log_prob(lm_ref, ids_chosen.to(ref_dev)).to(lm_dev)
            log_p_ref_r = _unconditional_log_prob(lm_ref, ids_rejected.to(ref_dev)).to(lm_dev)

            # implicit reward 差值（β scale 不影响 sign）
            diff = (log_p_lm_c - log_p_ref_c) - (log_p_lm_r - log_p_ref_r)
            if diff.item() > 0:
                n_correct += 1
    lm.train()  # ref 始终 eval
    return {"n": len(val_rows), "accuracy": n_correct / max(1, len(val_rows))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--instruction-tuned-ckpt", required=True,
                   help="SFT 后的 ckpt，policy 和 ref 都加载这份（PDF §5.4 starter）")
    p.add_argument("--hh-data", required=True, help="#26 load_hh.py 输出的 combined jsonl")
    p.add_argument("--output-dir", required=True)

    # PDF §5.4 推荐 hyperparams
    p.add_argument("--beta", type=float, default=0.1, help="PDF §5.4 推荐 β=0.1")
    p.add_argument("--learning-rate", type=float, default=1e-6, help="PDF §5.4 推荐 1e-6")
    p.add_argument("--train-batch-size", type=int, default=64, help="effective via grad accum")
    p.add_argument("--gradient-accumulation-steps", type=int, default=64,
                   help="DPO double-LM 显存大，micro=1 + accum=64 → effective=64")
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--val-size", type=int, default=200, help="PDF §5.4 推荐 200")
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--save-best", action=argparse.BooleanOptionalAction, default=True,
                   help="PDF: 'Save your model with the highest validation accuracy'")
    p.add_argument("--grad-clip", type=float, default=1.0)

    # device
    p.add_argument("--policy-device", default="cuda:0")
    p.add_argument("--ref-device", default="cuda:1")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--wandb-run-name", default=None)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
    set_seed(args.seed)

    # ---- 加载 tokenizer + policy + ref ----
    print(f"[load] policy on {args.policy_device}")
    tok = AutoTokenizer.from_pretrained(args.instruction_tuned_ckpt)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    policy = AutoModelForCausalLM.from_pretrained(
        args.instruction_tuned_ckpt, dtype=torch.bfloat16,
    ).to(args.policy_device)
    policy.gradient_checkpointing_enable()

    print(f"[load] ref on {args.ref_device}")
    ref = AutoModelForCausalLM.from_pretrained(
        args.instruction_tuned_ckpt, dtype=torch.bfloat16,
    ).to(args.ref_device)
    ref.eval()
    for p_ in ref.parameters():
        p_.requires_grad_(False)

    # PDF §5.4 明确："we won't be able to use AdamW unless we use other efficiency
    # tricks (such as quantization), so we will stick to the RMSprop optimizer"
    optimizer = torch.optim.RMSprop(policy.parameters(), lr=args.learning_rate)

    # ---- 数据 ----
    train_rows, val_rows = load_hh_combined(Path(args.hh_data), args.val_size, args.seed)
    print(f"[data] train: {len(train_rows)} | val: {len(val_rows)}")

    # ---- wandb ----
    wandb = None
    if args.wandb_project:
        try:
            import wandb as wb
            wb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
            wb.define_metric("train_step")
            wb.define_metric("eval_step")
            wb.define_metric("train/*", step_metric="train_step")
            wb.define_metric("eval/*", step_metric="eval_step")
            wandb = wb
        except Exception as e:
            print(f"  wandb unavailable: {e}")

    # ---- 主训练循环 ----
    rng = random.Random(args.seed)
    step = 0
    accum_loss = 0.0
    best_val_acc = -1.0
    best_step = 0
    t0 = time.perf_counter()

    while step < args.max_steps:
        # --- microbatch (实际是 single-instance) loop ---
        for _ in range(args.gradient_accumulation_steps):
            # DPO 训练单条单 example：双 LM forward 加 batch 太吃显存。
            # PDF default：micro_batch=1 + accum=64
            r = train_rows[rng.randrange(len(train_rows))]
            loss = compute_per_instance_dpo_loss(
                lm=policy, lm_ref=ref, tokenizer=tok,
                beta=args.beta,
                prompt=r["instruction"],
                response_chosen=r["chosen"],
                response_rejected=r["rejected"],
            )
            loss = loss / args.gradient_accumulation_steps
            loss.backward()
            accum_loss += loss.item()

        # optimizer step
        gnorm = torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1

        if step % 5 == 0 or step == 1:
            elapsed = time.perf_counter() - t0
            print(f"step {step:4d}/{args.max_steps} loss={accum_loss:.4f} "
                  f"gnorm={float(gnorm):.3f} elapsed={elapsed:.0f}s")
            if wandb:
                wandb.log({
                    "train/loss": accum_loss,
                    "train/grad_norm": float(gnorm),
                    "train_step": step,
                })
        accum_loss = 0.0

        # --- eval (classification acc) ---
        if step % args.eval_every == 0:
            print(f"[eval] step {step} ...")
            metrics = evaluate_dpo_classification_acc(policy, ref, tok, val_rows, args.beta)
            val_acc = metrics["accuracy"]
            print(f"  val classification acc = {val_acc:.3%}")
            if wandb:
                wandb.log({"eval/classification_acc": val_acc, "eval_step": step})

            # save best
            if args.save_best and val_acc > best_val_acc:
                best_val_acc = val_acc
                best_step = step
                ckpt_dir = out_dir / "best"
                ckpt_dir.mkdir(exist_ok=True)
                policy.save_pretrained(ckpt_dir)
                tok.save_pretrained(ckpt_dir)
                print(f"  [save best] step {step} acc={val_acc:.3%} → {ckpt_dir}")

    print(f"\n[done] best val acc = {best_val_acc:.3%} at step {best_step}")
    if wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
