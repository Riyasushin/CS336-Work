"""Expert Iteration（Algorithm 2，PDF §5）training loop.

PDF §5 任务一句话总结
====================
EI 用模型自己生成的 rollout 当 SFT 数据，每轮迭代:

    Algorithm 2 (Expert Iteration):
        policy ← base ckpt
        for ei_step in 1..n_ei_steps:
            sample question batch D_b ⊂ D
            π_old ← π                                       # 冻结 rollout 生成器
            for q in D_b:
                sample G outputs o^(1..G) ~ π_old(·|q)      # n=G via vLLM
                compute rewards r^(i) = R(q, o^(i))
            D_sft = filter (q, o) where r=1                 # 只保留答对的
            π ← SFT(π, D_sft)                               # 调 Algorithm 1
        return π

直觉：base 模型本身就有概率产生少数对的 rollout（虽然 r1_zero 模板下
零样本只有 1.56%）；filter 出对的，把模板 + 推理过程一起塞进 SFT，模型
逐步学到"自己能稳定走对的解题轨迹"。比纯 SFT 强在不需要外部 R1 模型生
成的高质量推理轨迹 —— 数据是模型自我引导出来的。

PDF §5 Deliverables（Problem `expert_iteration_experiment`，6 H100 hrs）：
    1. 至少 2 套 (rollout_count G, n_epochs_per_ei) config 的 val acc 曲线
    2. 模型达 ≥15% val acc on MATH
    3. 与 §4.3 SFT 的 2 句对比
    4. response 平均 entropy over training 的图

PDF 推荐 hyperparam（脚本 default 已填）：
    n_ei_steps = 5
    batch_size_questions ∈ {512, 1024, 2048}    （D_b 大小）
    G ∈ {2, 4, 8, ...}                          （每题 rollout 数）
    n_epochs_per_ei ∈ {1, 2, 4}                 （filtered SFT 跑几遍）
    sampling_min_tokens = 4                     （vLLM 防空串 NaN）
    grad_clip = 1.0

CLI（部署到 2 卡集群）：
    uv run python scripts/train_ei.py \\
        --model /data/a5-alignment/models/Qwen2.5-Math-1.5B \\
        --questions-data /data/a5-alignment/MATH/train.jsonl \\
        --val-data /data/a5-alignment/MATH/validation.jsonl \\
        --output-dir /data/runs/ei_n1024_G4_E2 \\
        --n-ei-steps 5 --batch-size-questions 1024 \\
        --rollouts-per-question 4 --n-sft-epochs-per-ei 2 \\
        --policy-device cuda:0 --vllm-device cuda:1

复用了什么
---------
- src/sft_train.py: SFTDataset / collate_pad / init_vllm /
  load_policy_into_vllm_instance / evaluate_policy_vllm / cosine_lr / set_lr
- src/sft.py:       6 个 §4.2 原语（SFT 训练逻辑、log_generations）
- src/grader.py:    r1_zero_reward_fn（rollout 评分）
- src/prompts:      r1_zero 模板

EI 与 SFT 的关键差异（实现层）
-----------------------------
1. 训练循环外多套 rollout 阶段：每个 EI step 调 vLLM.generate(n=G)
2. 每个 EI step 完了要 load_policy_into_vllm_instance(policy, llm) 同步权重
3. SFT 阶段的数据集是 filtered rollouts（每轮变化），不是固定的 sft.jsonl
4. SFT 阶段要跑 n_epochs_per_ei 遍 D_sft，因为 D_sft 通常比正经 sft.jsonl 小
5. PDF 明确要 log "entropy over training" —— 训练循环里把 token_entropy 累加
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from tiny_training_rl.sft_train import (
    SFTDataset,
    collate_pad,
    cosine_lr,
    evaluate_policy_vllm,
    infinite_iter,
    init_vllm,
    load_policy_into_vllm_instance,
    set_lr,
    set_seed,
)


# =============================================================================
# 1. 数据：questions.jsonl → list of {"question", "ground_truth"}
# =============================================================================
#
# 与 SFT 的数据格式区别：EI 的输入是**未配对**的问题，模型自己生成 rationale。
# val_rows 仍走 sft_train.evaluate_policy_vllm 的格式（自动识别 schema）。
# =============================================================================

def load_questions_jsonl(path: Path) -> list[dict]:
    """读训练用问题集；返回 [{"question", "ground_truth"}, ...]。

    schema 兼容 GSM8K / MATH-style，与 SFT 一致用 rsplit("####", 1) 取 final。
    """
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
# 2. EI 的核心：rollout 生成 + reward 过滤
# =============================================================================

def rollout_and_filter(
    llm,
    questions: list[dict],
    G: int,
    prompt_template: str,
    reward_fn,
    sampling_params,
) -> tuple[list[dict], dict]:
    """每个 question 跑 G 个 rollout，按 reward=1 过滤为 SFT 训练样例。

    返回：
        - sft_rows: list of {"prompt", "response"}（response = 模型自己生成的、
                    答对的那个 rollout，包含 think 段）
        - stats: 该 EI step 的 rollout 统计 {n_total, n_correct, n_format_only,
                 correct_rate, mean_response_len, ...}

    设计要点：
    - vLLM SamplingParams 的 n=G：单次调用一题生成 G 条；比 G 次单调用快很多
      （prefix cache 命中）
    - min_tokens=4：PDF 明确要求 —— vLLM 在 stop=["</answer>"] 下偶尔会生成
      空字符串（一开门就立刻 hit stop），下游 SFT 的 NLL 在空 response 上可能
      产 NaN，min_tokens 兜底
    - "正确"的判定：reward["answer_reward"] == 1.0；format_reward 单独统计但
      不当 SFT 数据 —— PDF 要求只筛 r=1 的（即 reward=1）
    - 每条 rollout 的 prompt 仍是 r1_zero 模板下的拼接，与 §4.3 一致；这让
      EI 的 D_sft 看起来就像 §4.3 的 sft.jsonl
    """
    # 拼接所有 prompt（每题一份；vLLM n=G 自己处理多 rollout）
    prompts = [prompt_template.format(question=q["question"]) for q in questions]

    outs = llm.generate(prompts, sampling_params)
    sft_rows: list[dict] = []
    n_total = n_correct = n_format = 0
    response_lens: list[int] = []

    for q, out in zip(questions, outs):
        # vLLM 在 n=G 模式下，out.outputs 是长度 G 的 list[CompletionOutput]
        for sample in out.outputs:
            n_total += 1
            text = sample.text
            r = reward_fn(text, q["ground_truth"])
            if r["format_reward"] == 1.0:
                n_format += 1
            if r["answer_reward"] == 1.0:
                n_correct += 1
                response_lens.append(len(text.split()))
                # 把 (prompt, correct_rollout) 当 SFT 训练样例
                # 注意：prompt 是已经拼好 r1_zero 模板的（含 "Assistant: <think>"），
                # response 是模型生成的（带 "...</think> <answer>...</answer>"），
                # 直接给 SFTDataset 用
                sft_rows.append({
                    "prompt": prompt_template.format(question=q["question"]),
                    "response": text,
                })

    stats = {
        "n_total": n_total,
        "n_correct": n_correct,
        "n_format_only": n_format - n_correct,
        "correct_rate": n_correct / max(1, n_total),
        "format_rate": n_format / max(1, n_total),
        "mean_response_len_correct": sum(response_lens) / max(1, len(response_lens)),
    }
    return sft_rows, stats


# =============================================================================
# 3. SFT 内层循环（在 D_sft 上跑 n_epochs_per_ei 遍）
# =============================================================================

def run_sft_inner_loop(
    policy,
    optimizer,
    sft_rows: list[dict],
    tokenizer,
    args,
    global_step_counter: int,
    total_optimizer_steps: int,
    wandb=None,
) -> tuple[int, float]:
    """在 D_sft 上跑 n_epochs_per_ei 遍，返回 (新 global step, 平均 entropy)。

    复用 §4.3 train_sft 的"microbatch + grad_accum + clip + cosine LR"循环；
    主要差别：
    - 输入是 in-memory list[dict]（rollout 结果），不是 jsonl 文件路径
    - 没有 in-the-loop eval —— eval 由外层 EI 循环负责
    - global_step_counter 跨 EI step 累加（让 wandb x 轴连续）

    返回 entropy 给外层 EI 画"entropy over training"图（PDF deliverable 4）。
    """
    from tiny_training_rl.sft import (
        get_response_log_probs,
        sft_microbatch_train_step,
    )

    if not sft_rows:
        # 这一轮 rollout 一个对的都没有 —— 跳过 SFT，但仍计 step
        # （实践上：base 模型 r1_zero 命中率 1-5%，G=4 + 1024 questions 至少能拿
        #  ~50-200 条；若真为 0，多半是 r1_zero 模板与 base 模型严重 mismatch）
        return global_step_counter, 0.0

    train_ds = SFTDataset(sft_rows, tokenizer, max_seq_len=args.max_seq_len)
    grad_accum = max(1, args.train_batch_size // args.micro_batch_size)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.micro_batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_pad(b, tokenizer.pad_token_id),
        num_workers=2,
        drop_last=True,
        pin_memory=True,
    )

    # 这一轮 SFT 内层共多少 optimizer step
    steps_per_epoch = max(1, len(train_ds) // (args.micro_batch_size * grad_accum))
    inner_steps = steps_per_epoch * args.n_sft_epochs_per_ei
    print(f"  [sft] D_sft={len(sft_rows)} | grad_accum={grad_accum} | "
          f"steps_per_epoch={steps_per_epoch} × {args.n_sft_epochs_per_ei} epochs = {inner_steps} optim steps")

    train_iter = infinite_iter(train_loader)
    sum_entropy = 0.0
    n_microbatches = 0

    for inner_step in range(inner_steps):
        accumulated_loss = 0.0
        accumulated_entropy = 0.0
        for _ in range(grad_accum):
            batch = next(train_iter)
            input_ids = batch["input_ids"].to(args.policy_device, non_blocking=True)
            labels = batch["labels"].to(args.policy_device, non_blocking=True)
            mask = batch["response_mask"].to(args.policy_device, non_blocking=True)

            out = get_response_log_probs(policy, input_ids, labels, return_token_entropy=True)
            log_probs = out["log_probs"]
            tok_entropy = out["token_entropy"]
            mask_f = mask.float()

            loss, _ = sft_microbatch_train_step(
                log_probs, mask_f, grad_accum, normalize_constant=1.0
            )
            accumulated_loss += loss.item()
            n_resp = mask_f.sum().clamp(min=1)
            accumulated_entropy += ((tok_entropy * mask_f).sum() / n_resp).item()
            sum_entropy += accumulated_entropy
            n_microbatches += 1

        gnorm = torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
        # cosine LR 在整个 EI 训练的 total optimizer steps 上 schedule，
        # 不是每个 EI step 内部独立 schedule —— 让 lr 平滑下降
        warmup_steps = int(total_optimizer_steps * args.warmup_ratio)
        lr = cosine_lr(global_step_counter, warmup_steps, total_optimizer_steps, args.learning_rate)
        set_lr(optimizer, lr)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        global_step_counter += 1

        if wandb and (global_step_counter % 10 == 0):
            wandb.log({
                "train/loss": accumulated_loss,
                "train/lr": lr,
                "train/grad_norm": float(gnorm),
                "train/avg_token_entropy": accumulated_entropy / grad_accum,
                "train_step": global_step_counter,
            })

    avg_entropy = sum_entropy / max(1, n_microbatches)
    return global_step_counter, avg_entropy


# =============================================================================
# 4. EI 主循环
# =============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--questions-data", required=True, help="EI 训练问题集（MATH train.jsonl 或 GSM8K train.jsonl）")
    p.add_argument("--val-data", default=None)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--prompt-name", default="r1_zero")

    # EI 主参
    p.add_argument("--n-ei-steps", type=int, default=5, help="PDF §5 推荐 5")
    p.add_argument("--batch-size-questions", type=int, default=1024,
                   help="每个 EI step 采样的 question 数 D_b ∈ {512,1024,2048}")
    p.add_argument("--rollouts-per-question", "-G", type=int, default=4,
                   help="每题 rollout 数 G")
    p.add_argument("--n-sft-epochs-per-ei", type=int, default=1,
                   help="每轮 EI 内 SFT 跑几遍 D_sft")

    # 采样参
    p.add_argument("--sampling-temperature", type=float, default=1.0)
    p.add_argument("--sampling-top-p", type=float, default=1.0)
    p.add_argument("--sampling-max-tokens", type=int, default=1024)
    p.add_argument("--sampling-min-tokens", type=int, default=4,
                   help="PDF §5 hint：vLLM 防空串 NaN")

    # SFT 内层循环参（与 train_sft.py 同款）
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--train-batch-size", type=int, default=32)
    p.add_argument("--micro-batch-size", type=int, default=2)
    p.add_argument("--max-seq-len", type=int, default=1024,
                   help="EI 默认 1024（rollout 可能比 GSM8K 长，如 MATH 长题）")
    p.add_argument("--grad-clip", type=float, default=1.0)

    # eval / save / device
    p.add_argument("--eval-num", type=int, default=512)
    p.add_argument("--policy-device", default="cuda:0")
    p.add_argument("--vllm-device", default="cuda:1")
    p.add_argument("--vllm-gpu-mem", type=float, default=0.85)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--no-grad-checkpoint", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
    set_seed(args.seed)

    # ---- 4.1 加载 policy + tokenizer + vLLM ----
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

    # ---- 4.2 加载数据 ----
    from tiny_training_rl.prompts import load as load_prompt
    from tiny_training_rl import grader
    prompt_template = load_prompt(args.prompt_name)
    reward_fn = grader.r1_zero_reward_fn

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

    # vLLM SamplingParams：rollout 用（n=G），val 用（n=1，stop=]"</answer>"]）
    from vllm import SamplingParams
    rollout_params = SamplingParams(
        temperature=args.sampling_temperature,
        top_p=args.sampling_top_p,
        max_tokens=args.sampling_max_tokens,
        min_tokens=args.sampling_min_tokens,
        n=args.rollouts_per_question,
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
            wb.define_metric("ei_step")
            wb.define_metric("train/*", step_metric="train_step")
            wb.define_metric("eval/*", step_metric="eval_step")
            wb.define_metric("rollout/*", step_metric="ei_step")
            wandb = wb
        except Exception as e:
            print(f"  wandb unavailable: {e}")

    # ---- 4.4 EI 外层循环 ----
    # cosine LR 全局 schedule：估算 total optimizer steps（= n_ei_steps × 平均
    # 每轮 sft inner steps）。每轮 D_sft 大小变化大，先按 questions × G ×
    # baseline correct_rate ~= 0.05 估个粗值，让 LR 平滑下降。
    rough_total_steps = max(
        100,
        args.n_ei_steps * args.batch_size_questions * args.rollouts_per_question // args.train_batch_size,
    )

    global_step = 0
    rng = torch.Generator().manual_seed(args.seed)
    t_start = time.perf_counter()

    for ei_step in range(1, args.n_ei_steps + 1):
        ei_t0 = time.perf_counter()
        print(f"\n=== EI step {ei_step}/{args.n_ei_steps} ===")

        # ---- (a) sample question batch D_b ----
        idx = torch.randint(0, len(questions), (args.batch_size_questions,), generator=rng).tolist()
        D_b = [questions[i] for i in idx]

        # ---- (b) rollout：π_old ← π，把当前 policy 灌进 vllm，n=G ----
        policy.eval()
        with torch.inference_mode():
            load_policy_into_vllm_instance(policy, llm)
            sft_rows, rollout_stats = rollout_and_filter(
                llm, D_b, args.rollouts_per_question,
                prompt_template, reward_fn, rollout_params,
            )
        policy.train()
        rollout_t = time.perf_counter() - ei_t0
        print(f"  [rollout] {rollout_stats['n_correct']}/{rollout_stats['n_total']} correct "
              f"(rate={rollout_stats['correct_rate']:.3%}), {rollout_t:.1f}s")
        if wandb:
            wandb.log({
                "rollout/n_total": rollout_stats["n_total"],
                "rollout/n_correct": rollout_stats["n_correct"],
                "rollout/correct_rate": rollout_stats["correct_rate"],
                "rollout/format_rate": rollout_stats["format_rate"],
                "rollout/mean_response_len_correct": rollout_stats["mean_response_len_correct"],
                "rollout/wall_time_sec": rollout_t,
                "ei_step": ei_step,
            })

        # ---- (c) SFT 内层循环 ----
        sft_t0 = time.perf_counter()
        global_step, avg_entropy = run_sft_inner_loop(
            policy, optimizer, sft_rows, tok, args,
            global_step_counter=global_step,
            total_optimizer_steps=rough_total_steps,
            wandb=wandb,
        )
        sft_t = time.perf_counter() - sft_t0
        print(f"  [sft] global_step={global_step} avg_entropy={avg_entropy:.3f} ({sft_t:.1f}s)")

        # ---- (d) eval ----
        if val_rows:
            policy.eval()
            with torch.inference_mode():
                load_policy_into_vllm_instance(policy, llm)
                metrics, samples = evaluate_policy_vllm(
                    llm, val_rows, prompt_template, reward_fn, val_params, log_n_samples=8,
                )
            policy.train()
            print(f"  [eval] acc={metrics['accuracy']:.3%} format={metrics['format_rate']:.3%}")
            if wandb:
                wandb.log({
                    "eval/accuracy": metrics["accuracy"],
                    "eval/format_rate": metrics["format_rate"],
                    "eval/avg_entropy_at_step": avg_entropy,  # PDF deliverable: entropy over training
                    "eval_step": ei_step,
                })

        # ---- (e) ckpt ----
        ckpt_dir = out_dir / f"ei_{ei_step}"
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
