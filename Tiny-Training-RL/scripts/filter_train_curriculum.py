"""§9 leaderboard 准备：用 base policy 跑一遍 MATH train.jsonl，按难度筛选 curriculum。

PDF §9 给的可选优化方向之一是"design a curriculum over the data"。一个简单的
curriculum：用 base policy（未训过）对每题跑 K 个 rollout，统计正确率：
    correct_rate = mean(reward(rollout, gt))

按 correct_rate 把题目分三档：
    EASY:   > 0.7  —— base 都能做对，留下浪费 GRPO 训练步
    MEDIUM: 0.05 - 0.7 —— 正面"信号题"，rollout 里能拿到非平凡的 advantage
    HARD:   < 0.05 —— base 几乎做不对，rollout 全 0 → A=0 全部 → 没梯度

保留 MEDIUM 档当 GRPO 训练池：每个 batch 都更可能产生 G 个 rollout 里有
对有错（non-zero std），advantage 信号最强。

这是 §9 leaderboard 4 小时窗口里最实在的 wall-time 节省 —— rollout 占 GRPO
训练 80% 的时间，过滤掉那些"100% rollout 都对"和"100% rollout 都错"的题，
等价于把 rollout 时间预算花在最有学习价值的题目上。

CLI:
    uv run python scripts/filter_train_curriculum.py \\
        --model /data/a5-alignment/models/Qwen2.5-Math-1.5B \\
        --questions-data /data/a5-alignment/MATH/train.jsonl \\
        --rollouts-per-question 8 \\
        --output /data/a5-alignment/MATH/train_filtered_curriculum.jsonl \\
        --keep-low 0.05 --keep-high 0.7
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tiny_training_rl.sft_train import init_vllm, set_seed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--questions-data", required=True)
    p.add_argument("--output", required=True, help="保留 MEDIUM 档的 jsonl 输出")
    p.add_argument("--prompt-name", default="r1_zero")
    p.add_argument("--reward-fn-name", choices=["r1_zero", "question_only"], default="r1_zero")
    p.add_argument("--rollouts-per-question", type=int, default=8,
                   help="每题用多少个 rollout 估计正确率；K 越大估计越稳但越慢")
    p.add_argument("--keep-low", type=float, default=0.05,
                   help="正确率低于此值视为 HARD，剔掉")
    p.add_argument("--keep-high", type=float, default=0.7,
                   help="正确率高于此值视为 EASY，剔掉")
    p.add_argument("--vllm-device", default="cuda:0")
    p.add_argument("--vllm-gpu-mem", type=float, default=0.85)
    p.add_argument("--sampling-temperature", type=float, default=1.0)
    p.add_argument("--sampling-max-tokens", type=int, default=1024)
    p.add_argument("--sampling-min-tokens", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-questions", type=int, default=-1, help="-1 全量")
    args = p.parse_args()

    set_seed(args.seed)

    # 1) 加载题目集
    rows: list[dict] = []
    with Path(args.questions_data).open() as f:
        for i, line in enumerate(f):
            if args.max_questions > 0 and i >= args.max_questions:
                break
            obj = json.loads(line)
            q = obj.get("question") or obj.get("problem") or obj.get("prompt")
            a = obj.get("answer", obj.get("response", ""))
            gt = a.rsplit("####", 1)[-1].strip() if "####" in a else a.strip()
            rows.append({"raw": obj, "question": q, "ground_truth": gt})
    print(f"[load] {len(rows)} questions")

    # 2) 加载 vLLM + reward fn + prompt
    from tiny_training_rl.prompts import load as load_prompt
    from tiny_training_rl import grader
    from vllm import SamplingParams

    prompt_template = load_prompt(args.prompt_name)
    reward_fn = {
        "r1_zero": grader.r1_zero_reward_fn,
        "question_only": grader.question_only_reward_fn,
    }[args.reward_fn_name]

    print(f"[load] vllm on {args.vllm_device}")
    llm = init_vllm(args.model, args.vllm_device, args.seed, args.vllm_gpu_mem)
    sampling = SamplingParams(
        temperature=args.sampling_temperature,
        max_tokens=args.sampling_max_tokens,
        min_tokens=args.sampling_min_tokens,
        n=args.rollouts_per_question,
        stop=["</answer>"],
        include_stop_str_in_output=True,
        seed=args.seed,
    )

    # 3) 一次 batch 跑 vLLM（n=K），统计每题正确率
    prompts = [prompt_template.format(question=r["question"]) for r in rows]
    print(f"[rollout] {len(prompts)} prompts × n={args.rollouts_per_question}")
    outs = llm.generate(prompts, sampling)

    correct_rates = []
    for r, out in zip(rows, outs):
        n_correct = 0
        for sample in out.outputs:
            res = reward_fn(sample.text, r["ground_truth"])
            n_correct += int(res["answer_reward"] == 1.0)
        rate = n_correct / args.rollouts_per_question
        correct_rates.append(rate)
        r["base_correct_rate"] = rate

    # 4) 分档统计
    easy = sum(1 for r in correct_rates if r > args.keep_high)
    hard = sum(1 for r in correct_rates if r < args.keep_low)
    medium = len(correct_rates) - easy - hard
    print(f"[stats] EASY (>{args.keep_high}): {easy}  "
          f"MEDIUM ({args.keep_low}-{args.keep_high}): {medium}  "
          f"HARD (<{args.keep_low}): {hard}")

    # 5) 保留 MEDIUM 档输出（保留原 row + 加个 base_correct_rate metadata）
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with Path(args.output).open("w") as f:
        for r, rate in zip(rows, correct_rates):
            if args.keep_low <= rate <= args.keep_high:
                obj = {**r["raw"], "base_correct_rate": rate}
                f.write(json.dumps(obj, ensure_ascii=False))
                f.write("\n")
                n_written += 1
    print(f"[save] {n_written} medium questions → {args.output}")


if __name__ == "__main__":
    main()
