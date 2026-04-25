"""MMLU zero-shot evaluation（supplement PDF §2.1 Problem `mmlu_baseline`）.

支持两种数据布局：
    1. supplement PDF 原版：MMLU jsonl，每行 {"subject", "question", "options", "answer"}
    2. 本机原始 CS336 csv：subject_val.csv（无 header），列序：question,A,B,C,D,answer

PDF 任务 (a)-(f)：
    (a) parse_mmlu_response —— 已在 src/metrics.py 实装（test 4/4 过）
    (b) ★ 本脚本：load + prompt + generate + reward + 序列化
    (c) 解析失败的 generation 数 + 例子（输出在 failures.jsonl）
    (d) examples/sec 吞吐（summary.json::throughput_examples_per_sec）
    (e) accuracy（summary.json::accuracy）
    (f) 错误样本 10 例（failures.jsonl 由调用方 /grep 抽样审查）

关键 prompt 模板（PDF §2.1 给出，照抄 + 加 system prompt）：
    # Instruction
    Below is a list of conversations between a human and an AI assistant (you).
    Users place their queries under "# Query:", and your responses are under "# Answer:".
    You are a helpful, respectful, and honest assistant.
    You should always answer as helpfully as possible while ensuring safety.
    Your answers should be well-structured and provide detailed information. They should also have an engaging tone.
    Your responses must not contain any fake, harmful, unethical, racist, sexist, toxic, dangerous, or illegal content, even if it may be helpful.
    Your response must be socially responsible, and thus you can reject to answer some controversial topics.

    # Query:
    ```Answer the following multiple choice question about {subject}. Respond with a single
    sentence of the form "The correct answer is _", filling the blank with the letter
    corresponding to the correct answer (i.e., A, B, C or D).

    Question: {question}
    A. {options[0]}
    B. {options[1]}
    C. {options[2]}
    D. {options[3]}
    Answer:```

    # Answer:
    ```

→ 模型在 ``` （code block 起始符）后开始生成；当看到 ``` （结束符）或 "# Query:" 时停止。

PDF §2.1 hyperparams：
    temperature=0.0 (greedy), top_p=1.0, max_tokens=1024
    模型：Llama 3.1 8B base（本机不可得，可用 Qwen 2.5 Math 1.5B 替换跑 demo）

CLI:
    uv run python scripts/eval_mmlu.py \\
        --model /data/a5-alignment/models/Llama-3.1-8B \\
        --mmlu-dir /data/a5-alignment/mmlu \\
        --split val \\
        --out /data/runs/mmlu_baseline_8b \\
        --backend vllm
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Iterable


# =============================================================================
# 数据加载
# =============================================================================

def load_mmlu_csv_dir(csv_dir: Path) -> list[dict]:
    """加载 CS336 风格的 csv 数据：每个 subject 一个 csv 文件，无 header。

    每行 5+1 列：question, A, B, C, D, answer_letter
    返回 list of {subject, question, options, answer}。
    """
    rows: list[dict] = []
    for csv_path in sorted(csv_dir.glob("*.csv")):
        # 文件名格式：{subject}_{split}.csv → subject = stem.rsplit('_', 1)[0]
        subject = csv_path.stem.rsplit("_", 1)[0].replace("_", " ")
        with csv_path.open() as f:
            for line in csv.reader(f):
                if len(line) < 6:
                    continue
                question, A, B, C, D, ans = line[:6]
                rows.append({
                    "subject": subject,
                    "question": question,
                    "options": [A, B, C, D],
                    "answer": ans.strip(),
                })
    return rows


def load_mmlu_jsonl(path: Path) -> list[dict]:
    """加载 supplement PDF 原版 jsonl：每行 {"subject", "question", "options", "answer"}。"""
    rows = []
    with path.open() as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


# =============================================================================
# Prompt 拼接
# =============================================================================

# supplement PDF §2 给的 system prompt
_SYSTEM_PROMPT = """# Instruction
Below is a list of conversations between a human and an AI assistant (you).
Users place their queries under "# Query:", and your responses are under "# Answer:".
You are a helpful, respectful, and honest assistant.
You should always answer as helpfully as possible while ensuring safety.
Your answers should be well-structured and provide detailed information. They should also \
have an engaging tone.
Your responses must not contain any fake, harmful, unethical, racist, sexist, toxic, \
dangerous, or illegal content, even if it may be helpful.
Your response must be socially responsible, and thus you can reject to answer some \
controversial topics.

"""

# §2.1 question template
_QUESTION_TEMPLATE = (
    'Answer the following multiple choice question about {subject}. Respond with a single\n'
    'sentence of the form "The correct answer is _", filling the blank with the letter\n'
    "corresponding to the correct answer (i.e., A, B, C or D).\n"
    "\n"
    "Question: {question}\n"
    "A. {A}\n"
    "B. {B}\n"
    "C. {C}\n"
    "D. {D}\n"
    "Answer:"
)


def make_prompt(example: dict) -> str:
    """拼接 system + query + answer 起始 ``` 结构。"""
    query = _QUESTION_TEMPLATE.format(
        subject=example["subject"],
        question=example["question"],
        A=example["options"][0],
        B=example["options"][1],
        C=example["options"][2],
        D=example["options"][3],
    )
    return _SYSTEM_PROMPT + "# Query:\n```" + query + "```\n\n# Answer:\n```"


# =============================================================================
# 推理后端（HF / vLLM 双支，复用 eval_zeroshot 思路）
# =============================================================================

def generate_vllm(model_path, prompts, max_tokens, temperature, top_p, stop, gpu_mem, max_model_len, dtype):
    from vllm import LLM, SamplingParams
    sampling = SamplingParams(
        temperature=temperature, top_p=top_p, max_tokens=max_tokens,
        stop=stop, include_stop_str_in_output=False,
    )
    llm = LLM(
        model=model_path, dtype=dtype,
        gpu_memory_utilization=gpu_mem,
        max_model_len=max_model_len, enable_prefix_caching=True,
    )
    outs = llm.generate(prompts, sampling)
    return [o.outputs[0].text for o in outs]


def generate_hf(model_path, prompts, max_tokens, temperature, top_p, stop, dtype, batch_size):
    """复用 eval_zeroshot.py 的 HF backend；这里简化版，stop 字符串截断在 caller 做。"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dt = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dt).to("cuda").eval()

    outs: list[str] = []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i:i + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True).to("cuda")
        with torch.inference_mode():
            gen_ids = model.generate(
                **enc, do_sample=temperature > 0,
                temperature=max(temperature, 1e-5), top_p=top_p,
                max_new_tokens=max_tokens, pad_token_id=tok.pad_token_id,
            )
        for ids in gen_ids:
            text = tok.decode(ids[enc.input_ids.shape[1]:], skip_special_tokens=True)
            for s in stop:
                idx = text.find(s)
                if idx >= 0:
                    text = text[:idx]
                    break
            outs.append(text)
        torch.cuda.empty_cache()
    return outs


# =============================================================================
# 评测主函数
# =============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--mmlu-dir", help="csv-per-subject 目录（CS336 layout）")
    p.add_argument("--mmlu-jsonl", help="单 jsonl 文件（supplement PDF 风格）")
    p.add_argument("--split", choices=["dev", "val", "test"], default="val",
                   help="csv-dir 模式下选 dev/val/test 子目录")
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--backend", choices=["hf", "vllm"], default="vllm")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="PDF §2.1 推荐 greedy decoding (T=0)")
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=1024,
                   help="PDF 推荐 1024；本地 demo 可降到 256")
    p.add_argument("--batch-size", type=int, default=8, help="HF backend only")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--dtype", default="bfloat16")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 加载数据
    if args.mmlu_dir:
        split_dir = Path(args.mmlu_dir) / args.split
        rows = load_mmlu_csv_dir(split_dir)
    elif args.mmlu_jsonl:
        rows = load_mmlu_jsonl(Path(args.mmlu_jsonl))
    else:
        raise ValueError("must give --mmlu-dir or --mmlu-jsonl")
    if args.limit:
        rows = rows[: args.limit]
    print(f"[data] {len(rows)} examples")

    # 2) 拼 prompts
    prompts = [make_prompt(r) for r in rows]

    # 3) 跑推理
    # PDF 模板：模型在 ``` 后生成；停止条件 ``` 或 "# Query:"（下一轮对话起始）
    stop = ["```", "# Query:"]
    t0 = time.perf_counter()
    if args.backend == "vllm":
        gens = generate_vllm(
            args.model, prompts, args.max_tokens, args.temperature, args.top_p,
            stop, args.gpu_memory_utilization, args.max_model_len, args.dtype,
        )
    else:
        gens = generate_hf(
            args.model, prompts, args.max_tokens, args.temperature, args.top_p,
            stop, args.dtype, args.batch_size,
        )
    elapsed = time.perf_counter() - t0
    print(f"[gen] {len(gens)} in {elapsed:.1f}s ({len(gens)/elapsed:.2f} ex/s)")

    # 4) 解析 + 评分
    from tiny_training_rl.metrics import parse_mmlu_response

    records = []
    n_correct = n_parse_fail = 0
    for ex, gen in zip(rows, gens):
        parsed = parse_mmlu_response(ex, gen)
        if parsed is None:
            n_parse_fail += 1
            correct = False
        else:
            correct = (parsed == ex["answer"])
            if correct:
                n_correct += 1
        records.append({
            "subject": ex["subject"],
            "question": ex["question"],
            "options": ex["options"],
            "answer": ex["answer"],
            "generation": gen,
            "parsed": parsed,
            "correct": correct,
        })

    # 5) 序列化
    n = len(rows)
    summary = {
        "model": args.model,
        "n_examples": n,
        "elapsed_sec": elapsed,
        "throughput_examples_per_sec": n / elapsed if elapsed > 0 else 0.0,
        "n_correct": n_correct,
        "n_parse_fail": n_parse_fail,
        "accuracy": n_correct / n if n else 0.0,
        "parse_fail_rate": n_parse_fail / n if n else 0.0,
    }
    with (out_dir / "predictions.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False))
            f.write("\n")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # 6) 抽样 (f) 错误分析：从 incorrect 里 seed=0 取 10 条
    rng = random.Random(0)
    incorrect = [r for r in records if not r["correct"]]
    parse_fail = [r for r in records if r["parsed"] is None]
    sampled_incorrect = rng.sample(incorrect, min(10, len(incorrect)))
    sampled_parse_fail = rng.sample(parse_fail, min(10, len(parse_fail)))
    with (out_dir / "failures.jsonl").open("w") as f:
        for r in sampled_incorrect:
            f.write(json.dumps({"bucket": "incorrect", **r}, ensure_ascii=False) + "\n")
        for r in sampled_parse_fail:
            f.write(json.dumps({"bucket": "parse_fail", **r}, ensure_ascii=False) + "\n")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
