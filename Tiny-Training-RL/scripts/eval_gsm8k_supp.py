"""GSM8K zero-shot evaluation（supplement PDF §2.2 Problem `gsm8k_baseline`）.

与 §3 主 PDF eval_zeroshot.py 的差别：
    主 PDF (§3):   r1_zero prompt（`<think>...</think> <answer>...</answer>`），
                   r1_zero_reward_fn（严格模板 + sympy 判等）
    supplement (§2.2): 简单 prompt `{question}\\nAnswer:`，parser 取**末尾数字** + 字符串判等

prompt 模板（supplement §2.2 给出）：
    # Instruction
    Below is a list of conversations between a human and an AI assistant (you).
    ... (system prompt 同 §2)

    # Query:
    ```{question}
    Answer:```

    # Answer:
    ```

CLI:
    uv run python scripts/eval_gsm8k_supp.py \\
        --model /data/a5-alignment/models/Llama-3.1-8B \\
        --dataset /data/a5-alignment/gsm8k/test.jsonl \\
        --out /data/runs/gsm8k_baseline_8b \\
        --backend vllm
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path


# 复用 supplement §2 通用 system prompt
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


def make_prompt(question: str) -> str:
    """supplement §2.2 prompt：system + ```question\\nAnswer:``` + ``` 起始符。"""
    return f"{_SYSTEM_PROMPT}# Query:\n```{question}\nAnswer:```\n\n# Answer:\n```"


def load_dataset(path: Path, limit: int | None) -> list[dict]:
    """读 GSM8K jsonl。schema 容错：`question`/`problem` 任一 + `answer`。
    GSM8K answer 字段是 `"...steps...#### final"`，取 #### 后部分作为 GT。
    """
    rows = []
    with path.open() as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            obj = json.loads(line)
            q = obj.get("question") or obj.get("problem")
            a = obj["answer"]
            gt = a.rsplit("####", 1)[-1].strip() if "####" in a else a.strip()
            rows.append({"question": q, "ground_truth": gt})
    return rows


# ---------- 推理后端（与 eval_mmlu.py 同款，简化复制）----------
def generate_vllm(model_path, prompts, max_tokens, temperature, top_p, stop, gpu_mem, max_model_len, dtype):
    from vllm import LLM, SamplingParams
    sampling = SamplingParams(
        temperature=temperature, top_p=top_p, max_tokens=max_tokens,
        stop=stop, include_stop_str_in_output=False,
    )
    llm = LLM(model=model_path, dtype=dtype, gpu_memory_utilization=gpu_mem,
              max_model_len=max_model_len, enable_prefix_caching=True)
    outs = llm.generate(prompts, sampling)
    return [o.outputs[0].text for o in outs]


def generate_hf(model_path, prompts, max_tokens, temperature, top_p, stop, dtype, batch_size):
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True, help="GSM8K test.jsonl")
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--backend", choices=["hf", "vllm"], default="vllm")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="PDF §2.2 推荐 greedy")
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--dtype", default="bfloat16")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_dataset(Path(args.dataset), args.limit)
    prompts = [make_prompt(r["question"]) for r in rows]
    print(f"[data] {len(rows)} examples")

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

    # 解析 + 评分
    from tiny_training_rl.metrics import parse_gsm8k_response

    records = []
    n_correct = n_parse_fail = 0
    for r, gen in zip(rows, gens):
        parsed = parse_gsm8k_response(gen)
        if parsed is None:
            n_parse_fail += 1
            correct = False
        else:
            # GSM8K 答案恒为整数；字符串判等（去 trailing zeros 之类的边界没处理）
            correct = (parsed == r["ground_truth"])
            if correct:
                n_correct += 1
        records.append({
            "question": r["question"],
            "ground_truth": r["ground_truth"],
            "generation": gen,
            "parsed": parsed,
            "correct": correct,
        })

    n = len(rows)
    summary = {
        "model": args.model, "n_examples": n, "elapsed_sec": elapsed,
        "throughput_examples_per_sec": n / elapsed if elapsed > 0 else 0.0,
        "n_correct": n_correct, "n_parse_fail": n_parse_fail,
        "accuracy": n_correct / n if n else 0.0,
        "parse_fail_rate": n_parse_fail / n if n else 0.0,
    }
    with (out_dir / "predictions.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # (f) 错误抽样
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
