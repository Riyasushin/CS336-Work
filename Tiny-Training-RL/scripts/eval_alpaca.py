"""AlpacaEval generation collection（supplement PDF §2.3 Problem `alpaca_eval_baseline`）.

PDF 任务流程：
    1. ★ 本脚本：用 base/SFT 模型在 alpaca_eval.jsonl 上跑生成；
       output JSON array, 每条 {instruction, output, generator, dataset}
    2. (脚本不做) 调 alpaca_eval CLI 跑 70B annotator 计算 winrate vs GPT-4 Turbo

PDF §2 通用 system prompt + AlpacaEval prompt 直接是 instruction 本身：

    # Query:
    ```{instruction}```

    # Answer:
    ```

注意 alpaca_eval CLI 期望的 JSON 结构是 array of dict（不是 jsonl）。

CLI（步骤 1）：
    uv run python scripts/eval_alpaca.py \\
        --model /data/a5-alignment/models/Llama-3.1-8B \\
        --dataset /data/a5-alignment/alpaca_eval/alpaca_eval.jsonl \\
        --out /data/runs/alpaca_baseline_8b/predictions.json \\
        --generator-name llama-3.1-8b-base \\
        --backend vllm

跑步骤 2（70B annotator，2×80GB H100，本地完全不能跑）：
    uv run alpaca_eval \\
        --model_outputs /data/runs/alpaca_baseline_8b/predictions.json \\
        --annotators_config 'scripts/alpaca_eval_vllm_llama3_3_70b_fn' \\
        --base-dir '.'
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


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


def make_prompt(instruction: str) -> str:
    return f"{_SYSTEM_PROMPT}# Query:\n```{instruction}```\n\n# Answer:\n```"


def load_alpaca_eval(path: Path) -> list[dict]:
    """读 alpaca_eval.jsonl；返回 list of dict（含 instruction + dataset metadata）。"""
    rows = []
    with path.open() as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True, help="alpaca_eval.jsonl")
    p.add_argument("--out", required=True, help="输出 JSON array 路径，alpaca_eval CLI 吃这个")
    p.add_argument("--generator-name", required=True,
                   help="entries['generator'] 字段；alpaca_eval CLI 用这个标识 model")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--backend", choices=["vllm"], default="vllm",
                   help="HF 路径理论上也行，但 alpaca_eval 题量 805 + max_tokens 1024 用 HF 太慢")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--dtype", default="bfloat16")
    args = p.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = load_alpaca_eval(Path(args.dataset))
    if args.limit:
        rows = rows[: args.limit]
    prompts = [make_prompt(r["instruction"]) for r in rows]
    print(f"[data] {len(rows)} alpaca examples")

    stop = ["```", "# Query:"]
    t0 = time.perf_counter()
    gens = generate_vllm(
        args.model, prompts, args.max_tokens, args.temperature, args.top_p,
        stop, args.gpu_memory_utilization, args.max_model_len, args.dtype,
    )
    elapsed = time.perf_counter() - t0
    print(f"[gen] {len(gens)} in {elapsed:.1f}s ({len(gens)/elapsed:.2f} ex/s)")

    # alpaca_eval CLI 期望 JSON array，每个 entry 必含 keys：
    # - instruction (str)
    # - output      (str)
    # - generator   (str, 该模型的 id)
    # - dataset     (str, AlpacaEval 原数据集自带的 dataset 字段)
    out_records = []
    for r, gen in zip(rows, gens):
        out_records.append({
            "instruction": r["instruction"],
            "output": gen,
            "generator": args.generator_name,
            "dataset": r.get("dataset", "alpaca_eval"),
        })
    out_path.write_text(json.dumps(out_records, ensure_ascii=False, indent=2))
    print(f"[save] {out_path}")
    print(f"\n下一步（70B annotator，本地不可跑）:")
    print(f"  uv run alpaca_eval \\")
    print(f"      --model_outputs {out_path} \\")
    print(f"      --annotators_config 'scripts/alpaca_eval_vllm_llama3_3_70b_fn' \\")
    print(f"      --base-dir '.'")


if __name__ == "__main__":
    main()
