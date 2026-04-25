"""Zero-shot evaluation of a base LM on a math reasoning dataset.

PDF: assignment5_alignment §3.2 (math_baseline).

本地资源约束：
- 没有 MATH（版权封锁）→ 默认数据集为 GSM8K（PDF 在 "Tip for Open-Source
  Auditors" 一节明确允许；GSM8K 的 ground truth 是整数，drgrpo 的 grade()
  能用 sympy 判等处理）。
- 单卡 8GB，跑 1.5B bf16；本机 CUDA 13.0 与 vLLM 编译版本不匹配，所以默认
  backend = "hf"（HuggingFace transformers）。vLLM 路径保留供 2 卡机使用。
- HF backend 用 model.generate(do_sample=True, temperature=1, top_p=1)；
  GSM8K 1.5B + bf16 + max_new_tokens=512，单卡可跑。

输出：
- <out>/predictions.jsonl —— 每行：question / ground_truth / generation /
  format_reward / answer_reward / reward
- <out>/summary.json     —— 三类计数 + 准确率 + 吞吐
- <out>/failures.jsonl   —— 抽样的错误样例（每类 ≤ 10 个，固定 seed=0）

跑法（HF backend，本地）：
    uv run --package tiny-training-rl python scripts/eval_zeroshot.py \
        --model /data/CS336-use/models/Qwen2.5-Math-1.5B \
        --dataset /data/CS336-use/assignment5-alignment/gsm8k/test.jsonl \
        --out /tmp/zeroshot_qwen25math \
        --limit 64 --backend hf

跑法（vLLM backend，2 卡机）：
    ... --backend vllm --gpu-memory-utilization 0.85
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable


def load_dataset(path: Path, limit: int | None) -> list[dict]:
    """读 jsonl 数据集，统一字段为 {question, ground_truth}。

    支持的字段名：
        - GSM8K      : {"question": ..., "answer": "...#### <gt>"}
        - GSM8K-my   : {"problem": ..., "answer": "...#### <gt>"}
        - MATH-style : {"problem"|"question": ..., "answer": "<gt>"}（无 #### 分隔）
    """
    rows: list[dict] = []
    with path.open() as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            obj = json.loads(line)
            q = obj.get("question") or obj.get("problem")
            ans = obj["answer"]
            # GSM8K 的 answer 字段是 "...#### 18" 风格 —— 提取最后的 ####
            if "####" in ans:
                gt = ans.split("####")[-1].strip()
            else:
                gt = ans.strip()
            rows.append({"question": q, "ground_truth": gt})
    return rows


# --------------------------------------------------------------------------------
# 推理后端
# --------------------------------------------------------------------------------

def generate_vllm(
    model_path: str,
    prompts: list[str],
    max_tokens: int,
    temperature: float,
    top_p: float,
    stop: list[str],
    gpu_memory_utilization: float,
    max_model_len: int,
    dtype: str,
) -> list[str]:
    from vllm import LLM, SamplingParams
    sampling = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop=stop,
        include_stop_str_in_output=True,
    )
    llm = LLM(
        model=model_path,
        dtype=dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        enable_prefix_caching=True,
    )
    outs = llm.generate(prompts, sampling)
    return [o.outputs[0].text for o in outs]


def generate_hf(
    model_path: str,
    prompts: list[str],
    max_tokens: int,
    temperature: float,
    top_p: float,
    stop: list[str],
    dtype: str,
    batch_size: int,
) -> list[str]:
    """HF transformers 推理。

    实现细节：
    - dtype 映射成 torch.bfloat16；attn_implementation="sdpa"（默认）—— 不强制
      flash-attn，因为本机不一定装。
    - 对每个 stop 字符串生成一个 StoppingCriteria，碰到任意一个就截。这里
      stop 通常只有 "</answer>"。
    - 左 padding，让 batch 内不同长度的 prompt 都能 generate（HF 默认右
      padding 会让 attention_mask 错位）。
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch_dtype)
    model = model.to("cuda").eval()

    # —— stop string 的实现：对每个 stop 取 token id 序列，每步 decode 末尾
    # 看是否包含；这里做最简实现：每步 decode 解出末 32 个 token 看是否
    # 含 stop 字符串。代价小，1.5B 单 batch 每步增量推理可接受。
    class StopOnString(StoppingCriteria):
        def __init__(self, stop_strs: list[str], prompt_lens: list[int]):
            self.stop = stop_strs
            self.prompt_lens = prompt_lens
            self.tok = tok

        def __call__(self, input_ids, scores, **kw):
            # 只有当 batch 中所有样本都已碰到 stop 才返回 True；否则继续。
            # 简化：实际 batch=1 走，多 batch 时各自截断由 caller 处理。
            B = input_ids.shape[0]
            done = []
            for b in range(B):
                gen_ids = input_ids[b, self.prompt_lens[b]:]
                text = self.tok.decode(gen_ids, skip_special_tokens=True)
                done.append(any(s in text for s in self.stop))
            return all(done)

    outs: list[str] = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        enc = tok(batch_prompts, return_tensors="pt", padding=True).to("cuda")
        prompt_lens = enc.input_ids.shape[1]  # 左 padding 后所有样本起点对齐
        prompt_lens_list = [prompt_lens] * len(batch_prompts)
        stopping = StoppingCriteriaList([StopOnString(stop, prompt_lens_list)]) if stop else None
        with torch.inference_mode():
            gen_ids = model.generate(
                **enc,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                top_p=top_p,
                max_new_tokens=max_tokens,
                pad_token_id=tok.pad_token_id,
                stopping_criteria=stopping,
            )
        # 切掉 prompt 部分；按 stop 字符串再裁一次（generate 一旦触发 stop 会
        # 让全 batch 一起停，但单条可能多生成一点）
        for b, ids in enumerate(gen_ids):
            text = tok.decode(ids[prompt_lens:], skip_special_tokens=True)
            for s in stop:
                idx = text.find(s)
                if idx >= 0:
                    text = text[: idx + len(s)]
                    break
            outs.append(text)
        torch.cuda.empty_cache()
    return outs


# --------------------------------------------------------------------------------
# 评测主体
# --------------------------------------------------------------------------------

def evaluate(
    backend: str,
    model_path: str,
    prompts: list[str],
    ground_truths: list[str],
    reward_fn: Callable[[str, str], dict[str, float]],
    cfg: argparse.Namespace,
) -> tuple[list[dict], dict]:
    """跑推理 + reward 评分，返回 records 和 metadata（吞吐 + 三类计数 + 准确率）。"""
    t0 = time.perf_counter()
    if backend == "vllm":
        gens = generate_vllm(
            model_path, prompts, cfg.max_tokens, cfg.temperature, cfg.top_p,
            stop=["</answer>"],
            gpu_memory_utilization=cfg.gpu_memory_utilization,
            max_model_len=cfg.max_model_len,
            dtype=cfg.dtype,
        )
    elif backend == "hf":
        gens = generate_hf(
            model_path, prompts, cfg.max_tokens, cfg.temperature, cfg.top_p,
            stop=["</answer>"],
            dtype=cfg.dtype,
            batch_size=cfg.batch_size,
        )
    else:
        raise ValueError(backend)
    elapsed = time.perf_counter() - t0

    records, counts = [], Counter()
    for prompt, gt, text in zip(prompts, ground_truths, gens):
        r = reward_fn(text, gt)
        bucket = (int(r["format_reward"]), int(r["answer_reward"]))
        counts[bucket] += 1
        records.append({
            "prompt": prompt,
            "ground_truth": gt,
            "generation": text,
            "format_reward": r["format_reward"],
            "answer_reward": r["answer_reward"],
            "reward": r["reward"],
        })

    n = len(prompts)
    meta = {
        "backend": backend,
        "n_examples": n,
        "elapsed_sec": elapsed,
        "throughput_examples_per_sec": n / elapsed if elapsed > 0 else 0.0,
        "count_format1_answer1": counts[(1, 1)],
        "count_format1_answer0": counts[(1, 0)],
        "count_format0_answer0": counts[(0, 0)],
        "accuracy": counts[(1, 1)] / n if n else 0.0,
    }
    return records, meta


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False))
            f.write("\n")


def sample_failures(records: list[dict], k: int = 10) -> list[dict]:
    """每类抽样 k 条用于 PDF (b) 的人眼审查。"""
    rng = random.Random(0)
    by_bucket: dict[tuple[int, int], list[dict]] = {}
    for r in records:
        bucket = (int(r["format_reward"]), int(r["answer_reward"]))
        by_bucket.setdefault(bucket, []).append(r)
    sampled: list[dict] = []
    for bucket, rows in by_bucket.items():
        chosen = rng.sample(rows, min(k, len(rows)))
        for r in chosen:
            sampled.append({"bucket": list(bucket), **r})
    return sampled


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--prompt-name", default="r1_zero")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--backend", choices=["hf", "vllm"], default="hf")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=512,
                   help="HF 单卡 8GB 默认 512；vLLM 可上 1024")
    p.add_argument("--batch-size", type=int, default=8, help="HF backend only")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.70)
    p.add_argument("--max-model-len", type=int, default=1280)
    p.add_argument("--dtype", default="bfloat16")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    from tiny_training_rl import grader
    from tiny_training_rl.prompts import load as load_prompt

    template = load_prompt(args.prompt_name)
    rows = load_dataset(Path(args.dataset), args.limit)
    prompts = [template.format(question=r["question"]) for r in rows]
    ground_truths = [r["ground_truth"] for r in rows]

    records, meta = evaluate(
        args.backend, args.model, prompts, ground_truths,
        grader.r1_zero_reward_fn, args
    )

    write_jsonl(out_dir / "predictions.jsonl", records)
    write_jsonl(out_dir / "failures.jsonl", sample_failures(records, k=10))
    (out_dir / "summary.json").write_text(json.dumps({
        "model": args.model,
        "dataset": args.dataset,
        "prompt_name": args.prompt_name,
        "limit": args.limit,
        **meta,
    }, indent=2))

    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
