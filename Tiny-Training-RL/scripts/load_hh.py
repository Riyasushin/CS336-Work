"""Load Anthropic HH dataset（supplement PDF §5.2）.

PDF §5.2 任务（Problem `look_at_hh`，2 pts）:
    1. 写函数加载 4 个 jsonl.gz 文件（harmless-base / helpful-base /
       helpful-online / helpful-rejection-sampled），合并成一个训练集
    2. 处理：(a) 跳过 multi-turn 对话；(b) 拆成 (instruction, chosen, rejected)
       三元组；(c) 记录每条来自哪个文件（给后续分析用）

PDF §5.2 数据格式（Anthropic HH 原版）:
    每行 JSON 含两个 key: "chosen" 和 "rejected"，各是一段 conversation 字符串：

        Human: <人类第一句>

        Assistant: <chosen 回答>

    或 multi-turn:

        Human: <第一句>

        Assistant: <第一回答>

        Human: <第二句>

        Assistant: <chosen 第二回答>

    multi-turn 在第二个 "Human:" 段需要被剔除（per PDF "ignore multi-turn"）。

CLI:
    uv run python scripts/load_hh.py \\
        --hh-dir /data/a5-alignment/hh \\
        --out /tmp/hh_combined.jsonl \\
        --seed 0 \\
        --inspect 6
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import re
from pathlib import Path


# 4 个 HH 子文件名（PDF §5.2 列出）
HH_FILES = [
    "harmless-base.jsonl.gz",
    "helpful-online.jsonl.gz",
    "helpful-base.jsonl.gz",
    "helpful-rejection-sampled.jsonl.gz",
]


def parse_hh_conversation(text: str) -> list[tuple[str, str]]:
    """把 HH 的 conversation 字符串切成 [(role, message), ...] list。

    HH 原版格式：
        "\\n\\nHuman: msg1\\n\\nAssistant: msg2\\n\\nHuman: msg3\\n\\nAssistant: msg4"

    用 findall + 显式 anchored regex（更稳，不需要 strip / split 协作）。
    每个 turn 模式: "(Human|Assistant): " 直到下一个 "\\n\\n(Human|Assistant): " 或字符串末尾。
    """
    # findall 模式：role + message，message 直到下一个 turn 起始或文本末尾
    # 用 re.DOTALL 让 . 匹配换行
    pattern = re.compile(
        r"(Human|Assistant): (.*?)(?=\n\n(?:Human|Assistant): |\Z)",
        re.DOTALL,
    )
    msgs: list[tuple[str, str]] = []
    for m in pattern.finditer(text):
        role, msg = m.group(1), m.group(2).strip()
        msgs.append((role, msg))
    return msgs


def is_single_turn(text: str) -> bool:
    """判断是否单轮对话（恰好 1 Human + 1 Assistant）。"""
    msgs = parse_hh_conversation(text)
    return len(msgs) == 2 and msgs[0][0] == "Human" and msgs[1][0] == "Assistant"


def load_hh_jsonl_gz(path: Path) -> list[dict]:
    rows = []
    with gzip.open(path, "rt") as f:
        for line in f:
            obj = json.loads(line)
            rows.append(obj)
    return rows


def load_hh_dataset(hh_dir: Path) -> list[dict]:
    """加载并合并 4 个 HH 子文件，过滤 multi-turn。

    返回 list of {instruction, chosen, rejected, source_file}。
    PDF deliverable 1: "A Python function that loads the dataset in a convenient
    data structure for you to use it for training."
    """
    out = []
    for fname in HH_FILES:
        path = hh_dir / fname
        if not path.exists():
            print(f"  [skip] {path} not found")
            continue
        rows = load_hh_jsonl_gz(path)
        n_total = len(rows)
        n_keep = 0
        for r in rows:
            chosen = r["chosen"]
            rejected = r["rejected"]
            # PDF 要求：跳过 multi-turn
            if not (is_single_turn(chosen) and is_single_turn(rejected)):
                continue
            chosen_msgs = parse_hh_conversation(chosen)
            rejected_msgs = parse_hh_conversation(rejected)
            instruction = chosen_msgs[0][1]  # Human 第一句
            # chosen 和 rejected 应该共享同一个 instruction
            assert chosen_msgs[0][1] == rejected_msgs[0][1], \
                f"instruction mismatch: {chosen_msgs[0][1][:50]!r} vs {rejected_msgs[0][1][:50]!r}"
            out.append({
                "instruction": instruction,
                "chosen": chosen_msgs[1][1],     # Assistant 答
                "rejected": rejected_msgs[1][1],
                "source_file": fname,
            })
            n_keep += 1
        print(f"  [{fname}] {n_keep}/{n_total} kept (single-turn only)")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hh-dir", required=True)
    p.add_argument("--out", required=True, help="combined jsonl 输出路径")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--inspect", type=int, default=0,
                   help="抽样 N 条打印（每来源 N/4，方便人眼审查）")
    args = p.parse_args()

    print(f"[load] hh dir: {args.hh_dir}")
    rows = load_hh_dataset(Path(args.hh_dir))
    print(f"[total] {len(rows)} single-turn pairs after merge + filter")

    # 写出
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[save] {out_path}")

    # PDF deliverable 2: 抽样 6 条（3 helpful + 3 harmless）
    if args.inspect > 0:
        rng = random.Random(args.seed)
        by_source: dict[str, list[dict]] = {}
        for r in rows:
            by_source.setdefault(r["source_file"], []).append(r)
        print(f"\n=== inspect {args.inspect} samples per source ===")
        for source, items in by_source.items():
            sampled = rng.sample(items, min(args.inspect, len(items)))
            for i, r in enumerate(sampled):
                print(f"\n--- {source} sample {i+1} ---")
                print(f"INSTRUCTION: {r['instruction'][:200]}")
                print(f"CHOSEN:      {r['chosen'][:200]}")
                print(f"REJECTED:    {r['rejected'][:200]}")


if __name__ == "__main__":
    main()
