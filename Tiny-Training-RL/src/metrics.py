"""答案解析（supplement PDF §2.1 / §2.2）。

两个公开函数：
    parse_mmlu_response(mmlu_example, model_output) -> "A"|"B"|"C"|"D"|None
    parse_gsm8k_response(model_output)               -> str|None  （末尾数字）

两个都对 LM 自由文本输出做"取最像答案的片段"。
"""

from __future__ import annotations

import re
from typing import Any


# =============================================================================
# §2.1 (a) parse_mmlu_response
# =============================================================================

# 按 supplement PDF §2.1 prompt 模板，model 期望生成 "The correct answer is X"。
# 但 base 模型可能写成各种变体；按"prompt 期待的 anchor 词更可信"原则排优先级。
# 优先匹配"answer is/answer:" 之类的明确 anchor，其次退到任何独立的 A-D 大写
# 字母（最 lenient）。
_MMLU_ANCHOR_PATTERNS = [
    # "The correct answer is A" / "the correct answer is (A)" / 各种空格
    re.compile(r"(?:correct\s+)?answer\s+is\s*[:\(]?\s*([A-D])\b", re.IGNORECASE),
    # "Answer: A" / "answer (A)"
    re.compile(r"answer\s*[:\(]?\s*([A-D])\b", re.IGNORECASE),
    # "Option A" / "Choice A"
    re.compile(r"(?:option|choice)\s*[:\(]?\s*([A-D])\b", re.IGNORECASE),
]
# 最后兜底：任何独立位置出现的大写 A/B/C/D 字母（注意大小写敏感 —— 防止
# 把英文里的 "a" "an" 之类的误识别）
_MMLU_FALLBACK = re.compile(r"\b([A-D])\b")


def parse_mmlu_response(mmlu_example: dict[str, Any], model_output: str) -> str | None:
    """把模型输出 parse 成 'A'/'B'/'C'/'D'，否则 None。

    搜索策略（按特异性递降）：
        1. "(correct\\s+)?answer is X" / "answer: X"
        2. "answer X"
        3. "option X" / "choice X"
        4. fallback：任何独立的 A-D 大写字母

    case-handling：
        - anchor 部分（"answer is" 等）case-insensitive
        - 选项字母虽然 IGNORECASE 也匹配，但用 .upper() 标准化为大写
        - fallback **大小写敏感** —— 避免把小写 "a" "an" 误判成 A 选项

    返回值：
        - "A"/"B"/"C"/"D"
        - None：找不到合理的选项字母
    """
    for pat in _MMLU_ANCHOR_PATTERNS:
        m = pat.search(model_output)
        if m:
            return m.group(1).upper()
    m = _MMLU_FALLBACK.search(model_output)
    if m:
        return m.group(1)
    return None


# =============================================================================
# §2.2 (a) parse_gsm8k_response
# =============================================================================

# GSM8K 答案恒为整数；但 LM 输出可能含中间计算式 "48/2 = 24" / "$10" 等等。
# PDF 明确规定：take the **last number** in the predicted output。
#
# 数字定义：
#   - 可选负号
#   - 整数部分（千分位逗号 "1,234" 也接受）
#   - 可选小数部分
# 这个 regex 匹配整数 / 浮点 / 带千分位的形式
_NUMBER_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")


def parse_gsm8k_response(model_output: str) -> str | None:
    """取 model_output 里**最后一个数字**作为答案；找不到返 None。

    返回 str（保留原数字字符串，去除千分位逗号），便于 grader 后续判等。
    """
    nums = _NUMBER_RE.findall(model_output)
    if not nums:
        return None
    return nums[-1].replace(",", "")
