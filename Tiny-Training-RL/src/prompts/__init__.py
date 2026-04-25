"""prompt 模板的访问入口。

把 .prompt 文件作为包数据放在这里，通过 importlib.resources 读取，避免脚本
里写绝对路径。
"""

from __future__ import annotations

from importlib.resources import files


def load(name: str) -> str:
    """按名字读取 prompt 模板（不含 `.prompt` 后缀）。"""
    return (files(__package__) / f"{name}.prompt").read_text(encoding="utf-8")
