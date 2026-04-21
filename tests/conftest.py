"""Root pytest config.

Skips test subdirectories whose optional dependencies are not installed, so
``uv run pytest`` works out of the box on the base env and picks up more
tests as you install extras:

    uv sync                    # base: tt.nn / tt.moe / tt.parallel tests
    uv sync --extra data       # + tests/data (pipeline deps)
    uv sync --extra alignment  # + tests/alignment (transformers, etc.)
"""

from __future__ import annotations

import importlib.util


def _missing(mod: str) -> bool:
    return importlib.util.find_spec(mod) is None


collect_ignore: list[str] = []
collect_ignore_glob: list[str] = []

# --- tests/alignment needs transformers (imported at conftest top level) ---
if _missing("transformers"):
    collect_ignore.append("alignment")

# --- tests/data/test_deduplication uses xopen ---
if _missing("xopen"):
    collect_ignore.append("data/test_deduplication.py")

# --- tests/data/test_extract uses resiliparse / fastwarc ---
if _missing("resiliparse") or _missing("fastwarc"):
    collect_ignore.append("data/test_extract.py")

# --- tests/data language / classifier tests need fasttext / nltk ---
if _missing("fasttext"):
    collect_ignore.extend([
        "data/test_langid.py",
        "data/test_quality.py",
        "data/test_toxicity.py",
    ])

# --- tests/data/test_pii uses only stdlib regex; always collected ---
