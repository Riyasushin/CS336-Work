from __future__ import annotations

import contextlib
import os
from functools import lru_cache

import fasttext

# fasttext 0.9.3's python wrapper calls `np.array(probs, copy=False)`, which
# raises on numpy>=2. The pybind object `model.f.predict` still works; we call
# it directly to stay numpy-2 compatible without pinning down.


@contextlib.contextmanager
def _silence_stderr():
    saved = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)
    try:
        yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)
        os.close(devnull)


@lru_cache(maxsize=8)
def load(path: str):
    with _silence_stderr():
        return fasttext.load_model(path)


def predict_top1(model, text: str) -> tuple[str, float]:
    cleaned = text.replace("\n", " ").strip()
    predictions = model.f.predict(cleaned, 1, 0.0, "strict")
    prob, label = predictions[0]
    return label.removeprefix("__label__"), float(prob)
