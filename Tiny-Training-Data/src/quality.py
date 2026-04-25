from __future__ import annotations

from . import _fasttext
from .assets import asset_path

_CLASSIFIER_FILENAME = "quality_classifier.bin"

# Gopher paper, Appendix A.1.1 — "MassiveWeb" filtering heuristics.
_STOPWORDS = {"the", "be", "to", "of", "and", "that", "have", "with"}


def classify_quality(text: str) -> tuple[str, float]:
    model = _fasttext.load(str(asset_path(_CLASSIFIER_FILENAME)))
    return _fasttext.predict_top1(model, text)


def gopher_quality_filter(text: str) -> bool:
    words = text.split()
    n_words = len(words)
    if n_words < 50 or n_words > 100_000:
        return False

    mean_len = sum(len(w) for w in words) / n_words
    if mean_len < 3 or mean_len > 10:
        return False

    lines = text.split("\n")
    if lines:
        ellipsis = sum(1 for ln in lines if ln.rstrip().endswith("..."))
        if ellipsis / len(lines) > 0.30:
            return False

    alpha = sum(1 for w in words if any(c.isalpha() for c in w))
    if alpha / n_words < 0.80:
        return False

    seen = {w.lower() for w in words} & _STOPWORDS
    if len(seen) < 2:
        return False

    return True
