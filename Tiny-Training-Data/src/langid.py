from __future__ import annotations

from . import _fasttext
from .assets import asset_path

_FILENAME = "lid.176.bin"


def identify_language(text: str) -> tuple[str, float]:
    model = _fasttext.load(str(asset_path(_FILENAME)))
    return _fasttext.predict_top1(model, text)
