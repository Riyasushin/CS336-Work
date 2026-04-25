from __future__ import annotations

from . import _fasttext
from .assets import asset_path

_NSFW_FILENAME = "dolma_fasttext_nsfw_jigsaw_model.bin"
_HATESPEECH_FILENAME = "dolma_fasttext_hatespeech_jigsaw_model.bin"


def classify_nsfw(text: str) -> tuple[str, float]:
    model = _fasttext.load(str(asset_path(_NSFW_FILENAME)))
    return _fasttext.predict_top1(model, text)


def classify_toxic_speech(text: str) -> tuple[str, float]:
    model = _fasttext.load(str(asset_path(_HATESPEECH_FILENAME)))
    return _fasttext.predict_top1(model, text)
