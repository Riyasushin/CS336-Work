"""Data-processing pipeline: extraction, language ID, PII masking, quality
classifiers, deduplication. Stubs only — these cover assignment4-data.

Implementation hint free by design (CS336 CLAUDE.md). See
assignment4-data/cs336_data_assignment.pdf for the full spec.
"""

from __future__ import annotations

import os
from typing import Any


def extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    raise NotImplementedError("tt.data.pipeline.extract_text_from_html_bytes")


def identify_language(text: str) -> tuple[Any, float]:
    raise NotImplementedError("tt.data.pipeline.identify_language")


def mask_emails(text: str) -> tuple[str, int]:
    raise NotImplementedError("tt.data.pipeline.mask_emails")


def mask_phone_numbers(text: str) -> tuple[str, int]:
    raise NotImplementedError("tt.data.pipeline.mask_phone_numbers")


def mask_ips(text: str) -> tuple[str, int]:
    raise NotImplementedError("tt.data.pipeline.mask_ips")


def classify_nsfw(text: str) -> tuple[Any, float]:
    raise NotImplementedError("tt.data.pipeline.classify_nsfw")


def classify_toxic_speech(text: str) -> tuple[Any, float]:
    raise NotImplementedError("tt.data.pipeline.classify_toxic_speech")


def classify_quality(text: str) -> tuple[Any, float]:
    raise NotImplementedError("tt.data.pipeline.classify_quality")


def gopher_quality_filter(text: str) -> bool:
    raise NotImplementedError("tt.data.pipeline.gopher_quality_filter")


def exact_line_deduplication(
    input_files: list[os.PathLike], output_directory: os.PathLike
) -> None:
    raise NotImplementedError("tt.data.pipeline.exact_line_deduplication")


def minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
) -> None:
    raise NotImplementedError("tt.data.pipeline.minhash_deduplication")


__all__ = [
    "extract_text_from_html_bytes",
    "identify_language",
    "mask_emails",
    "mask_phone_numbers",
    "mask_ips",
    "classify_nsfw",
    "classify_toxic_speech",
    "classify_quality",
    "gopher_quality_filter",
    "exact_line_deduplication",
    "minhash_deduplication",
]
