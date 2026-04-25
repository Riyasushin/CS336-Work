from __future__ import annotations

import hashlib
import os
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import mmh3


# ---------------- exact line deduplication ----------------


def _line_key(line: str) -> bytes:
    # md5 is stable across processes; Python's built-in hash() is salted and
    # would be fine within a single run but makes this harder to reason about.
    return hashlib.md5(line.rstrip("\n").encode("utf-8")).digest()


def exact_line_deduplication(
    input_files: list[os.PathLike], output_directory: os.PathLike
) -> None:
    out_dir = Path(output_directory)
    out_dir.mkdir(parents=True, exist_ok=True)

    counts: Counter[bytes] = Counter()
    for p in input_files:
        with open(p, encoding="utf-8") as f:
            for line in f:
                counts[_line_key(line)] += 1

    for p in input_files:
        p = Path(p)
        out_path = out_dir / p.name
        with open(p, encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
            for line in fin:
                if counts[_line_key(line)] == 1:
                    fout.write(line)


# ---------------- minhash deduplication ----------------

_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    text = text.lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = _PUNCT_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def _word_shingles(text: str, n: int) -> set[str]:
    words = _normalize(text).split()
    if len(words) < n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def _minhash_signature(shingles: set[str], num_hashes: int) -> list[int]:
    if not shingles:
        return [0] * num_hashes
    _MAX = 2**32 - 1
    sig = [_MAX] * num_hashes
    for s in shingles:
        b = s.encode("utf-8")
        for i in range(num_hashes):
            h = mmh3.hash(b, i, signed=False)
            if h < sig[i]:
                sig[i] = h
    return sig


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
) -> None:
    if num_hashes % num_bands != 0:
        raise ValueError(
            f"num_hashes ({num_hashes}) must be divisible by num_bands ({num_bands})"
        )
    rows_per_band = num_hashes // num_bands

    out_dir = Path(output_directory)
    out_dir.mkdir(parents=True, exist_ok=True)

    contents: list[str] = []
    shingle_sets: list[set[str]] = []
    signatures: list[list[int]] = []
    for p in input_files:
        with open(p, encoding="utf-8") as f:
            text = f.read()
        contents.append(text)
        sh = _word_shingles(text, ngrams)
        shingle_sets.append(sh)
        signatures.append(_minhash_signature(sh, num_hashes))

    n = len(input_files)

    buckets: dict[tuple[int, tuple[int, ...]], list[int]] = defaultdict(list)
    for i, sig in enumerate(signatures):
        for b in range(num_bands):
            band = tuple(sig[b * rows_per_band : (b + 1) * rows_per_band])
            buckets[(b, band)].append(i)

    uf = _UnionFind(n)
    checked: set[tuple[int, int]] = set()
    for docs in buckets.values():
        if len(docs) < 2:
            continue
        for i in range(len(docs)):
            for j in range(i + 1, len(docs)):
                a, b = docs[i], docs[j]
                pair = (a, b) if a < b else (b, a)
                if pair in checked:
                    continue
                checked.add(pair)
                if uf.find(a) == uf.find(b):
                    continue
                if _jaccard(shingle_sets[a], shingle_sets[b]) >= jaccard_threshold:
                    uf.union(a, b)

    clusters: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        clusters[uf.find(i)].append(i)
    keepers = {min(group) for group in clusters.values()}

    for i in sorted(keepers):
        p = Path(input_files[i])
        with open(out_dir / p.name, "w", encoding="utf-8") as fout:
            fout.write(contents[i])
