import json
import os
import regex as re
from typing import BinaryIO
from multiprocessing import Pool, set_start_method
from collections import defaultdict
import time
from tqdm import tqdm

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None):
        self.vocab = vocab
        self.vocab_reversed = {v: k for k, v in self.vocab.items()}
        self.merges = merges
        self.special_tokens = sorted(special_tokens or [], key=lambda x: -len(x))  # 长 token 优先匹配

    @staticmethod
    def save_bpe_model(vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], output_dir: str):
        os.makedirs(output_dir, exist_ok=True)

        vocab_serialized = {str(i): token.decode('utf-8', errors='replace') for i, token in vocab.items()}
        with open(os.path.join(output_dir, "vocab.json"), "w", encoding="utf-8") as f:
            json.dump(vocab_serialized, f, ensure_ascii=False, indent=2)

        with open(os.path.join(output_dir, "merges.txt"), "w", encoding="utf-8") as f:
            for a, b in merges:
                f.write(f"{a.decode('utf-8', errors='replace')} {b.decode('utf-8', errors='replace')}\n")


def process_chunk(args: tuple[str, list[str], bool]) -> list[list[bytes]]:
    """
    将一个文本 chunk 转为 token 的 bytes 列表
    保留每个字符的 bytes 对象，不合并，满足存储对象要求
    """
    chunk, special_tokens, keep_special_tokens = args
    pattern = "|".join(re.escape(tok) for tok in special_tokens)
    if keep_special_tokens and pattern:
        pattern = f"({pattern})"

    segments = re.split(pattern, chunk) if pattern else [chunk]
    pre_tokens_bytes: list[list[bytes]] = []

    for segment in segments:
        if keep_special_tokens and segment in special_tokens:
            pre_tokens_bytes.append([segment.encode('utf-8')])
        else:
            tokens = [match.group(0).encode("utf-8") for match in re.finditer(PAT, segment)]
            for token in tokens:
                # 每个字符仍保留为独立 bytes 对象
                pre_tokens_bytes.append([bytes([b]) for b in token])
    return pre_tokens_bytes


def find_chunk_boundaries(file: BinaryIO, desired_num_chunks: int, split_special_token: bytes) -> list[int]:
    """
    按特殊 token 分块文件，返回分块位置列表
    保证每个 chunk 都完整，不切断特殊 token
    """
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096
    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))


def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str], num_processes: int = 8):
    """
    多进程 BPE 训练函数（安全的 parallel tokenization）
    """
    # --- 初始化 vocab ---
    vocab = {i: bytes([i]) for i in range(256)}
    for tok in special_tokens:
        vocab[len(vocab)] = tok.encode("utf-8")
    special_tokens = sorted(special_tokens, key=lambda x: -len(x))

    counts = defaultdict(int)
    pair_to_indices = defaultdict(set)
    pre_tokens_bytes = []

    # --- 构建任务 ---
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        tasks = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            tasks.append((chunk, special_tokens, False))

    # --- 多进程 tokenization ---
    print("Tokenizing chunks in parallel...")
    # spawn 启动方式避免 fork stdout 冲突
    try:
        set_start_method("spawn")
    except RuntimeError:
        # 如果已经设置过 start_method 会报错，忽略即可
        pass

    with Pool(processes=num_processes) as pool:
        pbar = tqdm(total=len(tasks), desc="Tokenizing chunks")
        for chunk_tokens in pool.imap(process_chunk, tasks):
            # 收集 token 到全局列表，并统计 pair
            for token_bytes in chunk_tokens:
                idx = len(pre_tokens_bytes)
                pre_tokens_bytes.append(token_bytes)
                for i in range(len(token_bytes)-1):
                    pair = (token_bytes[i], token_bytes[i+1])
                    counts[pair] += 1
                    pair_to_indices[pair].add(idx)
            pbar.update(1)
        pbar.close()

    # --- merge 阶段（单进程） ---
    print("Performing BPE merges...")
    merges = []
    idx = len(vocab)
    while idx < vocab_size and counts:
        max_pair = max(counts.items(), key=lambda x: (x[1], x[0]))[0]
        merges.append(max_pair)
        a, b = max_pair
        new_token = a + b
        vocab[idx] = new_token
        idx += 1

        affected_indices = pair_to_indices[max_pair].copy()
        for j in affected_indices:
            token = pre_tokens_bytes[j]
            # 删除旧 pair
            for i in range(len(token)-1):
                old_pair = (token[i], token[i+1])
                counts[old_pair] -= 1
                pair_to_indices[old_pair].discard(j)
                if counts[old_pair] == 0:
                    counts.pop(old_pair)
                    pair_to_indices.pop(old_pair, None)
            # 合并 pair
            merged = []
            i = 0
            while i < len(token):
                if i < len(token)-1 and token[i]==a and token[i+1]==b:
                    merged.append(new_token)
                    i += 2
                else:
                    merged.append(token[i])
                    i += 1
            pre_tokens_bytes[j] = merged
            # 更新新 pair
            for i in range(len(merged)-1):
                pair = (merged[i], merged[i+1])
                counts[pair] += 1
                pair_to_indices[pair].add(j)

    return vocab, merges

def main():
    start_time = time.time()
    vocab, merges = train_bpe(
        input_path='/data/CS336-use/TinyStoriesV2-GPT4-train.txt',
        vocab_size=10000,
        special_tokens=["<|endoftext|>"]
    )
    end_time = time.time()

    print(f"Training completed in {(end_time - start_time):.2f} seconds.")
    print(f"Vocab size: {len(vocab)}")
    print(f"Longest token: {max(vocab.values(), key=len)} (length={len(max(vocab.values(), key=len))})")
    Tokenizer.save_bpe_model(vocab, merges, "bpe_on_TinyStories_train")


if __name__ == "__main__":
    main()
