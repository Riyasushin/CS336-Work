import json         # 用于读取和保存 JSON 文件（比如 vocab）
import os           # 用于操作文件路径和目录
import regex as re  # 使用增强版正则库，支持 Unicode 类别匹配
from typing import BinaryIO  # 用于类型提示：二进制文件对象
from multiprocessing import Pool  # 多进程并行处理
from collections import defaultdict  # 用于统计计数，自动初始化字典
import time         # 用于计时
from tqdm import tqdm  # 用于进度条显示

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None):
        """
        初始化 Tokenizer
        vocab: {id: bytes}，把 token ID 对应到具体的 byte 字节
        merges: BPE merge 列表 [(b'a', b'b'), ...]
        special_tokens: 用户自定义的特殊 token 列表
        """
        self.vocab = vocab
        self.vocab_reversed = {v: k for k, v in self.vocab.items()}  # bytes -> int，方便查找 ID
        self.merges = merges
        # special token 按长度从大到小排序，方便在预处理时优先匹配长 token
        self.special_tokens = sorted(special_tokens or [], key=lambda x: -len(x)) # TODO Why??
    
    @classmethod  
    def from_files(cls, vocab_file_path: str, merges_filepath: str, special_tokens: list[str] | None) -> "Tokenizer":
        vocab: dict[int, bytes] = {}
        with open(vocab_file_path, "r", encoding="utf-8") as f:
            for line in f:
                id_str, token_str = line.strip().split("\t")
                vocab[int(id_str)] = token_str.encode("utf-8")  # 存为 bytes
        
        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    merges.append((parts[0].encode("utf-8"), parts[1].encode("utf-8")))
        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)
    
    @staticmethod
    def save_bpe_model(vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], output_dir: str):
        os.makedirs(output_dir, exist_ok=True)

        # 保存 vocab
        vocab_serialized = {str(i): token.decode('utf-8', errors='replace') for i, token in vocab.items()}
        with open(os.path.join(output_dir, "vocab.json"), "w", encoding="utf-8") as f:
            json.dump(vocab_serialized, f, ensure_ascii=False, indent=2)

        # 保存 merges
        with open(os.path.join(output_dir, "merges.txt"), "w", encoding="utf-8") as f:
            for a, b in merges:
                f.write(f"{a.decode('utf-8', errors='replace')} {b.decode('utf-8', errors='replace')}\n")
    
    def encode(self):
        _


def train_bpe_old(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int = 8
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    使用 BPE 训练 tokenizer
    """
    
    # 根据任务要求，这个顺序
    vocab = {i: bytes([i]) for i in range(256)}
    for tok in special_tokens:
        vocab[len(vocab)] = tok.encode("utf-8")
    special_tokens = sorted(special_tokens, key=lambda x: -len(x)) # TODO
    
    # 读取文件，分块处理
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, "<|endoftext|>".encode("utf-8"))
        chunk_list = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunk_list.append(chunk)
    task_args = [(chunk, special_tokens, False) for chunk in chunk_list]
    with Pool(processes=num_processes) as pool:
        chunk_results = pool.map(process_chunk, task_args)
    
    # 计算 bpe merges
    merges : list[tuple[bytes, bytes]] = []
    pre_tokens_bytes : list[list[bytes]] = [token for chunk in  chunk_results for token in chunk]
    counts = defaultdict(int)           # pair 出现次数
    pair_to_indices = defaultdict(set)  # pair 出现的 token index
    
    # 统计每个 pair 出现位置和次数
    for idx, token in enumerate(pre_tokens_bytes):
        for i in range(len(token) - 1):
            pair = (token[i], token[i + 1])
            counts[pair] += 1
            pair_to_indices[pair].add(idx)
    
    idx = len(vocab)
    while idx < vocab_size:
        if not counts:
            # TODO 为什么这里会这样
            break
        max_pair : tuple[bytes, bytes] = None
        max_cnt = -1
        
        # 艹怎么说是这种优化方法，蚌埠住了，这不还是爽循环吗
        # 少处理一个神奇最大堆，sign
        for pair, cnt in counts.items():
            if cnt > max_cnt:
                max_pair = pair
                max_cnt = cnt
            elif cnt == max_cnt:
                if max_pair is None or pair > max_pair:
                    max_pair = pair
        merges.append(max_pair)           # 记录 merge
        a, b = max_pair
        new_token = a + b
        vocab[idx] = new_token
        idx += 1
        
        # 更新 pre_tokens_bytes 中所有受影响的 token
        affected_indices = pair_to_indices[max_pair].copy() # TODO 为什么 copy 
        
        for j in affected_indices:
            token = pre_tokens_bytes[j]
            # 删除旧 pair 的计数
            for i in range(len(token) - 1):
                old_pair = (token[i], token[i+1])
                pair_to_indices[old_pair].discard(j)
                counts[old_pair] -= 1
                if counts[old_pair] == 0:
                    counts.pop(old_pair)
                    pair_to_indices.pop(old_pair, None)
            
            # 合并pair
            merged = []
            i = 0
            while i < len(token):
                if i < len(token) - 1 and token[i] == a and token[i+1]==b:
                    merged.append(new_token)
                    i += 2
                else:
                    merged.append(token[i])
                    i += 1
            pre_tokens_bytes[j]=merged
            
            # 重新统计新的 pair
            # TODO 为什么
            token = pre_tokens_bytes[j]
            for i in range(len(token) - 1):
                pair = (token[i], token[i + 1])
                counts[pair] += 1
                pair_to_indices[pair].add(j)
    return vocab, merges
        
def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str], num_processes: int = 8):
    vocab = {i: bytes([i]) for i in range(256)}
    for tok in special_tokens:
        vocab[len(vocab)] = tok.encode("utf-8")
    special_tokens = sorted(special_tokens, key=lambda x: -len(x))

    counts = defaultdict(int)
    pair_to_indices = defaultdict(set)
    pre_tokens_bytes = []  # 仅保存 token 的引用（index）

    # 分块读取文件
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        print("Tokenizing chunks...")
        for start, end in tqdm(zip(boundaries[:-1], boundaries[1:])):
            f.seek(start)
            # XXX 一次只读一个 chunk → 内存只占当前块大小
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # process_chunk 返回的是 list[list[bytes]]，每个 token 是 bytes 列表
            chunk_tokens = process_chunk((chunk, special_tokens, False))
            
            for token_bytes in chunk_tokens:
                idx = len(pre_tokens_bytes)
                pre_tokens_bytes.append(token_bytes)
                for i in range(len(token_bytes) - 1):
                    pair = (token_bytes[i], token_bytes[i + 1])
                    counts[pair] += 1
                    pair_to_indices[pair].add(idx)

    # 后续 BPE merge 逻辑同之前
    print("Performing BPE merges...")
    merges = []
    idx = len(vocab)
    # while idx < vocab_size and counts:
    for _ in tqdm(range(vocab_size - idx)):
        # 找最频繁 pair
        max_pair = max(counts.items(), key=lambda x: (x[1], x[0]))[0]
        merges.append(max_pair)
        a, b = max_pair
        new_token = a + b
        vocab[idx] = new_token
        idx += 1

        affected_indices = pair_to_indices[max_pair].copy()
        for j in affected_indices:
            token = pre_tokens_bytes[j]
            # 删除旧 pair 的计数
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
       
    

def find_chunk_boundaries(file: BinaryIO, desired_num_chunks: int, split_special_token: bytes) -> list[int]:
    """
    按特殊 token 分块文件，返回分块位置列表
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


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def process_chunk(args: tuple[str, list[str], bool]) -> list[list[bytes]]:
    '''
    pre tokenization
    '''
    
    chunk, special_tokens, keep_special_tokens = args
    
    # 构造正则匹配特殊 token
    pattern = "|".join(re.escape(tok) for tok in special_tokens)
    if keep_special_tokens and pattern:
        pattern = f"({pattern})"
    
    segments = re.split(pattern, chunk) if pattern else [chunk]
    
    pre_tokens_bytes: list[list[bytes]] = []
    
    for segment in segments:
        if keep_special_tokens and segment in special_tokens:
            token_bytes = [segment.encode('utf-8')]
            pre_tokens_bytes.append(token_bytes)
        else:
            # 使用正则 PAT 匹配 segment 中的所有 token
            # match.group(0) 拿到匹配到的字符串
            # 把这个字符串转换成 UTF-8 字节序列  'sth' ->  b'sth'
            tokens = [match.group(0).encode("utf-8") for match in re.finditer(PAT, segment)]
            for token in tokens:
                token_bytes = [bytes([b]) for b in token]
                pre_tokens_bytes.append(token_bytes)
    return pre_tokens_bytes
    


def main():
    start_time = time.time()
    vocab, merges = train_bpe(
        input_path='/data/CS336-use/TinyStoriesV2-GPT4-train.txt',
        # input_path='/home/rj/WorkingOn/1-CS336/assignment1/cs336_basics/in.txt',
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
