import json         # 用于读取和保存 JSON 文件（比如 vocab）
import os           # 用于操作文件路径和目录
import regex as re  # 使用增强版正则库，支持 Unicode 类别匹配
from multiprocessing import Pool  # 多进程并行处理
from collections import defaultdict  # 用于统计计数，自动初始化字典
import time         # 用于计时
import multiprocessing as mp
import argparse
import tempfile
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from typing import Iterable, Iterator

CHUNK_SIZE = 1024 *  50
N_BYTES = 256

Test_pretokenize_and_count_mp = False

class BPE_Tokenizer:
    
    
    def __init__(self, vocab: dict[int, bytes] = {}, merges: list[tuple[bytes, bytes]] = [], special_tokens: list[str] | None = ["<|endoftext|>"]):
        self.vocab = vocab
        self.vocab_reversed = {v: k for k, v in self.vocab.items()}  # bytes: int
        self.merges = merges
        self.merge_ranks = {pair: i for i, pair in enumerate(merges)} # 越小越先合并
        self.special_tokens = sorted(special_tokens, key=len, reverse=True) if special_tokens is not None else special_tokens
    
    @classmethod
    def from_file(cls, name: str, special_tokens: list[str] = ["<|endoftext|>"]):
        vocab, merges = BPE_Tokenizer.load_bpe_model(name)
        return cls(vocab, merges, special_tokens)
    
    def encode(self, text: str) -> list[int]:
        '''
        special tokens 不参加encode
        '''
        token_ids = []
        
        # pre-tokenize the text to chunks, handling special tokens.
        for tokens in BPE_Tokenizer._get_pre_tokenize_list_iter(text, self.special_tokens):
            # merge and work
            if len(tokens) == 1 and tokens[0] in self.vocab_reversed:
                token_id = self.vocab_reversed.get(tokens[0])
                if token_id is not None:
                    token_ids.append(token_id)
                    continue
            while len(tokens) >= 2:
                # 找到pair中rank最小的那个（先合并它）
                # 如果 pair 不在 merge 规则里， 返回 inf
                best_pair = min(zip(tokens, tokens[1:]), key=lambda p: self.merge_ranks.get(p, float('inf')))
                # If the best pair is not in our merge rules, no more merges are possible.
                if best_pair not in self.merge_ranks:
                    break

                # Merge the best pair into a new token
                new_tok = best_pair[0] + best_pair[1]
                tokens = BPE_Tokenizer._merge_a_pair_for_encode(tokens, best_pair, new_tok)
                
            # Convert the final merged tokens to their corresponding IDs
            for token in tokens:
                token_id = self.vocab_reversed.get(token)
                if token_id is not None:
                    token_ids.append(token_id)
                    # XXX 其实不会识别错误吧
        return token_ids
    
    @staticmethod
    def _merge_a_pair_for_encode(tokens:list[bytes], pair: tuple[bytes, bytes], new_tok:bytes)->list[bytes]:
        i = 0
        new_token_bytes = []
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == pair:
                new_token_bytes.append(new_tok)
                i += 2
            else:
                new_token_bytes.append(tokens[i])
                i += 1
        return new_token_bytes
        
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for line in iterable:
            # token_ids = self.encode(line)
            # yield from token_ids
            # 流式 pre-tokenize，每次只返回一个 token 子列表
            for tokens in BPE_Tokenizer._get_pre_tokenize_list_iter(line, self.special_tokens):
                # 直接 merge 并 yield token_ids
                while len(tokens) >= 2:
                    best_pair = min(zip(tokens, tokens[1:]), key=lambda p: self.merge_ranks.get(p, float('inf')))
                    if best_pair not in self.merge_ranks:
                        break
                    new_tok = best_pair[0] + best_pair[1]
                    tokens = BPE_Tokenizer._merge_a_pair_for_encode(tokens, best_pair, new_tok)

                for token in tokens:
                    token_id = self.vocab_reversed.get(token)
                    if token_id is not None:
                        yield token_id
    
    
    def _encode_chunk_to_tmpfile(self, lines: list[str], dtype=np.uint16):
        """
        Worker 函数：把一块 lines 编码成 token IDs，并写入临时 npy 文件
        返回临时文件路径和 token 数量
        """
        token_ids = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            token_ids.extend(self.encode(line))
        # 创建临时文件保存 token_ids
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".npy")
        np.save(tmp_file, np.array(token_ids, dtype=dtype))
        tmp_file.close()
        return tmp_file.name, len(token_ids)

    
    def encode_to_npfile(self, input_path: str, output_path: str, chunk_size: int=100_000, max_workers: int=8, dtype=np.uint16):
        """
        将输入文本文件编码为 token IDs, 并保存为 .npy 文件
        Args:
            input_path: 文本文件路径
            output_path: 保存的 npy 文件路径
            chunk_size: 每个 chunk 的行数
            max_workers: 并行进程数
            dtype: 保存 token ID 的类型, uint16 足够大多数 vocab
        """

        tmp_files = []
        total_tokens = 0

        # 1. 创建进程池
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []

            # 2. 按行分块提交给进程池
            chunk_lines = []
            with open(input_path, 'r', encoding='utf-8') as f:
                for line in f:
                    chunk_lines.append(line)
                    if len(chunk_lines) >= chunk_size:
                        futures.append(executor.submit(BPE_Tokenizer._encode_chunk_to_tmpfile, self, chunk_lines, dtype))
                        chunk_lines = []
                # 处理剩余行
                if chunk_lines:
                    futures.append(executor.submit(BPE_Tokenizer._encode_chunk_to_tmpfile, self, chunk_lines, dtype))

            # 3. 收集每个 chunk 的结果
            for future in tqdm(futures, desc="多进程编码进度"):
                tmp_file, n_tokens = future.result()
                tmp_files.append((tmp_file, n_tokens))
                total_tokens += n_tokens

        # 4. 按顺序合并临时文件到最终 npy
        print(f"总 token 数量: {total_tokens}, 正在合并到 {output_path} ...")
        # memmap 方式创建最终文件
        out_array = np.memmap(output_path, dtype=dtype, mode='w+', shape=(total_tokens,))
        cursor = 0
        for tmp_file, n_tokens in tmp_files:
            tmp_array = np.load(tmp_file)
            out_array[cursor:cursor + n_tokens] = tmp_array
            cursor += n_tokens
            os.remove(tmp_file)  # 删除临时文件
        del out_array  # 确保 memmap 刷新写入磁盘
        print(f"✅ 保存完成: {output_path}")
    @staticmethod
    def _get_pre_tokenize_list_iter(text: str, special_tokens: list[str]| None) -> Iterator[list[bytes]]:
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        def yield_normal_tokens(segment: str) -> Iterator[list[bytes]]:
            """把普通文本按 PAT 分词，并拆成字节 yield"""
            for match in re.finditer(PAT, segment):
                token = match.group(0).encode("utf-8")
                yield [bytes([b]) for b in token]

        if not special_tokens:
            yield from yield_normal_tokens(text)
            return

        # special token 从长到短排序，防止重叠 token 被拆开
        special_tokens_sorted = sorted(special_tokens, key=len, reverse=True)
        special_pattern = '(' + '|'.join(re.escape(tok) for tok in special_tokens_sorted) + ')'

        for segment in re.split(special_pattern, text):
            if not segment:
                continue
            if segment in special_tokens:
                yield [segment.encode("utf-8")]
            else:
                yield from yield_normal_tokens(segment)

    @staticmethod
    def _get_pre_tokenize_list(text: str, special_tokens: list[str] | None) -> list[list[bytes]]:
        '''
        把输入的字符串pre-tokenize然后返回顺序的list[list[bytes]]
        保留special token
        '''
        result = []
        
        pattern = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        # 保留special tokens，需要用捕获组 `()`
        if special_tokens is None:
            blocks = [text]
        else:
            special_pattern_save_specials = '(' + "|".join(re.escape(token) for token in special_tokens) + ')'
            blocks = re.split(special_pattern_save_specials, text)  # 按特殊符号分割文本
        for block in blocks:
            if not block:
                continue
            if special_tokens is not None and block in special_tokens:
                result.append([block.encode("utf-8")])
            else:
                
                for match in re.finditer(pattern, block):
                    # text_b = b'你'
                    # token_bytes = [b'\xe4', b'\xbd', b'\xa0']
                    # ------
                    # token_b = b'Hello'
                    # token_bytes = [b'H', b'e', b'l', b'l', b'o']
                    text = match.group(0)
                    text_b = text.encode("utf-8")
                    token_bytes = [bytes([b]) for b in text_b]
                    result.append(token_bytes)
        return result        
        
    
    def decode(self, ids: list[int]) -> str:
        '''
        把输入的token_ids([int])根据vocab的出对应的真实字符结果
        '''
        tokens = []
        for token_id in ids:
            token_bytes = self.vocab.get(token_id, b'\xef\xbf\xbd') # 若不存在则用 � 的 UTF-8 编码（替换字符）
            tokens.append(token_bytes)
        full_bytes = b"".join(tokens)
        return full_bytes.decode(encoding='utf-8', errors='replace') # errors='replace' 用 � 替换错误部分
    
    ################################################################
    #     Train 过程中用的函数
    ################################################################
    def train(self, input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str], num_counter: int, num_merger: int, do_monitor: bool):
        st = time.perf_counter()
        # special_tokens 要按照长度降序排列，不然special tokens overlap 了
        special_tokens = sorted(special_tokens, key=len, reverse=True)
        word_counts = self._pretokenize_and_count_mp(input_path, special_tokens, num_counter, num_merger, do_monitor)
        ed = time.perf_counter()
        print(f'_pretokenize_and_count takes: {ed - st}')
        
        
        vocab = {i: bytes([i]) for i in range(N_BYTES)}
        for i, token in enumerate(special_tokens):
            vocab[N_BYTES + i] = token.encode('utf-8')
        size= len(vocab)
        merges = []
        
        if Test_pretokenize_and_count_mp:
            print('[test used] only test _pretokenize_and_count_mp. skip')
            return vocab, merges
        
        word_encodings = {}
        for word in word_counts:
            word_encodings[word] = list(word.encode('utf-8'))
            
        pair_strings = {}
        pair_to_words = defaultdict(set)
        pair_counts = BPE_Tokenizer._count_pairs(word_counts, word_encodings, pair_strings, vocab, pair_to_words)
        
        # Merge
        st = time.perf_counter()
        update_count = 0
        while size < vocab_size:
            BPE_Tokenizer._merge_a_pair(pair_counts, pair_strings, vocab,
                                   pair_to_words, word_counts, word_encodings,
                                   merges, size)
            size += 1
            update_count += 1
            if update_count % 1000 == 0:
                ed = time.perf_counter()
                print(f"merge {update_count}: {ed - st}")
        ed = time.perf_counter()
        print(f"merge time: {ed - st}")

        return vocab, merges
    
    @staticmethod
    def _merge_a_pair(pair_counts, pair_strings, vocab,
                                   pair_to_words, word_counts, word_encodings,
                                   merges, size):
        merge_pair, max_count = max(pair_counts.items(), key = lambda x: (x[1], pair_strings[x[0]]))
        merge_bytes = vocab[merge_pair[0]] + vocab[merge_pair[1]]
        
        vocab[size] = merge_bytes
        new_id = size
        
        affected_words = pair_to_words[merge_pair]
        BPE_Tokenizer._update_affected_word_count(merge_pair, affected_words, word_encodings,
                                                    word_counts, pair_counts,
                                                    pair_to_words, new_id, pair_strings, vocab)
        
        merges.append((vocab[merge_pair[0]], vocab[merge_pair[1]]))
    
    @staticmethod
    def _update_affected_word_count(merge_pair, affected_words, word_encodings,
                                                    word_counts, pair_counts,
                                                    pair_to_words, new_id, pair_strings, vocab):
        # we may update/delete words when iterate it.
        affected_words = affected_words.copy()
        
        for word in affected_words:
            word_tokens = word_encodings[word]
            wc = word_counts[word]
            
            for i in range(len(word_tokens) - 1):
                old_pair = (word_tokens[i], word_tokens[i + 1])
                pair_counts[old_pair] -= wc
                if pair_counts[old_pair] <= 0:
                    del pair_counts[old_pair]
                    pair_to_words.pop(old_pair)
                else:
                    pair_to_words[old_pair].discard(word)
            
            i = 0
            new_tokens = []
            
            while i < len(word_tokens):
                if i < len(word_tokens) - 1 and (word_tokens[i], word_tokens[i + 1]) == merge_pair:
                    new_tokens.append(new_id)
                    i += 2
                else:
                    new_tokens.append(word_tokens[i])
                    i += 1
            word_encodings[word] = new_tokens
            
            for i in range(len(new_tokens) - 1):
                new_pair = (new_tokens[i], new_tokens[i + 1])
                
                pair_counts[new_pair] += wc
                pair_to_words[new_pair].add(word)
                if new_pair not in pair_strings:
                    pair_strings[new_pair] = (vocab[new_pair[0]], vocab[new_pair[1]])
    
    @staticmethod
    def _count_pairs(word_counts, word_encodings, pair_strings, vocab, pair_to_words):
        pair_counts = defaultdict(int)
        for word, count in word_counts.items():
            encoding = word_encodings[word]
            for i in range(len(encoding) - 1):
                pair = encoding[i], encoding[i + 1]
                pair_counts[pair] += count
                if pair not in pair_strings:
                    # TODO
                    pair_strings[pair] = (vocab[pair[0]], vocab[pair[1]]) # pair_strings[pair] = (token_1_int, token_2_int)
                pair_to_words[pair].add(word) # 反序
        
        return pair_counts


    def _pretokenize_and_count(self, input_path: str, special_tokens: list[str], keep_special_tokens: bool = False) -> dict[str, int]:
        pattern = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        special_pattern = "|".join(re.escape(token) for token in special_tokens) # TODO ?????
        word_counts = defaultdict(int)

        for chunk in BPE_Tokenizer._chunk_documents_streaming(input_path):
            blocks = re.split(special_pattern, chunk)  # 按特殊符号分割文本
            for block in blocks:
                for match in re.finditer(pattern, block):
                    text = match.group(0)
                    word_counts[text] += 1
        
        return word_counts
    
    def _pretokenize_and_count_mp(self, input_path: str | os.PathLike, special_tokens: list[str], num_counter: int, num_merger: int, do_monitor: bool, keep_special_tokens: bool = False):
        pattern = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        special_pattern = "|".join(re.escape(token) for token in special_tokens)
        
        # 构建队列
        # 在 Python 的 multiprocessing 模块中，Queue 是线程安全的，
        # 因为它使用了内部锁来确保在多个进程之间安全地传递数据
        # 这意味着多个进程可以同时向队列中放入或取出数据，而不会导致数据损坏或竞争条件。
        chunk_queue = mp.Queue(maxsize=1_000_000)
        counter_queue = mp.Queue(maxsize=1_000_000)
        merged_queue = mp.Queue(maxsize=1_000_000)
        
        # 启动 _chunk_counter_process进程和_merge_counter_process进程
        counter_processes = []
        for i in range(num_counter):
            p = mp.Process(target=BPE_Tokenizer._chunk_counter_process,
                           args=(chunk_queue, counter_queue, pattern, special_pattern),
                           name=f'counter-process-{i}')
            p.start()
            counter_processes.append(p)
        merge_processes = []
        for i in range(num_merger):
            p = mp.Process(target=BPE_Tokenizer._merge_counter_process, 
                        args=(counter_queue, merged_queue),
                        name=f"merge_process-{i+1}")
            p.start()
            merge_processes.append(p)

        if do_monitor:
            # 这里做了什么，不理解
            stop_event = mp.Event()
            monitor_process = mp.Process(target=BPE_Tokenizer._queue_moniter_process, 
                                  args=(chunk_queue, counter_queue, merged_queue, stop_event))
            monitor_process.start()
        
        # 主进程读取文件生成chunk
        # 注意：这里我们的chunk_queue是有大小限制的(我们设置成了1_000_000)
        # 这样如果后面的流程比较慢，队列满了之后chunk_queue.put(chunk)就会阻塞，从而避免内存爆掉
        # 用yield来实现生成器函数的好处
        for chunk in BPE_Tokenizer._chunk_documents_streaming(input_path):
            chunk_queue.put(chunk)
        
        for i in range(num_counter):
            chunk_queue.put(None) # 补齐 上界
        
        # 主进程等待_chunk_counter_process结束并且通知_merge_counter_process进程
        for p in counter_processes:
            # 为什么放到这里，放到下一个循环后面可以吗?
            # 确保每个计数进程在主进程继续执行之前都已完成
            p.join()

        for _ in range(num_merger):
            # 这里加的
            counter_queue.put(None)
        
        # 主进程合并
        if num_counter == 1:
            word_counts = merged_queue.get()
        else:
            word_counts = merged_queue.get()
            for _ in range(num_merger - 1):
                counter = merged_queue.get()
                for k,v in counter.items():
                    word_counts[k] += v
        
        # 主进程等待所有合并进程结束，如果启动了监控进程，通过Event通知它结束自己
        for p in merge_processes:
            p.join() 
        if do_monitor:
            stop_event.set()
            monitor_process.join() 
        
        return word_counts  
        
    
    @staticmethod
    def _queue_moniter_process(chunk_queue, counter_queue, merged_queue, stop_event):
        while not stop_event.is_set():
            print(f'chunk_queu: {chunk_queue.qsize()}, counter_queue: {counter_queue.qsize()}, merged_queue: {merged_queue.qsize()}')
            # TODO 为什么要sleep 10s
            time.sleep(10)
    
    @staticmethod
    def _merge_counter_process(counter_queue, merged_queue):
        merge_counter = defaultdict(int)
        while True:
            counter = counter_queue.get()
            if counter == None:
                break
            for k, v in counter.items():
                merge_counter[k] += v
        merged_queue.put(merge_counter)
                
    
    @staticmethod
    def _chunk_counter_process(chunk_queue, counter_queue, pattern, special_pattern):
        while True:
            chunk = chunk_queue.get()
            if chunk == None:
                # TODO 这个None作为哨兵，如何工作的
                break
            blocks = re.split(special_pattern, chunk)
            counter = defaultdict(int)
            for block in blocks:
                for match in re.finditer(pattern, block):
                    # match.group(0) 为什么得到了text
                    # group(0) 是正则表达式匹配的基础用法，表示整个匹配的内容
                    text = match.group(0)
                    counter[text] += 1
            # 为什么线程安全
            counter_queue.put(counter)
            
    
    @staticmethod
    def _chunk_documents_streaming(path: str | os.PathLike, chunk_size: int = CHUNK_SIZE, special_token: str = "<|endoftext|>"):
        '''
        流式读取大文件，并将文件内容分割成以特定特殊标记（默认是 <|endoftext|>)为边界的文本块(chunk), 带特殊token
        '''
        leftover = ""
        token_len = len(special_token)
        
        with open(path, "r", encoding='utf-8') as f:
            while True:
                block = f.read(chunk_size)
                
                if not block:
                    break
                
                block = leftover + block
                leftover = ""
                
                last_eot_idx = block.rfind(special_token) # 查找当前 block 中最后一次出现特殊标记的位置（用 rfind 找最后一个，尽可能多包含完整内容）
                
                if last_eot_idx == -1:
                    # 情况1：当前 block 中没有找到特殊标记
                    # 说明所有内容都需要保留到下一次读取（与下一次的 block 合并后再找）
                    leftover = block
                else:
                    # 情况2：找到特殊标记，截取从开头到 "特殊标记末尾" 的内容作为完整块
                    # 截取范围：[0, last_eot_idx + token_len)，确保包含完整的特殊标记
                    yield block[: last_eot_idx + token_len]
                    # 将特殊标记之后的剩余内容存到 leftover，供下一次合并处理
                    leftover = block[last_eot_idx + token_len:]
            
        if leftover:
            # 文件读完后，如果还有未处理的 leftover（可能没有特殊标记结尾），也返回它
            yield leftover

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
    
    @staticmethod
    def load_bpe_model(name: str) ->  tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        """
        Load model saved by save_bpe_model(output_dir).
        - vocab.json is expected to be either:
            * a dict mapping str(id) -> token_string
            * a list where index -> token_string
          In both cases token_string was written using .decode('utf-8', errors='replace'),
          so we re-encode with utf-8 to obtain bytes.
        - merges.txt is expected to contain one merge per line in the form:
            <token_a_string><space><token_b_string>
          We split on the first space (tokens may include leading spaces).
        Returns (vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]])
        """
        if not name:
            raise ValueError("name must be a non-empty directory")
        vocab_path = os.path.join(name, "vocab.json")
        merges_path = os.path.join(name, "merges.txt")

        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"vocab file not found: {vocab_path}")
        if not os.path.exists(merges_path):
            raise FileNotFoundError(f"merges file not found: {merges_path}")

        try:
            # load vocab (support dict or list formats)
            with open(vocab_path, "r", encoding="utf-8") as f:
                vocab_data = json.load(f)

            vocab: dict[int, bytes] = {}
            if isinstance(vocab_data, dict):
                for k_str, token_str in vocab_data.items():
                    try:
                        idx = int(k_str)
                    except Exception as e:
                        raise ValueError(f"Invalid vocab key (expected int string): {k_str}") from e
                    if not isinstance(token_str, str):
                        raise ValueError(f"Invalid vocab value for id {k_str}: expected string")
                    vocab[idx] = token_str.encode("utf-8")
            elif isinstance(vocab_data, list):
                for idx, token_str in enumerate(vocab_data):
                    if not isinstance(token_str, str):
                        raise ValueError(f"Invalid vocab list element at index {idx}: expected string")
                    vocab[idx] = token_str.encode("utf-8")
            else:
                raise ValueError("vocab.json must be a dict or a list")

            # load merges (split on first space to allow tokens with internal spaces)
            merges: list[tuple[bytes, bytes]] = []
            with open(merges_path, "r", encoding="utf-8") as f:
                for lineno, raw in enumerate(f, start=1):
                    line = raw.rstrip("\n")
                    if not line:
                        continue
                    # split on first space only
                    parts = line.split(" ", 1)
                    if len(parts) != 2:
                        raise ValueError(f"Invalid merges line {lineno}: expected two tokens separated by space")
                    a_str, b_str = parts
                    merges.append((a_str.encode("utf-8"), b_str.encode("utf-8")))

            return vocab, merges

        except json.JSONDecodeError as e:
            raise IOError(f"Invalid JSON in vocab file {vocab_path}: {e}") from e
        except Exception as e:
            raise IOError(f"Failed to load BPE model from {name}: {e}") from e
    
    

def train_bpe(input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str]):
    trainer = BPE_Tokenizer()
    return trainer.train(input_path, vocab_size, special_tokens, num_counter=16, num_merger=4, do_monitor=False)

def main():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--test_pretokenize_and_count_mp",
                        action="store_true",
                        help="Enable queue monitor. (default: False)"
    )
    args = parser.parse_args()
    global Test_pretokenize_and_count_mp
    Test_pretokenize_and_count_mp = args.test_pretokenize_and_count_mp

    start_time = time.time()
    vocab, merges = train_bpe(
        input_path='/root/autodl-tmp/TinyStoriesV2-GPT4-train.txt',
        # input_path='/home/rj/WorkingOn/1-CS336/assignment1/cs336_basics/in.txt',
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
    )
    end_time = time.time()
    
    print(f"Training completed in {(end_time - start_time):.2f} seconds.")
    print(f"Vocab size: {len(vocab)}")
    print(f"Longest token: {max(vocab.values(), key=len)} (length={len(max(vocab.values(), key=len))})")
    BPE_Tokenizer.save_bpe_model(vocab, merges, "bpe_on_TinyStories_train")

if __name__ == "__main__":
    main()
