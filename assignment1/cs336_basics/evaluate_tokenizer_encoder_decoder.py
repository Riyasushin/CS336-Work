import os
import time
import random
import numpy as np
from typing import List, Tuple
from cs336_basics.train_bpe import BPE_Tokenizer

def evaluate_tokenizers(
    tiny_stories_path: str,
    openwebtext_path: str,
    n_samples: int = 20,
    sample_size: int = 1024,  # bytes per sample
) -> Tuple[dict, List[np.ndarray]]:
    """
    评估 TinyStories 和 OpenWebText tokenizer 的性能。
    
    Args:
        tiny_stories_path: TinyStories 数据集路径
        openwebtext_path: OpenWebText 数据集路径
        n_samples: 要采样的文档数量
        sample_size: 每个样本的大小（字节）
        
    Returns:
        metrics: dict 包含压缩率和吞吐量指标
        encoded_samples: List[np.ndarray] 编码后的样本（uint16）
    """
    # 加载两个 tokenizer
    try:
        tiny_tokenizer = BPE_Tokenizer.from_file("bpe_on_TinyStories_train")
        owt_tokenizer = BPE_Tokenizer.from_file("bpe_on_OpenWebText_train")
    except Exception as e:
        raise RuntimeError(f"Failed to load tokenizers: {e}")

    # 从两个数据集采样
    samples_tiny = []
    samples_owt = []
    
    def sample_from_file(filepath: str) -> List[str]:
        samples = []
        total_size = os.path.getsize(filepath)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for _ in range(n_samples):
                # 随机定位并读取，确保不跨文档边界
                while True:
                    pos = random.randint(0, max(0, total_size - sample_size))
                    f.seek(pos)
                    f.readline()  # 跳过可能不完整的行
                    sample = f.read(sample_size)
                    if sample and "<|endoftext|>" not in sample:
                        samples.append(sample)
                        break
        return samples

    try:
        samples_tiny = sample_from_file(tiny_stories_path)
        samples_owt = sample_from_file(openwebtext_path)
    except Exception as e:
        raise RuntimeError(f"Failed to sample from datasets: {e}")

    # 计算指标
    metrics = {
        "tiny_on_tiny": {"tokens": 0, "bytes": 0, "time": 0.0},
        "owt_on_owt": {"tokens": 0, "bytes": 0, "time": 0.0},
        "tiny_on_owt": {"tokens": 0, "bytes": 0, "time": 0.0},
    }
    
    def measure_tokenizer(tokenizer: BPE_Tokenizer, samples: List[str], key: str) -> List[np.ndarray]:
        encoded = []
        start = time.perf_counter()
        
        for sample in samples:
            sample_bytes = len(sample.encode('utf-8'))
            tokens = tokenizer.encode(sample)
            metrics[key]["tokens"] += len(tokens)
            metrics[key]["bytes"] += sample_bytes
            encoded.append(np.array(tokens, dtype=np.uint16))
            
        metrics[key]["time"] = time.perf_counter() - start
        return encoded

    # 评估 TinyStories tokenizer 在 TinyStories 上的表现
    encoded_tiny = measure_tokenizer(tiny_tokenizer, samples_tiny, "tiny_on_tiny")
    
    # 评估 OpenWebText tokenizer 在 OpenWebText 上的表现
    encoded_owt = measure_tokenizer(owt_tokenizer, samples_owt, "owt_on_owt")
    
    # 评估 TinyStories tokenizer 在 OpenWebText 上的表现（交叉评估）
    encoded_tiny_on_owt = measure_tokenizer(tiny_tokenizer, samples_owt, "tiny_on_owt")

    # 计算压缩率和吞吐量
    for key in metrics:
        stats = metrics[key]
        if stats["tokens"] > 0:
            stats["compression_ratio"] = stats["bytes"] / stats["tokens"]
            stats["throughput"] = stats["bytes"] / stats["time"] if stats["time"] > 0 else 0

    # 估算处理 Pile 数据集所需时间
    pile_size = 825 * 1024 * 1024 * 1024  # 825GB in bytes
    for key in metrics:
        if metrics[key]["throughput"] > 0:
            metrics[key]["pile_eta_hours"] = pile_size / (metrics[key]["throughput"] * 3600)

    return metrics, encoded_tiny + encoded_owt + encoded_tiny_on_owt

def main():
    """运行评估并打印结果"""
    metrics, encoded = evaluate_tokenizers(
        tiny_stories_path="/data/CS336-use/owt_valid.txt",
        openwebtext_path="/data/CS336-use/TinyStoriesV2-GPT4-valid.txt"
    )
    
    print("\nCompression Ratios:")
    print(f"TinyStories on TinyStories: {metrics['tiny_on_tiny']['compression_ratio']:.2f} bytes/token")
    print(f"OpenWebText on OpenWebText: {metrics['owt_on_owt']['compression_ratio']:.2f} bytes/token")
    print(f"TinyStories on OpenWebText: {metrics['tiny_on_owt']['compression_ratio']:.2f} bytes/token")
    
    print("\nThroughput:")
    print(f"TinyStories: {metrics['tiny_on_tiny']['throughput']/1024/1024:.2f} MB/s")
    print(f"OpenWebText: {metrics['owt_on_owt']['throughput']/1024/1024:.2f} MB/s")
    
    print("\nEstimated time to process Pile (825GB):")
    print(f"TinyStories: {metrics['tiny_on_tiny']['pile_eta_hours']:.1f} hours")
    print(f"OpenWebText: {metrics['owt_on_owt']['pile_eta_hours']:.1f} hours")


if __name__ == "__main__":
    main()
