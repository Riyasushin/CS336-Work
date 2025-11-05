import numpy.typing as npt
import numpy as np
import torch



def data_loading(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.超过这个长度的文本会被截断.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    
    
    # 确保可采样范围（留出预测 token 空间）
    n = len(dataset)
    max_start = n - context_length - 1 # 索引从 0 开始, 目标序列需要右移一位
    if max_start <= 0:
        raise ValueError("dataset 太短，无法采样 context_length 序列")
    
    # 随机选 batch_size 个起点, 使用 np.random.randint 的向量化采样，避免 Python 循环
    # 采样范围是 [low, high)（右边开区间)
    starts = np.random.randint(0, max_start + 1, size=batch_size)
    
    # 直接利用 numpy 的广播切片生成 batch（避免循环）
    # 构造索引矩阵：每行对应一个样本的 token 索引
    offsets = np.arange(context_length) # [0, 1, ... , context_length - 1]
    idx = starts[:, None] + offsets # (batch, context_length)
    
    x_batch = dataset[idx]
    y_batch = dataset[idx + 1]
    
    # 使用 torch.from_numpy 避免额外拷贝（比 torch.tensor 快）
    # 注意：memmap 也可以直接转为 tensor，不需要额外载入内存
    x_tensor = torch.from_numpy(x_batch).to(device=device, dtype=torch.long)
    y_tensor = torch.from_numpy(y_batch).to(device=device, dtype=torch.long)
    return x_tensor, y_tensor
    
    