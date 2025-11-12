import argparse
import torch
from cs336_basics.model import BasicsTransformerLM 
from cs336_basics.optimizer import AdamW, get_cosine_lr
from cs336_basics.nn_utils import cross_entropy, clip_gradient
import timeit
import pandas as pd
import gc
import torch.cuda.nvtx as nvtx
from contextlib import nullcontext

model_specs = [
    ("small", 768, 3072, 12, 12),
    ("medium", 1024, 4096, 24, 16),
    ("large", 1280, 5120, 36, 20),
    ("xl", 1600, 6400, 48, 25),
    ("2.7B", 2560, 10240, 32, 32),
]
# 用字典推导式生成CONFIGS
CONFIGS = {
    size: {
        "d_model": d_model,
        "d_ff": d_ff,
        "num_layers": num_layers,
        "num_heads": num_heads,
    }
    for size, d_model, d_ff, num_layers, num_heads in model_specs
}

TYPE_CHOICES = {
    'fp32': torch.float32,
    'fp16': torch.float16,
    'bf16': torch.bfloat16,
}

def benchmarking(args):
    print('------------------------------------------------------------')
    print(f"Model: {args.model_size}, Mode: {args.mode}")
    print(f"Batch={args.batch_size}, Context={args.context_length}")
    # 环境和随机种子设ben置
    torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'select device {device}')
    
    # 检查 bf16 支持
    if args.dtype == 'bf16' and device == 'cuda':
        if not torch.cuda.is_bf16_supported():
            print("Warning: BF16 not supported on this device, falling back to FP16")
            args.dtype = 'fp16'

    # 模型init
    model = BasicsTransformerLM(
        vocab_size=args.vocabulary_size,
        context_length=args.context_length,
        **CONFIGS[args.model_size],
        rope_theta=10000,
    ).to(device)

    # 数据和优化器
    x = torch.randint(args.vocabulary_size, (args.batch_size, args.context_length), device=device)
    target = torch.randint(args.vocabulary_size, (args.batch_size, args.context_length), device=device, dtype=torch.long)
    
    optimizer: torch.optim.optimizer | None = None
    if args.mode == 'fwd_bwd':
        optimizer = AdamW(model.parameters(), lr=1e-3, betas=(0.99, 0.9))
    
    # 根据 dtype 选择 autocast
    if args.dtype == 'fp32':
        autocast_context = nullcontext()
    else:
        autocast_context = torch.autocast(device_type=device, dtype=TYPE_CHOICES[args.dtype])
    
    # warm up
    for _ in range(args.warm_up_steps):
        torch.cuda.synchronize()
        st = timeit.default_timer()
        with autocast_context:
            logits = model(x)
        if optimizer is not None:
            loss = cross_entropy(logits, target)
            
            optimizer.zero_grad()
            loss.backward()
            clip_gradient(model.parameters(), max_norm=2.0)
            optimizer.step()
        torch.cuda.synchronize()
        ed = timeit.default_timer()
        print(f'warm cost: {ed - st}')
    
    
    # memory profile flag
    if device == 'cuda' and args.memory_profile:
        print('Start benchmarking memory...')
        torch.cuda.memory._record_memory_history(max_entries=1000000)
    
    # 正式benchmarking
    fwd_times = []
    bwd_times = []
    for _ in range(args.timing_steps):
        torch.cuda.synchronize() if device.startswith("cuda") else None
        st = timeit.default_timer()
        with nvtx.range("forward stage of llm"):
            with autocast_context:
                logits = model(x)
        torch.cuda.synchronize() if device.startswith("cuda") else None
        ed_fwd = timeit.default_timer()
        fwd_times.append(ed_fwd - st)
        print(f'forward cost : {ed_fwd - st}')
        if optimizer is not None:
            with nvtx.range("backward:"):
                with nvtx.range("compute loss"):
                    loss = cross_entropy(logits, target)
                
                optimizer.zero_grad()
                with nvtx.range("loss.backward compute:"):
                    loss.backward()
                    clip_gradient(model.parameters(), max_norm=2.0)
                with nvtx.range("optimize step:"):
                    optimizer.step()
                # Important
                torch.cuda.synchronize() if device.startswith("cuda") else None
                ed_bwd = timeit.default_timer()
                bwd_times.append(ed_bwd - ed_fwd)
                print(f'backward cost: {ed_bwd - ed_fwd}')
                
    # 内存分析：保存快照并停止记录
    if device == 'cuda' and args.memory_profile:
        print("Dumping memory snapshot...")
        torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)
        print("Memory snapshot saved to memory_snapshot.pickle")
        print("You can visualize it at: https://pytorch.org/memory_viz")

    # 统计结果
    avg_fwd = sum(fwd_times) / len(fwd_times)
    std_fwd = (sum((t - avg_fwd) ** 2 for t in fwd_times) / max(1, len(fwd_times) - 1)) ** 0.5
    print(f"Average forward: {avg_fwd*1000:.2f} ms, Std: {std_fwd*1000:.2f} ms")

    avg_bwd = 0.0
    std_bwd = 0.0
    if optimizer is not None: # 等价与 args.mode == 'fwd_bwd'
        avg_bwd = sum(bwd_times) / len(bwd_times)
        std_bwd = (sum((t - avg_bwd) ** 2 for t in bwd_times) / max(1, len(bwd_times) - 1)) ** 0.5
        print(f"Average backward: {avg_bwd*1000:.2f} ms, Std: {std_bwd*1000:.2f} ms")
    
    # 显存分析和输出
    max_memory = 0
    current_memory = 0
    if device == 'cuda' and args.memory_profile:
        max_memory = torch.cuda.max_memory_allocated() / 1024**3 # GB
        current_memory = torch.cuda.memory_allocated() / 1024**3
        print(f'Max GPU memory used    : {max_memory} GB')
        print(f'Current GPU memory used: {current_memory} GB')

    # ---- ✅ 释放显存 ----
    del model, x, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return dict(
                Model=args.model_size, Context=args.context_length, Batch=args.batch_size,
                Mode=args.mode, Avg_fwd=avg_fwd, Std_fwd=std_fwd, Avg_bwd=avg_bwd, Std_bwd=std_bwd,
                max_memory=max_memory
            )
    
    
    
# 对§1.1.2中描述的模型大小进行正向和反向传递的时间测量。使用5个预热步骤，并计算10个测量步骤的平均值和标准差。正向传递需要多长时间？反向传递呢？你在测量中看到高变异性吗？或者标准差很小？3

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--mode', type=str, default='fwd_bwd', choices=['fwd', 'fwd_bwd'])
    
    parser.add_argument("--model_size", type=str, default="small", choices=CONFIGS.keys())
    parser.add_argument("--warm_up_steps", type=int, default=5)
    parser.add_argument("--timing_steps", type=int, default=10)
    
    parser.add_argument("--vocabulary_size", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--context_length", type=int, default=512)
    
    parser.add_argument('--seed', type=int, default=42)
    
    parser.add_argument("--compile", action="store_true")
    
    parser.add_argument("--dtype", type=str, default="fp32", choices=TYPE_CHOICES.keys())
    
    parser.add_argument("--memory_profile", action="store_true")
    

    args = parser.parse_args()
    results = []
    # for model_size in CONFIGS.keys():
    for model_size in [args.model_size]:
        args.model_size = model_size
        with nvtx.range(f'model size: {model_size}'):
            results.append(benchmarking(args))
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    df = pd.DataFrame(results)
    print(df.to_markdown(index=False, floatfmt=".6f"))
    

