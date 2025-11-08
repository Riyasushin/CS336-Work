from tqdm import tqdm
import argparse


import torch
import numpy as np
from cs336_basics.my_data import *
from cs336_basics.my_utils import *
from cs336_basics.my_loss import *
from cs336_basics.my_module import *




'''
编写一个脚本，运行训练循环以使用户提供的输入训练您的模型
特别是，我们建议您的训练脚本允许以下功能（至少）：

- 配置和控制各种模型和优化器超参数的能力。
- 使用np.memmap高效加载训练和验证大型数据集。
- 将检查点序列化到用户提供的路径。
- 定期记录训练和验证性能(例如，输出到控制台和/或外部服务如Weights and Biases)
'''


def main(args):
    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Loading dataset
    train_data = np.memmap(args.train_data_path, dtype=np.uint16, mode='r')
    val_data = np.memmap(args.val_data_path, dtype=np.uint16, mode='r')
    
    # Init model
    model = RJTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        d_model=args.d_model,
        rope_theta=args.rope_theta,
        device=device
    )
    
    # Init Optimizer
    optimizer = RJAdamW(
        params=model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
        eps=args.eps,
    )
    
    
    # Init lr scheduler
    lr_scheule = lambda t: lr_cosine_schedule_with_warmup(
        it=t,
        max_learning_rate=args.lr,
        min_learning_rate=args.lr_min,
        warmup_iters=args.warmup_iters,
        cosine_cycle_iters=args.cosine_iters,
    )
    
    # Load from checkpoint(optional)
    start_iter = 0
    best_loss = 1e100
    if args.checkpoint_path and args.resume:
        start_iter, best_loss = load_checkpoint(
            src=Path(args.checkpoint_path) / "best.pt",
            model=model,
            optimizer=optimizer,
            has_loss=True,
        )
    
    
    # Main Training
    model.train()
    for iteration in range(start_iter, args.max_iters):
    
        # Get batch
        inputs, targets = data_loading(train_data, batch_size=args.batch_size, context_length=args.context_length, device=device)
    
        # forward
        logits = model(inputs)
        loss = cross_entropy(logits, targets)
    
        # backward
        optimizer.zero_grad()
        loss.backward()
        
        # clip gradient
        gradient_clipping(model.parameters(), max_norm=args.max_grad_norm)
    
        # update parameters
        optimizer.step()
    
        # update lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_scheule(iteration)
    
        # log
        if iteration % args.log_interval == 0:
            print(f"Iteration {iteration}, Loss: {loss.item():.4f}")
    
        # Verification
        if iteration % args.val_interval == 0:
            model.eval()
            val_inputs, val_targets = data_loading(val_data, batch_size=args.batch_size, context_length=args.context_length, device=device)
            with torch.no_grad():
                val_logits = model(val_inputs)
                val_loss = cross_entropy(val_logits, val_targets)
            print(f"Iteration {iteration}, Validation Loss: {val_loss.item():.4f}")
            model.train()
    
        # Save checkpoint
        if iteration % args.log_interval == 0:
            checkpoint_file = os.path.join(args.checkpoint_path, f"model_checkpoint_{iteration}.pt")
            save_checkpoint(model, optimizer, iteration, checkpoint_file, loss=loss.item())
            if best_loss > loss:
                best_checkpoint_file = os.path.join(args.checkpoint_path, "best.pt")
                save_checkpoint(model, optimizer, iteration, best_checkpoint_file, loss=loss.item())
                best_loss = loss
            
    print("Training complete!")
    
    
    

if __name__ == "__main__":
    # parser
    parser = argparse.ArgumentParser(description="Train a Transformer language model.")
    parser.add_argument("--train_data_path", type=str, default="/data/CS336-use/Tokens/TinyStories_train_10000_token_ids.npy", help="Path to training data (np.memmap).")
    parser.add_argument("--val_data_path", type=str, default="/data/CS336-use/Tokens/TinyStories_valid_10000_token_ids.npy", help="Path to validation data (np.memmap).")
    
    parser.add_argument("--vocab_size", type=int, default=10000, help="Vocabulary size.")
    parser.add_argument("--context_length", type=int, default=1024, help="Context length.")
    parser.add_argument("--num_layers", type=int, default=12, help="Number of Transformer layers.")
    parser.add_argument("--num_heads", type=int, default=12, help="Number of attention heads.")
    parser.add_argument("--d_ff", type=int, default=3072, help="Feedforward dimension.")
    parser.add_argument("--d_model", type=int, default=768, help="Model dimension.")
    parser.add_argument("--rope_theta", type=float, default=10000.0, help="RoPE theta parameter.")
    
    parser.add_argument("--lr_min", type=float, default=1e-5, help="Minimum learning rate.")
    parser.add_argument("--warmup_iters", type=int, default=1000, help="Number of warmup iterations.")
    parser.add_argument("--cosine_iters", type=int, default=100000, help="Number of cosine annealing iterations.")
    
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--beta1", type=float, default=0.9, help="AdamW beta1.")
    parser.add_argument("--beta2", type=float, default=0.999, help="AdamW beta2.")
    parser.add_argument("--eps", type=float, default=1e-8, help="AdamW epsilon.")
    
    parser.add_argument("--max_iters", type=int, default=120000, help="Maximum number of iterations.")
    
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
    
    parser.add_argument("--max_grad_norm", type=float, default=2.0, help="Maximum gradient norm for clipping.")
    
    parser.add_argument("--log_interval", type=int, default=100, help="Log interval.")
    parser.add_argument("--val_interval", type=int, default=1000, help="Validation interval.")
    
    parser.add_argument("--checkpoint_interval", type=int, default=1000, help="Checkpoint save interval.")
    parser.add_argument("--checkpoint_path", type=str, default="./checkpoints", help="Path to save/load checkpoints.")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint.")
    
    parser.add_argument("--device", type=str, default="cuda", help="Device to train on (e.g., 'cpu', 'cuda:0').")

    args = parser.parse_args()
    main(args)
