"""Low-level fused kernels consumed by `tiny_training_basic.layers`.

模块划分：
- `fa2`         — FlashAttention2（pytorch 参考 + triton 实现）
- `gemm`        — fused matmul（+ bias / + activation / epilogue）
- `fused_norm`  — fused RMSNorm / LayerNorm forward + backward
"""
