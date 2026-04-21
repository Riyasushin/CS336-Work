# Tiny-Training

## 项目目标

搭一个能跑起来的 LLM training 框架，覆盖：
- 对 training 本身的理解（model / optim / train loop）
- training / RL 优化加速（kernel 融合、分布式、FlashAttention 等）
- MoE 的经验

学习素材：CS336 Assignment 1/2/4/5 + CSE234 PA2/PA3 的脚手架（PDF、tests、fixtures）。

## 最终形态要求

1. **一个 uv 环境、一个整体仓库** —— 不是每个 assignment 自带 pyproject + venv
2. **模块化** —— 可以一个 stub 一个 stub 实现 + 测试，而不是"整 assignment 完工才能跑"
3. **保留 Adapter 模式** —— CS336 测试是这种形式，延续
4. **把 PA2/PA3 的算子优化、TP、PP、MoE 整合进来** —— 测试在未实现时必须 FAIL（不允许 stub 返回 zeros 造假绿）
5. **原 assignment 目录是只读参考**：整合完成即删除

## 包命名（最终形态）

| 包名                   | 来源                                                  | 内容                                                                   |
| ---------------------- | ----------------------------------------------------- | ---------------------------------------------------------------------- |
| `Tiny-Training-basic`  | assignment1-basics（用户自己的实现，不用 staff 版本） | tokenizer / transformer / optim / train loop                           |
| `Tiny-Training-System` | assignment2-systems + PA2 Part 1 + PA3                | profiling / Triton kernel / FlashAttention / DDP / TP / MoE / 算子优化 |
| `Tiny-Training-Data`   | assignment4-data                                      | 数据过滤 / dedup / 质量分类                                            |
| `Tiny-Training-RL`     | assignment5-alignment                                 | SFT / Expert Iteration / GRPO / DPO / rollout                          |

**canonical `cs336_basics` 源**：用户自己的 Assignment 1 实现。assignment2/4 里自带的 staff `cs336-basics/` 在整合时丢弃。

## 当前状态（2026-04-21）

### 目录
```
Tiny-Training/
├── readme.md                         # 项目目标说明
├── CLAUDE.md                         # 本文件
├── pyproject.toml                    # uv workspace root（虚拟，不打包）
├── uv.lock
├── Tiny-Training-basic/              # 包 tiny-training-basic（含 A1 tests/fixtures，25 tests FAIL@NotImpl）
│   ├── pyproject.toml                # setuptools，package_dir 把 src/ 映射为 tiny_training_basic
│   ├── README.md
│   ├── docs/cs336_assignment1_basics.pdf
│   ├── src/                          # 代码直接在这里（仅 __init__.py，待实现）
│   └── tests/                        # adapters / conftest / test_*.py / fixtures / _snapshots
├── Tiny-Training-System/             # 空壳（待整合 A2 + PA2 + PA3）
│   ├── pyproject.toml
│   ├── README.md
│   └── src/__init__.py
├── Tiny-Training-Data/               # 空壳（待整合 A4）
│   ├── pyproject.toml
│   ├── README.md
│   └── src/__init__.py
├── Tiny-Training-RL/                 # 空壳（待整合 A5）
│   ├── pyproject.toml
│   ├── README.md
│   └── src/__init__.py
├── assignment1-basics/               # 只读参考（待整合完删除）
├── assignment2-systems/              # 只读参考
├── assignment4-data/                 # 只读参考
├── assignment5-alignment/            # 只读参考
└── cse234-w25-PA/
    ├── pa2/                          # 只读参考（Triton matmul / TP+DP）
    └── pa3/                          # 只读参考（MoE TP+EP / 成本分析 / 投机解码）
```

### 进度
- [x] 确定 PA2 通信栈处理方式（NumPy+MPI 保留为参考，torch.distributed 为实际后端）
- [x] 新建顶层 uv workspace + `pyproject.toml`（virtual workspace，4 个 members）
- [x] 搭 `Tiny-Training-*` 四个包骨架（src layout + `uv_build`）
- [x] 迁移 Assignment 1 tests + fixtures + snapshots → `Tiny-Training-basic/tests/`（`uv run pytest` 确认 48 tests 全 FAIL@NotImplementedError）
- [ ] 实现 `tiny-training-basic`：逐 adapter stub 填充（Linear / Embedding / RMSNorm / SwiGLU / Attention / Transformer / AdamW / lr schedule / BPE train / Tokenizer / data batching / serialization）
- [ ] 整合 `assignment2-systems` + PA2 + PA3 → `tiny-training-system`
- [ ] 整合 `assignment4-data` → `tiny-training-data`
- [ ] 整合 `assignment5-alignment` → `tiny-training-rl`
- [ ] 整合完成后删除原 assignment/cse234-w25-PA 目录

### 命名约定（与 workspace 实际一致）

| 文件夹 | 发行名（PyPI） | import 名 |
|---|---|---|
| `Tiny-Training-basic/` | `tiny-training-basic` | `tiny_training_basic` |
| `Tiny-Training-System/` | `tiny-training-system` | `tiny_training_system` |
| `Tiny-Training-Data/` | `tiny-training-data` | `tiny_training_data` |
| `Tiny-Training-RL/` | `tiny-training-rl` | `tiny_training_rl` |

每个包的 `src/` 目录通过 setuptools 的 `package_dir` 被映射为对应 import 名（避免 4 个包的目录都叫 `src` 互相冲突）。Flat，无 `src/<pkg>/` 嵌套。

## 参考 repo 覆盖面清单

### 已覆盖（有参考实现，直接搬/改）
- BPE tokenizer / Transformer / optimizer / train loop → assignment1-basics
- profiling / Triton / FlashAttention / 单机优化 / DDP / 优化器 state 分片（ZeRO-1 级别）→ assignment2-systems
- 数据过滤 / dedup / 质量分类 / scaling-law 数据侧 → assignment4-data
- SFT / Expert Iteration / GRPO（naive PG, reinforce-with-baseline, grpo-clip）/ DPO → assignment5-alignment
- Triton matmul+ReLU+add fused kernel → PA2 Part 1
- 朴素 TP（Megatron 风格 column×row parallel on fc_q/k/v/o）+ DP，2D 并行通信布线 → PA2 Part 2
- MoE TP / MoE EP（all-to-all）+ benchmark → PA3 Part 1
- Llama-7B / DeepSeek-V3 参数/FLOPs/显存 + scaling-law 最优 N/D → PA3 Part 2
- Speculative decoding → PA3 Part 3

### readme.md 目标里列出、但无参考实现（需自己从零设计）
- **Pipeline Parallelism（PP）**：PA2 README 明确排除，PA3、assignment2 也无。候选方案：1F1B / interleaved 1F1B / zero-bubble
- **Ray / 分布式编排层**：RL rollout/train 分离的编排，无参考
- **FSDP / FSDP2 full-shard**：assignment2 只到 ZeRO-1（optimizer state shard），无 full-shard 参考

## 关键决策记录

### PA2 的通信栈
- PA2 使用 NumPy + mpi4py 实现 TP/DP 的通信布线（fc_q/k/v column-parallel, fc_o row-parallel）
- **保留** PA2 的 NumPy+MPI 实现作参考（`myAllreduce` / `myAlltoall` 的从零实现思路）
- 整合进 Tiny-Training 时**替换为 `torch.distributed`（NCCL backend）**，布线逻辑一致

### Tiny-Training-RL vs verl 的定位
- **算法层对齐 verl**：GRPO loss、advantage、response mask 这些按 assignment5 adapter 做全，完成即对齐 verl 算法层
- **编排层简化**：不复刻 Ray + HybridFlow；先解决 policy FSDP 权重 ↔ vLLM rollout 同步这一个核心问题
- **并行层复用**：TP/DP/MoE 通信直接用 `Tiny-Training-System` 的产出
- **暂缓**：RM / adaptive KL / multi-turn agentic / Megatron 后端

## AI 协作约定

- 整合工作在**顶层**进行（创建 `Tiny-Training-*/`、顶层 `pyproject.toml`、顶层 `tests/`）
- 不要在 `assignment*/` 或 `cse234-w25-PA/` 里改代码（它们是只读参考，整合完就删）
- 各 assignment 子目录内有 CS336 TA 风格的 `CLAUDE.md`，那是针对在校学生、约束 AI 不写代码的规则；本项目是用户自己的训练框架工程，不适用于顶层整合工作
- 整合测试时，测试必须在未实现时真 FAIL；不接受 stub 返回 zeros 造成假绿

## 相关记忆文件

在 `~/.claude/projects/-home-rj-WorkingOn-LLM-Training-Tiny-Training/memory/`：
- `project_package_layout.md` — 包命名和整合原则
- `project_reference_scope.md` — 各 repo 覆盖面和缺口
- `project_rl_vs_verl.md` — Tiny-Training-RL 的定位取舍
