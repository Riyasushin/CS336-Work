#!/usr/bin/env bash
# §9 Leaderboard 4-hour pipeline.
#
# 三段：
#   (1) curriculum 过滤（~20 min）：用 base policy 跑一次 train.jsonl，按正确
#       率 0.05-0.7 区间留下 ~30-40% 的题（MEDIUM 档）作为 GRPO 训练池
#   (2) GRPO 训练（~3.0 hr）：在过滤后的 train 上跑 off-policy GRPO-Clip
#       + group-norm advantage（含 std）+ Dr.GRPO 长度归一化（fixed K）
#   (3) 全 5K MATH validation 评测（~30 min）：PDF 强制 r1_zero prompt + T=1.0
#       + max_tokens=1024 + r1_zero_reward_fn
#
# 总时间预算 ≤ 4 hours on 2×H100。
#
# Usage:
#   bash scripts/leaderboard.sh \
#     /data/a5-alignment/models/Qwen2.5-Math-1.5B \
#     /data/a5-alignment/MATH \
#     /data/runs/leaderboard

set -e
MODEL="${1:?path to base model}"
DATA_DIR="${2:?dir containing train.jsonl + validation.jsonl}"
RUN_DIR="${3:?output dir}"

TRAIN_FULL="${DATA_DIR}/train.jsonl"
TRAIN_FILTERED="${RUN_DIR}/train_curriculum.jsonl"
VAL_DATA="${DATA_DIR}/validation.jsonl"

mkdir -p "${RUN_DIR}"

# ---------- 1. curriculum 过滤 ----------
echo "=== STEP 1: curriculum filter ==="
uv run python scripts/filter_train_curriculum.py \
  --model "${MODEL}" \
  --questions-data "${TRAIN_FULL}" \
  --output "${TRAIN_FILTERED}" \
  --rollouts-per-question 8 \
  --keep-low 0.05 --keep-high 0.7 \
  --vllm-device cuda:1

# ---------- 2. GRPO 训练 ----------
# 选型说明（详见 docs/leaderboard.md 决策表）：
#   - off-policy GRPO-Clip: 4 epochs / rollout, train_batch=128
#     → 一次 rollout 摊销 4 个 grad step，wall-clock 加速 ~3×
#   - learning_rate 5e-6: off-policy 必须降 lr 防 ratio 爆
#   - cliprange 0.2: PPO 标准
#   - rollout_batch_size 256, group_size 16: 每题 16 rollout，advantage 估计稳
#   - length-norm-method fixed: Dr.GRPO 防 length collapse
#   - use_std_normalization True: 默认；某些题 group std=0 时 advantage_eps 兜底
echo "=== STEP 2: GRPO training ==="
uv run python scripts/train_grpo.py \
  --model "${MODEL}" \
  --questions-data "${TRAIN_FILTERED}" \
  --val-data "${VAL_DATA}" \
  --output-dir "${RUN_DIR}/grpo" \
  --n-grpo-steps 200 \
  --learning-rate 5e-6 \
  --rollout-batch-size 256 \
  --group-size 16 \
  --train-batch-size 128 \
  --gradient-accumulation-steps 64 \
  --epochs-per-rollout-batch 4 \
  --loss-type grpo_clip \
  --cliprange 0.2 \
  --use-std-normalization \
  --length-norm-method fixed \
  --reward-fn-name r1_zero \
  --max-seq-len 1024 \
  --eval-every 20 \
  --eval-num 256 \
  --save-every 50 \
  --policy-device cuda:0 --vllm-device cuda:1 \
  --vllm-gpu-mem 0.85 \
  --grad-clip 1.0 \
  --wandb-project tiny-training-rl --wandb-run-name leaderboard

# ---------- 3. 全 5K validation 评测（PDF 强制设定）----------
# 找 grpo 训练出的最后一个 ckpt
FINAL_CKPT="$(ls -d "${RUN_DIR}/grpo"/step_* | sort -V | tail -1)"
echo "=== STEP 3: full 5K validation on ${FINAL_CKPT} ==="
uv run python scripts/eval_zeroshot.py \
  --model "${FINAL_CKPT}" \
  --dataset "${VAL_DATA}" \
  --out "${RUN_DIR}/final_eval" \
  --backend vllm \
  --temperature 1.0 --top-p 1.0 --max-tokens 1024 \
  --gpu-memory-utilization 0.85

echo "=== DONE ==="
echo "Final accuracy: $(jq .accuracy "${RUN_DIR}/final_eval/summary.json")"
