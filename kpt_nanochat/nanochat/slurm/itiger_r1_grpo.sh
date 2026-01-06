#!/bin/bash
#SBATCH --job-name=nanochat-r1-grpo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --nodelist=itiger01
#SBATCH --output=r1_grpo_%j.out
#SBATCH --error=r1_grpo_%j.err
#
# Phase 2: Mixed-task GRPO (JustRL recipe defaults).

set -euo pipefail

# Resolve repo root even when sbatch stages the script in /var/spool/slurmd
find_nanochat_root() {
  local base="$1"
  local candidate
  for candidate in \
    "$base" \
    "$base/.." \
    "$base/nanogpu/kpt_nanochat/nanochat" \
    "$base/kpt_nanochat/nanochat" \
    "$base/nanochat"; do
    if [ -d "$candidate/scripts" ]; then
      (cd "$candidate" && pwd)
      return 0
    fi
  done
  return 1
}

if [ -z "${NANOCHAT_ROOT:-}" ] && [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  NANOCHAT_ROOT="$(find_nanochat_root "$SLURM_SUBMIT_DIR" || true)"
fi
if [ -z "${NANOCHAT_ROOT:-}" ] && [ -n "${PWD:-}" ]; then
  NANOCHAT_ROOT="$(find_nanochat_root "$PWD" || true)"
fi
if [ -z "${NANOCHAT_ROOT:-}" ]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  NANOCHAT_ROOT="$(find_nanochat_root "$SCRIPT_DIR" || true)"
  if [ -z "${NANOCHAT_ROOT:-}" ]; then
    NANOCHAT_ROOT="$SCRIPT_DIR"
  fi
fi
if [ ! -d "$NANOCHAT_ROOT/scripts" ]; then
  echo "ERROR: could not find nanochat repo (NANOCHAT_ROOT=$NANOCHAT_ROOT); set NANOCHAT_ROOT to the path containing scripts/."
  exit 1
fi
cd "$NANOCHAT_ROOT"
export PYTHONPATH="$NANOCHAT_ROOT:${PYTHONPATH:-}"
mkdir -p logs

echo "========================================"
echo "nanochat R1 Phase 2: GRPO (mixed tasks)"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID:-}"
echo "Node: ${SLURM_NODELIST:-$(hostname)}"
echo "PWD: $(pwd)"
echo "========================================"

# Activate conda env: moe
CONDA_BASE="${CONDA_BASE:-}"
if [ -z "$CONDA_BASE" ] && command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
elif [ -z "$CONDA_BASE" ] && [ -d "$HOME/miniconda3" ]; then
  CONDA_BASE="$HOME/miniconda3"
elif [ -z "$CONDA_BASE" ] && [ -d "$HOME/anaconda3" ]; then
  CONDA_BASE="$HOME/anaconda3"
fi
if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate moe
  echo "Activated conda env: ${CONDA_DEFAULT_ENV:-}"
else
  echo "WARNING: conda not found; proceeding with current python"
fi

echo ""
echo "Environment:"
echo "  Python: $(which python)"
python -c "import torch; print('  torch:', torch.__version__); print('  cuda:', torch.version.cuda); print('  gpus:', torch.cuda.device_count())"
echo ""

# Runtime / NCCL knobs (single-node)
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_BLOCKING_WAIT="${NCCL_BLOCKING_WAIT:-1}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-1200}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-2}"

# Cache base
PROJECT_CACHE_ROOT="${PROJECT_CACHE_ROOT:-/project/inniang/.cache}"
PROJECT_CONFIG_ROOT="${PROJECT_CONFIG_ROOT:-${PROJECT_CACHE_ROOT%/.cache}/.config}"
CACHE_ROOT=""
CONFIG_ROOT=""
if [ -n "$PROJECT_CACHE_ROOT" ] && mkdir -p "$PROJECT_CACHE_ROOT" 2>/dev/null; then
  CACHE_ROOT="$PROJECT_CACHE_ROOT"
  CONFIG_ROOT="$PROJECT_CONFIG_ROOT"
elif [ -n "${SCRATCH:-}" ]; then
  CACHE_ROOT="$SCRATCH"
  CONFIG_ROOT="$SCRATCH/.config"
elif [ "${ALLOW_SLURM_TMPDIR:-0}" = "1" ] && [ -n "${SLURM_TMPDIR:-}" ]; then
  CACHE_ROOT="$SLURM_TMPDIR"
  CONFIG_ROOT="$SLURM_TMPDIR/.config"
else
  CACHE_ROOT="$HOME/.cache"
  CONFIG_ROOT="$HOME/.config"
fi
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CACHE_ROOT}"
export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-$CONFIG_ROOT}"
mkdir -p "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME"

if [ -z "${NANOCHAT_BASE_DIR:-}" ]; then
  export NANOCHAT_BASE_DIR="$CACHE_ROOT/nanochat"
fi
mkdir -p "$NANOCHAT_BASE_DIR"
echo "NANOCHAT_BASE_DIR=$NANOCHAT_BASE_DIR"
TOKENIZER_CHOICE="${NANOCHAT_TOKENIZER:-}"
if [ -z "$TOKENIZER_CHOICE" ]; then
  if [ -d "$NANOCHAT_BASE_DIR/tokenizer_qwen25" ]; then
    TOKENIZER_CHOICE="qwen25"
  else
    TOKENIZER_CHOICE="rustbpe"
  fi
fi
if [ "$TOKENIZER_CHOICE" = "qwen25" ] && [ ! -d "$NANOCHAT_BASE_DIR/tokenizer_qwen25" ]; then
  echo "WARNING: tokenizer_qwen25 not found; falling back to rustbpe."
  TOKENIZER_CHOICE="rustbpe"
fi
if [ "$TOKENIZER_CHOICE" = "rustbpe" ] && [ ! -d "$NANOCHAT_BASE_DIR/tokenizer" ]; then
  if [ -d "$HOME/.cache/nanochat/tokenizer" ]; then
    ln -s "$HOME/.cache/nanochat/tokenizer" "$NANOCHAT_BASE_DIR/tokenizer" || true
  fi
fi
export NANOCHAT_TOKENIZER="$TOKENIZER_CHOICE"
echo "NANOCHAT_TOKENIZER=$NANOCHAT_TOKENIZER"

MID_CHECKPOINT_SRC="${MID_CHECKPOINT_SRC:-/home/inniang/.cache/nanochat/mid_checkpoints}"
MID_CHECKPOINT_DST="$NANOCHAT_BASE_DIR/mid_checkpoints"
if [ ! -e "$MID_CHECKPOINT_DST" ] && [ -d "$MID_CHECKPOINT_SRC" ]; then
  if ln -s "$MID_CHECKPOINT_SRC" "$MID_CHECKPOINT_DST"; then
    echo "Linked mid checkpoints: $MID_CHECKPOINT_DST -> $MID_CHECKPOINT_SRC"
  else
    echo "WARNING: failed to link mid checkpoints from $MID_CHECKPOINT_SRC"
  fi
fi

if [ -n "${NANOCHAT_RUN_ID:-}" ]; then
  LATEST_SFT_FILE="$NANOCHAT_BASE_DIR/latest_sft_${NANOCHAT_RUN_ID}.json"
  LATEST_RL_FILE="$NANOCHAT_BASE_DIR/latest_rl_${NANOCHAT_RUN_ID}.json"
else
  LATEST_SFT_FILE="$NANOCHAT_BASE_DIR/latest_sft.json"
  LATEST_RL_FILE="$NANOCHAT_BASE_DIR/latest_rl.json"
fi

read_latest_checkpoint() {
  local json_path="$1"
  python - "$json_path" <<'PY'
import json
import sys

data = json.load(open(sys.argv[1]))
print(data.get("model_tag", ""), data.get("step", ""))
PY
}

write_latest_checkpoint() {
  local root="$1"
  local out_path="$2"
  python - "$root" "$out_path" <<'PY'
import glob
import json
import os
import re
import sys

root, out_path = sys.argv[1], sys.argv[2]
if not os.path.isdir(root):
    raise SystemExit(f"Missing checkpoint root: {root}")
tags = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
if not tags:
    raise SystemExit(f"No model tags in {root}")
candidates = []
for tag in tags:
    match = re.match(r"d(\\d+)$", tag)
    if match:
        candidates.append((int(match.group(1)), tag))
if candidates:
    model_tag = sorted(candidates)[-1][1]
else:
    model_tag = max(tags, key=lambda t: os.path.getmtime(os.path.join(root, t)))
ckpt_dir = os.path.join(root, model_tag)
steps = []
for path in glob.glob(os.path.join(ckpt_dir, "model_*.pt")):
    match = re.search(r"model_(\\d+)\\.pt$", path)
    if match:
        steps.append(int(match.group(1)))
if not steps:
    raise SystemExit(f"No checkpoints in {ckpt_dir}")
step = max(steps)
payload = {"model_tag": model_tag, "step": step, "checkpoint_dir": ckpt_dir}
with open(out_path, "w") as handle:
    json.dump(payload, handle)
print(out_path)
PY
}

# wandb
export WANDB_DIR="${WANDB_DIR:-$NANOCHAT_BASE_DIR/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$XDG_CACHE_HOME/wandb}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-$XDG_CONFIG_HOME/wandb}"
export WANDB_LOG_DIR="${WANDB_LOG_DIR:-$WANDB_CACHE_DIR/logs}"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" "$WANDB_LOG_DIR"
WANDB_RUN_BASE="${WANDB_RUN:-r1_grpo}"
export WANDB_PROJECT="${WANDB_PROJECT:-nanochat-r1}"
echo "WANDB_DIR=$WANDB_DIR"
echo "WANDB_RUN_BASE=$WANDB_RUN_BASE (set WANDB_DISABLED=true to disable)"

# HF cache
if [ -z "${HF_TOKEN:-}" ]; then
  for token_path in "${HF_TOKEN_PATH:-}" "$HOME/.cache/huggingface/token" "$HOME/.huggingface/token"; do
    if [ -n "$token_path" ] && [ -f "$token_path" ]; then
      HF_TOKEN="$(head -n 1 "$token_path" | tr -d '\r\n')"
      break
    fi
  done
fi
export HF_TOKEN="${HF_TOKEN:-}"
export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-$HF_TOKEN}"
# Use /project/inniang/.cache/hf to match warmup script cache location
export HF_HOME="${HF_HOME:-/project/inniang/.cache/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"

# Disable offline mode - the DDP-safe loader in tasks/common.py handles caching
unset HF_HUB_OFFLINE
unset HF_DATASETS_OFFLINE

# GPU count
NGPUS="${SLURM_GPUS_ON_NODE:-}"
if ! [[ "$NGPUS" =~ ^[0-9]+$ ]]; then
  NGPUS="$(python -c 'import torch; print(torch.cuda.device_count())')"
fi
if [ -z "$NGPUS" ] || [ "$NGPUS" -le 0 ]; then
  NGPUS="4"
fi
echo "Using NGPUS=$NGPUS"

# Config
GRPO_SOURCE="${GRPO_SOURCE:-sft}"
REF_SOURCE="${REF_SOURCE:-sft}"
# Note: humaneval is eval-only (no training split), so we use mbpp for code training
TASK_MIX="${TASK_MIX:-dolci:1.0,gsm8k:0.45,math:0.20,mmlu_science:0.10,mbpp:0.25}"
DOLCI_DATASET_ID="${DOLCI_DATASET_ID:-allenai/Dolci-Think-RL-32B}"
DOLCI_SPLIT="${DOLCI_SPLIT:-train}"
DOLCI_MODE="${DOLCI_MODE:-cot}"
DOLCI_STOP="${DOLCI_STOP:--1}"
DOLCI_STREAMING="${DOLCI_STREAMING:-0}"
DOLCI_STREAM_CACHE="${DOLCI_STREAM_CACHE:-}"
NUM_STEPS="${NUM_STEPS:-500}"
TOTAL_EXAMPLES="${TOTAL_EXAMPLES:-}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
PPO_MINIBATCH_SIZE="${PPO_MINIBATCH_SIZE:-64}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-1}"
NUM_SAMPLES="${NUM_SAMPLES:-4}"
EXAMPLES_PER_STEP="${EXAMPLES_PER_STEP:-}"
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-1024}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_K="${TOP_K:-50}"
KL_COEF="${KL_COEF:-0.0}"
KL_MAX_THRESHOLD="${KL_MAX_THRESHOLD:-50.0}"  # Warn if KL exceeds this
REWARD_SCALE="${REWARD_SCALE:-1.0}"
REWARD_MODE="${REWARD_MODE:-dapo}"
FORMAT_HINT_MODE="${FORMAT_HINT_MODE:-eval}"
GROUP_DYNAMIC_SAMPLING="${GROUP_DYNAMIC_SAMPLING:-0}"
GROUP_DYNAMIC_SAMPLING_MAX_TRIES="${GROUP_DYNAMIC_SAMPLING_MAX_TRIES:-50}"
USE_BEST_OF_N="${USE_BEST_OF_N:-0}"
SAVE_EVERY="${SAVE_EVERY:-100}"
# Temperature schedule: fixed by default
TEMP_START="${TEMP_START:-1.0}"
TEMP_END="${TEMP_END:-1.0}"
TEMP_SCHEDULE="${TEMP_SCHEDULE:-none}"  # "linear", "cosine", or "none"
# Length penalty (DAPO overlong shaping)
LENGTH_PENALTY_MODE="${LENGTH_PENALTY_MODE:-dapo}"
LENGTH_PENALTY_COEF="${LENGTH_PENALTY_COEF:-1.0}"
LENGTH_PENALTY_TARGET="${LENGTH_PENALTY_TARGET:-$MAX_NEW_TOKENS}"  # Penalty starts above this
LENGTH_PENALTY_FLOOR="${LENGTH_PENALTY_FLOOR:-0.0}"
# PPO/GRPO clipping ratio range
CLIP_RATIO_LOW="${CLIP_RATIO_LOW:-0.8}"
CLIP_RATIO_HIGH="${CLIP_RATIO_HIGH:-1.28}"
# LR schedule
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
LR_SCHEDULE="${LR_SCHEDULE:-constant}"
GRPO_LR_SCALE="${GRPO_LR_SCALE:-1.0}"
MODEL_TAG="${MODEL_TAG:-}"
STEP="${STEP:-}"
if [ -f "$LATEST_SFT_FILE" ]; then
  read -r LATEST_MODEL_TAG LATEST_STEP < <(read_latest_checkpoint "$LATEST_SFT_FILE")
if [ -z "$MODEL_TAG" ]; then
  MODEL_TAG="$LATEST_MODEL_TAG"
fi
if [ -z "$STEP" ]; then
  STEP="$LATEST_STEP"
fi
fi
REF_MODEL_TAG="${REF_MODEL_TAG:-$MODEL_TAG}"
REF_STEP="${REF_STEP:-$STEP}"

if [ -z "$EXAMPLES_PER_STEP" ]; then
  if [ $((TRAIN_BATCH_SIZE % NUM_SAMPLES)) -ne 0 ]; then
    echo "ERROR: TRAIN_BATCH_SIZE ($TRAIN_BATCH_SIZE) must be divisible by NUM_SAMPLES ($NUM_SAMPLES)."
    exit 1
  fi
  EXAMPLES_PER_STEP=$((TRAIN_BATCH_SIZE / NUM_SAMPLES))
fi

if [ $((NUM_SAMPLES % DEVICE_BATCH_SIZE)) -ne 0 ]; then
  echo "ERROR: NUM_SAMPLES ($NUM_SAMPLES) must be divisible by DEVICE_BATCH_SIZE ($DEVICE_BATCH_SIZE)."
  exit 1
fi
if [ $((EXAMPLES_PER_STEP % NGPUS)) -ne 0 ]; then
  echo "ERROR: EXAMPLES_PER_STEP ($EXAMPLES_PER_STEP) must be divisible by NGPUS ($NGPUS)."
  exit 1
fi

MODEL_ARGS=()
if [ -n "$MODEL_TAG" ]; then
  MODEL_ARGS+=(--model_tag="$MODEL_TAG")
fi
if [ -n "$STEP" ]; then
  MODEL_ARGS+=(--step="$STEP")
fi

REF_ARGS=(--ref_source="$REF_SOURCE")
if [ -n "$REF_MODEL_TAG" ]; then
  REF_ARGS+=(--ref_model_tag="$REF_MODEL_TAG")
fi
if [ -n "$REF_STEP" ]; then
  REF_ARGS+=(--ref_step="$REF_STEP")
fi

GRPO_ARGS=(
  --task_mix="$TASK_MIX"
  --dolci_dataset_id="$DOLCI_DATASET_ID"
  --dolci_split="$DOLCI_SPLIT"
  --dolci_mode="$DOLCI_MODE"
  --dolci_stop="$DOLCI_STOP"
  --dolci_streaming="$DOLCI_STREAMING"
  --dolci_stream_cache="$DOLCI_STREAM_CACHE"
  --num_steps="$NUM_STEPS"
  --device_batch_size="$DEVICE_BATCH_SIZE"
  --ppo_minibatch_size="$PPO_MINIBATCH_SIZE"
  --examples_per_step="$EXAMPLES_PER_STEP"
  --num_samples="$NUM_SAMPLES"
  --max_prompt_tokens="$MAX_PROMPT_TOKENS"
  --max_new_tokens="$MAX_NEW_TOKENS"
  --temperature="$TEMPERATURE"
  --top_k="$TOP_K"
  --dtype="float32"
  --kl_coef="$KL_COEF"
  --kl_max_threshold="$KL_MAX_THRESHOLD"
  --reward_scale="$REWARD_SCALE"
  --reward_mode="$REWARD_MODE"
  --format_hint_mode="$FORMAT_HINT_MODE"
  --group_dynamic_sampling="$GROUP_DYNAMIC_SAMPLING"
  --group_dynamic_sampling_max_tries="$GROUP_DYNAMIC_SAMPLING_MAX_TRIES"
  --use_best_of_n="$USE_BEST_OF_N"
  --save_every="$SAVE_EVERY"
  --temp_start="$TEMP_START"
  --temp_end="$TEMP_END"
  --temp_schedule="$TEMP_SCHEDULE"
  --length_penalty_mode="$LENGTH_PENALTY_MODE"
  --length_penalty_coef="$LENGTH_PENALTY_COEF"
  --length_penalty_target="$LENGTH_PENALTY_TARGET"
  --length_penalty_floor="$LENGTH_PENALTY_FLOOR"
  --clip_ratio_low="$CLIP_RATIO_LOW"
  --clip_ratio_high="$CLIP_RATIO_HIGH"
  --learning_rate="$LEARNING_RATE"
  --lr_schedule="$LR_SCHEDULE"
  --grpo_lr_scale="$GRPO_LR_SCALE"
)
if [ -n "$TOTAL_EXAMPLES" ]; then
  GRPO_ARGS+=(--total_examples="$TOTAL_EXAMPLES")
fi

echo ""
echo "Run config:"
echo "  SOURCE=$GRPO_SOURCE REF_SOURCE=$REF_SOURCE MODEL_TAG=${MODEL_TAG:-auto} STEP=${STEP:-latest}"
echo "  TASK_MIX=$TASK_MIX"
echo "  DOLCI_DATASET_ID=$DOLCI_DATASET_ID DOLCI_SPLIT=$DOLCI_SPLIT"
echo "  DOLCI_MODE=$DOLCI_MODE DOLCI_STOP=$DOLCI_STOP"
echo "  DOLCI_STREAMING=$DOLCI_STREAMING DOLCI_STREAM_CACHE=${DOLCI_STREAM_CACHE:-none}"
echo "  NUM_STEPS=$NUM_STEPS TOTAL_EXAMPLES=${TOTAL_EXAMPLES:-none}"
echo "  TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE PPO_MINIBATCH_SIZE=$PPO_MINIBATCH_SIZE"
echo "  DEVICE_BATCH_SIZE=$DEVICE_BATCH_SIZE EXAMPLES_PER_STEP=$EXAMPLES_PER_STEP NUM_SAMPLES=$NUM_SAMPLES"
echo "  MAX_PROMPT_TOKENS=$MAX_PROMPT_TOKENS MAX_NEW_TOKENS=$MAX_NEW_TOKENS"
echo "  TEMPERATURE=$TEMPERATURE TOP_K=$TOP_K"
echo "  KL_COEF=$KL_COEF KL_MAX_THRESHOLD=$KL_MAX_THRESHOLD"
echo "  REWARD_MODE=$REWARD_MODE GROUP_DYNAMIC_SAMPLING=$GROUP_DYNAMIC_SAMPLING"
echo "  FORMAT_HINT_MODE=$FORMAT_HINT_MODE"
echo "  GROUP_DYNAMIC_SAMPLING_MAX_TRIES=$GROUP_DYNAMIC_SAMPLING_MAX_TRIES"
echo "  CLIP_RATIO=[$CLIP_RATIO_LOW,$CLIP_RATIO_HIGH]"
echo "  LR=$LEARNING_RATE LR_SCHEDULE=$LR_SCHEDULE GRPO_LR_SCALE=$GRPO_LR_SCALE"
echo "  TEMP_SCHEDULE=$TEMP_SCHEDULE ($TEMP_START -> $TEMP_END) USE_BEST_OF_N=$USE_BEST_OF_N"
echo "  LENGTH_PENALTY_MODE=$LENGTH_PENALTY_MODE COEF=$LENGTH_PENALTY_COEF"
echo "  LENGTH_PENALTY_TARGET=$LENGTH_PENALTY_TARGET LENGTH_PENALTY_FLOOR=$LENGTH_PENALTY_FLOOR"
echo ""

torchrun --standalone --nproc_per_node="$NGPUS" -m scripts.chat_grpo -- \
  --run="${WANDB_RUN_BASE}_grpo" \
  --source="$GRPO_SOURCE" \
  "${MODEL_ARGS[@]}" \
  "${REF_ARGS[@]}" \
  "${GRPO_ARGS[@]}"

write_latest_checkpoint "$NANOCHAT_BASE_DIR/chatrl_checkpoints" "$LATEST_RL_FILE"
echo "Updated latest RL pointer: $LATEST_RL_FILE"

echo ""
echo "Done."
