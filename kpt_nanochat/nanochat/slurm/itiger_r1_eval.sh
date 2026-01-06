#!/bin/bash
#SBATCH --job-name=nanochat-r1-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=6
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=r1_eval_%j.out
#SBATCH --error=r1_eval_%j.err
#
# Phase 4: Downstream evaluation after Phase 3.

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  NANOCHAT_ROOT="$SLURM_SUBMIT_DIR"
else
  NANOCHAT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
PROJECT_ROOT="$(cd "$NANOCHAT_ROOT/../../.." && pwd)"
cd "$NANOCHAT_ROOT"
mkdir -p logs

echo "========================================"
echo "nanochat R1 Phase 4: Downstream evaluation"
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
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-2}"

# Cache base
if [ -z "${NANOCHAT_BASE_DIR:-}" ]; then
  if [ -d "$PROJECT_ROOT/.cache/nanochat" ]; then
    export NANOCHAT_BASE_DIR="$PROJECT_ROOT/.cache/nanochat"
  elif [ -n "${SCRATCH:-}" ]; then
    export NANOCHAT_BASE_DIR="$SCRATCH/nanochat"
  elif [ "${ALLOW_SLURM_TMPDIR:-0}" = "1" ] && [ -n "${SLURM_TMPDIR:-}" ]; then
    export NANOCHAT_BASE_DIR="$SLURM_TMPDIR/nanochat"
  else
    export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
  fi
fi
mkdir -p "$NANOCHAT_BASE_DIR"
echo "NANOCHAT_BASE_DIR=$NANOCHAT_BASE_DIR"

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

# wandb
export WANDB_DIR="${WANDB_DIR:-$NANOCHAT_BASE_DIR/wandb}"
mkdir -p "$WANDB_DIR"
export WANDB_PROJECT="${WANDB_PROJECT:-nanochat-r1}"

# HF cache
export HF_TOKEN="${HF_TOKEN:-}"
export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-$HF_TOKEN}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE"

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
EVAL_SOURCE="${EVAL_SOURCE:-rl}"
MODEL_TAG="${MODEL_TAG:-d28}"
STEP="${STEP:-200}"
TASKS="${TASKS:-ARC-Easy|ARC-Challenge|MMLU|GSM8K|MATH|BBH|HumanEval|MBPP}"
DTYPE="${DTYPE:-bfloat16}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_K="${TOP_K:-50}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_PROBLEMS="${MAX_PROBLEMS:-}"

if [ -z "$MODEL_TAG" ] || [ -z "$STEP" ]; then
  if [ "$EVAL_SOURCE" = "rl" ] && [ -f "$LATEST_RL_FILE" ]; then
    read -r LATEST_MODEL_TAG LATEST_STEP < <(read_latest_checkpoint "$LATEST_RL_FILE")
  elif [ -f "$LATEST_SFT_FILE" ]; then
    read -r LATEST_MODEL_TAG LATEST_STEP < <(read_latest_checkpoint "$LATEST_SFT_FILE")
  fi
  if [ -z "$MODEL_TAG" ]; then
    MODEL_TAG="$LATEST_MODEL_TAG"
  fi
  if [ -z "$STEP" ]; then
    STEP="$LATEST_STEP"
  fi
fi

if [ -z "${WANDB_RUN:-}" ]; then
  WANDB_RUN="r1_eval_${EVAL_SOURCE}"
  if [ -n "$MODEL_TAG" ]; then
    WANDB_RUN+="_${MODEL_TAG}"
  fi
  if [ -n "$STEP" ]; then
    WANDB_RUN+="_${STEP}"
  fi
  export WANDB_RUN
fi

echo ""
echo "Run config:"
echo "  SOURCE=$EVAL_SOURCE MODEL_TAG=${MODEL_TAG:-auto} STEP=${STEP:-latest}"
echo "  TASKS=$TASKS"
echo "  NUM_SAMPLES=$NUM_SAMPLES MAX_NEW_TOKENS=$MAX_NEW_TOKENS TEMPERATURE=$TEMPERATURE"
echo ""

EVAL_ARGS=(
  --source="$EVAL_SOURCE"
  --dtype="$DTYPE"
  --temperature="$TEMPERATURE"
  --max-new-tokens="$MAX_NEW_TOKENS"
  --num-samples="$NUM_SAMPLES"
  --top-k="$TOP_K"
  --batch-size="$BATCH_SIZE"
)
if [ -n "$TASKS" ]; then
  EVAL_ARGS+=(--task-name="$TASKS")
fi
if [ -n "$MODEL_TAG" ]; then
  EVAL_ARGS+=(--model-tag="$MODEL_TAG")
fi
if [ -n "$STEP" ]; then
  EVAL_ARGS+=(--step="$STEP")
fi
if [ -n "$MAX_PROBLEMS" ]; then
  EVAL_ARGS+=(--max-problems="$MAX_PROBLEMS")
fi

torchrun --standalone --nproc_per_node="$NGPUS" -m scripts.chat_eval -- "${EVAL_ARGS[@]}"

echo ""
echo "Done."
