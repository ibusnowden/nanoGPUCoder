#!/bin/bash
#SBATCH --job-name=nanochat-r1-ot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=5
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --nodelist=itiger01
#SBATCH --output=r1_ot_%j.out
#SBATCH --error=r1_ot_%j.err
#
# Phase 1 (Cold start): SFT on OpenThoughts (reasoning traces).

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
echo "nanochat R1 Phase 1: OpenThoughts SFT"
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
else
  LATEST_SFT_FILE="$NANOCHAT_BASE_DIR/latest_sft.json"
fi

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
WANDB_RUN_BASE="${WANDB_RUN:-r1_ot}"
export WANDB_PROJECT="${WANDB_PROJECT:-nanochat-r1}"
echo "WANDB_DIR=$WANDB_DIR"
echo "WANDB_RUN_BASE=$WANDB_RUN_BASE (set WANDB_RUN=dummy to disable)"

# HF cache - use home directory cache where datasets are already downloaded
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
# Use project cache to avoid home quota issues
export HF_HOME="${HF_HOME:-/project/inniang/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"
echo "HF_DATASETS_CACHE=$HF_DATASETS_CACHE"

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
SFT_SOURCE="${SFT_SOURCE:-sft}"  # sft=d32, mid=pretrained
MODEL_TAG="${MODEL_TAG:-d32}"
STEP="${STEP:-20832}"
SFT_RECIPE="${SFT_RECIPE:-r1_ot_mixed}"    # r1_ot | r1_ot_mixed | dolci_hf
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-4}"
NUM_STEPS="${NUM_STEPS:-}"  # empty = dataset default

# OpenThoughts config (for r1_ot / r1_ot_mixed recipes)
OT_DATASET="${OT_DATASET:-open-thoughts/OpenThoughts3-1.2M}"
OT_STOP="${OT_STOP:-1000000}"
CHAT_RATIO="${CHAT_RATIO:-0.30}"
GSM8K_STOP="${GSM8K_STOP:--1}"

# Dolci-Think config (for dolci_hf recipe)
DOLCI_STOP="${DOLCI_STOP:-500000}"
DOLCI_STREAMING="${DOLCI_STREAMING:-0}"
DOLCI_STREAM_CACHE="${DOLCI_STREAM_CACHE:-}"

echo ""
echo "Run config:"
echo "  SFT_SOURCE=$SFT_SOURCE"
echo "  MODEL_TAG=$MODEL_TAG"
echo "  STEP=$STEP"
echo "  SFT_RECIPE=$SFT_RECIPE"
if [ "$SFT_RECIPE" = "dolci_hf" ]; then
  echo "  DOLCI_STOP=$DOLCI_STOP (loading from HF cache)"
  echo "  DOLCI_STREAMING=$DOLCI_STREAMING"
  if [ -n "$DOLCI_STREAM_CACHE" ]; then
    echo "  DOLCI_STREAM_CACHE=$DOLCI_STREAM_CACHE"
  fi
else
  echo "  OT_DATASET=$OT_DATASET"
  echo "  OT_STOP=$OT_STOP"
  if [ "$SFT_RECIPE" = "r1_ot_mixed" ]; then
    echo "  CHAT_RATIO=$CHAT_RATIO"
    echo "  GSM8K_STOP=$GSM8K_STOP"
  fi
fi
echo ""

# Pre-cache Dolci streaming subset to avoid DDP timeouts.
if [ "$SFT_RECIPE" = "dolci_hf" ] && [ "$DOLCI_STREAMING" = "1" ]; then
  CACHE_PATH="${DOLCI_STREAM_CACHE:-$HOME/.cache/nanochat/dolci_think_streamed.jsonl}"
  if [ ! -f "$CACHE_PATH" ]; then
    echo "Pre-caching Dolci-Think subset to: $CACHE_PATH"
    python - <<'PY'
import os
from tasks.dolci_think import DolciThink

cache_path = os.environ.get("DOLCI_STREAM_CACHE")
if not cache_path:
    cache_path = os.path.expanduser("~/.cache/nanochat/dolci_think_streamed.jsonl")

stop_env = os.environ.get("DOLCI_STOP", "")
if stop_env and stop_env != "-1":
    stop_val = int(stop_env)
else:
    stop_val = None

DolciThink(
    dataset_id="allenai/Dolci-Think-SFT-7B",
    mode="cot",
    stop=stop_val,
    streaming=True,
    stream_cache_path=cache_path,
)
print(f"Done caching Dolci-Think to {cache_path}")
PY
  else
    echo "Found existing Dolci cache: $CACHE_PATH"
  fi
fi

# Build torchrun args
EXTRA_ARGS=""
if [ "$SFT_RECIPE" = "dolci_hf" ]; then
  EXTRA_ARGS="--dolci_stop=$DOLCI_STOP --dolci_streaming=$DOLCI_STREAMING"
  if [ -n "$DOLCI_STREAM_CACHE" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --dolci_stream_cache=$DOLCI_STREAM_CACHE"
  fi
else
  EXTRA_ARGS="--ot_dataset=$OT_DATASET --ot_stop=$OT_STOP"
  if [ "$SFT_RECIPE" = "r1_ot_mixed" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --chat_ratio=$CHAT_RATIO"
    EXTRA_ARGS="$EXTRA_ARGS --gsm8k_stop=$GSM8K_STOP"
  fi
fi
if [ -n "$NUM_STEPS" ]; then
  EXTRA_ARGS="$EXTRA_ARGS --num_steps=$NUM_STEPS"
fi
if [ -n "$MODEL_TAG" ]; then
  EXTRA_ARGS="$EXTRA_ARGS --model_tag=$MODEL_TAG"
fi
if [ -n "$STEP" ]; then
  EXTRA_ARGS="$EXTRA_ARGS --step=$STEP"
fi

torchrun --standalone --nproc_per_node="$NGPUS" -m scripts.chat_sft -- \
  --run="${WANDB_RUN_BASE}_${SFT_RECIPE}" \
  --source="$SFT_SOURCE" \
  --sft_recipe="$SFT_RECIPE" \
  --device_batch_size="$DEVICE_BATCH_SIZE" \
  $EXTRA_ARGS

write_latest_checkpoint "$NANOCHAT_BASE_DIR/chatsft_checkpoints" "$LATEST_SFT_FILE"
echo "Updated latest SFT pointer: $LATEST_SFT_FILE"

echo ""
echo "Done."
