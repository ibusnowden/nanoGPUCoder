#!/bin/bash
#SBATCH --job-name=nanochat-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=smoke_%j.out
#SBATCH --error=smoke_%j.err
#
# iTiger single-node smoke test (8x H100).
# Runs a tiny dense run + tiny MoE run on synthetic tokens (no dataset/tokenizer required).
#
# Usage:
#   sbatch slurm/itiger_smoke_base_train.sbatch
#   # override partition/account as needed, e.g.
#   # sbatch -p <partition> -A <account> slurm/itiger_smoke_base_train.sbatch
#

set -euo pipefail

NANOCHAT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$NANOCHAT_ROOT"
mkdir -p logs

echo "========================================"
echo "nanochat smoke (base_train)"
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

# Put nanochat cache/checkpoints on fast storage if available
if [ -z "${NANOCHAT_BASE_DIR:-}" ]; then
  if [ -n "${SCRATCH:-}" ]; then
    export NANOCHAT_BASE_DIR="$SCRATCH/nanochat"
  elif [ -n "${SLURM_TMPDIR:-}" ]; then
    export NANOCHAT_BASE_DIR="$SLURM_TMPDIR/nanochat"
  else
    export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
  fi
fi
mkdir -p "$NANOCHAT_BASE_DIR"
echo "NANOCHAT_BASE_DIR=$NANOCHAT_BASE_DIR"

# wandb: keep artifacts/logs off $HOME when possible
export WANDB_DIR="${WANDB_DIR:-$NANOCHAT_BASE_DIR/wandb}"
mkdir -p "$WANDB_DIR"
WANDB_RUN_BASE="${WANDB_RUN:-smoke_${SLURM_JOB_ID:-manual}}"
echo "WANDB_DIR=$WANDB_DIR"
echo "WANDB_RUN_BASE=$WANDB_RUN_BASE (set WANDB_RUN=dummy to disable)"

# GPU count
NGPUS="${SLURM_GPUS_ON_NODE:-}"
if ! [[ "$NGPUS" =~ ^[0-9]+$ ]]; then
  NGPUS="$(python -c 'import torch; print(torch.cuda.device_count())')"
fi
if [ -z "$NGPUS" ] || [ "$NGPUS" -le 0 ]; then
  NGPUS="8"
fi
echo "Using NGPUS=$NGPUS"

# Keep the math explicit so divisibility constraints are always met:
# world_tokens_per_fwdbwd = device_batch_size * max_seq_len * world_size
MAX_SEQ_LEN=256
DEVICE_BATCH_SIZE=1
TOTAL_BATCH_SIZE=$((DEVICE_BATCH_SIZE * MAX_SEQ_LEN * NGPUS))
EVAL_TOKENS=$TOTAL_BATCH_SIZE
NUM_ITERS=5

COMMON_ARGS=(
  --run=$WANDB_RUN_BASE
  --synthetic_data=1
  --synthetic_vocab_size=65536
  --skip_core_metric=1
  --skip_sampling=1
  --architecture_style=qwen25_small
  --depth=2
  --max_seq_len=$MAX_SEQ_LEN
  --device_batch_size=$DEVICE_BATCH_SIZE
  --total_batch_size=$TOTAL_BATCH_SIZE
  --eval_tokens=$EVAL_TOKENS
  --num_iterations=$NUM_ITERS
)

echo ""
echo "---- Dense smoke ----"
torchrun --standalone --nproc_per_node="$NGPUS" -m scripts.base_train -- \
  "${COMMON_ARGS[@]}" \
  --run="${WANDB_RUN_BASE}_dense" \
  --model_tag=smoke_dense

echo ""
echo "---- MoE smoke ----"
torchrun --standalone --nproc_per_node="$NGPUS" -m scripts.base_train -- \
  "${COMMON_ARGS[@]}" \
  --run="${WANDB_RUN_BASE}_moe" \
  --model_tag=smoke_moe \
  --moe_num_experts=4 \
  --moe_top_k=2 \
  --moe_layer_start=0 \
  --moe_layer_end=-1 \
  --moe_layer_stride=1

echo ""
echo "Done."
