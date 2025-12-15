#!/bin/bash
#SBATCH --job-name=nanochat-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=eval_%j.out
#SBATCH --error=eval_%j.err
#
# Run chat_eval on a chosen checkpoint.
# Example:
#   sbatch slurm/itiger_eval.sh
#   sbatch --export=SOURCE=mid,MODEL_TAG=d4,STEP=000020,TASKS=\"ARC-Easy,ARC-Challenge,GSM8K\" slurm/itiger_eval.sh

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  NANOCHAT_ROOT="$SLURM_SUBMIT_DIR"
else
  NANOCHAT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$NANOCHAT_ROOT"

SOURCE="${SOURCE:-sft}"       # base|mid|sft|rl
MODEL_TAG="${MODEL_TAG:-d4}"  # checkpoint tag, e.g., d4
STEP_OPT=()
if [ -n "${STEP:-}" ]; then
  STEP_OPT=(--step="$STEP")
fi
TASKS_RAW="${TASKS:-ARC-Easy,ARC-Challenge,GSM8K}"
IFS=',' read -r -a TASK_ARR <<< "$TASKS_RAW"
TASK_FLAGS=()
for t in "${TASK_ARR[@]}"; do
  TASK_FLAGS+=(-a "$t")
done

echo "========================================"
echo "nanochat eval"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID:-}"
echo "Node: ${SLURM_NODELIST:-$(hostname)}"
echo "PWD: $(pwd)"
echo "SOURCE=$SOURCE MODEL_TAG=$MODEL_TAG STEP=${STEP:-latest}"
echo "TASKS=${TASKS_RAW}"
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

# Runtime / NCCL knobs (single-node)
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"

# Base dirs
if [ -z "${NANOCHAT_BASE_DIR:-}" ]; then
  export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
fi
export WANDB_DIR="${WANDB_DIR:-$NANOCHAT_BASE_DIR/wandb}"

# GPU count
NGPUS="${SLURM_GPUS_ON_NODE:-}"
if ! [[ "$NGPUS" =~ ^[0-9]+$ ]]; then
  NGPUS="$(python -c 'import torch; print(torch.cuda.device_count())')"
fi
if [ -z "$NGPUS" ] || [ "$NGPUS" -le 0 ]; then
  NGPUS="4"
fi
echo "Using NGPUS=$NGPUS"

torchrun --standalone --nproc_per_node="$NGPUS" -m scripts.chat_eval -- \
  --source="$SOURCE" \
  --model_tag="$MODEL_TAG" \
  "${STEP_OPT[@]}" \
  "${TASK_FLAGS[@]}"

echo ""
echo "Done."
