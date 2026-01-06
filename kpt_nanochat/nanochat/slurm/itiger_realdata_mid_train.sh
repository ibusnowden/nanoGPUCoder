#!/bin/bash
#SBATCH --job-name=nanochat-mid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=realdata_mid_%j.out
#SBATCH --error=realdata_mid_%j.err
#
# Short midtraining on real-data base checkpoint.

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  NANOCHAT_ROOT="$SLURM_SUBMIT_DIR"
else
  NANOCHAT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$NANOCHAT_ROOT"
mkdir -p logs

echo "========================================"
echo "nanochat midtraining"
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

# Fix Intel library conflicts (iJIT_NotifyEvent undefined symbol)
export MKL_THREADING_LAYER=GNU
unset LD_PRELOAD

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
WANDB_RUN_BASE="${WANDB_RUN:-realdata_mid}"
export WANDB_PROJECT="${WANDB_PROJECT:-nanochat-mid}"
echo "WANDB_DIR=$WANDB_DIR"
echo "WANDB_RUN_BASE=$WANDB_RUN_BASE (set WANDB_RUN=dummy to disable)"

# GPU count
NGPUS="${SLURM_GPUS_ON_NODE:-}"
if ! [[ "$NGPUS" =~ ^[0-9]+$ ]]; then
  NGPUS="$(python -c 'import torch; print(torch.cuda.device_count())')"
fi
if [ -z "$NGPUS" ] || [ "$NGPUS" -le 0 ]; then
  NGPUS="4"
fi
echo "Using NGPUS=$NGPUS"

# Checkpoint selection
DEPTH="${DEPTH:-4}"
MODEL_TAG="${MODEL_TAG:-d$DEPTH}"   # matches base checkpoints directory name
STEP_OPT=()
if [ -n "${STEP:-}" ]; then
  STEP_OPT=(--step="$STEP")
fi

echo ""
echo "Run config:"
echo "  MODEL_TAG=$MODEL_TAG STEP=${STEP:-latest}"
echo ""

echo "========================================"
echo "Step 1: Midtraining"
echo "========================================"
torchrun --standalone --nproc_per_node="$NGPUS" -m scripts.mid_train -- \
  --run="${WANDB_RUN_BASE}_mid" \
  --model_tag="$MODEL_TAG" \
  --max_seq_len="${MAX_SEQ_LEN:-4096}" \
  --device_batch_size="${DEVICE_BATCH_SIZE:-2}" \
  --total_batch_size="${TOTAL_BATCH_SIZE:-524288}" \
  --length_buckets="[(2048,0.6),(4096,0.4)]" \
  "${STEP_OPT[@]}"

echo ""
echo "========================================"
echo "Step 2: Chat Eval on mid checkpoint"
echo "========================================"
export WANDB_RUN="${WANDB_RUN_EVAL:-${WANDB_RUN_BASE}_eval}"
torchrun --standalone --nproc_per_node="$NGPUS" -m scripts.chat_eval -- -i mid

echo ""
echo "Done."
