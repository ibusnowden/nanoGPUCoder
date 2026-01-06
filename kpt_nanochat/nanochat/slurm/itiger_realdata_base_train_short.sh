#!/bin/bash
#SBATCH --job-name=nanochat-realdata
#SBATCH --nodes=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:rtx_6000:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=300G

#SBATCH --output=realdata_%j.out
#SBATCH --error=realdata_%j.err
#
# iTiger single-node short real-data base pretrain + eval.
# Assumes you've already staged:
#   $NANOCHAT_BASE_DIR/tokenizer/{tokenizer.pkl,token_bytes.pt}
#   $NANOCHAT_BASE_DIR/base_data/*.parquet   (need >=2 shards: train+val)
#
# Usage:
#   sbatch slurm/itiger_realdata_base_train_short.sbatch
#

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  NANOCHAT_ROOT="$SLURM_SUBMIT_DIR"
else
  NANOCHAT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$NANOCHAT_ROOT"
mkdir -p logs

echo "========================================"
echo "nanochat short real-data (base_train)"
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

# wandb (writes to scratch by default)
export WANDB_DIR="${WANDB_DIR:-$NANOCHAT_BASE_DIR/wandb}"
mkdir -p "$WANDB_DIR"
WANDB_RUN_BASE="${WANDB_RUN:-realdata_${SLURM_JOB_ID:-manual}}"
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

# ---------------------------------------------------------------------------
# Preflight checks (no network): tokenizer + parquet shards

TOKENIZER_DIR="$NANOCHAT_BASE_DIR/tokenizer"
DATA_DIR="$NANOCHAT_BASE_DIR/base_data"

if [ ! -f "$TOKENIZER_DIR/tokenizer.pkl" ] || [ ! -f "$TOKENIZER_DIR/token_bytes.pt" ]; then
  echo "ERROR: missing tokenizer artifacts under $TOKENIZER_DIR"
  echo "Expected:"
  echo "  $TOKENIZER_DIR/tokenizer.pkl"
  echo "  $TOKENIZER_DIR/token_bytes.pt"
  echo ""
  echo "If you haven't trained/staged a tokenizer yet, do it first (requires data access):"
  echo "  python -m scripts.tok_train --max_chars=2000000000"
  exit 2
fi

NUM_PARQUETS=0
if [ -d "$DATA_DIR" ]; then
  NUM_PARQUETS="$(ls -1 "$DATA_DIR"/*.parquet 2>/dev/null | wc -l | tr -d ' ')"
fi
if [ "$NUM_PARQUETS" -lt 2 ]; then
  echo "ERROR: need >=2 parquet shards under $DATA_DIR (train+val). Found: $NUM_PARQUETS"
  echo "Stage at least 2 shards named like shard_00000.parquet ..."
  exit 3
fi
echo "Found $NUM_PARQUETS parquet shards in $DATA_DIR"

python -c "import pyarrow, tokenizers, tiktoken, rustbpe; print('Deps OK: pyarrow/tokenizers/tiktoken/rustbpe')"

# ---------------------------------------------------------------------------
# Training knobs (override via sbatch --export=VAR=VALUE,...)

ARCH_STYLE="${ARCH_STYLE:-qwen25_small}"   # qwen25_small|qwen25_1.5b|qwen25_7b|original
DEPTH="${DEPTH:-4}"                       # used by qwen25_small/original
MODEL_TAG="${MODEL_TAG:-realdata_short}"
SKIP_CORE_METRIC="${SKIP_CORE_METRIC:-1}" # 0 requires eval_bundle + pandas/yaml
SKIP_SAMPLING="${SKIP_SAMPLING:-0}"       # 0 will sample prompts using tokenizer/Engine

MOE_NUM_EXPERTS="${MOE_NUM_EXPERTS:-0}"
MOE_TOP_K="${MOE_TOP_K:-1}"
MOE_LAYER_STRIDE="${MOE_LAYER_STRIDE:-1}"

MAX_SEQ_LEN="${MAX_SEQ_LEN:-256}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-1}"
NUM_ITERS="${NUM_ITERS:-20}"
USE_DEEPSPEED="${USE_DEEPSPEED:-0}"
USE_FSDP="${USE_FSDP:-0}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-slurm/deepspeed_zero3.json}"
FSDP_MIN_PARAMS="${FSDP_MIN_PARAMS:-1000000}"
FSDP_CPU_OFFLOAD="${FSDP_CPU_OFFLOAD:-0}"

# Make divisibility constraints explicit:
TOTAL_BATCH_SIZE=$((DEVICE_BATCH_SIZE * MAX_SEQ_LEN * NGPUS))
EVAL_TOKENS="${EVAL_TOKENS:-$TOTAL_BATCH_SIZE}"

echo ""
echo "Run config:"
echo "  ARCH_STYLE=$ARCH_STYLE DEPTH=$DEPTH MODEL_TAG=$MODEL_TAG"
echo "  MOE_NUM_EXPERTS=$MOE_NUM_EXPERTS MOE_TOP_K=$MOE_TOP_K MOE_LAYER_STRIDE=$MOE_LAYER_STRIDE"
echo "  MAX_SEQ_LEN=$MAX_SEQ_LEN DEVICE_BATCH_SIZE=$DEVICE_BATCH_SIZE TOTAL_BATCH_SIZE=$TOTAL_BATCH_SIZE"
echo "  NUM_ITERS=$NUM_ITERS EVAL_TOKENS=$EVAL_TOKENS"
echo "  USE_DEEPSPEED=$USE_DEEPSPEED USE_FSDP=$USE_FSDP"
echo "  DEEPSPEED_CONFIG=$DEEPSPEED_CONFIG FSDP_MIN_PARAMS=$FSDP_MIN_PARAMS FSDP_CPU_OFFLOAD=$FSDP_CPU_OFFLOAD"
echo ""

if [ "$USE_DEEPSPEED" -eq 1 ] && [ "$USE_FSDP" -eq 1 ]; then
  echo "ERROR: USE_DEEPSPEED and USE_FSDP cannot both be 1"
  exit 4
fi

if [ "$USE_DEEPSPEED" -eq 1 ]; then
  LAUNCHER=(deepspeed --num_gpus="$NGPUS")
else
  LAUNCHER=(torchrun --standalone --nproc_per_node="$NGPUS")
fi

"${LAUNCHER[@]}" -m scripts.base_train -- \
  --run="${WANDB_RUN_BASE}_${MODEL_TAG}" \
  --architecture_style="$ARCH_STYLE" \
  --depth="$DEPTH" \
  --max_seq_len="$MAX_SEQ_LEN" \
  --device_batch_size="$DEVICE_BATCH_SIZE" \
  --total_batch_size="$TOTAL_BATCH_SIZE" \
  --eval_tokens="$EVAL_TOKENS" \
  --num_iterations="$NUM_ITERS" \
  --skip_core_metric="$SKIP_CORE_METRIC" \
  --skip_sampling="$SKIP_SAMPLING" \
  --model_tag="$MODEL_TAG" \
  --moe_num_experts="$MOE_NUM_EXPERTS" \
  --moe_top_k="$MOE_TOP_K" \
  --moe_layer_start=0 \
  --moe_layer_end=-1 \
  --moe_layer_stride="$MOE_LAYER_STRIDE" \
  --use_deepspeed="$USE_DEEPSPEED" \
  --deepspeed_config="$DEEPSPEED_CONFIG" \
  --use_fsdp="$USE_FSDP" \
  --fsdp_min_num_params="$FSDP_MIN_PARAMS" \
  --fsdp_cpu_offload="$FSDP_CPU_OFFLOAD"

echo ""
echo "Done."
