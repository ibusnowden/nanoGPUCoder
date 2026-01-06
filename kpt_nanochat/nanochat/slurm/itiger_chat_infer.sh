#!/bin/bash
#SBATCH --job-name=nanochat-chat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=3
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=chat_%j.out
#SBATCH --error=chat_%j.err
#SBATCH --partition=bigTiger
#SBATCH --nodelist=itiger01
#
# Simple inference wrapper to load a checkpoint and run chat_cli (single prompt) or chat_web.
# Usage:
#   sbatch slurm/itiger_chat_infer.sh                  # defaults to chat_cli with PROMPT
#   sbatch --export=MODE=web,PORT=8000 slurm/itiger_chat_infer.sh
#   sbatch --export=SOURCE=mid,MODEL_TAG=d4,STEP=000020,PROMPT="Hi" slurm/itiger_chat_infer.sh

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  NANOCHAT_ROOT="$SLURM_SUBMIT_DIR"
else
  NANOCHAT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$NANOCHAT_ROOT"

MODE="${MODE:-cli}"          # cli|web
PROMPT="${PROMPT:-\"Hello!\"}"
PORT="${PORT:-8000}"
SOURCE="${SOURCE:-sft}"       # base|mid|sft|rl
MODEL_TAG="${MODEL_TAG:-d4}"  # which checkpoint directory to load (e.g., d4)
STEP_OPT=()
if [ -n "${STEP:-}" ]; then
  STEP_OPT=(-s "$STEP")
fi

echo "========================================"
echo "nanochat inference (${MODE})"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID:-}"
echo "Node: ${SLURM_NODELIST:-$(hostname)}"
echo "PWD: $(pwd)"
echo "SOURCE=$SOURCE MODEL_TAG=$MODEL_TAG STEP=${STEP:-latest}"
echo "MODE=$MODE PROMPT=$PROMPT PORT=$PORT"
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

# Ensure repo on PYTHONPATH (set after conda activate)
export PYTHONPATH="$NANOCHAT_ROOT/nanochat:$NANOCHAT_ROOT:${PYTHONPATH:-}"

# Share base dir / wandb dir
if [ -z "${NANOCHAT_BASE_DIR:-}" ]; then
  export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
fi
export WANDB_DIR="${WANDB_DIR:-$NANOCHAT_BASE_DIR/wandb}"

if [ "$MODE" = "web" ]; then
  echo "Launching chat_web on port $PORT (SOURCE=$SOURCE MODEL_TAG=$MODEL_TAG STEP=${STEP:-latest})"
  python -m scripts.chat_web -i "$SOURCE" -g "$MODEL_TAG" "${STEP_OPT[@]}" -p "$PORT"
else
  echo "Running chat_cli with prompt: $PROMPT"
  python -m scripts.chat_cli -i "$SOURCE" -g "$MODEL_TAG" "${STEP_OPT[@]}" -p "$PROMPT"
fi
