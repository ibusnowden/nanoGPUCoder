#!/bin/bash
#
# Setup HuggingFace cache and pre-download datasets to avoid 429 rate limiting.
#
# This implements Fix 1 from the HF Hub rate-limiting solution:
# 1. Set up a shared cache directory
# 2. Warm the cache with a single process
# 3. Then run training with offline mode
#
# Usage:
#   source scripts/setup_hf_cache.sh        # Sets env vars and warms cache
#   torchrun --standalone --nproc_per_node=8 -m scripts.chat_grpo -- ...
#

set -e

# ============================================================================
# Step 1: Configure shared cache directories
# ============================================================================
export HF_HOME="${HF_HOME:-/project/inniang/.cache/hf}"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"

echo "Setting up HuggingFace cache..."
echo "  HF_HOME=$HF_HOME"
echo "  HF_DATASETS_CACHE=$HF_DATASETS_CACHE"
echo "  TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE"

mkdir -p "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"

# Additional settings to reduce API load
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_ENABLE_HF_TRANSFER=1

# ============================================================================
# Step 2: Warm the cache (single process, NOT torchrun)
# ============================================================================
echo ""
echo "Warming HuggingFace cache (this may take a while on first run)..."
python -m scripts.warmup_hf_cache "$@"

# ============================================================================
# Step 3: Set offline mode for subsequent runs
# ============================================================================
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo ""
echo "============================================================"
echo "Cache setup complete! Offline mode enabled."
echo ""
echo "Environment variables set:"
echo "  HF_HOME=$HF_HOME"
echo "  HF_DATASETS_CACHE=$HF_DATASETS_CACHE"
echo "  HF_HUB_OFFLINE=$HF_HUB_OFFLINE"
echo "  HF_DATASETS_OFFLINE=$HF_DATASETS_OFFLINE"
echo ""
echo "You can now run distributed training without 429 errors:"
echo "  torchrun --standalone --nproc_per_node=8 -m scripts.chat_grpo -- ..."
echo "============================================================"
