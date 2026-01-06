#!/usr/bin/env python
"""
Pre-download HuggingFace datasets to local cache.

Run this ONCE with a single process (NOT torchrun) BEFORE distributed training
to avoid 429 rate-limiting errors when multiple ranks hit the HF Hub simultaneously.

Usage:
    # 1. Set shared cache directory (important for cluster)
    export HF_HOME=/project/inniang/.cache/hf
    export HF_DATASETS_CACHE=$HF_HOME/datasets
    export TRANSFORMERS_CACHE=$HF_HOME/transformers
    mkdir -p $HF_DATASETS_CACHE $TRANSFORMERS_CACHE

    # 2. Run this script with a single process
    python -m scripts.warmup_hf_cache

    # 3. Then run training with offline mode
    export HF_HUB_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
    torchrun --standalone --nproc_per_node=8 -m scripts.chat_grpo -- ...
"""

import os
import sys


def warmup_datasets():
    """Pre-download all datasets used by nanochat tasks."""
    from datasets import load_dataset

    print("=" * 60)
    print("Warming up HuggingFace dataset cache...")
    print(f"HF_HOME: {os.environ.get('HF_HOME', '(not set)')}")
    print(f"HF_DATASETS_CACHE: {os.environ.get('HF_DATASETS_CACHE', '(not set)')}")
    print("=" * 60)

    datasets_to_load = [
        # GSM8K (math reasoning)
        ("openai/gsm8k", {"name": "main", "split": "train"}),
        ("openai/gsm8k", {"name": "main", "split": "test"}),
        ("openai/gsm8k", {"name": "socratic", "split": "train"}),
        ("openai/gsm8k", {"name": "socratic", "split": "test"}),

        # AQuA-RAT (math multiple choice)
        ("aqua_rat", {"split": "train"}),
        ("aqua_rat", {"split": "test"}),

        # HumanEval (code generation)
        ("openai/openai_humaneval", {"split": "test"}),

        # MMLU (knowledge/reasoning)
        ("cais/mmlu", {"name": "all", "split": "test"}),
        ("cais/mmlu", {"name": "all", "split": "validation"}),
        ("cais/mmlu", {"name": "auxiliary_train", "split": "train"}),

        # MBPP (code generation)
        ("mbpp", {"split": "train"}),
        ("mbpp", {"split": "test"}),

        # ARC (reasoning)
        ("allenai/ai2_arc", {"name": "ARC-Easy", "split": "train"}),
        ("allenai/ai2_arc", {"name": "ARC-Easy", "split": "test"}),
        ("allenai/ai2_arc", {"name": "ARC-Challenge", "split": "train"}),
        ("allenai/ai2_arc", {"name": "ARC-Challenge", "split": "test"}),

        # HumanEval+ (harder code generation)
        ("bigcode/humanevalpack", {"name": "humanevalplus", "split": "test"}),

        # SmolTalk (conversational)
        ("HuggingFaceTB/smol-smoltalk", {"split": "train"}),
        ("HuggingFaceTB/smol-smoltalk", {"split": "test"}),
    ]

    # BBH tasks (Big-Bench Hard)
    bbh_tasks = [
        "boolean_expressions", "causal_judgement", "date_understanding",
        "disambiguation_qa", "formal_fallacies", "geometric_shapes",
        "hyperbaton", "logical_deduction_five_objects", "logical_deduction_seven_objects",
        "logical_deduction_three_objects", "movie_recommendation", "navigate",
        "object_counting", "penguins_in_a_table", "salient_translation_error_detection",
        "snarks", "sports_understanding", "temporal_sequences",
        "tracking_shuffled_objects_five_objects", "tracking_shuffled_objects_seven_objects",
        "tracking_shuffled_objects_three_objects", "web_of_lies", "word_sorting",
    ]
    for task in bbh_tasks:
        datasets_to_load.append(("lukaemon/bbh", {"name": task, "split": "test"}))

    # Load each dataset
    total = len(datasets_to_load)
    for i, (dataset_id, kwargs) in enumerate(datasets_to_load, 1):
        name = kwargs.get("name", "default")
        split = kwargs.get("split", "default")
        print(f"[{i}/{total}] Loading {dataset_id} (name={name}, split={split})...")
        try:
            load_dataset(dataset_id, **kwargs)
            print(f"  -> OK")
        except Exception as e:
            print(f"  -> FAILED: {e}")

    print("=" * 60)
    print("Cache warmup complete!")
    print()
    print("Now run your training with:")
    print("  export HF_HUB_OFFLINE=1")
    print("  export HF_DATASETS_OFFLINE=1")
    print("  torchrun --standalone --nproc_per_node=8 -m scripts.chat_grpo -- ...")
    print("=" * 60)


def warmup_openthoughts():
    """
    Pre-download OpenThoughts dataset (separate because it requires HF token
    and is very large).
    """
    from datasets import load_dataset

    print("=" * 60)
    print("Warming up OpenThoughts dataset...")
    print("NOTE: This dataset is large and may require a HuggingFace token.")
    print("=" * 60)

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not hf_token:
        print("WARNING: No HF_TOKEN found. Set HF_TOKEN env var if needed.")

    try:
        print("Loading open-thoughts/OpenThoughts3-1.2M (train)...")
        load_dataset("open-thoughts/OpenThoughts3-1.2M", split="train", token=hf_token)
        print("  -> OK")
    except Exception as e:
        print(f"  -> FAILED: {e}")


if __name__ == "__main__":
    # Check if we should include OpenThoughts
    include_openthoughts = "--include-openthoughts" in sys.argv

    warmup_datasets()

    if include_openthoughts:
        warmup_openthoughts()
    else:
        print()
        print("NOTE: OpenThoughts not included. Run with --include-openthoughts to add it.")
