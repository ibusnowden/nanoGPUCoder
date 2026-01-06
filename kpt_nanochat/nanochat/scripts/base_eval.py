"""
Evaluate the CORE metric for a given model.

Run on a single GPU:
python base_eval.py

Run with torchrun on e.g. 8 GPUs:
torchrun --nproc_per_node=8 base_eval.py

The script will print the CORE metric to the console.
"""
import os
import sys
import argparse
import time
import json
import random
import yaml

import pandas as pd
import torch

from scripts.backend_utils import (
    build_adamw_all_params,
    init_deepspeed_if_needed,
    select_backend,
    wrap_fsdp_if_needed,
)
from nanochat.common import compute_init, compute_cleanup, print0, get_base_dir
from nanochat.tokenizer import HuggingFaceTokenizer
from nanochat.checkpoint_manager import load_model
from nanochat.core_eval import evaluate_task

# -----------------------------------------------------------------------------
# nanoChat specific function dealing with I/O etc.

def evaluate_model(model, tokenizer, device, max_per_task=-1):
    """
    Evaluate a base model on the CORE benchmark.
    - max_per_task: crop the data to this many examples per task for testing (-1 = disable)
    TODO: clean up this function, delete the need for all the files, for pandas dependency, etc.
    """
    # Load config and task metadata
    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
    config_path = os.path.join(eval_bundle_dir, "core.yaml")
    data_base_path = os.path.join(eval_bundle_dir, "eval_data")
    eval_meta_data = os.path.join(eval_bundle_dir, "eval_meta_data.csv")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    tasks = config['icl_tasks']
    eval_metadata = pd.read_csv(eval_meta_data)

    # Evaluate each task
    results = {}
    centered_results = {}
    for task in tasks:
        start_time = time.time()
        label = task['label']
        task_meta = {
            'task_type': task['icl_task_type'],
            'dataset_uri': task['dataset_uri'],
            'num_fewshot': task['num_fewshot'][0],
            'continuation_delimiter': task.get('continuation_delimiter', ' ')
        }
        print0(f"Evaluating: {label} ({task_meta['num_fewshot']}-shot, type: {task_meta['task_type']})... ", end='')

        # Load data for this task
        data_path = os.path.join(data_base_path, task_meta['dataset_uri'])
        with open(data_path, 'r') as f:
            data = [json.loads(line.strip()) for line in f]

        # shuffle the data because in many cases it appears ordered but we want
        # the abillity to only run a subset of the data for debugging purposes etc.
        shuffle_rng = random.Random(1337)
        shuffle_rng.shuffle(data)
        if max_per_task > 0:
            data = data[:max_per_task]

        # run the evaluation for this task
        accuracy = evaluate_task(model, tokenizer, data, device, task_meta)

        results[label] = accuracy
        row = eval_metadata[eval_metadata["Eval Task"] == label]
        random_baseline = row["Random baseline"].values[0]
        centered_result = (accuracy - 0.01 * random_baseline) / (1.0 - 0.01 * random_baseline)
        centered_results[label] = centered_result
        end_time = time.time()
        print0(f"accuracy: {accuracy:.4f} | centered: {centered_result:.4f} | time: {end_time - start_time:.2f}s")

    core_metric = sum(centered_results.values()) / len(centered_results)
    out = {
        "results": results,
        "centered_results": centered_results,
        "core_metric": core_metric
    }
    return out

# -----------------------------------------------------------------------------
# HuggingFace loading utilities and light wrappers for a model

class ModelWrapper:
    """Lightweight wrapper for a HuggingFace model"""
    def __init__(self, model, max_seq_len=None):
        self.model = model
        self.max_seq_len = max_seq_len

    def __call__(self, input_ids):
        outputs = self.model(input_ids)
        logits = outputs.logits
        return logits

def load_hf_model(hf_path: str, device):
    print0(f"Loading model from: {hf_path}")
    # Load the model
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(hf_path)
    model.to(device)
    model.eval()
    max_seq_len = 1024 if "openai-community/gpt2" in hf_path else None
    model = ModelWrapper(model, max_seq_len=max_seq_len)
    # Load the tokenizer
    tokenizer = HuggingFaceTokenizer.from_pretrained(hf_path)
    return model, tokenizer

# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate base model (HF or nano checkpoint)")
    parser.add_argument("hf_path", nargs="?", default=None, help="Optional HF model path to eval")
    parser.add_argument("--nano-model-tag", default=None, help="nano checkpoint model_tag (defaults to largest if None)")
    parser.add_argument("--nano-step", type=int, default=None, help="nano checkpoint step (defaults to latest if None)")
    parser.add_argument("--max-seq-len", type=int, default=None, help="Override model sequence length (reduces mem use at eval)")
    parser.add_argument("--use-deepspeed", type=int, default=0, help="1 = DeepSpeed ZeRO-3 for nano checkpoint eval")
    parser.add_argument("--deepspeed-config", type=str, default="slurm/deepspeed_zero3.json", help="DeepSpeed config path")
    parser.add_argument("--use-fsdp", type=int, default=0, help="1 = Torch FSDP full-shard for nano checkpoint eval")
    parser.add_argument("--fsdp-min-num-params", type=int, default=1_000_000, help="Auto-wrap threshold for FSDP")
    parser.add_argument("--fsdp-cpu-offload", type=int, default=0, help="1 = CPU offload for FSDP params")
    return parser.parse_args()


def main():
    args = parse_args()

    # distributed / precision setup
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    # Load model and tokenizer from command line or from file system
    backend = "ddp" if args.hf_path else select_backend(args.use_deepspeed, args.use_fsdp)
    if backend != "ddp":
        print0(f"Using backend={backend}")
    if args.hf_path:
        hf_path = args.hf_path
        print0(f"Loading huggingface model from: {hf_path}")
        model, tokenizer = load_hf_model(hf_path, device)
        model_name = hf_path # just for logging
        model_slug = hf_path.replace("/", "-") # for the output csv file
    else:
        # load a local model from the file system
        model, tokenizer, meta = load_model("base", device, phase="eval", model_tag=args.nano_model_tag, step=args.nano_step)
        if args.max_seq_len is not None:
            # shrink buffers to reduce memory at eval
            model.config.sequence_len = int(args.max_seq_len)
        model, fsdp_state_dict_config = wrap_fsdp_if_needed(
            model,
            backend=backend,
            ddp_local_rank=ddp_local_rank,
            fsdp_min_num_params=args.fsdp_min_num_params,
            fsdp_cpu_offload=args.fsdp_cpu_offload,
        )
        if backend == "deepspeed":
            # reuse training-style optimizer setup; we won't step it, just need ZeRO-3 partitioning
            adamw_optimizer = build_adamw_all_params(
                model,
                embedding_lr=0.2,
                unembedding_lr=0.004,
                matrix_lr=0.02,
                weight_decay=0.0,
            )
            model = init_deepspeed_if_needed(
                backend=backend,
                model=model,
                orig_model=model,
                optimizer=adamw_optimizer,
                deepspeed_config=args.deepspeed_config,
                device_batch_size=1,
                grad_accum_steps=1,
            )
        model_name = f"base_model (step {meta['step']})" # just for logging
        model_slug = f"base_model_{meta['step']:06d}" # for the output csv file

    eval_model = model.module if backend == "deepspeed" else model
    # Evaluate the model
    with autocast_ctx:
        out = evaluate_model(eval_model, tokenizer, device)

    # Write out the results to a csv file
    core_metric = None
    centered_results = {}
    if ddp_rank == 0:
        base_dir = get_base_dir()
        output_csv_path = os.path.join(base_dir, "base_eval", f"{model_slug}.csv")
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        results = out["results"]
        centered_results = out["centered_results"]
        core_metric = out["core_metric"]
        with open(output_csv_path, 'w') as f:
            f.write(f"{'Task':<35}, {'Accuracy':<10}, {'Centered':<10}\n")
            for label in results:
                f.write(f"{label:<35}, {results[label]:<10.6f}, {centered_results[label]:<10.6f}\n")
            f.write(f"{'CORE':<35}, {'':<10}, {core_metric:<10.6f}\n")
        # Print the content of the csv file to console too
        print0("="*80)
        print0(f"Model: {model_name}")
        print0("="*80)
        with open(output_csv_path, 'r') as f:
            print0(f.read())

    # Log to report
    from nanochat.report import get_report
    get_report().log(section="Base model evaluation", data=[
        {
            "Model": model_name,
            "CORE metric": core_metric,
        },
        centered_results, # the full table
    ])

    compute_cleanup()

if __name__ == "__main__":
    main()
