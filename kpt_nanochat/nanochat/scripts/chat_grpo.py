"""
Mixed-task GRPO-style RL with optional KL regularization to a reference SFT model.

Example:
torchrun --standalone --nproc_per_node=8 -m scripts.chat_grpo -- \
  --task_mix=gsm8k:0.4,math:0.2,mmlu_science:0.1,mbpp:0.15,humaneval:0.15 \
  --num_steps=500 --kl_coef=0.02
"""

import contextlib
import hashlib
import math
import os
import itertools
import random
import re
from collections import defaultdict

try:
    import wandb
except ImportError:
    wandb = None
import torch
import torch.distributed as dist

from scripts.backend_utils import (
    build_adamw_all_params,
    init_deepspeed_if_needed,
    select_backend,
    wrap_fsdp_if_needed,
)
from nanochat.common import compute_init, compute_cleanup, print0, get_base_dir, DummyWandb
from nanochat.checkpoint_manager import save_checkpoint, load_model
from nanochat.engine import Engine

from tasks.gsm8k import GSM8K
from tasks.math import MATH
from tasks.mmlu import MMLU
from tasks.mbpp import MBPP
from tasks.humaneval import HumanEval
from tasks.dolci_think import DolciThink

# -----------------------------------------------------------------------------
# Thought/Answer splitting for logging

_ANSWER_MARKERS = ("Final Answer:", "Final:", "Answer:", "####")

def _split_thought_answer(text):
    """
    Split generated text into thought (reasoning) and answer sections.
    Returns (thought_text, answer_text).
    If no answer marker found, all text is considered "answer" (thought_text="").
    """
    if not text:
        return "", ""
    # Try <final>...</final> tags first
    match = re.search(r"<final>(.*?)</final>", text, re.DOTALL | re.IGNORECASE)
    if match:
        return text[:match.start()].strip(), match.group(1).strip()
    # Try standard markers
    for tag in _ANSWER_MARKERS:
        idx = text.rfind(tag)
        if idx != -1:
            return text[:idx].strip(), text[idx + len(tag):].strip()
    # No marker found - entire text is the answer (no explicit reasoning section)
    return "", text.strip()

def _append_format_hint_content(content, hint):
    if not hint:
        return content
    if isinstance(content, str):
        base = content.rstrip()
        sep = "\n\n" if base else ""
        return f"{base}{sep}{hint}"
    if isinstance(content, list):
        new_content = list(content)
        new_content.append({"type": "text", "text": "\n\n" + hint})
        return new_content
    return f"{content}\n\n{hint}"

def _append_format_hint(conversation, hint):
    if not hint or not isinstance(conversation, dict):
        return conversation
    messages = conversation.get("messages")
    if not isinstance(messages, list) or not messages:
        return conversation
    new_messages = []
    for msg in messages:
        new_messages.append(dict(msg) if isinstance(msg, dict) else msg)
    for idx in range(len(new_messages) - 1, -1, -1):
        msg = new_messages[idx]
        if isinstance(msg, dict) and msg.get("role") == "user":
            msg = dict(msg)
            msg["content"] = _append_format_hint_content(msg.get("content"), hint)
            new_messages[idx] = msg
            break
    updated = dict(conversation)
    updated["messages"] = new_messages
    return updated

def _get_eval_format_hint(task_name, conversation):
    if task_name == "gsm8k":
        return "Put your final answer after ####."
    if task_name in {"mmlu_science", "math"}:
        letters = None
        if isinstance(conversation, dict):
            letters = conversation.get("letters")
        if letters:
            letters_str = ", ".join(str(letter) for letter in letters)
            return f"Answer with a single letter ({letters_str})."
        return "Answer with a single letter."
    if task_name in {"mbpp", "humaneval"}:
        return "Write only valid Python code. Do not include explanations."
    return ""

# -----------------------------------------------------------------------------
# GRPO hyperparameters
run = "dummy"
source = "sft" # base|mid|sft
model_tag = None
step = None
ref_source = "sft"
ref_model_tag = None
ref_step = None
dtype = "bfloat16"
device_batch_size = 4
examples_per_step = 16 # across all ranks
num_samples = 8
ppo_minibatch_size = 64
max_prompt_tokens = 0  # 0 disables prompt truncation
max_new_tokens = 256
temperature = 0.7  # Lower temperature reduces policy drift (used if temp_schedule="none")
top_k = 50
kl_coef = 0.0  # Disable KL penalty by default
kl_max_threshold = 50.0  # Warn if KL exceeds this
reward_scale = 1.0
reward_mode = "task"  # "task" (use task.reward) or "dapo" (binary >0 reward)
group_dynamic_sampling = 1  # Drop prompt groups with all-correct or all-wrong rewards.
group_dynamic_sampling_max_tries = 50  # Prevent infinite loops when all prompts collapse.
use_best_of_n = 0  # If 1, use best reward among num_samples instead of all
active_sampling = 1  # If 1, keep only positive-advantage samples
zero_grad_filtering = 1  # If 1, drop zero-advantage samples from updates
zero_adv_eps = 1e-8
format_hint_mode = "none"  # "none" or "eval" (append task-specific eval format hints)
learning_rate = 0.0  # 0 uses optimizer defaults
lr_schedule = "linear"  # "linear" or "constant"
# Temperature schedule: ramp from temp_start to temp_end over training
temp_start = 0.3  # Starting temperature (exploitation)
temp_end = 0.7    # Ending temperature (exploration)
temp_schedule = "linear"  # "linear", "cosine", or "none"
# Length penalty: discourage overly long responses
length_penalty_mode = "linear"  # "linear" or "dapo"
length_penalty_coef = 0.001  # Penalty scale (0 = disabled)
length_penalty_target = 256  # Penalty starts above this many tokens
length_penalty_floor = 0.0  # Minimum reward after shaping
# PPO/GRPO clipping
clip_eps = 0.2  # Trust-region clipping for importance ratio (used if clip_ratio_* unset)
clip_ratio_low = 0.0  # Set both low/high to enable explicit ratio clipping
clip_ratio_high = 0.0
advantage_clip = 5.0  # Clip advantages to [-clip, +clip] for stability (0 = disabled)
grpo_epochs = 4  # Number of optimization epochs per step (like PPO); more epochs = more weight change
grpo_lr_scale = 0.25  # Scale down LR for GRPO (since we do grpo_epochs updates per step)
num_steps = 1000
total_examples = -1 # overrides num_steps if >0
save_every = 200
eval_every = 50
eval_num_per_task = 5
eval_seed = 123
eval_temperature = 0.0
eval_top_k = 0
eval_max_new_tokens = 256
use_deepspeed = 0
deepspeed_config = "slurm/deepspeed_zero3.json"
use_fsdp = 0
fsdp_min_num_params = 1_000_000
fsdp_cpu_offload = 0
task_mix = "dolci:1.0,gsm8k:0.45,math:0.20,mmlu_science:0.10,mbpp:0.25"
dolci_dataset_id = "allenai/Dolci-Think-RL-32B"
dolci_split = "train"
dolci_mode = "cot"
dolci_stop = -1
dolci_streaming = 0
dolci_stream_cache = ""

config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join('nanochat', 'configurator.py')).read()) # overrides from command line or config file
user_config = {k: globals()[k] for k in config_keys}
# GRPO bookkeeping
MAX_GRAD_NORM = 1.0
CURRENT_STEP = 0
EVAL_ONLY_TASK_NAMES = {"humaneval"}
# -----------------------------------------------------------------------------

ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
master_process = ddp_rank == 0
backend = select_backend(use_deepspeed, use_fsdp)
if backend != "ddp":
    print0(f"Using backend={backend}")
dtype = torch.float32 if dtype == 'float32' else torch.bfloat16
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=dtype)

wandb_disabled = os.environ.get("WANDB_DISABLED", "false").lower() in ("1", "true", "yes")
use_dummy_wandb = run == "dummy" or not master_process or wandb_disabled
if not use_dummy_wandb and wandb is None:
    print0("wandb not installed; proceeding without wandb logging")
wandb_project = os.environ.get("WANDB_PROJECT", "nanochat-grpo")
wandb_run = DummyWandb() if use_dummy_wandb or wandb is None else wandb.init(
    project=wandb_project,
    name=run,
    config=user_config,
    save_code=True,
)
# Define global step metric for W&B to ensure monotonic logging
if wandb is not None and not use_dummy_wandb:
    wandb.define_metric("trainer/global_step")
    wandb.define_metric("*", step_metric="trainer/global_step")

# Load policy model
model, tokenizer, meta = load_model(source, device, phase="eval", model_tag=model_tag, step=step)
orig_model = model
model, fsdp_state_dict_config = wrap_fsdp_if_needed(
    model,
    backend=backend,
    ddp_local_rank=ddp_local_rank,
    fsdp_min_num_params=fsdp_min_num_params,
    fsdp_cpu_offload=fsdp_cpu_offload,
)
train_model = model
eval_model = train_model

# Load reference model for KL (optional)
use_kl = kl_coef > 0
ref_model = None
if use_kl:
    ref_model, ref_tokenizer, _ = load_model(ref_source, device, phase="eval", model_tag=ref_model_tag, step=ref_step)
    ref_model.eval()
    ref_model.requires_grad_(False)

# -----------------------------------------------------------------------------
# Task sampler

EVAL_TASK_REGISTRY = {}

def _register_eval_task(name, task):
    if name in EVAL_TASK_REGISTRY:
        return
    EVAL_TASK_REGISTRY[name] = task

# DDP-safe eval task registration: rank 0 first, then others
import time as _time
if ddp_rank == 0:
    _register_eval_task("humaneval", HumanEval())
    _time.sleep(2.0)  # Allow NFS cache sync
if ddp:
    dist.barrier()
if ddp_rank != 0:
    _time.sleep(0.3 * ddp_rank)  # Stagger to reduce contention
    _register_eval_task("humaneval", HumanEval())

def _build_mbpp_train_eval_pair():
    train_task = MBPP(split="train")
    eval_task = MBPP(split="test")
    return train_task, eval_task

def _build_gsm8k_train_eval_pair():
    train_task = GSM8K(subset="main", split="train")
    eval_task = GSM8K(subset="main", split="test")
    return train_task, eval_task

def _build_math_train_eval_pair():
    train_task = MATH(subset="all", split="train")
    eval_task = MATH(subset="all", split="test")
    return train_task, eval_task

def _build_mmlu_train_eval_pair():
    train_task = MMLU(subset="auxiliary_train", split="train", subjects="science")
    eval_task = MMLU(subset="all", split="validation", subjects="science")
    return train_task, eval_task

def _parse_task_mix(spec):
    items = []
    total = 0.0
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            name, weight = part.split(":", 1)
            weight = float(weight)
        else:
            name, weight = part, 1.0
        name = name.strip().lower()
        items.append((name, weight))
        total += weight
    if total <= 0:
        raise ValueError("task_mix must have positive weights")
    return [(name, weight / total) for name, weight in items]


def _build_task(name):
    if name == "gsm8k":
        train_task, eval_task = _build_gsm8k_train_eval_pair()
        _register_eval_task(name, eval_task)
        return train_task
    if name == "math":
        train_task, eval_task = _build_math_train_eval_pair()
        _register_eval_task(name, eval_task)
        return train_task
    if name == "mmlu_science":
        train_task, eval_task = _build_mmlu_train_eval_pair()
        _register_eval_task(name, eval_task)
        return train_task
    if name == "mbpp":
        train_task, eval_task = _build_mbpp_train_eval_pair()
        _register_eval_task(name, eval_task)
        return train_task
    if name == "dolci":
        resolved_stop = None if dolci_stop < 0 else dolci_stop
        stream_cache = dolci_stream_cache or None
        task = DolciThink(
            dataset_id=dolci_dataset_id,
            split=dolci_split,
            mode=dolci_mode,
            stop=resolved_stop,
            streaming=bool(dolci_streaming),
            stream_cache_path=stream_cache,
        )
        _register_eval_task(name, task)
        return task
    if name == "humaneval":
        raise ValueError("HumanEval is eval-only; remove it from task_mix (use eval set only).")
    raise ValueError(f"Unknown task name: {name}")


class WeightedTaskSampler:
    def __init__(self, spec, seed):
        import time
        self.items = _parse_task_mix(spec)
        self.tasks = []
        self.cdf = []
        cumulative = 0.0

        # DDP-safe task building: rank 0 builds first, then others
        if ddp_rank == 0:
            for name, weight in self.items:
                task = _build_task(name)
                if len(task) == 0:
                    raise ValueError(f"Task {name} has zero length after filtering.")
                self.tasks.append((name, task))
                cumulative += weight
                self.cdf.append(cumulative)
            # Allow time for cache to sync
            time.sleep(3.0)

        # Barrier: all ranks wait for rank 0 to finish building all tasks
        if ddp:
            dist.barrier()

        # Non-rank-0: build tasks from cache
        if ddp_rank != 0:
            # Stagger to reduce NFS contention
            time.sleep(0.5 * ddp_rank)
            for name, weight in self.items:
                task = _build_task(name)
                if len(task) == 0:
                    raise ValueError(f"Task {name} has zero length after filtering.")
                self.tasks.append((name, task))
                cumulative += weight
                self.cdf.append(cumulative)

        self.rng = random.Random(seed)
        self.iters = {
            name: itertools.cycle(range(ddp_rank, len(task), ddp_world_size))
            for name, task in self.tasks
        }

    def sample(self):
        r = self.rng.random()
        for i, cutoff in enumerate(self.cdf):
            if r <= cutoff:
                name, task = self.tasks[i]
                idx = next(self.iters[name])
                return name, task, task[idx]
        name, task = self.tasks[-1]
        idx = next(self.iters[name])
        return name, task, task[idx]


task_sampler = WeightedTaskSampler(task_mix, seed=42 + ddp_rank)

# -----------------------------------------------------------------------------
# Eval set utilities

def _sample_eval_indices(task_len, num_samples, seed):
    if num_samples <= 0 or task_len <= 0:
        return []
    if task_len <= num_samples:
        return list(range(task_len))
    rng = random.Random(seed)
    return rng.sample(range(task_len), num_samples)


def _build_eval_set(tasks, num_per_task, seed):
    eval_items = []
    for name, task in tasks:
        task_seed = seed + sum(ord(ch) for ch in name)
        indices = _sample_eval_indices(len(task), num_per_task, task_seed)
        for idx in sorted(indices):
            eval_items.append((name, task, task[idx]))
    return eval_items


def _run_pass_at_1(eval_items):
    if not eval_items:
        return 0.0, {}
    eval_top_k_val = None if eval_top_k <= 0 else eval_top_k
    per_task = defaultdict(lambda: {"correct": 0, "total": 0})
    total = 0
    correct = 0
    eval_model.eval()
    with torch.no_grad():
        for idx, (name, task, conversation) in enumerate(eval_items):
            tokens = tokenizer.render_for_completion(conversation)
            prefix_length = len(tokens)
            seed = (eval_seed * 1000003 + idx) & 0x7FFFFFFF
            with autocast_ctx:
                generated, _ = engine.generate_batch(
                    tokens,
                    num_samples=1,
                    max_tokens=eval_max_new_tokens,
                    temperature=eval_temperature,
                    top_k=eval_top_k_val,
                    seed=seed,
                )
            generated_tokens = generated[0][prefix_length:]
            generated_text = tokenizer.decode(generated_tokens)
            reward = task.reward(conversation, generated_text)
            success = 1 if reward > 0 else 0
            per_task[name]["correct"] += success
            per_task[name]["total"] += 1
            total += 1
            correct += success
    pass_at_1 = (correct / total) if total else 0.0
    return pass_at_1, per_task


eval_set = None
if master_process and eval_every > 0 and eval_num_per_task > 0:
    eval_task_items = []
    seen_names = set()
    for name, _ in task_sampler.items:
        if name in seen_names:
            continue
        seen_names.add(name)
        eval_task = EVAL_TASK_REGISTRY.get(name)
        if eval_task is None:
            continue
        eval_task_items.append((name, eval_task))
    for name in sorted(EVAL_ONLY_TASK_NAMES):
        if name in seen_names:
            continue
        extra_task = EVAL_TASK_REGISTRY.get(name)
        if extra_task is None:
            continue
        eval_task_items.append((name, extra_task))
    eval_set = _build_eval_set(eval_task_items, eval_num_per_task, eval_seed)
    print0(f"Eval set size: {len(eval_set)}")

# -----------------------------------------------------------------------------
# Rollout generator

def _stable_seed(task_name, sampling_step, prefix_length, rollout_idx, step):
    key = f"{ddp_rank}:{task_name}:{step}:{rollout_idx}:{sampling_step}:{prefix_length}"
    digest = hashlib.sha256(key.encode()).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def get_batch():
    global CURRENT_STEP
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "pad_token", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", None)
    if pad_token_id is None and hasattr(tokenizer, "encode_special"):
        pad_token_id = tokenizer.encode_special("<|assistant_end|>")
    if pad_token_id is None:
        pad_token_id = 0

    rollout_counter = 0
    dynamic_sampling_attempts = 0
    while True:
        task_name, task, conversation = task_sampler.sample()
        if format_hint_mode == "eval":
            hint = _get_eval_format_hint(task_name, conversation)
            conversation = _append_format_hint(conversation, hint)
        if max_prompt_tokens and max_prompt_tokens > 0:
            tokens = tokenizer.render_for_completion(conversation, max_prompt_tokens=max_prompt_tokens)
        else:
            tokens = tokenizer.render_for_completion(conversation)
        prefix_length = len(tokens)

        eval_model.eval()
        generated_token_sequences = []
        masks = []
        assert num_samples % device_batch_size == 0, "num_samples must be divisible by device_batch_size"
        num_sampling_steps = num_samples // device_batch_size
        for sampling_step in range(num_sampling_steps):
            seed = _stable_seed(task_name, sampling_step, prefix_length, rollout_counter, CURRENT_STEP)
            current_temp = get_temperature(CURRENT_STEP)
            with autocast_ctx:
                generated_batch, masks_batch = engine.generate_batch(
                    tokens,
                    num_samples=device_batch_size,
                    max_tokens=max_new_tokens,
                    temperature=current_temp,
                    top_k=top_k,
                    seed=seed,
                )
            generated_token_sequences.extend(generated_batch)
            masks.extend(masks_batch)

        rewards = []
        raw_rewards = []  # Pre-penalty rewards for logging
        thought_token_lengths = []
        answer_token_lengths = []
        response_token_lengths = []
        length_penalties_applied = []

        for sample_tokens in generated_token_sequences:
            generated_tokens = sample_tokens[prefix_length:]
            generated_text = tokenizer.decode(generated_tokens)

            # Track thought vs answer token lengths (Feature 1)
            thought_text, answer_text = _split_thought_answer(generated_text)
            # Use actual token counts (accurate for MFU/length control)
            thought_len = len(tokenizer.encode(thought_text)) if thought_text else 0
            answer_len = len(tokenizer.encode(answer_text)) if answer_text else 0
            thought_token_lengths.append(thought_len)
            answer_token_lengths.append(answer_len)

            # Compute raw reward
            raw_reward = float(task.reward(conversation, generated_text))
            if reward_mode == "dapo":
                raw_reward = 1.0 if raw_reward > 0 else 0.0
            raw_reward *= reward_scale
            raw_rewards.append(raw_reward)

            # Apply length penalty (Feature 3)
            total_len = len(generated_tokens)
            response_token_lengths.append(total_len)
            penalty = 0.0
            if length_penalty_coef > 0 and total_len > length_penalty_target:
                over = total_len - length_penalty_target
                if length_penalty_mode == "dapo":
                    denom = max(length_penalty_target, 1)
                    penalty = length_penalty_coef * (over / denom)
                else:
                    penalty = length_penalty_coef * over
                reward = raw_reward - penalty
                if reward < length_penalty_floor:
                    reward = length_penalty_floor
            else:
                reward = raw_reward
            length_penalties_applied.append(penalty)

            rewards.append(reward)

        if group_dynamic_sampling and num_samples > 1 and raw_rewards:
            successes = [r > 0 for r in raw_rewards]
            all_correct = all(successes)
            all_wrong = not any(successes)
            if all_correct or all_wrong:
                dynamic_sampling_attempts += 1
                rollout_counter += 1
                if group_dynamic_sampling_max_tries > 0 and dynamic_sampling_attempts >= group_dynamic_sampling_max_tries:
                    dynamic_sampling_attempts = 0
                else:
                    continue
        dynamic_sampling_attempts = 0
        rollout_counter += 1

        # Compute group statistics from ALL samples BEFORE filtering
        all_rewards_tensor = torch.tensor(rewards, dtype=torch.float, device=device)
        if all_rewards_tensor.numel() == 0:
            group_mean = torch.tensor(0.0, device=device)
            group_std = torch.tensor(1.0, device=device)
        else:
            group_mean = all_rewards_tensor.mean()
            group_variance = all_rewards_tensor.var(unbiased=False)
            group_std = torch.sqrt(torch.clamp(group_variance, min=1e-6))

        # Advantages with mean-only baseline (no std normalization)
        advantages_all = all_rewards_tensor - group_mean
        if advantage_clip > 0:
            advantages_all = torch.clamp(advantages_all, -advantage_clip, advantage_clip)

        keep_indices = list(range(len(rewards)))
        if active_sampling:
            keep_indices = [i for i in keep_indices if advantages_all[i].item() > zero_adv_eps]
        if zero_grad_filtering:
            keep_indices = [i for i in keep_indices if abs(advantages_all[i].item()) > zero_adv_eps]

        # Best-of-N: only keep the sample with highest reward
        if use_best_of_n and len(keep_indices) > 1:
            best_idx = max(keep_indices, key=lambda i: rewards[i])
            keep_indices = [best_idx]

        if not keep_indices and rewards:
            # Fallback: keep the best sample with zero advantage (no gradient signal).
            best_idx = max(range(len(rewards)), key=lambda i: rewards[i])
            keep_indices = [best_idx]
            advantages_all = torch.zeros_like(advantages_all)

        if keep_indices:
            generated_token_sequences = [generated_token_sequences[i] for i in keep_indices]
            masks = [masks[i] for i in keep_indices]
            rewards = [rewards[i] for i in keep_indices]
            raw_rewards = [raw_rewards[i] for i in keep_indices]
            thought_token_lengths = [thought_token_lengths[i] for i in keep_indices]
            answer_token_lengths = [answer_token_lengths[i] for i in keep_indices]
            response_token_lengths = [response_token_lengths[i] for i in keep_indices]
            length_penalties_applied = [length_penalties_applied[i] for i in keep_indices]

        max_length = max(len(seq) for seq in generated_token_sequences)
        padded_generated_token_sequences = [seq + [pad_token_id] * (max_length - len(seq)) for seq in generated_token_sequences]
        padded_masks = [mask + [0] * (max_length - len(mask)) for mask in masks]
        ids = torch.tensor(padded_generated_token_sequences, dtype=torch.long, device=device)
        mask_ids = torch.tensor(padded_masks, dtype=torch.long, device=device)
        inputs = ids[:, :-1]
        targets = ids[:, 1:].clone()
        targets[mask_ids[:, 1:] == 0] = -1

        if prefix_length > 0:
            assert mask_ids[:, :prefix_length].sum().item() == 0, "Prompt tokens should be masked out"

        # Compute logp_old from current policy BEFORE any updates (for GRPO clipped objective)
        eval_model.eval()
        with torch.no_grad():
            with autocast_ctx:
                logp_old = -eval_model(inputs, targets, loss_reduction='none').view_as(inputs)
        valid_mask = (targets >= 0).float()
        token_counts = valid_mask.sum(dim=1).clamp(min=1.0)
        logp_old_sum = (logp_old * valid_mask).sum(dim=1)
        logp_old_mean = (logp_old_sum / token_counts).detach()  # [batch_size]

        rewards_tensor = torch.tensor(rewards, dtype=torch.float, device=device)
        advantages = advantages_all[keep_indices] if keep_indices else torch.zeros_like(rewards_tensor)

        # Package new metrics
        raw_rewards_tensor = torch.tensor(raw_rewards, dtype=torch.float, device=device)
        thought_lens_tensor = torch.tensor(thought_token_lengths, dtype=torch.float, device=device)
        answer_lens_tensor = torch.tensor(answer_token_lengths, dtype=torch.float, device=device)
        response_lens_tensor = torch.tensor(response_token_lengths, dtype=torch.float, device=device)
        length_penalties_tensor = torch.tensor(length_penalties_applied, dtype=torch.float, device=device)

        entropy_sum = (-logp_old * valid_mask).sum()
        entropy_count = valid_mask.sum()

        yield (task_name, generated_token_sequences, inputs, targets, rewards_tensor, advantages,
               raw_rewards_tensor, thought_lens_tensor, answer_lens_tensor, response_lens_tensor,
               length_penalties_tensor, logp_old, logp_old_mean, valid_mask, entropy_sum, entropy_count)

# -----------------------------------------------------------------------------
# Optimizer

optimizers = []
adamw_optimizer = None
if backend == "ddp":
    optimizers = model.setup_optimizers(
        unembedding_lr=0.004,
        embedding_lr=0.2,
        matrix_lr=0.02,
        weight_decay=0.0,
    )
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["lr"] * 0.05
            group["initial_lr"] = group["lr"]
else:
    adamw_target = model if backend == "fsdp" else orig_model
    adamw_optimizer = build_adamw_all_params(
        adamw_target,
        embedding_lr=0.2,
        unembedding_lr=0.004,
        matrix_lr=0.02,
        weight_decay=0.0,
    )
    optimizers = [adamw_optimizer]
if backend == "deepspeed":
    train_model = init_deepspeed_if_needed(
        backend=backend,
        model=model,
        orig_model=orig_model,
        optimizer=adamw_optimizer,
        deepspeed_config=deepspeed_config,
        device_batch_size=device_batch_size,
        grad_accum_steps=1,
    )
eval_model = train_model.module if backend == "deepspeed" else train_model
sampling_model = eval_model if backend != "ddp" else orig_model
engine = Engine(sampling_model, tokenizer)

# Optional LR override across all optimizer groups.
if learning_rate and learning_rate > 0:
    target_opts = [train_model.optimizer] if backend == "deepspeed" else optimizers
    for opt in target_opts:
        for group in opt.param_groups:
            group["lr"] = learning_rate
            group["initial_lr"] = learning_rate

# DDP-safe gradient clipping: clip exactly what the optimizer steps
def clip_from_optimizers(opts, max_norm):
    """Clip gradients from optimizer param groups (avoids DDP wrapper mismatch)."""
    params = []
    for opt in opts:
        for group in opt.param_groups:
            for p in group["params"]:
                if p is not None and p.grad is not None:
                    params.append(p)
    if params:
        torch.nn.utils.clip_grad_norm_(params, max_norm)

reward_hist_bins = [0.0, 0.25, 0.5, 0.75, 1.0]
reward_hist_labels = [
    "reward_hist_le_0",
    "reward_hist_0_0.25",
    "reward_hist_0.25_0.5",
    "reward_hist_0.5_0.75",
    "reward_hist_0.75_1",
    "reward_hist_gt_1",
]
reward_hist_bins_tensor = torch.tensor(reward_hist_bins, device=device)

# Learning rate scheduler
def get_lr_multiplier(it):
    if lr_schedule in ("constant", "none"):
        return 1.0
    return max(0.0, 1.0 - it / max(num_steps, 1))

# Temperature scheduler
def get_temperature(step):
    """
    Compute temperature at current step based on schedule.
    Returns temp_start at step 0, temp_end at step num_steps-1.
    """
    if temp_schedule == "none" or temp_start == temp_end:
        return temperature  # Use fixed temperature (backward compatible)
    progress = min(1.0, step / max(num_steps - 1, 1))
    if temp_schedule == "cosine":
        # Cosine schedule: starts slow, accelerates in middle, slows at end
        progress = 0.5 * (1 - math.cos(math.pi * progress))
    # For "linear", progress is already linear
    return temp_start + (temp_end - temp_start) * progress

assert examples_per_step % ddp_world_size == 0, \
    f"examples_per_step={examples_per_step} must be divisible by world_size={ddp_world_size}"
examples_per_rank = examples_per_step // ddp_world_size
print0(f"Examples per rank: {examples_per_rank}")

pad_token_id = getattr(tokenizer, "pad_token_id", None)
if pad_token_id is None:
    pad_token_id = getattr(tokenizer, "pad_token", None)
if pad_token_id is None:
    pad_token_id = getattr(tokenizer, "eos_token_id", None)
if pad_token_id is None and hasattr(tokenizer, "encode_special"):
    pad_token_id = tokenizer.encode_special("<|assistant_end|>")
if pad_token_id is None:
    pad_token_id = 0

if (clip_ratio_low > 0) ^ (clip_ratio_high > 0):
    raise ValueError("clip_ratio_low and clip_ratio_high must both be set to enable ratio clipping.")
if clip_ratio_low > 0 and clip_ratio_high <= clip_ratio_low:
    raise ValueError("clip_ratio_high must be greater than clip_ratio_low.")
if reward_mode not in {"task", "dapo"}:
    raise ValueError(f"reward_mode must be 'task' or 'dapo', got {reward_mode!r}")
if length_penalty_mode not in {"linear", "dapo"}:
    raise ValueError(f"length_penalty_mode must be 'linear' or 'dapo', got {length_penalty_mode!r}")
if group_dynamic_sampling_max_tries < 0:
    raise ValueError("group_dynamic_sampling_max_tries must be >= 0")
if format_hint_mode not in {"none", "eval"}:
    raise ValueError(f"format_hint_mode must be 'none' or 'eval', got {format_hint_mode!r}")

if total_examples > 0:
    num_steps = max(1, total_examples // max(examples_per_step, 1))
    print0(f"Overriding num_steps via total_examples={total_examples}: {num_steps}")

# -----------------------------------------------------------------------------
# Training loop

batch_iterator = get_batch()
for step in range(num_steps):
    CURRENT_STEP = step
    rewards_list = []
    sequence_lengths = []
    response_length_list = []
    kl_list = []  # kl_surrogate (non-negative)
    kl_abs_list = []
    kl_gap_list = []  # signed gap (can be negative)
    kl_max_tok_list = []  # max token-level deviation
    task_counts = defaultdict(int)
    reward_nonzero = 0
    reward_total = 0
    reward_min = None
    reward_max = None
    reward_hist = torch.zeros(len(reward_hist_bins) + 1, device=device, dtype=torch.float)
    # New metrics accumulators
    raw_rewards_list = []
    thought_length_list = []
    answer_length_list = []
    length_penalty_list = []
    length_penalty_applied_count = 0
    policy_entropy_sum = 0.0
    policy_entropy_count = 0.0
    # GRPO clipping metrics
    clip_frac_list = []
    ratio_list = []
    ratio_min_list = []
    ratio_max_list = []
    log_ratio_list = []
    adv_mean_list = []
    adv_std_list = []
    adv_max_list = []

    # =========================================================================
    # PHASE 1: Collect ALL rollouts with logp_old BEFORE any optimization
    # This ensures logp_old comes from the true old policy
    # =========================================================================
    collected_batches = []
    for example_step in range(examples_per_rank):
        (task_name, sequences_all, inputs_all, targets_all, rewards_all, advantages_all,
         raw_rewards_all, thought_lens_all, answer_lens_all, response_lens_all,
         length_penalties_all, logp_old_all, logp_old_mean_all, valid_mask_all,
         entropy_sum_all, entropy_count_all) = next(batch_iterator)

        collected_batches.append({
            'task_name': task_name,
            'sequences_all': sequences_all,
            'inputs_all': inputs_all,
            'targets_all': targets_all,
            'rewards_all': rewards_all,
            'advantages_all': advantages_all,
            'raw_rewards_all': raw_rewards_all,
            'thought_lens_all': thought_lens_all,
            'answer_lens_all': answer_lens_all,
            'response_lens_all': response_lens_all,
            'length_penalties_all': length_penalties_all,
            'logp_old_all': logp_old_all,
            'valid_mask_all': valid_mask_all,
        })

        # Accumulate stats during collection phase
        task_counts[task_name] += 1
        rewards_list.append(rewards_all.mean().item())
        sequence_lengths.extend(len(seq) for seq in sequences_all)
        response_length_list.extend(response_lens_all.tolist())
        reward_nonzero += int((rewards_all > 0).sum().item())
        reward_total += int(rewards_all.numel())
        rewards_min_val = rewards_all.min().item()
        rewards_max_val = rewards_all.max().item()
        reward_min = rewards_min_val if reward_min is None else min(reward_min, rewards_min_val)
        reward_max = rewards_max_val if reward_max is None else max(reward_max, rewards_max_val)
        if rewards_all.numel():
            reward_bucket_idx = torch.bucketize(rewards_all, reward_hist_bins_tensor, right=True)
            reward_hist += torch.bincount(
                reward_bucket_idx,
                minlength=reward_hist_bins_tensor.numel() + 1,
            ).to(dtype=reward_hist.dtype)
        raw_rewards_list.append(raw_rewards_all.mean().item())
        thought_length_list.extend(thought_lens_all.tolist())
        answer_length_list.extend(answer_lens_all.tolist())
        length_penalty_list.extend(length_penalties_all.tolist())
        length_penalty_applied_count += int((length_penalties_all > 0).sum().item())
        policy_entropy_sum += entropy_sum_all.item()
        policy_entropy_count += entropy_count_all.item()

    flat_inputs = []
    flat_targets = []
    flat_advantages = []
    flat_logp_old = []
    flat_valid_mask = []
    for batch in collected_batches:
        sequences_all = batch['sequences_all']
        inputs_all = batch['inputs_all']
        targets_all = batch['targets_all']
        advantages_all = batch['advantages_all']
        logp_old_all = batch['logp_old_all']
        valid_mask_all = batch['valid_mask_all']
        for idx, seq in enumerate(sequences_all):
            seq_len = len(seq)
            token_len = seq_len - 1
            if token_len <= 0:
                continue
            flat_inputs.append(inputs_all[idx, :token_len])
            flat_targets.append(targets_all[idx, :token_len])
            flat_advantages.append(advantages_all[idx])
            flat_logp_old.append(logp_old_all[idx, :token_len])
            flat_valid_mask.append(valid_mask_all[idx, :token_len])

    total_samples = len(flat_inputs)
    if total_samples == 0:
        continue
    minibatch_size = ppo_minibatch_size if ppo_minibatch_size > 0 else total_samples
    minibatch_size = min(minibatch_size, total_samples)

    # =========================================================================
    # PHASE 2: Optimization with GRPO clipped surrogate loss
    # Multiple epochs over collected data - after each epoch, weights change
    # so logp_new differs from logp_old, making ratio != 1.0
    # =========================================================================
    lrm = get_lr_multiplier(step)  # Get LR multiplier early for epoch loop

    for grpo_epoch in range(grpo_epochs):
        did_backward = False
        indices = list(range(total_samples))
        rng = random.Random(step * 1000 + grpo_epoch * 97 + ddp_rank)
        rng.shuffle(indices)

        for mb_start in range(0, total_samples, minibatch_size):
            mb_indices = indices[mb_start:mb_start + minibatch_size]
            batch_inputs = [flat_inputs[i] for i in mb_indices]
            max_len = max(t.size(0) for t in batch_inputs)

            def pad_1d(tensor, pad_value):
                if tensor.size(0) == max_len:
                    return tensor
                pad = torch.full((max_len - tensor.size(0),), pad_value, device=tensor.device, dtype=tensor.dtype)
                return torch.cat([tensor, pad], dim=0)

            inputs_all = torch.stack([pad_1d(t, pad_token_id) for t in batch_inputs], dim=0)
            targets_all = torch.stack([pad_1d(flat_targets[i], -1) for i in mb_indices], dim=0)
            logp_old_all = torch.stack([pad_1d(flat_logp_old[i], 0.0) for i in mb_indices], dim=0)
            valid_mask_all = torch.stack([pad_1d(flat_valid_mask[i], 0.0) for i in mb_indices], dim=0)
            advantages_all = torch.stack([flat_advantages[i] for i in mb_indices], dim=0)

            token_counts_all = valid_mask_all.sum(dim=1).clamp(min=1.0)
            logp_old_mean_all = (logp_old_all * valid_mask_all).sum(dim=1) / token_counts_all

            # IMPORTANT: Use eval() mode for logp_new to match logp_old (dropout consistency)
            # Gradients still flow through for backprop (eval() only disables dropout/batchnorm)
            train_model.eval()
            if use_kl:
                ref_model.eval()  # Ensure ref_model is also in eval mode

            batch_size = inputs_all.size(0)
            effective_batch_size = min(device_batch_size, batch_size)
            pass_kl_list = []  # Non-negative surrogate
            pass_kl_abs_list = []
            pass_kl_gap_list = []  # Signed gap (can be negative)
            pass_kl_max_tok_list = []  # Max token-level deviation
            pass_clip_frac_list = []
            pass_ratio_list = []
            pass_ratio_min_list = []
            pass_ratio_max_list = []
            pass_log_ratio_list = []
            pass_adv_mean_list = []
            pass_adv_std_list = []
            pass_adv_max_list = []

            # DDP optimization: only sync grads on last minibatch of last epoch
            is_last_minibatch = (grpo_epoch == grpo_epochs - 1) and (mb_start + minibatch_size >= total_samples)
            use_no_sync = ddp and not is_last_minibatch and hasattr(train_model, 'no_sync')
            sync_ctx = train_model.no_sync() if use_no_sync else contextlib.nullcontext()

            with sync_ctx:
                for b0 in range(0, batch_size, effective_batch_size):
                    b1 = min(b0 + effective_batch_size, batch_size)
                    inputs = inputs_all[b0:b1]
                    targets = targets_all[b0:b1]
                    advantages = advantages_all[b0:b1]
                    logp_old = logp_old_all[b0:b1]
                    logp_old_mean = logp_old_mean_all[b0:b1]
                    valid_mask = valid_mask_all[b0:b1]

                    # Compute logp_new from CURRENT policy (eval mode for dropout consistency)
                    with autocast_ctx:
                        logp_new = -train_model(inputs, targets, loss_reduction='none').view_as(inputs)
                    token_counts = valid_mask.sum(dim=1).clamp(min=1.0)
                    logp_new_sum = (logp_new * valid_mask).sum(dim=1)
                    logp_new_mean = logp_new_sum / token_counts

                    # Token-level truncated importance sampling
                    log_ratio = logp_new - logp_old
                    log_ratio_clamped = torch.clamp(log_ratio, -20.0, 20.0)
                    ratio = torch.exp(log_ratio_clamped)
                    if clip_ratio_low > 0 and clip_ratio_high > 0:
                        ratio_truncated = torch.clamp(ratio, min=clip_ratio_low, max=clip_ratio_high)
                    else:
                        ratio_truncated = torch.clamp(ratio, max=1.0 + clip_eps)

                    # Zero-gradient filtering
                    adv_nonzero = (advantages.abs() > zero_adv_eps)
                    if zero_grad_filtering:
                        adv_mask = adv_nonzero
                    else:
                        adv_mask = torch.ones_like(advantages, dtype=torch.bool)
                    adv_mask = adv_mask.float().unsqueeze(1)
                    token_mask = valid_mask * adv_mask
                    token_denom = token_mask.sum().clamp(min=1.0)

                    # Token-level loss: average over valid tokens
                    pg_loss = -(ratio_truncated * advantages.unsqueeze(1) * token_mask).sum() / token_denom

                    # Track clipping and diagnostic statistics (token-level)
                    if clip_ratio_low > 0 and clip_ratio_high > 0:
                        clip_frac = ((ratio < clip_ratio_low) | (ratio > clip_ratio_high)).float()
                    else:
                        clip_frac = ((ratio - 1.0) > clip_eps).float()
                    clip_frac = (clip_frac * token_mask).sum() / token_denom
                    ratio_masked = ratio[token_mask.bool()]
                    ratio_mean = ratio_masked.mean().detach().item() if ratio_masked.numel() else 0.0
                    ratio_min = ratio_masked.min().detach().item() if ratio_masked.numel() else 0.0
                    ratio_max = ratio_masked.max().detach().item() if ratio_masked.numel() else 0.0
                    log_ratio_mean = ((log_ratio * token_mask).sum() / token_denom).detach().item()
                    pass_clip_frac_list.append(clip_frac.detach().item())
                    pass_ratio_list.append(ratio_mean)
                    pass_ratio_min_list.append(ratio_min)
                    pass_ratio_max_list.append(ratio_max)
                    pass_log_ratio_list.append(log_ratio_mean)
                    pass_adv_mean_list.append(advantages.mean().detach().item())
                    pass_adv_std_list.append(advantages.std().detach().item() if advantages.numel() > 1 else 0.0)
                    pass_adv_max_list.append(advantages.abs().max().detach().item())

                    # KL regularization to SFT reference (robust sequence-level surrogate)
                    if use_kl:
                        # Always score with dropout OFF for stable KL measurements
                        ref_model.eval()
                        with torch.no_grad():
                            with autocast_ctx:
                                logp_ref = -ref_model(inputs, targets, loss_reduction='none').view_as(inputs)

                        # logp_new is per-token log-prob: [batch, seq_len]
                        # valid_mask: 1 on generated tokens, 0 on prompt: [batch, seq_len]
                        log_ratio_ref = logp_new - logp_ref  # [batch, seq_len]

                        # Compute sequence-level mean log ratio (not token-level!)
                        # This avoids exp() blow-up from rare token outliers
                        x_seq = (log_ratio_ref * valid_mask).sum(dim=1) / token_counts  # [batch]

                        # (A) Diagnostic: signed log-likelihood gap (can be negative)
                        kl_gap = x_seq.mean()  # can be ±

                        # (B) Diagnostic: absolute gap (>= 0)
                        kl_abs = x_seq.abs().mean()  # >= 0

                        # (C) Diagnostic: max token-level deviation (to detect outliers)
                        x_tok_masked = log_ratio_ref * valid_mask
                        kl_max_tok = x_tok_masked.abs().max().item()  # for debugging

                        # (D) Penalty: non-negative KL surrogate on SEQUENCE-LEVEL (robust)
                        # exp(x) - 1 - x >= 0, applied to sequence mean, not per-token
                        # This prevents rare token outliers from causing exp() explosion
                        x_seq_clamped = torch.clamp(x_seq, min=-10.0, max=10.0)
                        kl_surrogate = (torch.exp(x_seq_clamped) - 1.0 - x_seq_clamped).mean()  # >= 0
                    else:
                        kl_gap = torch.zeros((), device=inputs.device, dtype=torch.float32)
                        kl_abs = torch.zeros((), device=inputs.device, dtype=torch.float32)
                        kl_surrogate = torch.zeros((), device=inputs.device, dtype=torch.float32)
                        kl_max_tok = 0.0

                    # Use the non-negative surrogate in the loss (always acts as proper penalty)
                    loss = pg_loss + kl_coef * kl_surrogate

                    if backend == "deepspeed":
                        train_model.backward(loss)
                    else:
                        loss.backward()
                    did_backward = True

                    if use_kl:
                        pass_kl_list.append(kl_surrogate.detach().float().item())  # Non-negative surrogate
                        pass_kl_abs_list.append(kl_abs.detach().float().item())
                        pass_kl_gap_list.append(kl_gap.detach().float().item())  # Signed gap for diagnostics
                        pass_kl_max_tok_list.append(kl_max_tok)  # Max token deviation

                    # Debug logging for first few steps (rank 0 only)
                    # Log epoch 0 and epoch 1 to verify ratio changes after optimizer step
                    if step < 3 and mb_start == 0 and b0 == 0 and master_process:
                        logp_ref_mean_val = ((logp_ref * valid_mask).sum(dim=1) / token_counts).mean().item() if use_kl else 0.0
                        kl_surr_val = kl_surrogate.item() if use_kl else 0.0
                        kl_gap_val = kl_gap.item() if use_kl else 0.0
                        kl_max_val = kl_max_tok if use_kl else 0.0
                        print0(f"  [DEBUG step={step} epoch={grpo_epoch}] "
                               f"logp_new={logp_new_mean.mean().item():.4f}, "
                               f"logp_old={logp_old_mean.mean().item():.4f}, "
                               f"logp_ref={logp_ref_mean_val:.4f}, "
                               f"KL_surr={kl_surr_val:.4f}, "
                               f"KL_gap={kl_gap_val:.4f}, "
                               f"KL_max_tok={kl_max_val:.2f}, "
                               f"ratio=[{ratio.min().item():.3f},{ratio.max().item():.3f}]")

            # Clip gradients after all passes for this minibatch
            if backend == "ddp":
                clip_from_optimizers(optimizers, MAX_GRAD_NORM)
            elif backend != "deepspeed":
                torch.nn.utils.clip_grad_norm_(train_model.parameters(), MAX_GRAD_NORM)

            # Accumulate metrics
            if pass_kl_list:
                kl_list.append(sum(pass_kl_list) / len(pass_kl_list))  # kl_surrogate
                kl_abs_list.append(sum(pass_kl_abs_list) / len(pass_kl_abs_list))
                kl_gap_list.append(sum(pass_kl_gap_list) / len(pass_kl_gap_list))
                kl_max_tok_list.append(max(pass_kl_max_tok_list))  # Track max across passes
            if pass_clip_frac_list:
                clip_frac_list.append(sum(pass_clip_frac_list) / len(pass_clip_frac_list))
                ratio_list.append(sum(pass_ratio_list) / len(pass_ratio_list))
                ratio_min_list.append(min(pass_ratio_min_list))
                ratio_max_list.append(max(pass_ratio_max_list))
                log_ratio_list.append(sum(pass_log_ratio_list) / len(pass_log_ratio_list))
                adv_mean_list.append(sum(pass_adv_mean_list) / len(pass_adv_mean_list))
                adv_std_list.append(sum(pass_adv_std_list) / len(pass_adv_std_list))
                adv_max_list.append(max(pass_adv_max_list))

        # Optimizer step AFTER each epoch (not just at end of step!)
        # This is what makes ratio != 1.0 in subsequent epochs
        # Apply grpo_lr_scale since we do grpo_epochs updates per step
        if did_backward:
            effective_lr_mult = lrm * grpo_lr_scale
            if backend == "deepspeed":
                for group in train_model.optimizer.param_groups:
                    group["lr"] = group["initial_lr"] * effective_lr_mult
                train_model.step()
                train_model.zero_grad(set_to_none=True)
            else:
                for opt in optimizers:
                    for group in opt.param_groups:
                        group["lr"] = group["initial_lr"] * effective_lr_mult
                for opt in optimizers:
                    opt.step()
                train_model.zero_grad(set_to_none=True)
        else:
            train_model.zero_grad(set_to_none=True)

    mean_reward = sum(rewards_list) / len(rewards_list)
    mean_sequence_length = sum(sequence_lengths) / len(sequence_lengths)
    mean_response_length = sum(response_length_list) / len(response_length_list) if response_length_list else 0.0
    mean_policy_entropy = policy_entropy_sum / policy_entropy_count if policy_entropy_count > 0 else 0.0
    mean_kl = sum(kl_list) / len(kl_list) if kl_list else 0.0  # kl_surrogate (non-negative)
    mean_kl_abs = sum(kl_abs_list) / len(kl_abs_list) if kl_abs_list else 0.0
    mean_kl_gap = sum(kl_gap_list) / len(kl_gap_list) if kl_gap_list else 0.0  # signed gap
    max_kl_tok = max(kl_max_tok_list) if kl_max_tok_list else 0.0  # max token deviation
    reward_nonzero_rate = (reward_nonzero / reward_total) if reward_total else 0.0
    # New metrics means
    mean_raw_reward = sum(raw_rewards_list) / len(raw_rewards_list) if raw_rewards_list else 0.0
    mean_thought_length = sum(thought_length_list) / len(thought_length_list) if thought_length_list else 0.0
    mean_answer_length = sum(answer_length_list) / len(answer_length_list) if answer_length_list else 0.0
    mean_length_penalty = sum(length_penalty_list) / len(length_penalty_list) if length_penalty_list else 0.0
    total_gen_length = mean_thought_length + mean_answer_length
    mean_thought_frac = mean_thought_length / total_gen_length if total_gen_length > 0 else 0.0
    length_penalty_applied_rate = length_penalty_applied_count / reward_total if reward_total else 0.0
    # GRPO clipping metrics
    mean_clip_frac = sum(clip_frac_list) / len(clip_frac_list) if clip_frac_list else 0.0
    mean_ratio = sum(ratio_list) / len(ratio_list) if ratio_list else 1.0
    min_ratio = min(ratio_min_list) if ratio_min_list else 1.0
    max_ratio = max(ratio_max_list) if ratio_max_list else 1.0
    mean_log_ratio = sum(log_ratio_list) / len(log_ratio_list) if log_ratio_list else 0.0
    mean_adv = sum(adv_mean_list) / len(adv_mean_list) if adv_mean_list else 0.0
    mean_adv_std = sum(adv_std_list) / len(adv_std_list) if adv_std_list else 0.0
    max_adv = max(adv_max_list) if adv_max_list else 0.0

    if ddp:
        mean_reward_tensor = torch.tensor(mean_reward, dtype=torch.float, device=device)
        mean_sequence_tensor = torch.tensor(mean_sequence_length, dtype=torch.float, device=device)
        mean_response_tensor = torch.tensor(mean_response_length, dtype=torch.float, device=device)
        mean_kl_tensor = torch.tensor(mean_kl, dtype=torch.float, device=device)
        mean_kl_abs_tensor = torch.tensor(mean_kl_abs, dtype=torch.float, device=device)
        mean_kl_gap_tensor = torch.tensor(mean_kl_gap, dtype=torch.float, device=device)
        max_kl_tok_tensor = torch.tensor(max_kl_tok, dtype=torch.float, device=device)
        policy_entropy_sum_tensor = torch.tensor(policy_entropy_sum, dtype=torch.float, device=device)
        policy_entropy_count_tensor = torch.tensor(policy_entropy_count, dtype=torch.float, device=device)
        reward_nonzero_tensor = torch.tensor(reward_nonzero, dtype=torch.float, device=device)
        reward_total_tensor = torch.tensor(reward_total, dtype=torch.float, device=device)
        reward_min_tensor = torch.tensor(reward_min if reward_min is not None else float("inf"), dtype=torch.float, device=device)
        reward_max_tensor = torch.tensor(reward_max if reward_max is not None else float("-inf"), dtype=torch.float, device=device)
        dist.all_reduce(mean_reward_tensor, op=dist.ReduceOp.AVG)
        dist.all_reduce(mean_sequence_tensor, op=dist.ReduceOp.AVG)
        dist.all_reduce(mean_response_tensor, op=dist.ReduceOp.AVG)
        dist.all_reduce(mean_kl_tensor, op=dist.ReduceOp.AVG)
        dist.all_reduce(mean_kl_abs_tensor, op=dist.ReduceOp.AVG)
        dist.all_reduce(mean_kl_gap_tensor, op=dist.ReduceOp.AVG)
        dist.all_reduce(max_kl_tok_tensor, op=dist.ReduceOp.MAX)
        dist.all_reduce(policy_entropy_sum_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(policy_entropy_count_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(reward_nonzero_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(reward_total_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(reward_min_tensor, op=dist.ReduceOp.MIN)
        dist.all_reduce(reward_max_tensor, op=dist.ReduceOp.MAX)
        dist.all_reduce(reward_hist, op=dist.ReduceOp.SUM)
        mean_reward = mean_reward_tensor.item()
        mean_sequence_length = mean_sequence_tensor.item()
        mean_response_length = mean_response_tensor.item()
        mean_kl = mean_kl_tensor.item()
        mean_kl_abs = mean_kl_abs_tensor.item()
        mean_kl_gap = mean_kl_gap_tensor.item()
        max_kl_tok = max_kl_tok_tensor.item()
        policy_entropy_sum = policy_entropy_sum_tensor.item()
        policy_entropy_count = policy_entropy_count_tensor.item()
        mean_policy_entropy = policy_entropy_sum / policy_entropy_count if policy_entropy_count > 0 else 0.0
        reward_nonzero = int(reward_nonzero_tensor.item())
        reward_total = int(reward_total_tensor.item())
        reward_nonzero_rate = (reward_nonzero / reward_total) if reward_total else 0.0
        reward_min = reward_min_tensor.item()
        reward_max = reward_max_tensor.item()
        # Reduce new metrics
        raw_reward_tensor = torch.tensor(mean_raw_reward, dtype=torch.float, device=device)
        thought_len_tensor = torch.tensor(mean_thought_length, dtype=torch.float, device=device)
        answer_len_tensor = torch.tensor(mean_answer_length, dtype=torch.float, device=device)
        len_penalty_tensor = torch.tensor(mean_length_penalty, dtype=torch.float, device=device)
        len_penalty_count_tensor = torch.tensor(length_penalty_applied_count, dtype=torch.float, device=device)
        dist.all_reduce(raw_reward_tensor, op=dist.ReduceOp.AVG)
        dist.all_reduce(thought_len_tensor, op=dist.ReduceOp.AVG)
        dist.all_reduce(answer_len_tensor, op=dist.ReduceOp.AVG)
        dist.all_reduce(len_penalty_tensor, op=dist.ReduceOp.AVG)
        dist.all_reduce(len_penalty_count_tensor, op=dist.ReduceOp.SUM)
        mean_raw_reward = raw_reward_tensor.item()
        mean_thought_length = thought_len_tensor.item()
        mean_answer_length = answer_len_tensor.item()
        mean_length_penalty = len_penalty_tensor.item()
        length_penalty_applied_count = int(len_penalty_count_tensor.item())
        # Recompute derived metrics after reduction
        total_gen_length = mean_thought_length + mean_answer_length
        mean_thought_frac = mean_thought_length / total_gen_length if total_gen_length > 0 else 0.0
        length_penalty_applied_rate = length_penalty_applied_count / reward_total if reward_total else 0.0
        # Reduce GRPO clipping metrics
        clip_frac_tensor = torch.tensor(mean_clip_frac, dtype=torch.float, device=device)
        ratio_tensor = torch.tensor(mean_ratio, dtype=torch.float, device=device)
        ratio_min_tensor = torch.tensor(min_ratio, dtype=torch.float, device=device)
        ratio_max_tensor = torch.tensor(max_ratio, dtype=torch.float, device=device)
        log_ratio_tensor = torch.tensor(mean_log_ratio, dtype=torch.float, device=device)
        adv_tensor = torch.tensor(mean_adv, dtype=torch.float, device=device)
        adv_std_tensor = torch.tensor(mean_adv_std, dtype=torch.float, device=device)
        adv_max_tensor = torch.tensor(max_adv, dtype=torch.float, device=device)
        dist.all_reduce(clip_frac_tensor, op=dist.ReduceOp.AVG)
        dist.all_reduce(ratio_tensor, op=dist.ReduceOp.AVG)
        dist.all_reduce(ratio_min_tensor, op=dist.ReduceOp.MIN)
        dist.all_reduce(ratio_max_tensor, op=dist.ReduceOp.MAX)
        dist.all_reduce(log_ratio_tensor, op=dist.ReduceOp.AVG)
        dist.all_reduce(adv_tensor, op=dist.ReduceOp.AVG)
        dist.all_reduce(adv_std_tensor, op=dist.ReduceOp.AVG)
        dist.all_reduce(adv_max_tensor, op=dist.ReduceOp.MAX)
        mean_clip_frac = clip_frac_tensor.item()
        mean_ratio = ratio_tensor.item()
        min_ratio = ratio_min_tensor.item()
        max_ratio = ratio_max_tensor.item()
        mean_log_ratio = log_ratio_tensor.item()
        mean_adv = adv_tensor.item()
        mean_adv_std = adv_std_tensor.item()
        max_adv = adv_max_tensor.item()

    # KL explosion warning (now using non-negative surrogate)
    kl_warning = ""
    if mean_kl > kl_max_threshold:
        kl_warning = f" ⚠️  KL>{kl_max_threshold:.0f}!"
    print0(f"Step {step}/{num_steps} | Reward: {mean_reward:.4f} | KL_surr: {mean_kl:.4f} | KL_gap: {mean_kl_gap:.4f} | Ratio: [{min_ratio:.3f}, {max_ratio:.3f}] | ClipFrac: {mean_clip_frac:.3f}{kl_warning}")

    # lrm already computed at start of Phase 2 (line 690)
    current_temp = get_temperature(step)

    if master_process:
        # Build complete log dict ONCE per step to avoid W&B step conflicts
        # Use "train/" prefix for all training metrics so they appear in the same wandb section
        log_dict = {
            "trainer/global_step": step,
            # Reward metrics
            "train/reward": mean_reward,
            "train/reward_raw": mean_raw_reward,
            "train/reward_min": reward_min,
            "train/reward_max": reward_max,
            "train/reward_nonzero_rate": reward_nonzero_rate,
            "train/reward_nonzero": reward_nonzero,
            "train/reward_total": reward_total,
            # KL metrics (surrogate is non-negative, gap can be ±)
            "train/kl_surrogate": mean_kl,  # Non-negative, used in loss (sequence-level)
            "train/kl_gap": mean_kl_gap,    # Signed log-likelihood gap (can be negative)
            "train/kl_abs": mean_kl_abs,    # Absolute gap magnitude
            "train/kl_max_tok": max_kl_tok, # Max token-level deviation (outlier detector)
            # GRPO clipping metrics
            "train/clip_frac": mean_clip_frac,
            "train/ratio": mean_ratio,
            "train/ratio_min": min_ratio,
            "train/ratio_max": max_ratio,
            "train/log_ratio": mean_log_ratio,
            # Advantage diagnostics
            "train/adv_mean": mean_adv,
            "train/adv_std": mean_adv_std,
            "train/adv_max": max_adv,
            # Sequence/generation metrics
            "train/sequence_length": mean_sequence_length,
            "train/response_length": mean_response_length,
            "train/thought_length": mean_thought_length,
            "train/answer_length": mean_answer_length,
            "train/thought_frac": mean_thought_frac,
            "train/length_penalty": mean_length_penalty,
            "train/length_penalty_applied_rate": length_penalty_applied_rate,
            "train/policy_entropy": mean_policy_entropy,
            # Hyperparameters (for tracking schedules)
            "train/temperature": current_temp,
            "train/lrm": lrm,
            "train/lrm_effective": lrm * grpo_lr_scale,
            "train/grpo_epochs": grpo_epochs,
            "train/grpo_lr_scale": grpo_lr_scale,
        }
        # Add histogram metrics
        reward_hist_counts = reward_hist.tolist()
        for idx, label in enumerate(reward_hist_labels):
            count = reward_hist_counts[idx]
            rate = (count / reward_total) if reward_total else 0.0
            log_dict[f"train/{label}_count"] = count
            log_dict[f"train/{label}_rate"] = rate
        # Add task counts
        for name, count in task_counts.items():
            log_dict[f"task/{name}"] = count
        # Single log call per step
        wandb_run.log(log_dict)
    # NOTE: optimizer.step() now happens inside the grpo_epochs loop (after each epoch)
    # so we don't need another step here

    if master_process and eval_set and (step % eval_every == 0 or step == num_steps - 1):
        pass_at_1, per_task = _run_pass_at_1(eval_set)
        eval_log = {"trainer/global_step": step, "eval/pass_at_1": pass_at_1}
        for name, stats in per_task.items():
            total = stats["total"]
            correct = stats["correct"]
            rate = (correct / total) if total else 0.0
            eval_log[f"eval/{name}_pass_at_1"] = rate
            eval_log[f"eval/{name}_total"] = total
        wandb_run.log(eval_log)

    if master_process and ((step > 0 and step % save_every == 0) or step == num_steps - 1):
        base_dir = get_base_dir()
        depth = orig_model.config.n_layer
        model_tag = f"d{depth}"
        checkpoint_dir = os.path.join(base_dir, "chatrl_checkpoints", model_tag)
        model_config_kwargs = orig_model.config.__dict__
        if backend == "deepspeed":
            client_state = {"step": step, "model_config": model_config_kwargs, "user_config": user_config}
            train_model.save_checkpoint(checkpoint_dir, tag=str(step), client_state=client_state)
        elif backend == "fsdp":
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType

            with FSDP.state_dict_type(train_model, StateDictType.FULL_STATE_DICT, state_dict_config=fsdp_state_dict_config):
                full_state = train_model.state_dict()
            save_checkpoint(checkpoint_dir, step, full_state, [opt.state_dict() for opt in optimizers], {
                "model_config": model_config_kwargs,
                "user_config": user_config,
            })
        else:
            save_checkpoint(checkpoint_dir, step, orig_model.state_dict(), [opt.state_dict() for opt in optimizers], {
                "model_config": model_config_kwargs,
                "user_config": user_config,
            })
        print0(f"✅ Saved model checkpoint to {checkpoint_dir}")

wandb_run.finish()
compute_cleanup()
