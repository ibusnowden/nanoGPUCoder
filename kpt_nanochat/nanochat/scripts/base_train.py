"""
Train model. Run as:

python base_train.py

or distributed as:

torchrun --nproc_per_node=8 base_train.py
"""

import json
import os
import time
from pathlib import Path
try:
    import wandb
except ImportError:
    wandb = None
import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, print_banner, get_base_dir
from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint, find_last_step
from nanochat.loss_eval import evaluate_bpb
print_banner()


def build_adamw_all_params(model, embedding_lr, unembedding_lr, matrix_lr, weight_decay):
    """Single AdamW optimizer that mirrors the LR scaling logic of setup_optimizers."""
    model_dim = model.config.n_embd
    dmodel_lr_scale = (model_dim / 768) ** -0.5
    print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")
    embedding_params = list(model.transformer.wte.parameters())
    lm_head_params = list(model.lm_head.parameters())
    used_ids = {id(p) for p in embedding_params + lm_head_params}
    other_params = [p for p in model.parameters() if id(p) not in used_ids]
    adam_groups = [
        dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale),
        dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),
        dict(params=other_params, lr=matrix_lr * dmodel_lr_scale),
    ]
    adamw_kwargs = dict(betas=(0.8, 0.95), eps=1e-10, weight_decay=weight_decay, fused=True)
    optimizer = torch.optim.AdamW(adam_groups, **adamw_kwargs)
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]
    return optimizer

# Resume (optional defaults; can be overridden via configurator)
load_checkpoint_dir = "" # absolute path or relative to base_checkpoints; empty = no resume
load_checkpoint_step = -1 # if <0, auto-pick last step in the directory
load_checkpoint_optimizer = 0 # 1 = also load optimizer state (when available)

# -----------------------------------------------------------------------------
# User settings
run = "dummy" # wandb run name default ("dummy" is special - we won't log to wandb)
# Model architecture
architecture_style = "qwen25_small" # "qwen25_small", "qwen25_1.5b", "qwen25_7b", or "original" for backward compat
depth = 20 # the depth of the Transformer model to train (used for qwen25_small and original styles)
max_seq_len = 2048 # max context length
# Smoke test helpers
synthetic_data = 0 # 1 = use random tokens (no tokenizer/dataset required)
synthetic_vocab_size = 65536 # only used when synthetic_data=1
skip_core_metric = 0 # 1 = skip CORE eval_bundle metric (useful on clusters without staged eval_bundle)
skip_sampling = 0 # 1 = skip prompt sampling (useful for fast smoke tests)
# MoE (optional)
moe_num_experts = 0 # 0 disables MoE
moe_top_k = 1 # 1 or 2
moe_layer_start = 0
moe_layer_end = -1
moe_layer_stride = 1
moe_capacity_factor = 1.25
moe_aux_loss_coef = 0.01
# Training horizon. Only one of these 3 will be used, in this order of precedence.
num_iterations = -1 # explicit number of steps of the optimization (-1 = disable)
target_flops = -1.0 # calculate num_iterations to reach target_flops. Useful for scaling laws experiments (-1 = disable)
target_param_data_ratio = 20 # calculate num_iterations to maintain fixed data:param ratio (Chinchilla=20) (-1 = disable)
# Optimization
device_batch_size = 32 # per-device batch size (set to not OOM)
total_batch_size = 524288 # total desired batch size, in #tokens
embedding_lr = 0.2 # learning rate for the embedding parameters (Adam)
unembedding_lr = 0.004 # learning rate for the unembedding parameters (Adam)
weight_decay = 0.0 # weight decay for the embedding/unembedding parameters (Adam)
matrix_lr = 0.02 # learning rate for the matrix parameters (Muon)
grad_clip = 1.0 # gradient clipping value (0.0 = disabled)
# Optional sharding/offload backends
use_deepspeed = 0 # 1 = DeepSpeed ZeRO-3 (plain AdamW)
deepspeed_config = "slurm/deepspeed_zero3.json"
use_fsdp = 0 # 1 = Torch FSDP full-shard (plain AdamW)
fsdp_min_num_params = 1_000_000 # auto-wrap threshold
fsdp_cpu_offload = 0 # 1 = offload params to CPU (slow; last resort)
# Evaluation
eval_every = 250 # every how many steps to evaluate the model for val bpb
eval_tokens = 20*524288 # number of tokens to evaluate val loss on
core_metric_every = 2000 # every how many steps to evaluate the core metric
core_metric_max_per_task = 500 # examples per task in estimating the core metric
sample_every = 2000 # every how many steps to sample from the model
# Output
model_tag = "" # optionally override the model tag for the output checkpoint directory name
# now allow CLI to override the settings via the configurator lol
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join('nanochat', 'configurator.py')).read()) # overrides from command line or config file
user_config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# Compute init
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
use_deepspeed_flag = bool(use_deepspeed)
use_fsdp_flag = bool(use_fsdp)
if use_deepspeed_flag and use_fsdp_flag:
    raise ValueError("use_deepspeed and use_fsdp are mutually exclusive")
backend = "deepspeed" if use_deepspeed_flag else "fsdp" if use_fsdp_flag else "ddp"
if backend != "ddp":
    print0(f"Using backend={backend}")
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

# wandb logging init
use_dummy_wandb = run == "dummy" or not master_process
if not use_dummy_wandb and wandb is None:
    print0("wandb not installed; proceeding without wandb logging")
    use_dummy_wandb = True
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat", name=run, config=user_config)

# Tokenizer + token_bytes (for bpb). Optional in synthetic mode.
tokenizer = None
if synthetic_data:
    vocab_size = int(synthetic_vocab_size)
    token_bytes = torch.ones((vocab_size,), dtype=torch.int32, device=device)
    print0(f"Synthetic data enabled (vocab_size={vocab_size:,})")
    if skip_core_metric == 0:
        print0("Forcing skip_core_metric=1 because synthetic_data=1")
        skip_core_metric = 1
    if skip_sampling == 0:
        print0("Forcing skip_sampling=1 because synthetic_data=1")
        skip_sampling = 1
else:
    from nanochat.tokenizer import get_tokenizer, get_token_bytes
    tokenizer = get_tokenizer()
    token_bytes = get_token_bytes(device=device)
    vocab_size = tokenizer.get_vocab_size()
    print0(f"Vocab size: {vocab_size:,}")

# Model configuration based on architecture style
from nanochat.model_configs import (
    get_qwen25_small_config,
    get_qwen25_1_5b_config,
    get_qwen25_7b_config,
    get_nanochat_original_config,
)

if architecture_style == "qwen25_small":
    model_config = get_qwen25_small_config(vocab_size=vocab_size, sequence_len=max_seq_len, depth=depth)
    print0(f"Using Qwen2.5-style small model (depth={depth})")
elif architecture_style == "qwen25_1.5b":
    model_config = get_qwen25_1_5b_config(vocab_size=vocab_size, sequence_len=max_seq_len)
    print0("Using Qwen2.5-Coder-1.5B configuration")
elif architecture_style == "qwen25_7b":
    model_config = get_qwen25_7b_config(vocab_size=vocab_size, sequence_len=max_seq_len)
    print0("Using Qwen2.5-Coder-7B configuration")
elif architecture_style == "original":
    model_config = get_nanochat_original_config(vocab_size=vocab_size, sequence_len=max_seq_len, depth=depth)
    print0(f"Using original nanochat configuration (depth={depth})")
else:
    raise ValueError(f"Unknown architecture_style: {architecture_style}")

# Apply MoE settings (kept here so they can be CLI-overridden via configurator.py)
model_config.moe_num_experts = moe_num_experts
model_config.moe_top_k = moe_top_k
model_config.moe_layer_start = moe_layer_start
model_config.moe_layer_end = moe_layer_end
model_config.moe_layer_stride = moe_layer_stride
model_config.moe_capacity_factor = moe_capacity_factor
model_config.moe_aux_loss_coef = moe_aux_loss_coef
if moe_num_experts > 0:
    print0(
        f"MoE enabled: experts={moe_num_experts}, top_k={moe_top_k}, "
        f"layers=[{moe_layer_start}:{moe_layer_end}:{moe_layer_stride}], "
        f"capacity_factor={moe_capacity_factor}, aux_coef={moe_aux_loss_coef}"
    )

# -----------------------------------------------------------------------------
# Optional resume
resume_model_data = None
resume_optim_data = None
resume_meta = None
start_step = 0
if load_checkpoint_dir:
    checkpoint_dir = Path(load_checkpoint_dir)
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = Path(get_base_dir()) / "base_checkpoints" / load_checkpoint_dir
    resume_step = load_checkpoint_step
    if resume_step < 0:
        resume_step = find_last_step(str(checkpoint_dir))
    print0(f"Loading checkpoint from {checkpoint_dir} at step {resume_step}")
    resume_model_data, resume_optim_data, resume_meta = load_checkpoint(
        str(checkpoint_dir),
        resume_step,
        device="cpu",
        load_optimizer=bool(load_checkpoint_optimizer),
    )
    start_step = resume_meta.get("step", resume_step) if resume_meta else resume_step

# Save full model config (for checkpoint round-tripping)
model_config_kwargs = model_config.__dict__

# Extract model dimensions for logging
num_layers = model_config.n_layer
model_dim = model_config.n_embd
num_heads = model_config.n_head
num_kv_heads = model_config.n_kv_head
intermediate_size = model_config.intermediate_size if model_config.intermediate_size else 4 * model_dim
print0(f"num_layers: {num_layers}")
print0(f"model_dim: {model_dim}")
print0(f"num_heads: {num_heads}")
print0(f"num_kv_heads: {num_kv_heads}")

# Optimizer / data / training length related hyperparameters
# figure out the needed gradient accumulation to reach the desired total batch size
tokens_per_fwdbwd = device_batch_size * max_seq_len # tokens per iteration for a single rank
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size # total tokens per iteration for all ranks
assert total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens / micro-batch / rank: {device_batch_size} x {max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")
# -----------------------------------------------------------------------------
# Initialize the Model
with torch.device("meta"):
    model = GPT(model_config)
model.to_empty(device="cuda")
model.init_weights()
if resume_model_data is not None:
    model.load_state_dict(resume_model_data, strict=True)
orig_model = model # original, uncompiled model, for saving raw model state_dict
fsdp_state_dict_config = None
if backend == "fsdp":
    from torch.distributed.fsdp import CPUOffload, FullStateDictConfig, MixedPrecision, ShardingStrategy
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType
    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

    auto_wrap_policy = size_based_auto_wrap_policy(min_num_params=int(fsdp_min_num_params))
    mp_policy = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16)
    cpu_offload = CPUOffload(offload_params=bool(fsdp_cpu_offload))
    fsdp_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mp_policy,
        cpu_offload=cpu_offload,
        device_id=ddp_local_rank,
        sync_module_states=True,
        use_orig_params=True,
    )
elif backend == "ddp":
    model = torch.compile(model, dynamic=False) # TODO: dynamic True/False think through
num_params = sum(p.numel() for p in orig_model.parameters())
print0(f"Number of parameters: {num_params:,}")
num_flops_per_token = orig_model.estimate_flops()
print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")

# Initialize GPU metrics collector
from nanochat.gpu_metrics import GPUMetricsCollector
gpu_metrics = GPUMetricsCollector(device=device)
print0("GPU metrics collector initialized")

# Calculate number of iterations. Either it is given, or from target flops, or from target data:param ratio (in that order)
assert num_iterations > 0 or target_param_data_ratio > 0 or target_flops > 0
if num_iterations > 0:
    print0(f"Using user-provided number of iterations: {num_iterations:,}")
elif target_flops > 0:
    # calculate the number of iterations from the target flops
    num_iterations = round(target_flops / (num_flops_per_token * total_batch_size))
    print0(f"Calculated number of iterations from target FLOPs: {num_iterations:,}")
elif target_param_data_ratio > 0:
    # calculate the number of iterations from the target param data ratio
    target_tokens = target_param_data_ratio * num_params
    num_iterations = target_tokens // total_batch_size
    print0(f"Calculated number of iterations from target data:param ratio: {num_iterations:,}")
else:
    raise ValueError("No training horizon specified")
total_tokens = total_batch_size * num_iterations
print0(f"Total number of training tokens: {total_tokens:,}")
print0(f"Tokens : Params ratio: {total_batch_size * num_iterations / num_params:.2f}") # Chinchilla is ~20
print0(f"Total training FLOPs estimate: {num_flops_per_token * total_tokens:e}")

# -----------------------------------------------------------------------------
# Initialize the Optimizer (Muon for Linear layers, AdamW for embedding and lm_head)
optimizers = []
muon_optimizer = None
adamw_optimizer = None
if backend == "ddp":
    optimizers = model.setup_optimizers(unembedding_lr=unembedding_lr, embedding_lr=embedding_lr, matrix_lr=matrix_lr, weight_decay=weight_decay)
    adamw_optimizer, muon_optimizer = optimizers
    # If requested, load optimizer state after they are constructed
    if resume_optim_data:
        if isinstance(resume_optim_data, (list, tuple)) and len(resume_optim_data) == len(optimizers):
            for opt, state in zip(optimizers, resume_optim_data):
                opt.load_state_dict(state)
            print0("Loaded optimizer state from checkpoint")
        else:
            print0("Warning: optimizer state provided but shape/count did not match; skipping optimizer load")
elif backend in ("fsdp", "deepspeed"):
    # Single AdamW across all parameters (Muon/DistAdamW are not compatible with FSDP/ZeRO-3)
    adamw_target = model if backend == "fsdp" else orig_model
    adamw_optimizer = build_adamw_all_params(adamw_target, embedding_lr=embedding_lr, unembedding_lr=unembedding_lr, matrix_lr=matrix_lr, weight_decay=weight_decay)
    optimizers = [adamw_optimizer]

# Initialize the DataLoaders for train/val
base_dir = get_base_dir()
tokens_dir = os.path.join(base_dir, "tokenized_data")
if synthetic_data:
    from nanochat.dataloader import synthetic_distributed_data_loader
    train_loader = synthetic_distributed_data_loader(device_batch_size, max_seq_len, vocab_size=vocab_size, seed=42)
    build_val_loader = lambda: synthetic_distributed_data_loader(device_batch_size, max_seq_len, vocab_size=vocab_size, seed=123)
else:
    from nanochat.dataloader import tokenizing_distributed_data_loader
    train_loader = tokenizing_distributed_data_loader(device_batch_size, max_seq_len, split="train")
    build_val_loader = lambda: tokenizing_distributed_data_loader(device_batch_size, max_seq_len, split="val")
x, y = next(train_loader) # kick off load of the very first batch of data

# -----------------------------------------------------------------------------
# Optional DeepSpeed initialization (ZeRO-3)
if backend == "deepspeed":
    import deepspeed
    if not os.path.isfile(deepspeed_config):
        raise FileNotFoundError(f"DeepSpeed config not found at {deepspeed_config}")
    with open(deepspeed_config, "r", encoding="utf-8") as f:
        ds_config = json.load(f)
    ds_config["train_micro_batch_size_per_gpu"] = int(device_batch_size)
    ds_config["gradient_accumulation_steps"] = int(grad_accum_steps)
    model, _, _, _ = deepspeed.initialize(model=orig_model, optimizer=adamw_optimizer, config=ds_config)
    for group in model.optimizer.param_groups:
        group.setdefault("initial_lr", group["lr"])

# -----------------------------------------------------------------------------
# Set up hyperparameter schedulers

# Learning rate scheduler
# TODO: experiment with a short warmup for the AdamW params (expecting slight improvement)
warmup_ratio = 0.0 # ratio of iterations for LR warmup
warmdown_ratio = 0.2 # ratio of iterations for LR warmdown
final_lr_frac = 0.0 # final LR is this fraction of the initial LR
def get_lr_multiplier(it):
    warmup_iters = round(warmup_ratio * num_iterations)
    warmdown_iters = round(warmdown_ratio * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * final_lr_frac

# Momentum scheduler for Muon optimizer
def get_muon_momentum(it):
    frac = min(it / 300, 1)
    momentum = (1 - frac) * 0.85 + frac * 0.95
    return momentum

# Active models for different backends
train_model = model
if backend == "deepspeed":
    eval_model = model.module
    sampling_model = model.module
elif backend == "ddp":
    eval_model = model
    sampling_model = orig_model
else:
    eval_model = model
    sampling_model = model

# -----------------------------------------------------------------------------
# Training loop
min_val_bpb = float("inf")
smooth_train_loss = 0 # EMA of training loss
ema_beta = 0.9 # EMA decay factor
total_training_time = 0 # total wall-clock time of training
core_results = None
# note that we run +1 steps only so that we can eval and save at the end
for step in range(start_step, num_iterations + 1):
    last_step = step == num_iterations
    flops_so_far = num_flops_per_token * total_batch_size * step

    # once in a while: evaluate the val bpb (all ranks participate)
    if last_step or step % eval_every == 0:
        eval_model.eval()
        val_loader = build_val_loader()
        eval_steps = eval_tokens // (device_batch_size * max_seq_len * ddp_world_size)
        with autocast_ctx:
            val_bpb = evaluate_bpb(eval_model, val_loader, eval_steps, token_bytes)
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.4f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "val/bpb": val_bpb,
        })
        train_model.train()

    # once in a while: estimate the CORE metric (all ranks participate)
    # use the original uncompiled model because the inputs keep changing shape
    if not skip_core_metric and (last_step or (step > 0 and step % core_metric_every == 0)):
        assert tokenizer is not None, "Tokenizer is required for CORE metric evaluation"
        from scripts.base_eval import evaluate_model
        eval_model.eval()
        with autocast_ctx:
            core_results = evaluate_model(sampling_model, tokenizer, device, max_per_task=core_metric_max_per_task)
        print0(f"Step {step:05d} | CORE metric: {core_results['core_metric']:.4f}")
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "core_metric": core_results["core_metric"],
            "centered_results": core_results["centered_results"],
        })
        train_model.train()

    # once in a while: sample from the model (only on master process)
    # use the original uncompiled model because the inputs keep changing shape
    if not skip_sampling and tokenizer is not None and master_process and (last_step or (step > 0 and step % sample_every == 0)):
        from nanochat.engine import Engine
        eval_model.eval()
        prompts = [
            "The capital of France is",
            "The chemical symbol of gold is",
            "If yesterday was Friday, then tomorrow will be",
            "The opposite of hot is",
            "The planets of the solar system are:",
            "My favorite color is",
            "If 5*x + 3 = 13, then x is",
        ]
        engine = Engine(sampling_model, tokenizer)
        for prompt in prompts:
            tokens = tokenizer(prompt, prepend="<|bos|>")
            with autocast_ctx:
                sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
            print0(tokenizer.decode(sample[0]))
        train_model.train()

    # save checkpoint at the end of the run (only on master process)
    if master_process and last_step:
        output_dirname = model_tag if model_tag else f"d{depth}" # e.g. d12
        checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)
        if backend == "deepspeed":
            client_state = {
                "step": step,
                "val_bpb": val_bpb,
                "model_config": model_config_kwargs,
                "user_config": user_config,
                "device_batch_size": device_batch_size,
                "max_seq_len": max_seq_len,
            }
            model.save_checkpoint(checkpoint_dir, tag=str(step), client_state=client_state)
        elif backend == "fsdp":
            with FSDP.state_dict_type(train_model, StateDictType.FULL_STATE_DICT, state_dict_config=fsdp_state_dict_config):
                full_state = train_model.state_dict()
            save_checkpoint(
                checkpoint_dir,
                step,
                full_state,
                [opt.state_dict() for opt in optimizers],
                {
                    "step": step,
                    "val_bpb": val_bpb,
                    "model_config": model_config_kwargs,
                    "user_config": user_config,
                    "device_batch_size": device_batch_size,
                    "max_seq_len": max_seq_len,
                }
            )
        else:
            save_checkpoint(
                checkpoint_dir,
                step,
                orig_model.state_dict(),
                [opt.state_dict() for opt in optimizers], # TODO: make sure saving across ranks is done correctly
                {
                    "step": step,
                    "val_bpb": val_bpb, # loss at last step
                    "model_config": model_config_kwargs,
                    "user_config": user_config, # inputs to the training script
                    "device_batch_size": device_batch_size,
                    "max_seq_len": max_seq_len,
                }
            )

    if last_step:
        break

    # -------------------------------------------------------------------------
    # single training step
    # evaluate the gradient
    torch.cuda.synchronize()
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = train_model(x, y)
        train_loss = loss.detach() # for logging
        if backend == "deepspeed":
            train_model.backward(loss)
        else:
            loss = loss / grad_accum_steps # each .backward() is a grad sum => normalize loss here
            loss.backward()
        x, y = next(train_loader) # prefetch the next batch while the GPU is busy with forward/backward
    # gradient clipping and optimizer step
    lrm = get_lr_multiplier(step)
    if backend == "deepspeed":
        for group in train_model.optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
        train_model.step()
        train_model.zero_grad(set_to_none=True)
    else:
        # gradient clipping (TODO possibly experiment with)
        if grad_clip > 0.0:
            target_params = orig_model.parameters() if backend == "ddp" else train_model.parameters()
            torch.nn.utils.clip_grad_norm_(target_params, grad_clip)
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["initial_lr"] * lrm
        if backend == "ddp" and muon_optimizer is not None:
            muon_momentum = get_muon_momentum(step)
            for group in muon_optimizer.param_groups:
                group["momentum"] = muon_momentum
        for opt in optimizers:
            opt.step()
        train_model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0
    # -------------------------------------------------------------------------

    # logging
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item() # EMA the training loss
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1)) # debias the EMA
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(world_tokens_per_fwdbwd / dt)
    flops_per_sec = num_flops_per_token * total_batch_size / dt
    promised_flops_per_sec_h100 = 989e12 * ddp_world_size # bfloat16 H100 SXM and without 2:4 sparsity
    mfu = 100 * flops_per_sec / promised_flops_per_sec_h100 # in %
    if step > 10:
        total_training_time += dt # only count the time after the first 10 steps
    print0(f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.2f} | total time: {total_training_time/60:.2f}m")
    
    # Collect GPU metrics
    current_gpu_metrics = gpu_metrics.collect()
    
    if step % 100 == 0:
        log_data = {
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
            "gpu/memory_allocated_mb": current_gpu_metrics.gpu_memory_allocated_mb,
            "gpu/memory_reserved_mb": current_gpu_metrics.gpu_memory_reserved_mb,
        }
        if current_gpu_metrics.gpu_utilization_percent is not None:
            log_data["gpu/utilization_percent"] = current_gpu_metrics.gpu_utilization_percent
        wandb_run.log(log_data)

# print a few more stats
print0(f"Peak memory usage: {torch.cuda.max_memory_allocated() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
print0(f"Minimum validation bpb: {min_val_bpb:.4f}")

# Log to report
from nanochat.report import get_report
core_metric = None if core_results is None else core_results.get("core_metric", None)
get_report().log(section="Base model training", data=[
    user_config, # CLI args
    { # stats about the training setup
        "Number of parameters": num_params,
        "Number of FLOPs per token": f"{num_flops_per_token:e}",
        "Calculated number of iterations": num_iterations,
        "Number of training tokens": total_tokens,
        "Tokens : Params ratio": total_batch_size * num_iterations / num_params,
        "DDP world size": ddp_world_size,
        "warmup_ratio": warmup_ratio,
        "warmdown_ratio": warmdown_ratio,
        "final_lr_frac": final_lr_frac,
    },
    { # stats about training outcomes
        "Minimum validation bpb": min_val_bpb,
        "Final validation bpb": val_bpb,
        "CORE metric estimate": core_metric,
        "MFU %": f"{mfu:.2f}%",
        "Total training flops": f"{flops_so_far:e}",
        "Total training time": f"{total_training_time/60:.2f}m",
        "Peak memory usage": f"{torch.cuda.max_memory_allocated() / 1024 / 1024:.2f}MiB",
    }
])

# cleanup
wandb_run.finish() # wandb run finish
compute_cleanup()
