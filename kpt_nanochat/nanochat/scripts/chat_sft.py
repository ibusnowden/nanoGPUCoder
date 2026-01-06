"""
Finetune a base model to be a chat model.
Run on one GPU e.g. for debugging:

python -m scripts.chat_sft

Or torchrun for training:

torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft
"""

import os

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
from nanochat.common import compute_init, compute_cleanup, get_base_dir, print0, DummyWandb
from nanochat.checkpoint_manager import load_model
from nanochat.checkpoint_manager import save_checkpoint
from nanochat.engine import Engine
from scripts.chat_eval import run_chat_eval

from nanochat.data_recipes import build_sft_recipe

# -----------------------------------------------------------------------------
# SFT Hyperparameters
run = "dummy" # wandb run name default ("dummy" is special - we won't log to wandb)
# input model options
source = "mid" # base|mid|sft , which checkpoint to load the model from
model_tag = None # model tag to load the model from (base model or midtrained model)
step = None # step to load the model from (base model or midtrained model)
# data recipe
sft_recipe = "default" # default|r1_ot|r1_ot_mixed|r1_rs_sft
ot_dataset = "open-thoughts/OpenThoughts3-1.2M"
ot_split = "train"
ot_stop = 500_000 # use -1 for full dataset
rs_cot_path = "" # required for r1_rs_sft
rs_cot_stop = 100_000
# GSM8K config (for r1_ot_mixed recipe)
gsm8k_stop = -1 # use -1 for full dataset
# Dolci-Think config (for dolci_mid recipe)
dolci_path = "" # path to preprocessed Dolci-Think JSONL
dolci_stop = 500_000 # use -1 for full dataset
dolci_streaming = 0 # 1 = stream from HF and cache only up to dolci_stop
dolci_stream_cache = "" # JSONL cache path for streamed subset
chat_ratio = 0.30
chat_ot_answer_ratio = 0.10
chat_ot_trace_ratio = 0.05
val_smoltalk_stop = 2000
val_arc_stop = 400
# compute/precision
dtype = "bfloat16"
device_batch_size = 4 # max to avoid OOM
# optimization
num_epochs = 1
max_iterations = -1 # override number of iterations (-1 = use num_epochs * num_iterations)
target_examples_per_step = 32
unembedding_lr = 0.004
embedding_lr = 0.2
matrix_lr = 0.02
weight_decay = 0.0
init_lr_frac = 0.02
use_deepspeed = 0 # 1 = DeepSpeed ZeRO-3 (plain AdamW)
deepspeed_config = "slurm/deepspeed_zero3.json"
use_fsdp = 0 # 1 = Torch FSDP full-shard (plain AdamW)
fsdp_min_num_params = 1_000_000 # auto-wrap threshold
fsdp_cpu_offload = 0 # 1 = offload params to CPU (slow; last resort)
# evaluation and logging there of
eval_every = 100
eval_steps = 100
eval_metrics_every = 200
# now allow CLI to override the settings via the configurator lol
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join('nanochat', 'configurator.py')).read()) # overrides from command line or config file
user_config = {k: globals()[k] for k in config_keys} # possibly useful for logging
# -----------------------------------------------------------------------------

# Compute init
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
master_process = ddp_rank == 0
backend = select_backend(use_deepspeed, use_fsdp)
if backend != "ddp":
    print0(f"Using backend={backend}")
dtype = torch.float32 if dtype == 'float32' else torch.bfloat16
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=dtype)

# wandb logging init
wandb_disabled = os.environ.get("WANDB_DISABLED", "false").lower() in ("1", "true", "yes")
use_dummy_wandb = run == "dummy" or not master_process or wandb_disabled
if not use_dummy_wandb and wandb is None:
    print0("wandb not installed; proceeding without wandb logging")
    use_dummy_wandb = True
wandb_project = os.environ.get("WANDB_PROJECT", "nanochat-sft")
wandb_run = DummyWandb() if use_dummy_wandb or wandb is None else wandb.init(
    project=wandb_project,
    name=run,
    config=user_config,
    save_code=True,
)

# Load the model and tokenizer
model, tokenizer, meta = load_model(source, device, phase="train", model_tag=model_tag, step=step)
orig_model = model # original, uncompiled model
model, fsdp_state_dict_config = wrap_fsdp_if_needed(
    model,
    backend=backend,
    ddp_local_rank=ddp_local_rank,
    fsdp_min_num_params=fsdp_min_num_params,
    fsdp_cpu_offload=fsdp_cpu_offload,
)
# model = torch.compile(model, dynamic=True) # doesn't work super well because of variable lengths of inputs
train_model = model

# -----------------------------------------------------------------------------
# Task data mixture we'll train on
resolved_ot_stop = None if ot_stop < 0 else ot_stop
resolved_rs_cot_path = rs_cot_path or None
resolved_gsm8k_stop = None if gsm8k_stop < 0 else gsm8k_stop
resolved_dolci_path = dolci_path or None
resolved_dolci_stop = None if dolci_stop < 0 else dolci_stop
resolved_dolci_stream_cache = dolci_stream_cache or None
train_ds, val_ds = build_sft_recipe(
    sft_recipe,
    ot_dataset=ot_dataset,
    ot_split=ot_split,
    ot_stop=resolved_ot_stop,
    rs_cot_path=resolved_rs_cot_path,
    rs_cot_stop=rs_cot_stop,
    chat_ratio=chat_ratio,
    chat_ot_answer_ratio=chat_ot_answer_ratio,
    chat_ot_trace_ratio=chat_ot_trace_ratio,
    gsm8k_stop=resolved_gsm8k_stop,
    val_smoltalk_stop=val_smoltalk_stop,
    val_arc_stop=val_arc_stop,
    # Dolci-Think kwargs
    dolci_path=resolved_dolci_path,
    dolci_stop=resolved_dolci_stop,
    dolci_streaming=bool(dolci_streaming),
    dolci_stream_cache=resolved_dolci_stream_cache,
)

# -----------------------------------------------------------------------------
# DataLoader

def sft_data_generator(dataset, batch_size):
    pad_token_id = tokenizer.encode_special("<|assistant_end|>") # use <|assistant_end|> as the pad token is ok, these positions are masked in the loss
    # prepares a list of tokenized conversations into a batch and yields
    def collate_and_yield(batch):
        nrows = len(batch)
        ncols = max(len(ids) for ids, mask in batch) - 1 # seq of n creates inputs/targets of n-1
        inputs = torch.full((nrows, ncols), pad_token_id, dtype=torch.long)
        targets = torch.full((nrows, ncols), -1, dtype=torch.long) # -1 is ignore index
        for i, (ids, mask) in enumerate(batch):
            n = len(ids)
            ids_tensor = torch.tensor(ids, dtype=torch.long)
            inputs[i, :n-1] = ids_tensor[:-1]
            # recall -1 is the ignore index, so mask out targets where mask is 0
            row_targets = ids_tensor[1:]
            # mask[1:] omits the mask for the BOS token, which is never a target atm so it's ok
            mask_tensor = torch.tensor(mask[1:], dtype=torch.long)
            row_targets[mask_tensor == 0] = -1 # mask out targets where mask is 0
            targets[i, :n-1] = row_targets
        inputs = inputs.to(device) # move to device
        targets = targets.to(device)
        return inputs, targets
    # iterates over the dataset in epochs, tokenizes
    batch = []
    while True:
        for i in range(ddp_rank, len(dataset), ddp_world_size):
            doc = dataset[i]
            ids, mask = tokenizer.render_conversation(doc)
            batch.append((ids, mask))
            if len(batch) == batch_size:
                yield collate_and_yield(batch)
                batch = []

examples_per_step = device_batch_size * ddp_world_size
print0(f"Target examples per step: {target_examples_per_step}")
print0(f"Device batch size: {device_batch_size}")
print0(f"Examples per step is device_batch_size * ddp_world_size: {examples_per_step}")
if target_examples_per_step % examples_per_step != 0:
    adjusted_target = -(-target_examples_per_step // examples_per_step) * examples_per_step
    print0(f"Adjusted target_examples_per_step from {target_examples_per_step} to {adjusted_target} so it's divisible by {examples_per_step}")
    target_examples_per_step = adjusted_target
grad_accum_steps = target_examples_per_step // examples_per_step
print0(f"=> Setting grad accum steps: {grad_accum_steps}")

num_iterations = (len(train_ds) // target_examples_per_step) * num_epochs
if max_iterations >= 0 and num_iterations > max_iterations:
    print0(f"Number of iterations is too high: {num_iterations}, capping to {max_iterations}")
    num_iterations = max_iterations
train_loader = sft_data_generator(train_ds, batch_size=device_batch_size)
build_val_loader = lambda: sft_data_generator(val_ds, batch_size=device_batch_size)

# -----------------------------------------------------------------------------
# Initialize the Optimizer

optimizers = []
adamw_optimizer = None
if backend == "ddp":
    optimizers = model.setup_optimizers(
        unembedding_lr=unembedding_lr,
        embedding_lr=embedding_lr,
        matrix_lr=matrix_lr,
        weight_decay=weight_decay,
    )
    # Set the initial learning rate as a fraction of the base learning rate
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["lr"] * init_lr_frac
            group["initial_lr"] = group["lr"] # save the initial learning so we can decay easily later
else:
    adamw_target = model if backend == "fsdp" else orig_model
    adamw_optimizer = build_adamw_all_params(
        adamw_target,
        embedding_lr=embedding_lr,
        unembedding_lr=unembedding_lr,
        matrix_lr=matrix_lr,
        weight_decay=weight_decay,
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
        grad_accum_steps=grad_accum_steps,
    )
eval_model = train_model.module if backend == "deepspeed" else train_model
sampling_model = eval_model if backend != "ddp" else orig_model
engine = Engine(sampling_model, tokenizer) # will be used for inline model evaluation only

# -----------------------------------------------------------------------------
# Training loop

# Learning rate scheduler
def get_lr_multiplier(it):
    lrm = 1.0 - it / num_iterations
    return lrm

# Go!
metrics = {}
step = 0
train_iter = iter(train_loader)
for step in range(num_iterations):
    last_step = step == num_iterations - 1

    # evaluate the validation loss
    if last_step or step % eval_every == 0:
        eval_model.eval()
        val_iter = iter(build_val_loader())
        losses = []
        for _ in range(eval_steps):
            val_inputs, val_targets = next(val_iter)
            with torch.no_grad(), autocast_ctx:
                loss = eval_model(val_inputs, val_targets)
            losses.append(loss)
        val_loss = torch.stack(losses).mean() # average over eval_steps
        if ddp:
            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG) # average over ranks
        val_loss = val_loss.item()
        print0(f"Step {step:05d} | Validation loss: {val_loss:.6f}")
        wandb_run.log({
            "step": step,
            "val_loss": val_loss,
        })
        train_model.train()

    # evlauate accuracy of the multiple choice tasks (which are quick to run)
    if last_step or (step > 0 and step % eval_metrics_every == 0):
        eval_model.eval()
        metrics = {}
        with torch.no_grad(), autocast_ctx:
            # note that because these are inside no_grad, we can usually afford to at least ~2X the batch size
            mmlu_result = run_chat_eval("MMLU", eval_model, tokenizer, engine, batch_size=device_batch_size*2, max_problems=1024)
            arc_easy_result = run_chat_eval("ARC-Easy", eval_model, tokenizer, engine, batch_size=device_batch_size*2, max_problems=1024)
        if isinstance(mmlu_result, dict):
            for key, value in mmlu_result.items():
                metrics[f"mmlu_{key}"] = value
        else:
            metrics["mmlu_acc"] = mmlu_result
        if isinstance(arc_easy_result, dict):
            for key, value in arc_easy_result.items():
                metrics[f"arc_easy_{key}"] = value
        else:
            metrics["arc_easy_acc"] = arc_easy_result
        flat_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, dict):
                for subkey, subval in value.items():
                    flat_metrics[f"{key}_{subkey}"] = subval
            else:
                flat_metrics[key] = value
        metrics = flat_metrics

        def format_metric_value(value):
            if torch.is_tensor(value):
                value = value.item()
            if isinstance(value, (int, float)):
                return f"{value:.6f}"
            return str(value)

        metrics_str = ', '.join(f"{k}: {format_metric_value(v)}" for k, v in metrics.items())
        print0(f"Step {step:05d} | {metrics_str}")
        log_payload = {"step": step}
        for key, value in metrics.items():
            if torch.is_tensor(value):
                value = value.item()
            if isinstance(value, (int, float)):
                log_payload[key] = value
        wandb_run.log(log_payload)
        train_model.train()

    if last_step:
        break

    # evaluate the gradient
    num_tokens = torch.tensor(0, device=device) # the number of "active" tokens of supervision seen
    for micro_step in range(grad_accum_steps):
        train_inputs, train_targets = next(train_iter)
        with autocast_ctx:
            loss = train_model(train_inputs, train_targets)
        train_loss = loss.detach() # for logging
        if backend == "deepspeed":
            train_model.backward(loss)
        else:
            loss = loss / grad_accum_steps # each .backward() is a grad sum => normalize loss here
            loss.backward() # accumulate the gradient
        num_tokens += (train_targets >= 0).sum()
    if ddp:
        dist.all_reduce(num_tokens, op=dist.ReduceOp.SUM) # sum over ranks

    # learning rate scheduler
    lrm = get_lr_multiplier(step)
    if backend == "deepspeed":
        for group in train_model.optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
    else:
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["initial_lr"] * lrm

    # step the optimizers
    if backend == "deepspeed":
        train_model.step()
        train_model.zero_grad(set_to_none=True)
    else:
        for opt in optimizers:
            opt.step()
        train_model.zero_grad(set_to_none=True)

    # logging
    train_loss_item = train_loss.item()
    num_tokens_item = num_tokens.item()
    print0(f"Step {step:05d}/{num_iterations:05d} | Training loss: {train_loss_item:.6f}| lrm: {lrm:.6f}| num_tokens: {num_tokens_item:,}")
    wandb_run.log({
        "step": step,
        "lrm": lrm,
        "train_loss": train_loss_item,
        "num_tokens": num_tokens_item,
    })
    step += 1

# Save the model at the end of the run
if master_process:
    base_dir = get_base_dir()
    depth = orig_model.config.n_layer
    model_tag = f"d{depth}" # base the model tag on the depth of the base model
    checkpoint_dir = os.path.join(base_dir, "chatsft_checkpoints", model_tag)
    model_config_kwargs = orig_model.config.__dict__ # slightly naughty, abusing the simplicity of GPTConfig, TODO nicer
    if backend == "deepspeed":
        client_state = {
            "step": step,
            "val_loss": val_loss,
            **metrics,
            "model_config": model_config_kwargs,
            "user_config": user_config,
        }
        train_model.save_checkpoint(checkpoint_dir, tag=str(step), client_state=client_state)
    elif backend == "fsdp":
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType

        with FSDP.state_dict_type(train_model, StateDictType.FULL_STATE_DICT, state_dict_config=fsdp_state_dict_config):
            full_state = train_model.state_dict()
        save_checkpoint(
            checkpoint_dir,
            step,
            full_state,
            [opt.state_dict() for opt in optimizers],
            {
                "step": step,
                "val_loss": val_loss,
                **metrics,
                "model_config": model_config_kwargs,
                "user_config": user_config,
            }
        )
    else:
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(),
            [opt.state_dict() for opt in optimizers],
            {
                "step": step,
                "val_loss": val_loss,
                **metrics,
                "model_config": model_config_kwargs,
                "user_config": user_config,
            }
        )
    print(f"✅ Saved model checkpoint to {checkpoint_dir}")

# Log to report
from nanochat.report import get_report
get_report().log(section="Chat SFT", data=[
    user_config, # CLI args
    {
        "Training rows": len(train_ds),
        "Number of iterations": num_iterations,
        "Training loss": train_loss_item,
        "Validation loss": val_loss,
    },
])

# Cleanup
wandb_run.finish()
compute_cleanup()
