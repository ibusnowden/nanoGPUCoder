"""
Estimate parameter counts for nanochat GPT configs (dense or MoE).

Examples (run from repo root: nanogpu/kpt_nanochat/nanochat):

  # Count params for a Qwen2.5-1.5B-shaped config (using your vocab size)
  python -m scripts.param_count --architecture_style=qwen25_1.5b --vocab_size=65536

  # Add sparse MoE on ~3 layers (0,10,20) with 4 experts
  python -m scripts.param_count --architecture_style=qwen25_1.5b --vocab_size=65536 --moe_num_experts=4 --moe_layer_end=28 --moe_layer_stride=10

  # Ask for a target total parameter count (rough guidance for #MoE layers / stride)
  python -m scripts.param_count --architecture_style=qwen25_1.5b --vocab_size=65536 --moe_num_experts=4 --target_total_params=1900000000
"""

import os
import copy
import math

import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.common import print0, get_base_dir

# -----------------------------------------------------------------------------
# User settings (override via nanochat/configurator.py using --key=value)

architecture_style = "qwen25_1.5b"  # qwen25_small | qwen25_1.5b | qwen25_7b | original
depth = 20  # used for qwen25_small/original
max_seq_len = 2048  # sequence length (does NOT change parameter count)

vocab_size = -1  # if -1, try to read from local tokenizer; else use explicit vocab size

# MoE (optional)
moe_num_experts = 0  # 0 disables MoE
moe_top_k = 1  # 1 or 2 (does NOT change parameter count)
moe_layer_start = 0
moe_layer_end = -1
moe_layer_stride = 1
moe_capacity_factor = 1.25  # does NOT change parameter count
moe_aux_loss_coef = 0.01  # does NOT change parameter count

# Optional helper: print deltas vs a target total parameter budget
target_total_params = -1  # if >0, print suggestions toward this total

# allow CLI to override
config_keys = [k for k, v in globals().items() if not k.startswith("_") and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join("nanochat", "configurator.py")).read())

# -----------------------------------------------------------------------------

def _is_moe_layer(config: GPTConfig, layer_idx: int) -> bool:
    if config.moe_num_experts <= 0:
        return False
    start = max(0, int(config.moe_layer_start))
    end = config.n_layer if config.moe_layer_end < 0 else int(config.moe_layer_end)
    stride = max(1, int(config.moe_layer_stride))
    return start <= layer_idx < end and (layer_idx - start) % stride == 0


def _fmt_int(n: int) -> str:
    return f"{n:,}"


def _fmt_params(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n/1e9:.3f}B ({_fmt_int(n)})"
    if n >= 1_000_000:
        return f"{n/1e6:.3f}M ({_fmt_int(n)})"
    return _fmt_int(n)


def _load_vocab_size_from_tokenizer() -> int:
    try:
        from nanochat.tokenizer import get_tokenizer
    except Exception as e:
        raise RuntimeError(
            f"Failed to import nanochat tokenizer to infer vocab_size ({type(e).__name__}: {e}). "
            "Pass --vocab_size=... explicitly."
        ) from e
    try:
        tok = get_tokenizer()
    except Exception as e:
        base_dir = get_base_dir()
        raise RuntimeError(
            f"Failed to load tokenizer from {os.path.join(base_dir, 'tokenizer')} ({type(e).__name__}: {e}). "
            "Pass --vocab_size=... explicitly, or stage/train a tokenizer."
        ) from e
    return int(tok.get_vocab_size())


def _build_config() -> GPTConfig:
    vs = int(vocab_size)
    if vs <= 0:
        vs = _load_vocab_size_from_tokenizer()
        print0(f"Inferred vocab_size from tokenizer: {vs:,}")

    from nanochat.model_configs import (
        get_qwen25_small_config,
        get_qwen25_1_5b_config,
        get_qwen25_7b_config,
        get_nanochat_original_config,
    )

    if architecture_style == "qwen25_small":
        config = get_qwen25_small_config(vocab_size=vs, sequence_len=max_seq_len, depth=depth)
    elif architecture_style == "qwen25_1.5b":
        config = get_qwen25_1_5b_config(vocab_size=vs, sequence_len=max_seq_len)
    elif architecture_style == "qwen25_7b":
        config = get_qwen25_7b_config(vocab_size=vs, sequence_len=max_seq_len)
    elif architecture_style == "original":
        config = get_nanochat_original_config(vocab_size=vs, sequence_len=max_seq_len, depth=depth)
    else:
        raise ValueError(f"Unknown architecture_style: {architecture_style}")

    # Apply MoE knobs (mirrors scripts/base_train.py)
    config.moe_num_experts = int(moe_num_experts)
    config.moe_top_k = int(moe_top_k)
    config.moe_layer_start = int(moe_layer_start)
    config.moe_layer_end = int(moe_layer_end)
    config.moe_layer_stride = int(moe_layer_stride)
    config.moe_capacity_factor = float(moe_capacity_factor)
    config.moe_aux_loss_coef = float(moe_aux_loss_coef)
    return config


def _count_params_by_name(model: torch.nn.Module) -> dict[str, int]:
    buckets = {
        "embeddings": 0,
        "lm_head": 0,
        "attention": 0,
        "mlp_dense": 0,
        "moe_router": 0,
        "moe_experts": 0,
        "other": 0,
    }
    for name, p in model.named_parameters():
        n = int(p.numel())
        if name.startswith("transformer.wte."):
            buckets["embeddings"] += n
        elif name.startswith("lm_head."):
            buckets["lm_head"] += n
        elif ".attn." in name:
            buckets["attention"] += n
        elif ".mlp.router." in name:
            buckets["moe_router"] += n
        elif ".mlp.experts." in name:
            buckets["moe_experts"] += n
        elif ".mlp." in name:
            buckets["mlp_dense"] += n
        else:
            buckets["other"] += n
    buckets["total"] = sum(buckets.values())
    return buckets


def _count_params(config: GPTConfig) -> dict[str, int]:
    # Avoid allocating real weights by instantiating on meta.
    # Parameter count is independent of sequence_len, but GPT builds rotary buffers that scale with it;
    # keep the init sequence small to make this fast.
    cfg = copy.copy(config)
    cfg.sequence_len = min(int(cfg.sequence_len), 32)
    with torch.device("meta"):
        model = GPT(cfg)
    return _count_params_by_name(model)


def _suggest_stride_for_count(start: int, end: int, desired_layers: int) -> int | None:
    span = max(0, end - start)
    if span == 0:
        return None
    for stride in range(1, span + 1):
        count = len(range(start, end, stride))
        if count == desired_layers:
            return stride
    return None


def main():
    config = _build_config()
    n_moe_layers = sum(_is_moe_layer(config, i) for i in range(config.n_layer))
    start = max(0, int(config.moe_layer_start))
    end = config.n_layer if config.moe_layer_end < 0 else int(config.moe_layer_end)
    stride = max(1, int(config.moe_layer_stride))

    # Dense baseline (same arch, MoE disabled)
    dense_cfg = copy.copy(config)
    dense_cfg.moe_num_experts = 0

    dense = _count_params(dense_cfg)
    total = _count_params(config)

    print0("")
    print0("Model config")
    print0(f"  architecture_style: {architecture_style}")
    print0(f"  vocab_size        : {_fmt_int(int(config.vocab_size))}")
    print0(f"  max_seq_len       : {_fmt_int(int(max_seq_len))} (no impact on params)")
    print0(f"  n_layer           : {config.n_layer}")
    print0(f"  n_embd            : {config.n_embd}")
    print0(f"  n_head            : {config.n_head}")
    print0(f"  n_kv_head         : {config.n_kv_head}")
    print0(f"  intermediate_size : {config.intermediate_size if config.intermediate_size is not None else 4 * config.n_embd}")
    print0(f"  attention_bias    : {bool(config.attention_bias)}")

    print0("")
    print0("MoE")
    if config.moe_num_experts <= 0:
        print0("  disabled")
    else:
        print0(f"  experts           : {config.moe_num_experts}")
        print0(f"  top_k             : {config.moe_top_k} (no impact on params)")
        print0(f"  layers            : [{start}:{end}:{stride}] -> {n_moe_layers} layers")
        print0(f"  capacity_factor   : {config.moe_capacity_factor} (no impact on params)")
        print0(f"  aux_loss_coef     : {config.moe_aux_loss_coef} (no impact on params)")

    print0("")
    print0("Parameter count")
    print0(f"  dense total       : {_fmt_params(dense['total'])}")
    print0(f"  current total     : {_fmt_params(total['total'])}  (delta: {_fmt_params(total['total'] - dense['total'])})")
    print0(f"  breakdown current : emb {_fmt_params(total['embeddings'])} | attn {_fmt_params(total['attention'])} | mlp {_fmt_params(total['mlp_dense'] + total['moe_experts'] + total['moe_router'])}")
    if total["moe_experts"] or total["moe_router"]:
        print0(f"                    moe_experts {_fmt_params(total['moe_experts'])} | moe_router {_fmt_params(total['moe_router'])}")

    # Optional target helper
    if int(target_total_params) > 0 and config.moe_num_experts > 0:
        tgt = int(target_total_params)
        extra_needed = tgt - dense["total"]
        mlp_intermediate = config.intermediate_size if config.intermediate_size is not None else 4 * config.n_embd
        dense_mlp_per_layer = 3 * config.n_embd * mlp_intermediate
        extra_per_moe_layer = config.moe_num_experts * config.n_embd + (config.moe_num_experts - 1) * dense_mlp_per_layer
        ideal = extra_needed / max(1, extra_per_moe_layer)
        k_floor = max(0, min(config.n_layer, math.floor(ideal)))
        k_ceil = max(0, min(config.n_layer, math.ceil(ideal)))

        print0("")
        print0(f"Targeting total_params={_fmt_params(tgt)}")
        print0(f"  dense->target extra needed: {_fmt_params(extra_needed)}")
        print0(f"  approx extra per MoE layer: {_fmt_params(extra_per_moe_layer)} (experts={config.moe_num_experts})")
        print0(f"  ideal #MoE layers          : {ideal:.2f}")

        for k in sorted(set([k_floor, k_ceil])):
            approx_total = dense["total"] + k * extra_per_moe_layer
            stride_suggest = _suggest_stride_for_count(start, end, k) if k > 0 else None
            hint = ""
            if stride_suggest is not None:
                hint = f" (try --moe_layer_stride={stride_suggest} for {k} layers)"
            print0(f"  k={k:2d} -> approx total {_fmt_params(approx_total)} (diff {_fmt_params(approx_total - tgt)}){hint}")


if __name__ == "__main__":
    main()

