"""
Qwen2.5 model configuration presets for different model sizes.
Based on official Qwen2.5-Coder architecture specifications.
"""

from nanochat.gpt import GPTConfig


# Qwen2.5-Coder-1.5B configuration
def get_qwen25_1_5b_config(vocab_size=151646, sequence_len=32768):
    """
    Qwen2.5-Coder-1.5B configuration
    - 1.5B parameters
    - 28 layers
    - 1536 hidden dimension
    - 12 query heads, 2 KV heads (GQA)
    - 8960 intermediate size
    """
    return GPTConfig(
        vocab_size=vocab_size,
        sequence_len=sequence_len,
        n_layer=28,
        n_embd=1536,
        n_head=12,
        n_kv_head=2,
        intermediate_size=8960,
        rope_theta=10000.0,
        attention_bias=True,  # Qwen uses bias in attention
    )


# Qwen2.5-Coder-7B configuration
def get_qwen25_7b_config(vocab_size=151646, sequence_len=32768):
    """
    Qwen2.5-Coder-7B configuration
    - 7.6B parameters
    - 28 layers
    - 3584 hidden dimension
    - 28 query heads, 4 KV heads (GQA)
    - 18944 intermediate size
    """
    return GPTConfig(
        vocab_size=vocab_size,
        sequence_len=sequence_len,
        n_layer=28,
        n_embd=3584,
        n_head=28,
        n_kv_head=4,
        intermediate_size=18944,
        rope_theta=10000.0,
        attention_bias=True,
    )


# Small Qwen2.5-style model for budget training (similar to nanochat d20)
def get_qwen25_small_config(vocab_size=50304, sequence_len=2048, depth=20):
    """
    Small Qwen2.5-style model for budget training
    Uses Qwen2.5 architecture (SwiGLU, GQA) but with smaller dimensions
    Compatible with nanochat's depth-based scaling
    """
    model_dim = depth * 64  # aspect ratio 64
    num_heads = max(1, (model_dim + 127) // 128)  # head dim 128
    num_kv_heads = max(1, num_heads // 2)  # 2:1 GQA ratio (more efficient than 1:1)
    # Qwen uses non-standard intermediate sizes, approximate with 2.7x ratio
    intermediate_size = int(model_dim * 2.7)
    
    return GPTConfig(
        vocab_size=vocab_size,
        sequence_len=sequence_len,
        n_layer=depth,
        n_embd=model_dim,
        n_head=num_heads,
        n_kv_head=num_kv_heads,
        intermediate_size=intermediate_size,
        rope_theta=10000.0,
        attention_bias=True,  # Use Qwen-style attention bias
    )


# Backward compatible: original nanochat config (no Qwen features)
def get_nanochat_original_config(vocab_size=50304, sequence_len=2048, depth=20):
    """
    Original nanochat configuration (backward compatible)
    - ReLU² activation (via intermediate_size=None)
    - No attention bias
    - 1:1 MQA ratio
    """
    model_dim = depth * 64
    num_heads = max(1, (model_dim + 127) // 128)
    
    return GPTConfig(
        vocab_size=vocab_size,
        sequence_len=sequence_len,
        n_layer=depth,
        n_embd=model_dim,
        n_head=num_heads,
        n_kv_head=num_heads,  # 1:1 ratio (MQA)
        intermediate_size=None,  # None = use 4*n_embd with ReLU² (backward compat)
        mlp_type="relu2",
        rope_theta=10000.0,
        attention_bias=False,
    )
