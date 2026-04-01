# ИИ
"""Representative (N, K) weight shapes for Llama-3.2-1B-Instruct linear layers.

Config: hidden_size=2048, intermediate_size=8192, num_attention_heads=32,
num_key_value_heads=8, head_dim=64.

PyTorch nn.Linear stores weight as (out_features, in_features) = (N, K).
Forward uses y = x @ W.T, i.e. (M, K) @ (K, N).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LinearShape:
    name: str
    n: int
    k: int


# Llama-3.2-1B-Instruct (unsloth config)
LLAMA_1B_LINEAR_SHAPES: tuple[LinearShape, ...] = (
    LinearShape("q_proj", 2048, 2048),
    LinearShape("k_proj", 512, 2048),
    LinearShape("v_proj", 512, 2048),
    LinearShape("o_proj", 2048, 2048),
    LinearShape("gate_proj", 8192, 2048),
    LinearShape("up_proj", 8192, 2048),
    LinearShape("down_proj", 2048, 8192),
)

TOKEN_BATCH_SIZES: tuple[int, ...] = (128, 512, 2048)
