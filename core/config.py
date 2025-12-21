"""
Model configuration classes for Qwen3 models.

This module provides configuration classes for different Qwen3 model variants,
making it easy to instantiate models of various sizes.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for Qwen3 model architecture.

    Attributes:
        vocab_size: Size of the vocabulary
        context_length: Maximum sequence length
        emb_dim: Embedding dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        n_kv_groups: Number of key-value groups for grouped query attention
        rope_base: Base frequency for rotary position embeddings
        hidden_dim: Hidden dimension in feedforward network
        head_dim: Dimension per attention head
        dtype: Data type for model parameters (e.g., "bfloat16", "float32")
        rope_freq: Custom rope frequencies for extended context
    """

    vocab_size: int
    context_length: int
    emb_dim: int
    n_heads: int
    n_layers: int
    n_kv_groups: int
    rope_base: float
    hidden_dim: int
    head_dim: int
    dtype: str = "bfloat16"
    rope_freq: Optional[dict] = None

    @classmethod
    def qwen3_0_6b(cls) -> "ModelConfig":
        """Configuration for Qwen3 0.6B parameter model."""
        return cls(
            vocab_size=151_936,
            context_length=40_960,
            emb_dim=1024,
            n_heads=16,
            n_layers=28,
            n_kv_groups=8,
            rope_base=1_000_000.0,
            hidden_dim=3072,
            head_dim=128,
        )

    @classmethod
    def qwen3_1_7b(cls) -> "ModelConfig":
        """Configuration for Qwen3 1.7B parameter model."""
        return cls(
            vocab_size=151_936,
            context_length=40_960,
            emb_dim=2048,
            n_heads=16,
            n_layers=28,
            n_kv_groups=8,
            rope_base=1_000_000.0,
            hidden_dim=11008,
            head_dim=128,
        )

    @classmethod
    def qwen3_4b(cls) -> "ModelConfig":
        """Configuration for Qwen3 4B parameter model."""
        return cls(
            vocab_size=151_936,
            context_length=40_960,
            emb_dim=2560,
            n_heads=20,
            n_layers=36,
            n_kv_groups=4,
            rope_base=1_000_000.0,
            hidden_dim=13696,
            head_dim=128,
        )

    @classmethod
    def qwen3_8b(cls) -> "ModelConfig":
        """Configuration for Qwen3 8B parameter model."""
        return cls(
            vocab_size=151_936,
            context_length=40_960,
            emb_dim=4096,
            n_heads=32,
            n_layers=36,
            n_kv_groups=8,
            rope_base=1_000_000.0,
            hidden_dim=18944,
            head_dim=128,
        )

    @classmethod
    def qwen3_14b(cls) -> "ModelConfig":
        """Configuration for Qwen3 14B parameter model."""
        return cls(
            vocab_size=151_936,
            context_length=40_960,
            emb_dim=5120,
            n_heads=40,
            n_layers=48,
            n_kv_groups=8,
            rope_base=1_000_000.0,
            hidden_dim=27392,
            head_dim=128,
        )

    @classmethod
    def qwen3_32b(cls) -> "ModelConfig":
        """Configuration for Qwen3 32B parameter model."""
        return cls(
            vocab_size=151_936,
            context_length=40_960,
            emb_dim=5120,
            n_heads=40,
            n_layers=64,
            n_kv_groups=8,
            rope_base=1_000_000.0,
            hidden_dim=27392,
            head_dim=128,
        )

    def __repr__(self) -> str:
        """String representation of the configuration."""
        return (
            f"ModelConfig(\n"
            f"  vocab_size={self.vocab_size:,},\n"
            f"  context_length={self.context_length:,},\n"
            f"  emb_dim={self.emb_dim},\n"
            f"  n_heads={self.n_heads},\n"
            f"  n_layers={self.n_layers},\n"
            f"  n_kv_groups={self.n_kv_groups},\n"
            f"  head_dim={self.head_dim},\n"
            f"  rope_base={self.rope_base:,.0f},\n"
            f"  dtype={self.dtype}\n"
            f")"
        )
