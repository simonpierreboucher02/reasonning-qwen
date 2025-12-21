"""
Qwen3 model implementation.

This module implements the complete Qwen3 transformer architecture with
Grouped Query Attention and Rotary Position Embeddings.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from .attention import GroupedQueryAttention, RMSNorm, compute_rope_params
from .config import ModelConfig


class Qwen3Model(nn.Module):
    """Qwen3 transformer model.

    A decoder-only transformer model with Grouped Query Attention and
    Rotary Position Embeddings, designed for efficient text generation
    and reasoning tasks.

    Args:
        config: Model configuration
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        # Store config
        self.config = config
        dtype = getattr(torch, config.dtype)

        # Main model parameters
        self.tok_emb = nn.Embedding(config.vocab_size, config.emb_dim, dtype=dtype)

        # Transformer blocks
        self.trf_blocks = nn.ModuleList(
            [TransformerBlock(config, dtype) for _ in range(config.n_layers)]
        )

        self.final_norm = RMSNorm(config.emb_dim)
        self.out_head = nn.Linear(
            config.emb_dim, config.vocab_size, bias=False, dtype=dtype
        )

        # Precompute RoPE parameters
        cos, sin = compute_rope_params(
            head_dim=config.head_dim,
            theta_base=config.rope_base,
            context_length=config.context_length,
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        self.current_pos = 0  # Track current position in KV cache

    def forward(
        self, in_idx: torch.Tensor, cache: Optional["KVCache"] = None
    ) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            in_idx: Input token indices of shape (batch, seq_len)
            cache: Optional KV cache for efficient generation

        Returns:
            Logits of shape (batch, seq_len, vocab_size)
        """
        # Embed tokens
        tok_embeds = self.tok_emb(in_idx)
        x = tok_embeds

        num_tokens = x.shape[1]

        # Create attention mask
        if cache is not None:
            pos_start = self.current_pos
            pos_end = pos_start + num_tokens
            self.current_pos = pos_end
            mask = torch.triu(
                torch.ones(pos_end, pos_end, device=x.device, dtype=torch.bool),
                diagonal=1,
            )[pos_start:pos_end, :pos_end]
        else:
            pos_start = 0
            mask = torch.triu(
                torch.ones(num_tokens, num_tokens, device=x.device, dtype=torch.bool),
                diagonal=1,
            )

        # Add batch and head dimensions for broadcasting
        mask = mask[None, None, :, :]

        # Process through transformer blocks
        for i, block in enumerate(self.trf_blocks):
            blk_cache = cache.get(i) if cache else None
            x, new_blk_cache = block(
                x, mask, self.cos, self.sin, start_pos=pos_start, cache=blk_cache
            )
            if cache is not None:
                cache.update(i, new_blk_cache)

        # Final normalization and projection
        x = self.final_norm(x)
        logits = self.out_head(x.to(getattr(torch, self.config.dtype)))
        return logits

    def reset_kv_cache(self):
        """Reset the KV cache position counter."""
        self.current_pos = 0


class TransformerBlock(nn.Module):
    """Single transformer block with attention and feedforward layers.

    Implements pre-normalization (normalization before attention/FFN) with
    residual connections.

    Args:
        config: Model configuration
        dtype: Data type for parameters
    """

    def __init__(self, config: ModelConfig, dtype: torch.dtype):
        super().__init__()

        # Attention layer
        self.att = GroupedQueryAttention(
            d_in=config.emb_dim,
            num_heads=config.n_heads,
            head_dim=config.head_dim,
            num_kv_groups=config.n_kv_groups,
            qk_norm=True,  # Qwen3 uses QK normalization
            dtype=dtype,
        )

        # Feedforward layer
        self.ff = FeedForward(config, dtype)

        # Normalization layers
        self.norm1 = RMSNorm(config.emb_dim, eps=1e-6)
        self.norm2 = RMSNorm(config.emb_dim, eps=1e-6)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        start_pos: int = 0,
        cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through transformer block.

        Args:
            x: Input tensor
            mask: Attention mask
            cos: Cosine values for RoPE
            sin: Sine values for RoPE
            start_pos: Starting position for RoPE
            cache: Optional KV cache

        Returns:
            Output tensor and updated cache
        """
        # Attention with residual connection
        shortcut = x
        x = self.norm1(x)
        x, next_cache = self.att(x, mask, cos, sin, start_pos=start_pos, cache=cache)
        x = x + shortcut

        # Feedforward with residual connection
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + shortcut

        return x, next_cache


class FeedForward(nn.Module):
    """Feedforward network with SwiGLU activation.

    Uses the SwiGLU activation function (SiLU gating) which has been shown
    to improve model performance.

    Args:
        config: Model configuration
        dtype: Data type for parameters
    """

    def __init__(self, config: ModelConfig, dtype: torch.dtype):
        super().__init__()

        # Use hidden_dim from configuration
        hidden_dim = config.hidden_dim

        self.fc1 = nn.Linear(config.emb_dim, hidden_dim, dtype=dtype, bias=False)
        self.fc2 = nn.Linear(config.emb_dim, hidden_dim, dtype=dtype, bias=False)
        self.fc3 = nn.Linear(hidden_dim, config.emb_dim, dtype=dtype, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with SwiGLU activation.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        # SwiGLU: SiLU(x @ W1) * (x @ W2)
        x = nn.functional.silu(x_fc1) * x_fc2
        return self.fc3(x)


class KVCache:
    """Key-Value cache for efficient autoregressive generation.

    Stores cached keys and values from previous tokens to avoid
    recomputing them during generation.

    Args:
        n_layers: Number of transformer layers
    """

    def __init__(self, n_layers: int):
        self.cache = [None] * n_layers

    def get(self, layer_idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Get cache for a specific layer.

        Args:
            layer_idx: Index of the layer

        Returns:
            Cached (keys, values) tuple or None
        """
        return self.cache[layer_idx]

    def update(
        self, layer_idx: int, value: Tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Update cache for a specific layer.

        Args:
            layer_idx: Index of the layer
            value: New (keys, values) tuple to cache
        """
        self.cache[layer_idx] = value

    def get_all(self) -> list:
        """Get all cached values.

        Returns:
            List of cached values for all layers
        """
        return self.cache

    def reset(self) -> None:
        """Reset all cached values."""
        for i in range(len(self.cache)):
            self.cache[i] = None
