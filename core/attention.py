"""
Attention mechanisms for Qwen3 model.

This module implements Grouped Query Attention (GQA) with Rotary Position Embeddings (RoPE),
which are core components of the Qwen3 architecture.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention mechanism with optional QK normalization.

    GQA reduces memory and compute costs by sharing key-value pairs across multiple
    query heads, while maintaining model quality.

    Args:
        d_in: Input dimension
        num_heads: Number of query heads
        num_kv_groups: Number of key-value groups (fewer than num_heads)
        head_dim: Dimension per head (computed from d_in if not provided)
        qk_norm: Whether to apply RMS normalization to queries and keys
        dtype: Data type for parameters
    """

    def __init__(
        self,
        d_in: int,
        num_heads: int,
        num_kv_groups: int,
        head_dim: Optional[int] = None,
        qk_norm: bool = False,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        assert (
            num_heads % num_kv_groups == 0
        ), "num_heads must be divisible by num_kv_groups"

        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups

        if head_dim is None:
            assert (
                d_in % num_heads == 0
            ), "d_in must be divisible by num_heads if head_dim is not set"
            head_dim = d_in // num_heads

        self.head_dim = head_dim
        self.d_out = num_heads * head_dim

        # Projections
        self.W_query = nn.Linear(d_in, self.d_out, bias=False, dtype=dtype)
        self.W_key = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(
            d_in, num_kv_groups * head_dim, bias=False, dtype=dtype
        )
        self.out_proj = nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)

        # Optional normalization
        if qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=1e-6)
            self.k_norm = RMSNorm(head_dim, eps=1e-6)
        else:
            self.q_norm = self.k_norm = None

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        start_pos: int = 0,
        cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass with KV caching support.

        Args:
            x: Input tensor of shape (batch, seq_len, d_in)
            mask: Attention mask
            cos: Precomputed cosine values for RoPE
            sin: Precomputed sine values for RoPE
            start_pos: Starting position for RoPE (for cached generation)
            cache: Optional cached (keys, values) from previous steps

        Returns:
            Output tensor and updated cache
        """
        b, num_tokens, _ = x.shape

        # Apply projections
        queries = self.W_query(x)  # (b, num_tokens, num_heads * head_dim)
        keys = self.W_key(x)  # (b, num_tokens, num_kv_groups * head_dim)
        values = self.W_value(x)  # (b, num_tokens, num_kv_groups * head_dim)

        # Reshape to heads/kv-groups
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        keys_new = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(
            1, 2
        )
        values_new = values.view(
            b, num_tokens, self.num_kv_groups, self.head_dim
        ).transpose(1, 2)

        # Optional normalization
        if self.q_norm:
            queries = self.q_norm(queries)
        if self.k_norm:
            keys_new = self.k_norm(keys_new)

        # Apply RoPE
        queries = apply_rope(queries, cos, sin, offset=start_pos)
        keys_new = apply_rope(keys_new, cos, sin, offset=start_pos)

        # Use cache if available
        if cache is not None:
            prev_k, prev_v = cache
            keys = torch.cat([prev_k, keys_new], dim=2)
            values = torch.cat([prev_v, values_new], dim=2)
        else:
            keys, values = keys_new, values_new

        next_cache = (keys, values)

        # Expand K and V to match number of heads
        keys = keys.repeat_interleave(self.group_size, dim=1)
        values = values.repeat_interleave(self.group_size, dim=1)

        # Compute attention
        attn_scores = queries @ keys.transpose(2, 3)
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)
        attn_weights = torch.softmax(attn_scores / self.head_dim**0.5, dim=-1)

        # Apply attention to values
        context = (attn_weights @ values).transpose(1, 2).reshape(b, num_tokens, self.d_out)
        return self.out_proj(context), next_cache


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    RMSNorm is a simpler and more efficient alternative to LayerNorm that
    normalizes using only the RMS statistic without mean centering.

    Args:
        emb_dim: Embedding dimension
        eps: Small constant for numerical stability
        bias: Whether to include a learnable bias term
        qwen3_compatible: Whether to use float32 for computation (Qwen3 behavior)
    """

    def __init__(
        self,
        emb_dim: int,
        eps: float = 1e-6,
        bias: bool = False,
        qwen3_compatible: bool = True,
    ):
        super().__init__()
        self.eps = eps
        self.qwen3_compatible = qwen3_compatible
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization."""
        input_dtype = x.dtype

        if self.qwen3_compatible:
            x = x.to(torch.float32)

        variance = x.pow(2).mean(dim=-1, keepdim=True)
        norm_x = x * torch.rsqrt(variance + self.eps)
        norm_x = norm_x * self.scale

        if self.shift is not None:
            norm_x = norm_x + self.shift

        return norm_x.to(input_dtype)


def compute_rope_params(
    head_dim: int,
    theta_base: float = 10_000,
    context_length: int = 4096,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute precomputed cosine and sine values for Rotary Position Embeddings.

    Args:
        head_dim: Dimension per attention head (must be even)
        theta_base: Base frequency for the rotary embeddings
        context_length: Maximum sequence length
        dtype: Data type for computation

    Returns:
        Tuple of (cos, sin) tensors of shape (context_length, head_dim)
    """
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    # Compute the inverse frequencies
    inv_freq = 1.0 / (
        theta_base
        ** (torch.arange(0, head_dim, 2, dtype=dtype)[: (head_dim // 2)].float() / head_dim)
    )

    # Generate position indices
    positions = torch.arange(context_length, dtype=dtype)

    # Compute the angles
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(
        0
    )  # Shape: (context_length, head_dim // 2)

    # Expand angles to match the head_dim
    angles = torch.cat([angles, angles], dim=1)  # Shape: (context_length, head_dim)

    # Precompute sine and cosine
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin


def apply_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, offset: int = 0
) -> torch.Tensor:
    """Apply Rotary Position Embeddings to input tensor.

    Uses the split-halves style (compatible with Hugging Face Transformers):
    The embedding dimensions are split into two halves, and each half is rotated.

    Args:
        x: Input tensor of shape (batch_size, num_heads, seq_len, head_dim)
        cos: Precomputed cosine values
        sin: Precomputed sine values
        offset: Position offset for cached generation

    Returns:
        Tensor with RoPE applied
    """
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"

    # Split x into first half and second half
    x1 = x[..., : head_dim // 2]  # First half
    x2 = x[..., head_dim // 2 :]  # Second half

    # Adjust sin and cos shapes
    cos = cos[offset : offset + seq_len, :].unsqueeze(0).unsqueeze(
        0
    )  # Shape: (1, 1, seq_len, head_dim)
    sin = sin[offset : offset + seq_len, :].unsqueeze(0).unsqueeze(0)

    # Apply the rotary transformation
    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)

    # Convert to original dtype after rotation
    return x_rotated.to(dtype=x.dtype)
