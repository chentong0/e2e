"""
PyTorch Reimplementation of MetaModel from transformer.py

This is a standalone PyTorch implementation of the MetaModel architecture originally
implemented in JAX/Equinox. The implementation focuses on efficiency by leveraging:
- PyTorch's native FlashAttention when available
- Efficient RMSNorm implementation
- Optimized rotary position embeddings

Line-by-line analysis of the original JAX implementation:

1. **NormalLinear**: A linear layer with truncated normal initialization.
   - In PyTorch, we use nn.Linear and apply truncated normal init manually.
   
2. **precompute_freqs_cis**: Computes rotary embedding frequencies using complex numbers.
   - Standard RoPE implementation, widely used in modern transformers.
   
3. **apply_rotary_emb**: Applies rotary embeddings to query/key tensors.
   - Uses complex number multiplication for efficient rotation.
   
4. **SwiGLUMLP**: SwiGLU activation MLP block (w1 * silu + w3) * w2.
   - A gated MLP variant used in models like LLaMA.
   
5. **RMSNorm**: Root Mean Square Layer Normalization.
   - More efficient than LayerNorm, used in LLaMA-style models.
   
6. **AttentionBase**: Base class for attention mechanisms with Q/K/V projections.
   - Includes QK normalization and RoPE application.
   
7. **Attention**: Standard causal self-attention.
   - Uses dot product attention with causal masking.
   
8. **SWA/SWAFull**: Sliding Window Attention variants.
   - Limits attention to a local window for efficiency.
   
9. **Block**: Single transformer block with attention + FFN.
   - Implements pre-norm and post-norm configurations.
   
10. **BlockCollection/BlockCollectionSplit**: Collections of transformer blocks.
    - BlockCollectionSplit separates prefix and suffix blocks for meta-learning.
    
11. **TransformerModel**: Full transformer with embedding and normalization.
    - Orchestrates the forward pass through all blocks.
    
12. **CausalLM**: Language model head on top of transformer.
    - Supports tied word embeddings.
    
13. **MetaModel**: Meta-learning wrapper with inner loop training.
    - Implements test-time training with inner loop gradient steps.
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass, field
from enum import StrEnum, auto
from functools import partial
from typing import Any, Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange

# scaled_dot_product_attention is available in PyTorch 2.0+
# and automatically uses FlashAttention when available
from torch.nn.functional import scaled_dot_product_attention

# torch.func provides functional programming utilities for PyTorch
# Used for differentiable inner loop training
import torch.func as functorch


# =============================================================================
# Configuration Classes
# =============================================================================

@dataclass
class ModelConfig:
    """
    Configuration dataclass for the model architecture.
    
    Maps directly from the JAX ModelConfig, providing all hyperparameters
    needed to construct the model.
    """
    name: str = "unnamed"
    vocab_size: int = 32000
    hidden_size: int = 768
    intermediate_size: int = 2048
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    mini_batch_size: int = 1024
    sliding_window_size: int = 1024
    seq_len: int = 131072
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02
    bos_token_id: int = 1
    eos_token_id: int = 2
    resid_pdrop: float = 0.0
    embd_pdrop: float = 0.0
    attn_pdrop: float = 0.0
    tie_word_embeddings: bool = False
    seq_modeling_block: str = "self_attention"  # "self_attention", "SWA", "SWAFull"
    rope_theta: float = 10000.0
    output_size: Optional[int] = None
    compute_dtype: str = "bf16"
    param_dtype: str = "fp32"
    state_dtype: str = "fp32"
    force_flash: bool = False
    suffix_len: int = 0
    prime: bool = False
    qk_norm: bool = True
    pre_norm: bool = True
    post_norm: bool = True
    feed_forward_prime: str = "swiglu"
    
    def __post_init__(self):
        if self.output_size is None:
            self.output_size = self.vocab_size


@dataclass
class TrainingConfig:
    """Training configuration for meta-learning."""
    train_mode: str = "pretrain"  # "pretrain" or "meta"
    seq_length: int = 1024
    inner_remat_freq: int = 1
    ilr_warmup_steps: int = 0
    ilr_init: float = 1.0
    inner_lr: float = 0.01
    spec_outer: list = field(default_factory=lambda: ["**"])
    spec_inner: list = field(default_factory=lambda: ["**"])


@dataclass
class Config:
    """Top-level configuration combining model and training configs."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Batch:
    """
    Batch data container for model inputs.
    
    Corresponds to the JAX Batch class, containing:
    - input_ids: Token IDs for input sequence
    - target_tokens: Target token IDs for loss computation
    - loss_masks: Binary mask indicating which tokens to include in loss
    - attention_mask: Optional attention mask
    - position_ids: Optional position IDs for RoPE
    """
    input_ids: Tensor
    target_tokens: Tensor
    loss_masks: Tensor
    attention_mask: Optional[Tensor] = None
    position_ids: Optional[Tensor] = None
    
    @property
    def shape(self):
        return self.input_ids.shape
    
    def to(self, device):
        """Move batch to device."""
        return Batch(
            input_ids=self.input_ids.to(device),
            target_tokens=self.target_tokens.to(device),
            loss_masks=self.loss_masks.to(device),
            attention_mask=self.attention_mask.to(device) if self.attention_mask is not None else None,
            position_ids=self.position_ids.to(device) if self.position_ids is not None else None,
        )


@dataclass
class BaseModelOutput:
    """Output container from the transformer model."""
    last_hidden_state: Optional[Tensor] = None
    logits: Optional[Tensor] = None


# =============================================================================
# Utility Functions
# =============================================================================

def get_dtype(dtype_name: str) -> torch.dtype:
    """
    Convert dtype name string to torch dtype.
    
    JAX equivalent: get_float_dtype_by_name in jax_utils.py
    """
    dtype_map = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp64": torch.float64,
        "float64": torch.float64,
    }
    if dtype_name not in dtype_map:
        raise ValueError(f"Unknown dtype: {dtype_name}")
    return dtype_map[dtype_name]


def precompute_freqs_cis(
    dim: int,
    end: int,
    theta: float = 10000.0,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """
    Precompute the frequency tensor for rotary positional embeddings.
    
    JAX equivalent: precompute_freqs_cis in attention.py
    
    This computes frequencies as complex numbers for efficient rotation.
    The formula is: freq_i = 1 / (theta^(2i/dim))
    
    Args:
        dim: Head dimension (must be even)
        end: Maximum sequence length
        theta: Base for frequency computation
        dtype: Output dtype
        
    Returns:
        Complex tensor of shape (end, dim//2) containing cos + i*sin
    """
    # Compute frequency bands: theta^(-2i/dim) for i in [0, dim/2)
    # torch.arange(0, dim, 2) produces dim//2 elements
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=dtype) / dim))
    
    # Create position indices
    t = torch.arange(end, dtype=dtype)
    
    # Outer product: positions x frequencies
    freqs = torch.outer(t, freqs)
    
    # Convert to complex exponential: e^(i*freq) = cos(freq) + i*sin(freq)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    
    return freqs_cis


def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
    """
    Apply rotary positional embeddings to input tensor.
    
    JAX equivalent: apply_rotary_emb in attention.py
    
    The rotation is applied by treating pairs of features as complex numbers
    and multiplying by the frequency tensor.
    
    Args:
        x: Input tensor of shape (..., seq_len, num_heads, head_dim)
        freqs_cis: Precomputed frequencies of shape (seq_len, head_dim//2)
        
    Returns:
        Rotated tensor of same shape as input
    """
    input_dtype = x.dtype
    
    # Reshape freqs_cis for broadcasting: (seq_len, 1, head_dim//2)
    freqs_cis = freqs_cis.unsqueeze(-2)
    
    # View x as complex: (..., seq_len, num_heads, head_dim//2, 2) -> complex
    x_float = x.float()
    x_reshaped = x_float.reshape(*x.shape[:-1], -1, 2)
    x_complex = torch.view_as_complex(x_reshaped)
    
    # Apply rotation via complex multiplication
    x_rotated = x_complex * freqs_cis
    
    # Convert back to real: interleave real and imaginary parts
    x_out = torch.view_as_real(x_rotated).flatten(-2)
    
    return x_out.to(input_dtype)


# =============================================================================
# Layer Implementations
# =============================================================================

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    JAX equivalent: nn.RMSNorm from equinox
    
    RMSNorm normalizes by the root mean square of activations, without
    centering (no mean subtraction). This is more computationally efficient
    than LayerNorm.
    
    Formula: y = x / sqrt(mean(x^2) + eps) * weight
    """
    
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
    
    def forward(self, x: Tensor) -> Tensor:
        input_dtype = x.dtype
        x = x.float()
        
        # Compute RMS: sqrt(mean(x^2))
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        
        return (x * self.weight).to(input_dtype)


class NormalLinear(nn.Module):
    """
    Linear layer with truncated normal initialization.
    
    JAX equivalent: NormalLinear in attention.py
    
    This is a standard linear layer that initializes weights from a
    normal distribution with a specified standard deviation.
    """
    
    def __init__(
        self,
        config: ModelConfig,
        in_features: int,
        out_features: int,
        std: float,
        bias: bool = False,
    ):
        super().__init__()
        self.compute_dtype = get_dtype(config.compute_dtype)
        param_dtype = get_dtype(config.param_dtype)
        
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features, dtype=param_dtype) * std
        )
        self.bias = nn.Parameter(torch.zeros(out_features, dtype=param_dtype)) if bias else None
    
    def forward(self, x: Tensor) -> Tensor:
        x = x.to(self.compute_dtype)
        weight = self.weight.to(self.compute_dtype)
        
        if self.bias is not None:
            return F.linear(x, weight, self.bias.to(self.compute_dtype))
        return F.linear(x, weight)


class SwiGLUMLP(nn.Module):
    """
    SwiGLU MLP block used in LLaMA-style transformers.
    
    JAX equivalent: SwiGLUMLP in transformer.py
    
    SwiGLU applies: output = dropout(w2(silu(w1(x)) * w3(x)))
    
    This gated architecture has been shown to improve model quality.
    The gate (w3) modulates the hidden representation.
    """
    
    def __init__(
        self,
        config: ModelConfig,
    ):
        super().__init__()
        self.config = config
        self.compute_dtype = get_dtype(config.compute_dtype)
        
        # w1 and w3 project to intermediate size
        # w2 projects back to hidden size
        self.w1 = NormalLinear(
            config,
            in_features=config.hidden_size,
            out_features=config.intermediate_size,
            std=config.initializer_range,
        )
        self.w2 = NormalLinear(
            config,
            in_features=config.intermediate_size,
            out_features=config.hidden_size,
            std=config.initializer_range,
        )
        self.w3 = NormalLinear(
            config,
            in_features=config.hidden_size,
            out_features=config.intermediate_size,
            std=config.initializer_range,
        )
        self.dropout = nn.Dropout(p=config.resid_pdrop)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of SwiGLU MLP.
        
        Args:
            x: Input tensor of shape (..., hidden_size)
            
        Returns:
            Output tensor of shape (..., hidden_size)
        """
        # Compute gate and hidden
        z1 = self.w1(x)
        z1_act = F.silu(z1)  # SiLU activation on first projection
        z3 = self.w3(x)      # Gate projection
        
        # Element-wise gating
        x2 = z1_act * z3
        
        # Project back to hidden size
        z2 = self.w2(x2)
        
        return self.dropout(z2)


# =============================================================================
# Attention Mechanisms
# =============================================================================

class AttentionBase(nn.Module):
    """
    Base class for attention mechanisms.
    
    JAX equivalent: AttentionBase in attention.py
    
    This provides:
    - Q/K/V projections
    - Head splitting/merging
    - RoPE application
    - QK normalization (optional)
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.compute_dtype = get_dtype(config.compute_dtype)
        param_dtype = get_dtype(config.param_dtype)
        
        embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = embed_dim // self.num_heads
        
        # Q/K normalization layers
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, dtype=param_dtype)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, dtype=param_dtype)
        
        # Q/K/V/O projections
        self.wq = NormalLinear(config, embed_dim, embed_dim, std=config.initializer_range)
        self.wk = NormalLinear(config, embed_dim, embed_dim, std=config.initializer_range)
        self.wv = NormalLinear(config, embed_dim, embed_dim, std=config.initializer_range)
        self.wo = NormalLinear(config, embed_dim, embed_dim, std=config.initializer_range)
        
        self.resid_dropout = nn.Dropout(p=config.resid_pdrop)
        
        # Precompute RoPE frequencies (cached)
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(self.head_dim, 2 * config.seq_len, theta=config.rope_theta),
            persistent=False,
        )
    
    def _split_heads(self, x: Tensor) -> Tensor:
        """
        Split the last dimension into (num_heads, head_dim).
        
        Args:
            x: (..., hidden_size)
            
        Returns:
            (..., num_heads, head_dim)
        """
        return rearrange(x, "... (head head_dim) -> ... head head_dim", 
                         head=self.num_heads, head_dim=self.head_dim)
    
    def _merge_heads(self, x: Tensor) -> Tensor:
        """
        Merge heads back to hidden dimension.
        
        Args:
            x: (..., num_heads, head_dim)
            
        Returns:
            (..., hidden_size)
        """
        return rearrange(x, "... head head_dim -> ... (head head_dim)")
    
    def project_qkv(self, hidden_states: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Project input to query, key, value."""
        return self.wq(hidden_states), self.wk(hidden_states), self.wv(hidden_states)
    
    def get_attention_input(
        self,
        hidden_states: Tensor,
        position_ids: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Compute Q/K/V with head splitting, normalization, and RoPE.
        
        Args:
            hidden_states: (seq_len, hidden_size)
            position_ids: (seq_len,)
            
        Returns:
            xq, xk, xv each of shape (seq_len, num_heads, head_dim)
        """
        xq, xk, xv = self.project_qkv(hidden_states)
        xq, xk, xv = self._split_heads(xq), self._split_heads(xk), self._split_heads(xv)
        
        # Apply QK normalization if enabled
        if self.config.qk_norm:
            xq = self.q_norm(xq)
            xk = self.k_norm(xk)
        
        # Apply RoPE
        freqs_cis = self.freqs_cis[position_ids]
        xq = apply_rotary_emb(xq, freqs_cis)
        xk = apply_rotary_emb(xk, freqs_cis)
        
        return xq, xk, xv
    
    def get_attention_output(self, attn_output: Tensor) -> Tensor:
        """Apply output projection and dropout."""
        o_output = self.wo(attn_output)
        return self.resid_dropout(o_output)
    
    def forward(self, hidden_states: Tensor, seq: Batch, is_prefix: bool = False) -> Tensor:
        raise NotImplementedError


class Attention(AttentionBase):
    """
    Standard causal self-attention.
    
    JAX equivalent: Attention in attention.py
    
    Uses PyTorch's scaled_dot_product_attention which automatically uses
    FlashAttention when available.
    """
    
    def forward(
        self,
        hidden_states: Tensor,
        seq: Batch,
        is_prefix: bool = False,
    ) -> Tensor:
        """
        Forward pass of causal self-attention.
        
        Args:
            hidden_states: (seq_len, hidden_size)
            seq: Batch containing position_ids
            is_prefix: Whether this is prefix processing (enables optimizations)
            
        Returns:
            Attention output of shape (seq_len, hidden_size)
        """
        seq_len = hidden_states.shape[0]
        
        # Get position IDs
        if seq.position_ids is None:
            position_ids = torch.arange(seq_len, device=hidden_states.device)
        else:
            position_ids = seq.position_ids
        
        # Compute Q/K/V
        xq, xk, xv = self.get_attention_input(hidden_states, position_ids)
        
        # Add batch dimension for attention (PyTorch expects batch first)
        # Shape: (1, seq_len, num_heads, head_dim) -> (1, num_heads, seq_len, head_dim)
        xq = xq.unsqueeze(0).transpose(1, 2)
        xk = xk.unsqueeze(0).transpose(1, 2)
        xv = xv.unsqueeze(0).transpose(1, 2)
        
        # Compute attention with causal mask
        attn_output = scaled_dot_product_attention(
            xq, xk, xv,
            is_causal=True,
            dropout_p=self.config.attn_pdrop if self.training else 0.0,
        )
        
        # Remove batch dim and merge heads
        attn_output = attn_output.squeeze(0).transpose(0, 1)  # (seq_len, num_heads, head_dim)
        attn_output = self._merge_heads(attn_output)
        
        return self.get_attention_output(attn_output)


class SWAFull(Attention):
    """
    Sliding Window Attention with full attention computation.
    
    JAX equivalent: SWAFull in attention.py
    
    This uses the sliding window attention mask but computes attention
    over all positions (optimized for cases where flash attention handles it).
    """
    
    def forward(
        self,
        hidden_states: Tensor,
        seq: Batch,
        is_prefix: bool = False,
    ) -> Tensor:
        seq_len = hidden_states.shape[0]
        
        if seq.position_ids is None:
            position_ids = torch.arange(seq_len, device=hidden_states.device)
        else:
            position_ids = seq.position_ids
        
        xq, xk, xv = self.get_attention_input(hidden_states, position_ids)
        
        # Add batch dimension
        xq = xq.unsqueeze(0).transpose(1, 2)
        xk = xk.unsqueeze(0).transpose(1, 2)
        xv = xv.unsqueeze(0).transpose(1, 2)
        
        # Create sliding window causal mask
        # For each query position i, attend only to keys in range [max(0, i-window+1), i]
        window_size = self.config.sliding_window_size
        q_idx = torch.arange(seq_len, device=hidden_states.device).unsqueeze(1)
        k_idx = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
        # Causal + sliding window: key must be <= query and within window
        attn_mask = (k_idx <= q_idx) & (k_idx >= q_idx - window_size + 1)
        attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        
        attn_output = scaled_dot_product_attention(
            xq, xk, xv,
            attn_mask=attn_mask,
            dropout_p=self.config.attn_pdrop if self.training else 0.0,
        )
        
        attn_output = attn_output.squeeze(0).transpose(0, 1)
        attn_output = self._merge_heads(attn_output)
        
        return self.get_attention_output(attn_output)


class SWA(AttentionBase):
    """
    Sliding Window Attention with KV cache for efficient inference.
    
    JAX equivalent: SWA in attention.py
    
    This maintains a sliding window KV cache for autoregressive generation.
    The cache stores the last `window_size` key-value pairs.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.mini_batch_size = config.mini_batch_size
        self.window_size = config.sliding_window_size
        
        # Initialize KV cache
        self.register_buffer(
            "k_cache",
            torch.zeros(self.window_size, config.hidden_size),
            persistent=False,
        )
        self.register_buffer(
            "v_cache",
            torch.zeros(self.window_size, config.hidden_size),
            persistent=False,
        )
        self.register_buffer(
            "chunk_index",
            torch.tensor(0, dtype=torch.int32),
            persistent=False,
        )
    
    def reset_cache(self):
        """Reset the KV cache."""
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.chunk_index.zero_()
    
    def sw_causal_mask(self, chunk_id: int) -> Tensor:
        """
        Create sliding window causal attention mask.
        
        Args:
            chunk_id: Current chunk index
            
        Returns:
            Boolean mask of shape (mini_batch_size, window_size + mini_batch_size)
        """
        nk = self.window_size + self.mini_batch_size
        nq = self.mini_batch_size
        
        starting_query_idx = chunk_id * nq
        ending_query_idx = starting_query_idx + self.mini_batch_size
        ending_key_idx = ending_query_idx
        
        qi = (torch.arange(nq, device=self.k_cache.device) + starting_query_idx).unsqueeze(1)
        ki = (torch.arange(-nk, 0, device=self.k_cache.device) + ending_key_idx).unsqueeze(0)
        
        mask = (qi >= ki) & (qi < ki + self.window_size) & (ki >= 0)
        return mask
    
    def full_sw_attention(
        self,
        hidden_states: Tensor,
        seq: Batch,
    ) -> Tensor:
        """Full sliding window attention (used for prefix)."""
        seq_len = hidden_states.shape[0]
        
        if seq.position_ids is None:
            position_ids = torch.arange(seq_len, device=hidden_states.device)
        else:
            position_ids = seq.position_ids
        
        xq, xk, xv = self.get_attention_input(hidden_states, position_ids)
        
        xq = xq.unsqueeze(0).transpose(1, 2)
        xk = xk.unsqueeze(0).transpose(1, 2)
        xv = xv.unsqueeze(0).transpose(1, 2)
        
        attn_output = scaled_dot_product_attention(xq, xk, xv, is_causal=True)
        
        attn_output = attn_output.squeeze(0).transpose(0, 1)
        attn_output = self._merge_heads(attn_output)
        
        return self.get_attention_output(attn_output)
    
    def forward(
        self,
        hidden_states: Tensor,
        seq: Batch,
        is_prefix: bool = False,
    ) -> Tensor:
        if is_prefix:
            return self.full_sw_attention(hidden_states, seq)
        
        # Project to Q/K/V without RoPE (applied after concatenation)
        xq, xk, xv = self.project_qkv(hidden_states)
        xq, xk, xv = self._split_heads(xq), self._split_heads(xk), self._split_heads(xv)
        
        # Apply QK normalization
        if self.config.qk_norm:
            xq = self.q_norm(xq)
            xk = self.k_norm(xk)
        
        # Get cached K/V
        prev_k = self._split_heads(self.k_cache)
        prev_v = self._split_heads(self.v_cache)
        
        # Concatenate with cache
        xk_full = torch.cat([prev_k, xk], dim=0)
        xv_full = torch.cat([prev_v, xv], dim=0)
        
        # Update cache
        new_k_cache = self._merge_heads(xk_full[-self.window_size:])
        new_v_cache = self._merge_heads(xv_full[-self.window_size:])
        self.k_cache.copy_(new_k_cache)
        self.v_cache.copy_(new_v_cache)
        
        # Apply RoPE
        total_len = self.window_size + self.mini_batch_size
        q_positions = torch.arange(total_len, device=xq.device)[-self.mini_batch_size:]
        k_positions = torch.arange(total_len, device=xk.device)
        
        xq = apply_rotary_emb(xq, self.freqs_cis[q_positions])
        xk_full = apply_rotary_emb(xk_full, self.freqs_cis[k_positions])
        
        # Create attention mask
        chunk_id = self.chunk_index.item()
        causal_mask = self.sw_causal_mask(chunk_id)
        
        # Compute attention
        xq = xq.unsqueeze(0).transpose(1, 2)
        xk_full = xk_full.unsqueeze(0).transpose(1, 2)
        xv_full = xv_full.unsqueeze(0).transpose(1, 2)
        
        # Convert mask to attention mask format (True = attend, False = mask)
        attn_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, nq, nk)
        
        attn_output = scaled_dot_product_attention(
            xq, xk_full, xv_full,
            attn_mask=attn_mask,
        )
        
        attn_output = attn_output.squeeze(0).transpose(0, 1)
        attn_output = self._merge_heads(attn_output)
        
        # Update chunk index
        self.chunk_index.add_(1)
        
        return self.get_attention_output(attn_output)


# =============================================================================
# Transformer Block
# =============================================================================

class Block(nn.Module):
    """
    Single transformer block with attention and feed-forward.
    
    JAX equivalent: Block in transformer.py
    
    Structure:
    1. Attention sublayer with pre/post norm
    2. Optional prime FFN sublayer
    3. FFN sublayer with pre/post norm
    
    Uses residual connections around each sublayer.
    """
    
    def __init__(
        self,
        config: ModelConfig,
        feed_forward_prime: Optional[SwiGLUMLP] = None,
        ffn_prime_norm: Optional[RMSNorm] = None,
        ffn_prime_post_norm: Optional[RMSNorm] = None,
    ):
        super().__init__()
        self.config = config
        param_dtype = get_dtype(config.param_dtype)
        
        # Create attention layer based on config
        if config.seq_modeling_block == "self_attention":
            self.seq_modeling_block = Attention(config)
        elif config.seq_modeling_block == "SWA":
            self.seq_modeling_block = SWA(config)
        elif config.seq_modeling_block == "SWAFull":
            self.seq_modeling_block = SWAFull(config)
        else:
            raise NotImplementedError(f"Unknown seq_modeling_block: {config.seq_modeling_block}")
        
        # Main FFN
        self.feed_forward = SwiGLUMLP(config)
        
        # Normalization layers
        self.seq_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=param_dtype)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=param_dtype)
        self.seq_post_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=param_dtype)
        self.ffn_post_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=param_dtype)
        
        # Optional prime FFN (for meta-learning suffix blocks)
        self.ffn_prime_norm = ffn_prime_norm
        self.ffn_prime_post_norm = ffn_prime_post_norm
        self.feed_forward_prime = feed_forward_prime
    
    def forward(
        self,
        hidden_states: Tensor,
        seq: Batch,
        is_prefix: bool = False,
    ) -> Tensor:
        """
        Forward pass through the transformer block.
        
        Args:
            hidden_states: (seq_len, hidden_size)
            seq: Batch data
            is_prefix: Whether this is prefix processing
            
        Returns:
            Output hidden states of shape (seq_len, hidden_size)
        """
        # Attention sublayer
        if self.config.pre_norm:
            seq_modeling_input = self.seq_norm(hidden_states)
        else:
            seq_modeling_input = hidden_states
        
        seq_modeling_output = self.seq_modeling_block(seq_modeling_input, seq, is_prefix=is_prefix)
        
        if self.config.post_norm:
            seq_modeling_output = self.seq_post_norm(seq_modeling_output)
        
        hidden_states = hidden_states + seq_modeling_output
        
        # Optional prime FFN sublayer
        if self.feed_forward_prime is not None:
            if self.config.pre_norm:
                ff_prime_input = self.ffn_prime_norm(hidden_states)
            else:
                ff_prime_input = hidden_states
            
            ff_prime_output = self.feed_forward_prime(ff_prime_input)
            
            if self.config.post_norm:
                ff_prime_output = self.ffn_prime_post_norm(ff_prime_output)
            
            hidden_states = hidden_states + ff_prime_output
        
        # Main FFN sublayer
        if self.config.pre_norm:
            ff_input = self.ffn_norm(hidden_states)
        else:
            ff_input = hidden_states
        
        ff_output = self.feed_forward(ff_input)
        
        if self.config.post_norm:
            ff_output = self.ffn_post_norm(ff_output)
        
        hidden_states = hidden_states + ff_output
        
        return hidden_states


# =============================================================================
# Block Collections
# =============================================================================

class PrimeStorage(nn.Module):
    """
    Storage for prime (meta-learning) parameters.
    
    JAX equivalent: PrimeStorage in transformer.py
    
    Contains the extra FFN parameters used for test-time training.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        param_dtype = get_dtype(config.param_dtype)
        suffix_len = config.suffix_len
        
        self.feed_forward_prime = nn.ModuleList([
            SwiGLUMLP(config) for _ in range(suffix_len)
        ])
        self.ffn_prime_norm = nn.ModuleList([
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=param_dtype)
            for _ in range(suffix_len)
        ])
        self.ffn_prime_post_norm = nn.ModuleList([
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=param_dtype)
            for _ in range(suffix_len)
        ])


class BlockCollection(nn.Module):
    """
    Collection of transformer blocks.
    
    JAX equivalent: BlockCollection in transformer.py
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.blocks = nn.ModuleList([
            Block(config) for _ in range(config.num_hidden_layers)
        ])
        
        self.prime_storage = None
        if config.prime:
            self.prime_storage = PrimeStorage(config)
    
    def forward(self, hidden_states: Tensor, seq: Batch) -> Tensor:
        for block in self.blocks:
            hidden_states = block(hidden_states, seq)
        return hidden_states


class BlockCollectionSplit(nn.Module):
    """
    Split block collection for meta-learning.
    
    JAX equivalent: BlockCollectionSplit in transformer.py
    
    Separates blocks into prefix (frozen) and suffix (trainable) for
    test-time training. The suffix blocks can have additional prime FFN.
    """
    
    def __init__(
        self,
        config: ModelConfig,
        block_collection: BlockCollection,
    ):
        super().__init__()
        self.config = config
        suffix_len = config.suffix_len
        
        # Split into prefix and suffix
        if suffix_len > 0:
            self.prefix_blocks = nn.ModuleList(list(block_collection.blocks)[:-suffix_len])
            self.suffix_blocks = nn.ModuleList(list(block_collection.blocks)[-suffix_len:])
            
            # Add prime parameters to suffix blocks if available
            if block_collection.prime_storage is not None:
                for i, block in enumerate(self.suffix_blocks):
                    block.feed_forward_prime = block_collection.prime_storage.feed_forward_prime[i]
                    block.ffn_prime_norm = block_collection.prime_storage.ffn_prime_norm[i]
                    block.ffn_prime_post_norm = block_collection.prime_storage.ffn_prime_post_norm[i]
        else:
            self.prefix_blocks = nn.ModuleList(list(block_collection.blocks))
            self.suffix_blocks = nn.ModuleList()
    
    def prefix_call(self, hidden_states: Tensor, seq: Batch) -> Tensor:
        """Forward through prefix blocks only."""
        for block in self.prefix_blocks:
            hidden_states = block(hidden_states, seq, is_prefix=True)
        return hidden_states
    
    def suffix_call(self, hidden_states: Tensor, seq: Batch) -> Tensor:
        """Forward through suffix blocks only."""
        for block in self.suffix_blocks:
            hidden_states = block(hidden_states, seq, is_prefix=False)
        return hidden_states
    
    def forward(self, hidden_states: Tensor, seq: Batch) -> Tensor:
        hidden_states = self.prefix_call(hidden_states, seq)
        hidden_states = self.suffix_call(hidden_states, seq)
        return hidden_states


# =============================================================================
# Full Transformer Model
# =============================================================================

class TransformerModel(nn.Module):
    """
    Full transformer model with embedding and blocks.
    
    JAX equivalent: TransformerModel in transformer.py
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.compute_dtype = get_dtype(config.compute_dtype)
        param_dtype = get_dtype(config.param_dtype)
        
        # Token embeddings
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size, dtype=param_dtype)
        nn.init.normal_(self.wte.weight, std=config.initializer_range)
        
        self.dropout = nn.Dropout(p=config.embd_pdrop)
        
        # Transformer blocks
        self.h = BlockCollection(config)
        
        # Final layer norm
        self.ln_f = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=param_dtype)
    
    def wte_call(self, input_ids: Tensor) -> Tensor:
        """Embed input tokens."""
        input_embeds = self.wte(input_ids.long())
        input_embeds = input_embeds.to(self.compute_dtype)
        return self.dropout(input_embeds)
    
    def forward(self, seq: Batch) -> BaseModelOutput:
        """
        Full forward pass.
        
        Args:
            seq: Batch containing input_ids
            
        Returns:
            BaseModelOutput with last_hidden_state
        """
        hidden_states = self.wte_call(seq.input_ids)
        hidden_states = self.h(hidden_states, seq)
        hidden_states = self.ln_f(hidden_states)
        
        return BaseModelOutput(last_hidden_state=hidden_states)


# =============================================================================
# Causal Language Model
# =============================================================================

class CausalLM(nn.Module):
    """
    Causal language model with optional tied embeddings.
    
    JAX equivalent: CausalLM in transformer.py
    """
    
    @dataclass
    class Output:
        last_hidden_states: Tensor
        logits: Tensor
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.compute_dtype = get_dtype(config.compute_dtype)
        
        self.model = TransformerModel(config)
        
        if not config.tie_word_embeddings:
            self.lm_head = NormalLinear(
                config,
                in_features=config.hidden_size,
                out_features=config.output_size,
                std=config.initializer_range,
            )
        else:
            self.lm_head = None
    
    def wte_call(self, input_ids: Tensor) -> Tensor:
        """Get token embeddings."""
        return self.model.wte_call(input_ids)
    
    def wte_disembed_call(self, hidden_states: Tensor) -> Tensor:
        """Convert hidden states to logits."""
        if self.config.tie_word_embeddings:
            # Use transposed embedding weights
            return F.linear(hidden_states.to(self.compute_dtype), 
                           self.model.wte.weight.to(self.compute_dtype))
        else:
            return self.lm_head(hidden_states)
    
    def forward(self, seq: Batch) -> "CausalLM.Output":
        """
        Forward pass returning logits.
        
        Args:
            seq: Batch data
            
        Returns:
            Output with logits
        """
        outputs = self.model(seq)
        hidden_states = outputs.last_hidden_state
        logits = self.wte_disembed_call(hidden_states)
        
        return CausalLM.Output(
            last_hidden_states=hidden_states,
            logits=logits,
        )
    
    def suffix_call(
        self,
        prefix_outputs: Tensor,
        seq: Batch,
        suffix_blocks: nn.ModuleList,
        ln_f: nn.Module,
    ) -> "CausalLM.Output":
        """
        Continue forward pass from prefix outputs through suffix blocks.
        
        Used in meta-learning where prefix is processed once and cached,
        then suffix is processed with updated inner parameters.
        
        Args:
            prefix_outputs: Hidden states from prefix blocks (seq_len, hidden_size)
            seq: Batch data
            suffix_blocks: List of suffix transformer blocks
            ln_f: Final layer normalization
            
        Returns:
            Output with logits
        """
        hidden_states = prefix_outputs
        
        # Process through suffix blocks
        for block in suffix_blocks:
            hidden_states = block(hidden_states, seq, is_prefix=False)
        
        # Apply final layer norm
        hidden_states = ln_f(hidden_states)
        
        # Get logits
        logits = self.wte_disembed_call(hidden_states)
        
        return CausalLM.Output(
            last_hidden_states=hidden_states,
            logits=logits,
        )


# =============================================================================
# Loss Functions
# =============================================================================

def cross_entropy_loss_and_accuracy(
    logits: Tensor,
    tokens: Tensor,
    valid: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor]:
    """
    Compute cross-entropy loss with masking.
    
    JAX equivalent: cross_entropy_loss_and_accuracy in loss.py
    
    Args:
        logits: (seq_len, vocab_size)
        tokens: (seq_len,)
        valid: Optional (seq_len,) mask
        
    Returns:
        Tuple of (loss, pure_ce_loss)
    """
    if valid is None:
        valid = torch.ones_like(tokens, dtype=torch.float32)
    valid = valid.float()
    
    # Compute token-wise cross entropy
    log_probs = F.log_softmax(logits.float(), dim=-1)
    token_log_probs = log_probs.gather(-1, tokens.unsqueeze(-1).long()).squeeze(-1)
    token_log_probs = torch.where(valid > 0, token_log_probs, torch.zeros_like(token_log_probs))
    
    token_wise_loss = -token_log_probs
    
    # Compute mean loss over valid tokens
    valid_count = valid.sum(-1).clamp(min=1e-10)
    loss = (token_wise_loss.sum(-1) / valid_count).mean()
    
    return loss, loss


def token_log_probs(logits: Tensor, targets: Tensor) -> Tensor:
    """
    Compute log probabilities for target tokens.
    
    JAX equivalent: token_log_probs in loss.py
    """
    log_probs = F.log_softmax(logits.float(), dim=-1)
    return log_probs.gather(-1, targets.unsqueeze(-1).long()).squeeze(-1)


# =============================================================================
# Meta Model
# =============================================================================

class MetaModel(nn.Module):
    """
    Meta-learning model with inner loop training capability.
    
    JAX equivalent: MetaModel in transformer.py
    
    This model supports test-time training (TTT) where:
    1. The model processes a long context
    2. During forward pass, it performs gradient descent on "inner" parameters
    3. The outer model learns to produce good initial weights for TTT
    
    The model separates parameters into:
    - Outer parameters: Updated during normal training
    - Inner parameters: Updated during test-time (inner loop)
    """
    
    class MetricType(StrEnum):
        """Metrics tracked during training."""
        loss = auto()
        token_nll_loss = auto()
        outer_grad_norm = auto()
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.compute_dtype = get_dtype(config.model.compute_dtype)
        self.param_dtype = get_dtype(config.model.param_dtype)
        self.state_dtype = get_dtype(config.model.state_dtype)
        
        # Step counter for learning rate warmup
        self.register_buffer("step_index", torch.tensor(0, dtype=torch.int32))
        
        # The main language model
        self.language_model = CausalLM(config.model)
    
    def get_ilr_multiplier(self, step: int) -> float:
        """
        Compute inner learning rate multiplier for warmup.
        
        Args:
            step: Current training step
            
        Returns:
            Learning rate multiplier in [ilr_init, 1.0]
        """
        if self.config.training.ilr_warmup_steps == 0:
            return 1.0
        
        progress = min(1.0, (step + 1) / self.config.training.ilr_warmup_steps)
        ilr = (
            self.config.training.ilr_init +
            (self.config.training.inner_lr - self.config.training.ilr_init) * progress
        )
        return ilr / self.config.training.inner_lr
    
    def get_inner_parameters(self) -> list[nn.Parameter]:
        """
        Get parameters to be updated in inner loop.
        
        In the JAX version, this uses spec patterns to select parameters.
        Here we use a simpler approach: all prime FFN parameters.
        """
        inner_params = []
        
        # Get suffix block prime parameters
        blocks = self.language_model.model.h.blocks
        suffix_len = self.config.model.suffix_len
        
        if suffix_len > 0:
            for block in list(blocks)[-suffix_len:]:
                if block.feed_forward_prime is not None:
                    inner_params.extend(block.feed_forward_prime.parameters())
        
        return inner_params
    
    def clone_inner_params(self) -> dict[str, Tensor]:
        """Create a copy of inner parameters for inner loop training."""
        inner_param_set = set(id(p) for p in self.get_inner_parameters())
        return {
            name: param.clone()
            for name, param in self.named_parameters()
            if id(param) in inner_param_set
        }
    
    def lm_loss(
        self,
        seq: Batch,
        prefix_outputs: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Compute language modeling loss.
        
        Args:
            seq: Batch data
            prefix_outputs: Optional precomputed prefix hidden states
            
        Returns:
            Tuple of (loss, ce_loss, token_nll)
        """
        if prefix_outputs is None:
            lm_outputs = self.language_model(seq)
        else:
            # For meta-learning: continue from prefix outputs through suffix blocks
            # This requires the suffix blocks and final norm
            lm_outputs = self.language_model.suffix_call(
                prefix_outputs=prefix_outputs,
                seq=seq,
                suffix_blocks=self._suffix_blocks,
                ln_f=self.language_model.model.ln_f,
            )
        
        logits = lm_outputs.logits
        loss, ce_loss = cross_entropy_loss_and_accuracy(
            logits, seq.target_tokens, seq.loss_masks
        )
        token_nll = -token_log_probs(logits, seq.target_tokens)
        
        return loss, ce_loss, token_nll
    
    def _compute_suffix_loss(
        self,
        inner_params: dict[str, Tensor],
        seq: Batch,
        prefix_outputs: Tensor,
    ) -> Tensor:
        """
        Compute loss using suffix blocks with given inner parameters.
        
        This function is designed to be used with torch.func.grad for
        computing gradients with respect to inner_params.
        
        Args:
            inner_params: Dictionary of inner parameter tensors
            seq: Batch data for this chunk
            prefix_outputs: Precomputed prefix hidden states
            
        Returns:
            Scalar loss tensor
        """
        # Use functional_call to forward with custom parameters
        # We need to replace the inner params in the model temporarily
        
        # Get the suffix blocks
        suffix_blocks = self._suffix_blocks
        hidden_states = prefix_outputs
        
        # Process through suffix blocks using functional_call for inner params
        for i, block in enumerate(suffix_blocks):
            if block.feed_forward_prime is not None:
                # Create a dict of params for this block's prime FFN
                block_prefix = f"_suffix_blocks.{i}.feed_forward_prime."
                block_params = {
                    k[len(block_prefix):]: v 
                    for k, v in inner_params.items() 
                    if k.startswith(block_prefix)
                }
                
                if block_params:
                    # Use functional_call for the prime FFN with updated params
                    hidden_states = self._forward_block_with_inner_params(
                        block, hidden_states, seq, block_params
                    )
                else:
                    hidden_states = block(hidden_states, seq, is_prefix=False)
            else:
                hidden_states = block(hidden_states, seq, is_prefix=False)
        
        # Apply final layer norm
        hidden_states = self.language_model.model.ln_f(hidden_states)
        
        # Get logits
        logits = self.language_model.wte_disembed_call(hidden_states)
        
        # Compute loss
        loss, _ = cross_entropy_loss_and_accuracy(
            logits, seq.target_tokens, seq.loss_masks
        )
        
        return loss
    
    def _forward_block_with_inner_params(
        self,
        block: nn.Module,
        hidden_states: Tensor,
        seq: Batch,
        prime_ffn_params: dict[str, Tensor],
    ) -> Tensor:
        """
        Forward through a block, using functional_call for the prime FFN.
        
        Args:
            block: Transformer block
            hidden_states: Input hidden states
            seq: Batch data
            prime_ffn_params: Parameters for the prime FFN
            
        Returns:
            Output hidden states
        """
        config = block.config
        
        # Attention sublayer
        if config.pre_norm:
            seq_modeling_input = block.seq_norm(hidden_states)
        else:
            seq_modeling_input = hidden_states
        
        seq_modeling_output = block.seq_modeling_block(seq_modeling_input, seq, is_prefix=False)
        
        if config.post_norm:
            seq_modeling_output = block.seq_post_norm(seq_modeling_output)
        
        hidden_states = hidden_states + seq_modeling_output
        
        # Prime FFN sublayer - use functional_call with updated params
        if block.feed_forward_prime is not None:
            if config.pre_norm:
                ff_prime_input = block.ffn_prime_norm(hidden_states)
            else:
                ff_prime_input = hidden_states
            
            # Use functional_call to apply prime FFN with custom parameters
            ff_prime_output = functorch.functional_call(
                block.feed_forward_prime,
                prime_ffn_params,
                (ff_prime_input,),
            )
            
            if config.post_norm:
                ff_prime_output = block.ffn_prime_post_norm(ff_prime_output)
            
            hidden_states = hidden_states + ff_prime_output
        
        # Main FFN sublayer
        if config.pre_norm:
            ff_input = block.ffn_norm(hidden_states)
        else:
            ff_input = hidden_states
        
        ff_output = block.feed_forward(ff_input)
        
        if config.post_norm:
            ff_output = block.ffn_post_norm(ff_output)
        
        hidden_states = hidden_states + ff_output
        
        return hidden_states
    
    def inner_loop_step(
        self,
        inner_params: dict[str, Tensor],
        seq: Batch,
        prefix_outputs: Tensor,
        inner_lr: float,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """
        Perform a single step of inner loop training.
        
        JAX equivalent: inner_loop_step in transformer.py
        
        This computes gradients of the loss with respect to inner parameters
        and applies an SGD update. The update is differentiable, allowing
        gradients to flow back through the inner loop for meta-learning.
        
        Args:
            inner_params: Current inner parameter tensors (detached dict)
            seq: Batch for this chunk
            prefix_outputs: Precomputed prefix hidden states
            inner_lr: Inner loop learning rate
            
        Returns:
            Tuple of (updated_params, metrics_dict)
        """
        M = MetaModel.MetricType
        metrics: dict[str, Tensor] = {}
        
        # Compute loss and get logits for metrics
        hidden_states = prefix_outputs
        
        for i, block in enumerate(self._suffix_blocks):
            if block.feed_forward_prime is not None:
                block_prefix = f"_suffix_blocks.{i}.feed_forward_prime."
                block_params = {
                    k[len(block_prefix):]: v 
                    for k, v in inner_params.items() 
                    if k.startswith(block_prefix)
                }
                
                if block_params:
                    hidden_states = self._forward_block_with_inner_params(
                        block, hidden_states, seq, block_params
                    )
                else:
                    hidden_states = block(hidden_states, seq, is_prefix=False)
            else:
                hidden_states = block(hidden_states, seq, is_prefix=False)
        
        hidden_states = self.language_model.model.ln_f(hidden_states)
        logits = self.language_model.wte_disembed_call(hidden_states)
        
        # Compute loss
        loss, ce_loss = cross_entropy_loss_and_accuracy(
            logits, seq.target_tokens, seq.loss_masks
        )
        token_nll = -token_log_probs(logits, seq.target_tokens)
        
        metrics[M.loss] = ce_loss.detach()
        metrics[M.token_nll_loss] = token_nll.detach().mean()
        
        # Compute gradients with respect to inner params
        # We need to compute gradients manually since we're using a dict of params
        grads = torch.autograd.grad(
            loss,
            list(inner_params.values()),
            create_graph=True,  # Allow gradients to flow through for meta-learning
            allow_unused=True,
        )
        
        # Apply SGD update: param = param - lr * grad
        updated_params = {}
        for (name, param), grad in zip(inner_params.items(), grads):
            if grad is not None:
                updated_params[name] = param - inner_lr * grad
            else:
                updated_params[name] = param
        
        return updated_params, metrics
    
    def loss_for_sequence(
        self,
        seq: Batch,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """
        Process a sequence and compute loss.
        
        JAX equivalent: loss_for_sequence in transformer.py
        
        For meta-learning mode:
        1. Process prefix blocks once (frozen)
        2. Iterate over chunks, doing inner loop training on suffix blocks
        3. Return total loss for outer gradient
        
        For pretrain mode:
        1. Standard forward pass through all blocks
        2. Return loss
        
        Args:
            seq: Full sequence batch
            
        Returns:
            Tuple of (loss, metrics_dict)
        """
        cfg = self.config
        M = MetaModel.MetricType
        
        if cfg.training.train_mode == "pretrain":
            # Standard pretraining
            loss, ce_loss, token_nll = self.lm_loss(seq)
            
            return loss, {
                M.loss: ce_loss.detach(),
                M.token_nll_loss: token_nll.detach().mean(),
            }
        
        elif cfg.training.train_mode == "meta":
            # Meta-learning with inner loop
            
            # 1. Split blocks into prefix and suffix
            blocks = self.language_model.model.h.blocks
            suffix_len = cfg.model.suffix_len
            
            if suffix_len == 0:
                raise ValueError("Meta-learning requires suffix_len > 0")
            
            # Store references for use in inner loop
            self._prefix_blocks = nn.ModuleList(list(blocks)[:-suffix_len])
            self._suffix_blocks = nn.ModuleList(list(blocks)[-suffix_len:])
            
            # Add prime parameters to suffix blocks if available
            if self.language_model.model.h.prime_storage is not None:
                for i, block in enumerate(self._suffix_blocks):
                    block.feed_forward_prime = self.language_model.model.h.prime_storage.feed_forward_prime[i]
                    block.ffn_prime_norm = self.language_model.model.h.prime_storage.ffn_prime_norm[i]
                    block.ffn_prime_post_norm = self.language_model.model.h.prime_storage.ffn_prime_post_norm[i]
            
            # 2. Embed tokens and process through prefix blocks once
            input_embeds = self.language_model.wte_call(seq.input_ids)
            
            # Process prefix blocks with gradient checkpointing for memory efficiency
            # Gradients will flow through for outer parameter updates
            prefix_output = input_embeds
            for block in self._prefix_blocks:
                # Use gradient checkpointing to save memory
                prefix_output = torch.utils.checkpoint.checkpoint(
                    block, 
                    prefix_output, 
                    seq, 
                    True,  # is_prefix=True
                    use_reentrant=False,
                )
            
            # 3. Initialize inner parameters (clone from model)
            inner_params = {}
            for i, block in enumerate(self._suffix_blocks):
                if block.feed_forward_prime is not None:
                    for name, param in block.feed_forward_prime.named_parameters():
                        full_name = f"_suffix_blocks.{i}.feed_forward_prime.{name}"
                        inner_params[full_name] = param.clone().requires_grad_(True)
            
            # 4. Get inner learning rate
            step = self.step_index.item()
            ilr_multiplier = self.get_ilr_multiplier(step)
            inner_lr = cfg.training.inner_lr * ilr_multiplier
            
            # 5. Chunk sequence into mini-batches
            seqlen = seq.input_ids.shape[0]
            tokens_per_chunk = cfg.model.mini_batch_size
            
            if seqlen % tokens_per_chunk != 0:
                raise ValueError(
                    f"Sequence length {seqlen} must be divisible by "
                    f"mini_batch_size {tokens_per_chunk}"
                )
            
            num_chunks = seqlen // tokens_per_chunk
            
            # 6. Iterate over chunks, doing inner loop training
            all_losses = []
            all_metrics = {M.loss: [], M.token_nll_loss: []}
            
            for chunk_idx in range(num_chunks):
                start_idx = chunk_idx * tokens_per_chunk
                end_idx = start_idx + tokens_per_chunk
                
                # Slice the batch for this chunk
                chunk_seq = Batch(
                    input_ids=seq.input_ids[start_idx:end_idx],
                    target_tokens=seq.target_tokens[start_idx:end_idx],
                    loss_masks=seq.loss_masks[start_idx:end_idx],
                    attention_mask=seq.attention_mask[start_idx:end_idx] if seq.attention_mask is not None else None,
                    position_ids=seq.position_ids[start_idx:end_idx] if seq.position_ids is not None else None,
                )
                chunk_prefix = prefix_output[start_idx:end_idx]
                
                # Perform inner loop step
                inner_params, chunk_metrics = self.inner_loop_step(
                    inner_params=inner_params,
                    seq=chunk_seq,
                    prefix_outputs=chunk_prefix,
                    inner_lr=inner_lr,
                )
                
                all_metrics[M.loss].append(chunk_metrics[M.loss])
                all_metrics[M.token_nll_loss].append(chunk_metrics[M.token_nll_loss])
                
                # Compute loss for this chunk with updated params (for outer gradient)
                chunk_loss = self._compute_suffix_loss(
                    inner_params=inner_params,
                    seq=chunk_seq,
                    prefix_outputs=chunk_prefix,
                )
                all_losses.append(chunk_loss)
            
            # 7. Aggregate losses and metrics
            loss = torch.stack(all_losses).mean()
            metrics = {
                M.loss: torch.stack(all_metrics[M.loss]).mean(),
                M.token_nll_loss: torch.stack(all_metrics[M.token_nll_loss]).mean(),
            }
            
            # Clean up temporary references
            del self._prefix_blocks
            del self._suffix_blocks
            
            return loss, metrics
        
        else:
            raise NotImplementedError(f"Unknown train_mode: {cfg.training.train_mode}")
    
    def forward(self, seq: Batch) -> tuple[Tensor, dict[str, Tensor]]:
        """
        Main forward pass.
        
        Args:
            seq: Input batch
            
        Returns:
            Tuple of (loss, metrics)
        """
        return self.loss_for_sequence(seq)


# =============================================================================
# Model Factory
# =============================================================================

def create_metamodel(
    vocab_size: int = 32000,
    hidden_size: int = 768,
    intermediate_size: int = 2048,
    num_hidden_layers: int = 12,
    num_attention_heads: int = 12,
    suffix_len: int = 0,
    prime: bool = False,
    **kwargs,
) -> MetaModel:
    """
    Factory function to create a MetaModel.
    
    Args:
        vocab_size: Size of vocabulary
        hidden_size: Hidden dimension
        intermediate_size: FFN intermediate dimension
        num_hidden_layers: Number of transformer layers
        num_attention_heads: Number of attention heads
        suffix_len: Number of suffix layers for meta-learning
        prime: Whether to use prime FFN in suffix layers
        **kwargs: Additional ModelConfig parameters
        
    Returns:
        Initialized MetaModel
    """
    model_config = ModelConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        suffix_len=suffix_len,
        prime=prime,
        **kwargs,
    )
    
    config = Config(model=model_config)
    
    return MetaModel(config)


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Example: Create a small model and do a forward pass
    print("Creating MetaModel...")
    
    model = create_metamodel(
        vocab_size=32000,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=4,
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create dummy batch
    seq_len = 128
    batch = Batch(
        input_ids=torch.randint(0, 32000, (seq_len,)),
        target_tokens=torch.randint(0, 32000, (seq_len,)),
        loss_masks=torch.ones(seq_len),
    )
    
    # Forward pass
    print("Running forward pass...")
    with torch.no_grad():
        loss, metrics = model(batch)
    
    print(f"Loss: {loss.item():.4f}")
    print(f"Metrics: {metrics}")
    print("Done!")
