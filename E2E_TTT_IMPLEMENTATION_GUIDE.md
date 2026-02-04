# End-to-End Test-Time Training (E2E-TTT) Implementation Guide

This document provides a comprehensive overview of how the E2E-TTT algorithm is implemented in this codebase, answering key questions about the architecture and design choices.

## Table of Contents
1. [Overall Algorithm Implementation](#1-overall-algorithm-implementation)
2. [Forward Pass Implementation](#2-forward-pass-implementation)
3. [Inner Loop Backpropagation](#3-inner-loop-backpropagation)
4. [Multinode Sharding Support](#4-multinode-sharding-support)
5. [Learning Rate and Scheduler Control](#5-learning-rate-and-scheduler-control)
6. [Stability Improvements](#6-stability-improvements)

---

## 1. Overall Algorithm Implementation

### High-Level Overview
The E2E-TTT algorithm formulates long-context language modeling as a continual learning problem. The key ideas are:

- **Test-Time Training**: The model continues learning at test time via next-token prediction on the given context, compressing the context into its weights
- **Meta-Learning**: The model's initialization is optimized for better test-time learning via meta-learning at training time
- **End-to-End**: Both training and test time use next-token prediction (unlike previous TTT methods)

### Key Files
- **`ttt/train.py`**: Main training loop orchestration
- **`ttt/model/transformer.py`**: Core model implementation with MetaModel class (lines 528+)
- **`ttt/infra/loop.py`**: Training step implementation

### Architecture Details
The model uses a standard **Transformer with sliding-window attention (SWA)** rather than specialized long-context architectures. The innovation is in the training procedure, not the architecture itself.

---

## 2. Forward Pass Implementation

### Primary Location
**File**: `ttt/model/transformer.py`

### Key Methods

#### 2.1 MetaModel.loss_for_sequence() (lines 648-746)
This is the main forward pass for meta-learning:
```python
def loss_for_sequence(
    self,
    seq: Int[Array, "n_blocks ttt_blocksize"],
    suffix_state: SuffixState,
    inner_steps: int,
) -> tuple[Array, SuffixState, LossDict]:
```

**Key features:**
- Splits the model into **prefix blocks** (frozen during inner loop) and **suffix blocks** (updated during inner loop)
- Computes prefix outputs once for efficiency
- Performs inner loop optimization on suffix blocks
- Returns loss, updated state, and detailed loss metrics

#### 2.2 CausalLM.__call__() (lines 843-850)
Standard autoregressive forward pass:
```python
def __call__(
    self, input_ids: Int[Array, "n_seq"], *, key: PRNGKeyArray | None = None
) -> Float[Array, "n_seq vocab_size"]:
    x = self.embedding(input_ids)
    x = self.transformer(x, key=key)
    logits = self.lm_head(x)
    return logits
```

Flow: `embedding → transformer → lm_head`

#### 2.3 TransformerModel.__call__() (lines 505-525)
Processes embeddings through transformer blocks:
- Applies dropout if enabled
- Iterates through block collection (with optional rematerialization)
- Applies final layer normalization

---

## 3. Inner Loop Backpropagation

### Answer: JAX Autograd (NOT Manual Formulas)

The implementation uses **JAX's automatic differentiation** system, not manually derived gradient formulas.

### Evidence

**File**: `ttt/model/transformer.py`

#### Inner Loop Step (lines 594-630)
```python
def inner_loop_step(
    self,
    seq: Int[Array, "context"],
    prefix_outputs: PrefixState,
    suffix_state: SuffixState,
    step: int,
) -> tuple[SuffixState, LossDict]:
    # ... setup code ...
    
    # Use JAX autograd to compute gradients
    value_and_grad_fn = eqx.filter_value_and_grad(MetaModel.lm_loss, has_aux=True)
    (_loss_with_aux, (md[M.loss], md[M.token_nll_loss], new_suffix_state)), grads = value_and_grad_fn(
        self, seq, suffix_state, prefix_outputs=prefix_outputs
    )
    
    # Extract gradients for inner parameters
    inner_grads = grads.inner_parameters()
    
    # Apply optimizer update
    updates, opt_state = self.inner_opt.update(inner_grads, opt_state, inner_params)
    inner_params = eqx.apply_updates(inner_params, updates)
```

**Key observations:**
1. Uses `eqx.filter_value_and_grad()` from Equinox library (JAX wrapper)
2. Automatically differentiates through `MetaModel.lm_loss()`
3. No manual gradient computation code anywhere in the repository
4. Leverages JAX's efficient automatic differentiation

### Why Autograd?
- **Flexibility**: Easy to modify model architecture without rewriting gradients
- **Correctness**: Eliminates manual gradient bugs
- **Performance**: JAX's XLA compilation optimizes gradient computation
- **Maintainability**: Simpler codebase

---

## 4. Multinode Sharding Support

### Primary Implementation
**File**: `ttt/model/sharding.py`

### Key Components

#### 4.1 ModelSharding Class (lines 21-92)
```python
class ModelSharding:
    def __init__(self, cfg: Config):
        # Create JAX mesh with data and state axes
        self.mesh = jax.make_mesh(
            axis_shapes=(n_data_parallel, cfg.training.n_state_parallel),
            axis_names=("data", "state")
        )
```

**Two parallelism axes:**
- **`data`**: Data parallelism across devices
- **`state`**: Model/state parallelism (shards model parameters)

#### 4.2 shard_params() Method (lines 40-92)
Applies sharding constraints to model parameters:
```python
def shard_params(self, params: P) -> P:
    """Apply NamedSharding to model parameters"""
    # Uses jax.tree_util.tree_map with self.sharding_fn
    # Applies P("state") or None based on parameter type
```

**Sharding strategy:**
- Embedding parameters: Sharded along vocabulary dimension
- Attention parameters: Sharded based on configuration
- MLP parameters: Sharded appropriately
- Layer norms: Typically replicated (not sharded)

#### 4.3 Data Parallelism
**File**: `ttt/train.py` (line 128)
```python
data_sharding = jax.NamedSharding(mesh, P("data"))
```

**File**: `ttt/infra/loop.py` (line 133)
```python
@eqx.filter_vmap(in_axes=(None, 0, 0), out_axes=(0, 0))
def step_fn(model, batch, state):
    # Vectorized/parallelized step function
```

Uses `filter_vmap` (equivalent to `pmap`) to distribute computation across data parallel devices.

### NCCL Communication
From README.md:
- Requires **NCCL 2.26.2** for efficient multi-GPU communication
- Handles gradient synchronization across devices
- Supports multinode training via Slurm/Submitit launcher

### Configuration
**File**: `configs/deploy/submitit.yaml`
```yaml
hydra.launcher.nodes=4  # Number of nodes
```

---

## 5. Learning Rate and Scheduler Control

### Implementation Files
- **`ttt/optimizers.py`**: Optimizer and scheduler definitions
- **`ttt/model/transformer.py`**: Inner learning rate warmup
- **`ttt/config.py`**: Configuration parameters

### 5.1 Outer Loop Optimizer (lines 10-42)

**File**: `ttt/optimizers.py`

```python
def make_optimizer_outer(config: Config) -> optax.GradientTransformation:
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.training.optimizer_outer.lr,
        warmup_steps=config.training.warmup_steps,
        decay_steps=config.training.total_steps,
        end_value=config.training.optimizer_outer.lr * config.training.lr_decay_fraction,
    )
    
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.clip_gradient),
        optax.adamw(
            learning_rate=lr_schedule,
            b1=config.training.optimizer_outer.beta1,
            b2=config.training.optimizer_outer.beta2,
            weight_decay=config.training.optimizer_outer.weight_decay,
        ),
    )
```

**Schedule**: Warmup → Cosine decay with configurable end value

### 5.2 Inner Loop Optimizer (lines 45-55)

```python
def make_optimizer_inner(config: Config) -> optax.GradientTransformation:
    return optax.chain(
        optax.clip_by_global_norm(config.clip_gradient),
        optax.sgd(learning_rate=config.training.optimizer_inner.lr),
    )
```

**Simpler approach**: SGD with gradient clipping (no complex scheduling needed for inner loop)

### 5.3 Inner Learning Rate (ILR) Warmup

**File**: `ttt/model/transformer.py` (lines 565-574)

```python
def get_ilr_multiplier(self, step):
    """Compute inner learning rate warmup multiplier"""
    progress = jnp.minimum(
        1.0, 
        1.0 * (step + 1) / self.config.training.ilr_warmup_steps
    )
    ilr = (
        self.config.training.ilr_init 
        + (self.config.training.optimizer_inner.lr - self.config.training.ilr_init) * progress
    )
    return ilr / self.config.training.optimizer_inner.lr
```

**Purpose**: Gradually increase inner loop learning rate from `ilr_init` to full `lr` over `ilr_warmup_steps`

### Configuration Parameters

**File**: `ttt/config.py`

```python
@dataclass
class Training:
    # Outer loop
    optimizer_outer: OptimizerConfig
    warmup_steps: int = 1000
    lr_decay_fraction: float = 0.1
    
    # Inner loop
    optimizer_inner: OptimizerConfig
    ilr_warmup_steps: int = 10000
    ilr_init: float = 0.0
    
    # Gradient control
    clip_gradient: float = 1.0
```

---

## 6. Stability Improvements

The codebase implements multiple techniques to ensure stable training:

### 6.1 Gradient Clipping

**File**: `ttt/optimizers.py` (lines 31, 49)
```python
optax.clip_by_global_norm(config.clip_gradient)
```

Applied to both outer and inner loop optimizers to prevent gradient explosion.

### 6.2 RMS Normalization on Query/Key

**File**: `ttt/config.py` (line 115)
```python
qk_norm: bool = True  # Apply RMS norm to queries and keys
```

Stabilizes attention computation, especially important for long sequences.

### 6.3 Gradient Checkpointing (Rematerialization)

**File**: `ttt/config.py` (lines 94-102)
```python
@dataclass
class Model:
    block_remat: RematerializationConfig = field(default_factory=RematerializationConfig)
    attn_remat: RematerializationConfig = field(default_factory=RematerializationConfig)
    mlp_remat: RematerializationConfig = field(default_factory=RematerializationConfig)
    inner_remat_freq: int = 1  # Frequency of rematerialization in inner loop
```

**Levels of control:**
- Block-level rematerialization
- Attention-level rematerialization  
- MLP-level rematerialization
- Inner loop rematerialization frequency

**Trade-off**: Memory vs. computation time

### 6.4 Mixed Precision Training

**File**: `ttt/config.py` (lines 73-75)
```python
compute_dtype: DType = jnp.bfloat16  # Computation precision
param_dtype: DType = jnp.float32     # Parameter storage precision
state_dtype: DType = jnp.float32     # Optimizer state precision
```

**Strategy:**
- Compute in bf16 for speed and memory
- Store parameters in fp32 for precision
- Keep optimizer state in fp32 for accuracy

### 6.5 Safe Global Norm Computation

**File**: `ttt/infra/loop.py` (line 171)
```python
def global_norm_safe(tree):
    """Compute global norm with protection against zero gradients"""
    leaves, _ = jax.tree_util.tree_flatten(tree)
    squared_sum = sum(jnp.sum(jnp.square(x)) for x in leaves)
    return jnp.sqrt(squared_sum + 1e-10)  # Add epsilon for stability
```

Prevents NaN from zero gradients.

### 6.6 Gradient Accumulation

**File**: `ttt/infra/loop.py` (line 164)
```python
# Use Welford's online algorithm for stable mean computation
grad_mean = welfords_online_mean(grads, accum_steps)
```

**Benefits:**
- Effectively larger batch sizes
- More stable gradient estimates
- Reduced memory requirements

### 6.7 Inner Learning Rate Warmup

As described in section 5.3, gradually warming up the inner learning rate prevents destabilization at the start of meta-training.

### 6.8 Attention Normalization

Besides `qk_norm`, the model uses:
- Layer normalization before attention
- Layer normalization before MLP
- Final layer normalization

**File**: `ttt/model/attention.py` and `ttt/model/transformer.py`

---

## Summary

This E2E-TTT implementation demonstrates sophisticated engineering:

1. **Algorithm**: Meta-learns good initializations for test-time continual learning
2. **Forward Pass**: Efficient prefix/suffix split in `transformer.py`
3. **Backpropagation**: Leverages JAX autograd for flexibility and correctness
4. **Sharding**: Two-axis (data + state) parallelism for multinode scaling
5. **Learning Rate**: Warmup-cosine decay (outer) + ILR warmup (inner)
6. **Stability**: Gradient clipping, mixed precision, rematerialization, qk_norm, and more

The codebase prioritizes:
- **Correctness**: Autograd over manual gradients
- **Scalability**: Robust sharding infrastructure
- **Stability**: Multiple techniques for reliable training
- **Flexibility**: Configuration-driven design with Hydra

---

## References

### Key Files to Explore
1. `ttt/model/transformer.py` - Core model and meta-learning logic
2. `ttt/train.py` - Training orchestration
3. `ttt/infra/loop.py` - Training step implementation
4. `ttt/model/sharding.py` - Distributed training setup
5. `ttt/optimizers.py` - Optimizer and scheduler definitions
6. `ttt/config.py` - All configuration parameters

### External Documentation
- [JAX Documentation](https://jax.readthedocs.io/)
- [Equinox Library](https://docs.kidger.site/equinox/)
- [Optax Optimizers](https://optax.readthedocs.io/)
- [E2E-TTT Paper](https://test-time-training.github.io/e2e.pdf)
