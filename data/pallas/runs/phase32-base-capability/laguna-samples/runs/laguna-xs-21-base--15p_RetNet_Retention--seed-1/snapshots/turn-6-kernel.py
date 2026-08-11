import jax
import jax.numpy as jnp
import pallas as pl
import pytpu

def _multi_scale_retention_kernel(query_ref, key_ref, value_ref, out_ref):
    """Pallas kernel for multi-scale retention."""
    # Get shapes
    B, H, S, D = query_ref.shape
    
    # Compute gammas for each head: exp2(1.0 - 5.0 * h / H)
    # This creates different decay rates per head
    gammas = jnp.exp2(1.0 - 5.0 * jnp.arange(H, dtype=jnp.float32) / H)
    
    # Compute positions
    positions = jnp.arange(S, dtype=jnp.float32)
    
    # Compute distance matrix: positions[i] - positions[j]
    distance = positions[:, None] - positions[None, :]
    
    # Causal mask: 1 where distance >= 0, else 0
    causal_mask = jnp.where(distance >= 0, 1.0, 0.0)
    
    # Log gamma for numerical stability
    log_gamma = jnp.log(gammas)
    
    # Compute decay: exp(log_gamma * max(distance, 0)) * causal_mask
    # This gives exponential decay based on position difference
    decay = jnp.exp(log_gamma[:, None] * jnp.maximum(distance, 0.0)) * causal_mask
    
    # Convert to float32 for computation
    query_f32 = query_ref.astype(jnp.float32)
    key_f32 = key_ref.astype(jnp.float32)
    
    # Compute QK^T: einsum("bhsd,bhtd->bhst", query, key)
    qk = jnp.einsum("bhsd,bhtd->bhst", query_f32, key_f32)
    
    # Apply decay: qk * decay
    # decay has shape (S, S) for each head
    # We need to apply it per batch and head
    qk_decayed = qk * decay[None, :, :, None]  # broadcast decay
    
    # Compute retention sum for normalization
    retention_sum = jnp.maximum(jnp.abs(qk_decayed), 1.0).sum(axis=2, keepdims=True)
    
    # Normalize
    qk_normalized = qk_decayed / retention_sum
    
    # Compute output: einsum("bhst,bhtd->bhsd", qk_normalized, value)
    value_f32 = value_ref.astype(jnp.float32)
    output = jnp.einsum("bhst,bhtd->bhsd", qk_normalized, value_f32)
    
    # Store result
    out_ref[:] = output.astype(out_ref.dtype)


def workload(query, key, value):
    """Multi-scale retention workload."""
    # Define block size for Pallas
    block_size = 128
    
    # Get shapes
    B, H, S, D = query.shape
    
    # Create output shape
    out_shape = (B, H, S, D)
    
    # Define grid based on sequence length
    grid = (S // block_size,)
    
    # Call Pallas kernel
    return pl.pallas_call(
        _multi_scale_retention_kernel,
        out_shape=jax.ShapeDtypeStruct(out_shape, query.dtype),
        grid=grid,
        in_specs=(
            pl.BlockSpec((1, 1, S, D), lambda _: (0, 0, 0, 0)),
            pl.BlockSpec((1, 1, S, D), lambda _: (0, 0, 0, 0)),
            pl.BlockSpec((1, 1, S, D), lambda _: (0, 0, 0, 0)),
        ),
        out_specs=pl.BlockSpec((1, 1, S, D), lambda _: (0, 0, 0, 0)),
        compiler_params=pytpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "sequential", "parallel")
        ),
    )(query, key, value)
