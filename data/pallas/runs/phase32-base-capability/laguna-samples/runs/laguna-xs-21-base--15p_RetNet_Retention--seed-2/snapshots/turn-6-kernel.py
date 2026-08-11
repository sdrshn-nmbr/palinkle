import jax
import jax.numpy as jnp
import pallas as pl
import pytpu as pltpu

def workload(query, key, value):
    """Multi-scale retention kernel for RetNet.
    
    Retention(X) = (Q K^T ⊙ D) V
    where D[i,j] = γ^(i-j) if i >= j, else 0
    
    Each head has a different decay rate γ_h, creating multi-scale representation.
    """
    B, H, S, D = query.shape
    
    # Compute gammas for each head: exp2(1.0 - 5.0 * h / (H - 1))
    # This gives different decay rates per head
    gammas = jnp.exp2(1.0 - 5.0 * jnp.arange(H, dtype=jnp.float32) / (H - 1))
    
    # Compute positions
    positions = jnp.arange(S, dtype=jnp.float32)
    
    # Compute distance matrix: positions[i] - positions[j]
    # distance[s, t] = positions[s] - positions[t]
    distance = positions[:, None] - positions[None, :]
    
    # Causal mask: 1 if distance >= 0, else 0
    causal_mask = (distance >= 0).astype(jnp.float32)
    
    # Log gamma for each head
    log_gamma = jnp.log(gammas)
    
    # Decay: exp(log_gamma[h] * distance) * causal_mask
    # Shape: (S, S, H) -> need to broadcast properly
    # decay[s, t, h] = exp(log_gamma[h] * distance[s, t]) * causal_mask[s, t]
    decay = jnp.exp(log_gamma[None, None, :] * distance[:, :, None]) * causal_mask[:, :, None]
    
    # Compute QK: einsum("bhsd,bhtd->bhst", query, key)
    qk = jnp.einsum("bhsd,bhtd->bhst", query.astype(jnp.float32), key.astype(jnp.float32))
    
    # Apply decay: qk * decay
    # decay is (S, S, H), qk is (B, H, S, T)
    # Need to expand decay to (1, 1, S, S, H) for broadcasting
    qk_decayed = qk[:, :, :, None, :] * decay[None, None, :, :, :]
    
    # Sum over T dimension (axis=2): abs(qk * decay).sum(axis=-1)
    # This gives shape (B, H, S)
    retention_sum = jnp.sum(jnp.abs(qk_decayed), axis=-2)
    
    # Normalize: maximum(retention_sum, 1.0)
    retention_sum = jnp.maximum(retention_sum, 1.0)
    
    # Final QK normalization: qk / retention_sum
    qk_normalized = qk / retention_sum[:, :, None, :]
    
    # Compute output: einsum("bhst,bhtd->bhsd", qk_normalized, value)
    output = jnp.einsum("bhst,bhtd->bhsd", qk_normalized.astype(query.dtype), value)
    
    return output


def _multi_scale_retention_kernel(query_ref, key_ref, value_ref, gammas_ref, positions_ref, 
                                   distance_ref, causal_mask_ref, log_gamma_ref, decay_ref,
                                   qk_ref, retention_sum_ref, output_ref):
    """Pallas kernel for multi-scale retention."""
    B, H, S, D = query_ref.shape
    
    # Compute gammas: exp2(1.0 - 5.0 * arange(H) / (H - 1))
    h_idx = pl.program_id(0)
    gamma_val = jnp.exp2(1.0 - 5.0 * h_idx / (H - 1)) if H > 1 else 1.0
    gammas_ref[h_idx] = gamma_val
    
    # Compute positions
    s_idx = pl.program_id(1)
    positions_ref[s_idx] = s_idx


def workload_pallas(query, key, value):
    """Multi-scale retention using Pallas kernel."""
    B, H, S, D = query.shape
    
    # Define the kernel
    def retention_kernel(query_ref, key_ref, value_ref, out_ref):
        # Get shapes
        B, H, S, D = query_ref.shape
        
        # Compute gammas for each head
        # gamma_h = exp2(1.0 - 5.0 * h / (H - 1))
        h_idx = pl.program_id(0)
        gammas = jnp.exp2(1.0 - 5.0 * jnp.arange(H, dtype=jnp.float32) / (H - 1))
        
        # Compute positions
        positions = jnp.arange(S, dtype=jnp.float32)
        
        # Compute distance matrix
        distance = positions[:, None] - positions[None, :]
        
        # Causal mask
        causal_mask = (distance >= 0).astype(jnp.float32)
        
        # Log gamma
        log_gamma = jnp.log(gammas)
        
        # Decay tensor
        decay = jnp.exp(log_gamma[None, None, :] * distance[:, :, None]) * causal_mask[:, :, None]
        
        # Compute QK
        qk = jnp.einsum("bhsd,bhtd->bhst", 
                        query_ref.astype(jnp.float32), 
                        key_ref.astype(jnp.float32))
        
        # Apply decay
        qk_decayed = qk[:, :, :, None, :] * decay[None, None, :, :, :]
        
        # Sum and normalize
        retention_sum = jnp.sum(jnp.abs(qk_decayed), axis=-1)
        retention_sum = jnp.maximum(retention_sum, 1.0)
        
        qk_normalized = qk / retention_sum[:, :, None, :]
        
        # Compute output
        out_ref[...] = jnp.einsum("bhst,bhtd->bhsd", 
                                  qk_normalized.astype(query_ref.dtype), 
                                  value_ref.astype(query_ref.dtype))
    
    # Call the kernel
    return pl.pallas_call(
        retention_kernel,
        out_shape=jax.ShapeDtypeStruct((B, H, S, D), query.dtype),
        grid=(H, S),
        in_specs=(
            pl.BlockSpec((1, 1, S, D), lambda b, h, s, d: (b, h, s, d)),
            pl.BlockSpec((1, 1, S, D), lambda b, h, s, d: (b, h, s, d)),
            pl.BlockSpec((1, 1, S, D), lambda b, h, s, d: (b, h, s, d)),
        ),
        out_specs=pl.BlockSpec((B, H, S, D), lambda _: (0,) * 4),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "parallel", "parallel")
        ),
    )(query, key, value)
