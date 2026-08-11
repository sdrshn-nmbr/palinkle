import jax
import jax.numpy as jnp
import pallas as pl
import pytensor as pltpu

def flash_attention_kernel(
    query_ref,
    key_ref,
    value_ref,
    mask_ref,
    output_ref,
    scale_ref,
):
    """Flash attention kernel for TPU."""
    # Get dimensions from grid
    b = pl.program_id(0)  # batch
    h = pl.program_id(1)  # head
    q_idx = pl.program_id(2)  # query block index
    
    # Block sizes
    block_q = query_ref.shape[2]  # sequence dimension for this block
    block_k = key_ref.shape[2]
    d = query_ref.shape[3]  # head dimension
    
    # For simplicity, we'll compute attention in a tiled manner
    # This is a simplified implementation
    
    # Read query and key blocks
    q_block = query_ref[b, h, q_idx, :]  # [seq_block, head_dim]
    k_block = key_ref[b, h, :, :]  # [seq, head_dim]
    v_block = value_ref[b, h, :, :]  # [seq, head_dim]
    
    # Compute QK^T
    attn = jnp.dot(q_block, k_block.T)  # [seq_q, seq_k]
    
    # Apply causal mask (lower triangular)
    # Create mask for this block
    mask = jnp.tril(jnp.ones((attn.shape[0], attn.shape[1]))))
    attn = jnp.where(mask, attn, -1e9)
    
    # Softmax
    attn = jax.nn.softmax(attn, axis=-1)
    
    # Compute output
    out_block = jnp.dot(attn, v_block)
    
    # Write output
    output_ref[b, h, q_idx, :] = out_block


def workload(query, key, value):
    """Flash attention workload."""
    # Get shapes
    b, h, s, d = query.shape
    
    # Compute scale factor
    scale = 1.0 / jnp.sqrt(d)
    
    # Create causal mask (lower triangular)
    mask = jnp.tril(jnp.ones((s, s)))
    
    # Compute attention scores
    # QK^T / sqrt(d)
    attn_scores = jnp.einsum('bhed,bhkd->bhqk', query, key) * scale
    
    # Apply causal mask
    attn_scores = jnp.where(mask, attn_scores, -1e9)
    
    # Softmax along key dimension
    attn_weights = jax.nn.softmax(attn_scores, axis=-1)
    
    # Compute output
    output = jnp.einsum('bhqk,bhkd->bhed', attn_weights, value)
    
    return output
