import jax
import jax.numpy as jnp
import jax.pallas as pl
import jax.pallas as plp
from jax import lax
from jax.pallas.lib import SlabAllocator
import functools

# Configuration
CONFIG = {
    "batch": 4,
    "head_dim": 128,
    "model": "Llama-3.1-70B",
    "name": "llama3_70b_flex_attention",
    "num_heads": 64,
    "operator": "flex_attention",
    "seq_len": 4096
}

def workload(q, k, v, rel_pos_bias):
    """Flex attention workload with causal mask and relative position bias."""
    batch = CONFIG["batch"]
    num_heads = CONFIG["num_heads"]
    seq_len = CONFIG["seq_len"]
    head_dim = CONFIG["head_dim"]
    
    # Scale factor
    sm_scale = head_dim ** -0.5
    
    # Block size for tiling
    BLOCK_Q = 64
    BLOCK_K = 64
    
    def attention_kernel(
        q_ref, k_ref, v_ref, rel_pos_bias_ref, out_ref,
        m0, m1, m2, m3,
        *args
    ):
        """Pallas kernel for attention computation."""
        # Get indices
        b = m0
        h = m1
        q_start = m2 * BLOCK_Q
        q_end = min(q_start + BLOCK_Q, seq_len)
        k_start = m3 * BLOCK_K
        k_end = min(k_start + BLOCK_K, seq_len)
        
        # Load q slice: [BLOCK_Q, head_dim]
        q_block = q_ref[b, h, q_start:q_end, :]
        
        # Initialize output block
        out_block = jnp.zeros((q_end - q_start, head_dim), dtype=jnp.float32)
        
        # Compute attention for this q block
        for k_idx in range(k_start, k_end, BLOCK_K):
            k_block_end = min(k_idx + BLOCK_K, seq_len)
            
            # Load k and v blocks
            k_block = k_ref[b, h, k_idx:k_block_end, :]
            v_block = v_ref[b, h, k_idx:k_block_end, :]
            
            # Load relative position bias for this block
            # rel_pos_bias is [num_heads, seq_len, seq_len]
            bias_block = rel_pos_bias_ref[h, q_start:q_end, k_idx:k_block_end]
            
            # Compute attention scores: [BLOCK_Q, BLOCK_K]
            # q @ k.T * sm_scale
            attn_scores = jnp.dot(q_block, k_block.T) * sm_scale
            
            # Add relative position bias
            attn_scores = attn_scores + bias_block
            
            # Apply causal mask: only allow attention to positions <= current q position
            # For each q position, we need to mask out k positions > q position
            q_local = jnp.arange(q_end - q_start) + q_start
            k_local = jnp.arange(k_block_end - k_idx) + k_idx
            
            # Create causal mask: True where k <= q
            causal_mask = k_local[None, :] <= q_local[:, None]
            
            # Apply mask: set large negative value for masked positions
            attn_scores = jnp.where(causal_mask, attn_scores, -1e30)
            
            # Softmax along k dimension (axis 1)
            attn_probs = jax.nn.softmax(attn_scores, axis=1)
            
            # Compute output: attn @ v
            out_block = out_block + jnp.dot(attn_probs, v_block)
        
        # Write output
        out_ref[b, h, q_start:q_end, :] = out_block.astype(jnp.bfloat16)
    
    # Grid dimensions
    grid = (batch, num_heads, seq_len // BLOCK_Q, seq_len // BLOCK_K)
    
    # Define block specs
    q_spec = pl.BlockSpec(
        (batch, num_heads, BLOCK_Q, head_dim),
        lambda b, h, q_idx, k_idx: (b, h, q_idx * BLOCK_Q, slice(None))
    )
    k_spec = pl.BlockSpec(
        (batch, num_heads, BLOCK_K, head_dim),
        lambda b, h, q_idx, k_idx: (b, h, k_idx * BLOCK_K, slice(None))
    )
    v_spec = pl.BlockSpec(
        (batch, num_heads, BLOCK_K, head_dim),
        lambda b, h, q_idx, k_idx: (b, h, k_idx * BLOCK_K, slice(None))
    )
    rel_pos_bias_spec = pl.BlockSpec(
        (num_heads, BLOCK_Q, BLOCK_K),
        lambda b, h, q_idx, k_idx: (h, q_idx * BLOCK_Q, k_idx * BLOCK_K)
    )
    out_spec = pl.BlockSpec(
        (batch, num_heads, BLOCK_Q, head_dim),
        lambda b, h, q_idx, k_idx: (b, h, q_idx * BLOCK_Q, slice(None))
    )
    
    return pl.pallas_call(
        attention_kernel,
        out_shape=jax.ShapeDtypeStruct((batch, num_heads, seq_len, head_dim), jnp.bfloat16),
        grid=grid,
        in_specs=(q_spec, k_spec, v_spec, rel_pos_bias_spec),
        out_specs=out_spec,
        compiler_params=plp.CompilerParams(
            dimension_semantics=("parallel", "parallel", "parallel", "parallel")
        ),
    )(q, k, v, rel_pos_bias)
