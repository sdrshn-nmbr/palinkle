import jax
import jax.numpy as jnp
import jax.pallas as pl
import jax.pallas as plp
import jax.interpreters.tpu as pltpu

def flash_attention_kernel(
    query_ref,
    key_ref,
    value_ref,
    out_ref,
    *,
    num_heads: int,
    head_dim: int,
    seq_len: int,
    block_size_q: int,
    block_size_kv: int,
):
    """Flash attention kernel for causal multi-head attention."""
    # Get program IDs for batch and head
    b = pl.program_id(0)
    h = pl.program_id(1)
    
    # Get the query position for this block
    q_block_idx = pl.program_id(2)
    
    # Calculate the starting position for this query block
    q_start = q_block_idx * block_size_q
    q_end = min(q_start + block_size_q, seq_len)
    
    # For causal attention, we need to attend to all positions <= q_end
    # We'll process key/value blocks in chunks
    
    # Initialize accumulator for softmax (in float32 for numerical stability)
    # Shape: [block_size_q, head_dim]
    acc_shape = (q_end - q_start, head_dim)
    acc = jnp.zeros(acc_shape, dtype=jnp.float32)
    
    # Initialize max_score and sum_exp for stable softmax
    max_score = jnp.full((q_end - q_start,), -jnp.inf, dtype=jnp.float32)
    sum_exp = jnp.zeros((q_end - q_start,), dtype=jnp.float32)
    
    # Scale factor
    scale = 1.0 / jnp.sqrt(head_dim)
    
    # Process all key/value positions (causal: k <= q)
    for kv_block_idx in range(0, seq_len, block_size_kv):
        kv_start = kv_block_idx * block_size_kv
        kv_end = min(kv_start + block_size_kv, seq_len)
        
        # For causal attention, only consider kv positions <= q positions
        # We need to handle the causal mask
        
        # Load query block: [block_q, head_dim]
        q_block = query_ref[b, h, q_start:q_end, :]
        
        # Load key block: [block_kv, head_dim]
        k_block = key_ref[b, h, kv_start:kv_end, :]
        
        # Compute attention scores: [block_q, block_kv]
        # scores = q @ k.T * scale
        scores = jnp.dot(q_block, k_block.T) * scale
        
        # Apply causal mask: only allow attention to positions kv < q
        # Create mask for this block
        q_positions = jnp.arange(q_start, q_end)[:, None]  # [block_q, 1]
        kv_positions = jnp.arange(kv_start, kv_end)[None, :]  # [1, block_kv]
        
        # Causal mask: kv_positions <= q_positions
        causal_mask = kv_positions <= q_positions  # [block_q, block_kv]
        
        # Apply mask by setting invalid positions to -inf
        scores = jnp.where(causal_mask, scores, -1e9)
        
        # Update max_score and sum_exp for stable softmax
        new_max = jnp.maximum(max_score[:, None], scores)
        # Handle the case where max_score was -inf (first iteration)
        new_max = jnp.where(jnp.isneginf(max_score), scores, new_max)
        
        # Compute exp(scores - new_max) for numerical stability
        exp_scores = jnp.exp(scores - new_max)
        
        # Update sum_exp
        sum_exp = sum_exp[:, None] + exp_scores
        sum_exp = jnp.squeeze(sum_exp, axis=-1)
        
        # Update max_score
        max_score = new_max[:, 0] if new_max.shape[-1] == 1 else new_max.max(axis=-1)
        max_score = jnp.where(jnp.isneginf(max_score), new_max.max(axis=-1), max_score)
        
        # Load value block: [block_kv, head_dim]
        v_block = value_ref[b, h, kv_start:kv_end, :]
        
        # Update accumulator: acc += exp(scores) @ v
        acc = acc + jnp.dot(exp_scores, v_block)
    
    # Finalize softmax: divide by sum_exp
    acc = acc / sum_exp[:, None]
    
    # Write output
    out_ref[b, h, q_start:q_end, :] = acc.astype(out_ref.dtype)


def workload(query, key, value):
    """Compute causal multi-head attention."""
    B, H, S, D = query.shape
    
    # Block sizes for TPU efficiency
    # Use multiples of 8 for bf16 on TPU
    block_size_q = 64
    block_size_kv = 64
    
    # Calculate grid dimensions
    # Grid: [batch, heads, num_q_blocks]
    num_q_blocks = (S + block_size_q - 1) // block_size_q
    
    def kernel_ref(
        query_ref,
        key_ref,
        value_ref,
        out_ref,
    ):
        flash_attention_kernel(
            query_ref,
            key_ref,
            value_ref,
            out_ref,
            num_heads=H,
            head_dim=D,
            seq_len=S,
            block_size_q=block_size_q,
            block_size_kv=block_size_kv,
        )
    
    return pl.pallas_call(
        kernel_ref,
        out_shape=jax.ShapeDtypeStruct(query.shape, query.dtype),
        grid=(B, H, num_q_blocks),
        in_specs=(
            pl.BlockSpec((1, 1, block_size_q, D), lambda b, h, q_idx, s: (b, h, q_idx * block_size_q, 0)),
            pl.BlockSpec((1, 1, block_size_kv, D), lambda b, h, q_idx, s: (b, h, 0, 0)),
            pl.BlockSpec((1, 1, block_size_kv, D), lambda b, h, q_idx, s: (b, h, 0, 0)),
        ),
        out_specs=pl.BlockSpec((1, 1, block_size_q, D), lambda b, h, q_idx, s: (b, h, q_idx * block_size_q, 0)),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "parallel")
        ),
    )(query, key, value)
