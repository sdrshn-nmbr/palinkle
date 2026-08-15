import jax
import jax.numpy as jnp
import jax.pallas as pl
import jax.pallas.tpu as pltpu

# Configuration from the problem
BATCH = 4
NUM_QUERY_HEADS = 64
NUM_KV_HEADS = 8
SEQ_LEN = 4096
HEAD_DIM = 128
NUM_Q_PER_KV = NUM_QUERY_HEADS // NUM_KV_HEADS  # 8

# Tile sizes for TPU
TILE_HEADS = 64  # Process all heads in one tile
TILE_SEQ = 64    # Tile size for sequence positions
TILE_HEAD_DIM = 128  # Full head dimension (vectorized)


def attention_kernel(
    q_ref,  # [NUM_QUERY_HEADS, SEQ_LEN, HEAD_DIM]
    k_ref,  # [NUM_QUERY_HEADS, SEQ_LEN, HEAD_DIM] (repeated)
    v_ref,  # [NUM_QUERY_HEADS, SEQ_LEN, HEAD_DIM] (repeated)
    out_ref,  # [NUM_QUERY_HEADS, SEQ_LEN, HEAD_DIM]
):
    """Pallas kernel for attention computation."""
    h_tile = pl.program_id(0)
    q_tile = pl.program_id(1)
    
    h_start = h_tile * TILE_HEADS
    q_start = q_tile * TILE_SEQ
    
    # Determine actual sizes for this tile
    h_size = min(TILE_HEADS, NUM_QUERY_HEADS - h_start)
    q_size = min(TILE_SEQ, SEQ_LEN - q_start)
    
    # Initialize output tile with zeros
    out_ref[...] = jnp.zeros((h_size, q_size, HEAD_DIM), dtype=jnp.bfloat16)
    
    # For each query position in this tile
    for q_idx in range(q_size):
        q_pos = q_start + q_idx
        
        # For each head in this tile
        for h_idx in range(h_size):
            h = h_start + h_idx
            
            # Compute attention scores for all key positions (causal)
            # attn[q_pos, :] = q[h, q_pos, :] @ k[h, :, :].T
            # But only for key positions <= q_pos (causal mask)
            
            # Initialize attention scores
            attn_scores = jnp.zeros(SEQ_LEN, dtype=jnp.float32)
            
            # Compute dot products with all key positions
            for k_pos in range(SEQ_LEN):
                if k_pos <= q_pos:  # Causal mask
                    # Compute dot product
                    score = 0.0
                    for d in range(HEAD_DIM):
                        score += float(q_ref[h, q_pos, d]) * float(k_ref[h, k_pos, d])
                    attn_scores = attn_scores.at[k_pos].set(score)
                else:
                    attn_scores = attn_scores.at[k_pos].set(-1e30)
            
            # Apply softmax
            # Shift for numerical stability
            max_score = jnp.max(jnp.where(jnp.arange(SEQ_LEN) <= q_pos, attn_scores, -jnp.inf))
            shifted_scores = jnp.where(jnp.arange(SEQ_LEN) <= q_pos, attn_scores - max_score, 0.0)
            exp_scores = jnp.exp(shifted_scores)
            sum_exp = jnp.sum(exp_scores)
            attn_probs = exp_scores / sum_exp
            
            # Compute weighted sum of values
            for d in range(HEAD_DIM):
                weighted_sum = 0.0
                for k_pos in range(SEQ_LEN):
                    if k_pos <= q_pos:
                        weighted_sum += attn_probs[k_pos] * float(v_ref[h, k_pos, d])
                out_ref[h_idx, q_idx, d] = jnp.bfloat16(weighted_sum)


def workload(q, k, v):
    """Sparse attention workload with GQA and causal mask."""
    # Step 1: Compute num_q_per_kv
    num_q_per_kv = NUM_QUERY_HEADS // NUM_KV_HEADS  # 8
    
    # Step 2: Repeat k and v along axis 0
    k_repeated = jnp.repeat(k, num_q_per_kv, axis=0)  # [64, 4096, 128]
    v_repeated = jnp.repeat(v, num_q_per_kv, axis=0)  # [64, 4096, 128]
    
    # Step 3: Compute attention scores
    attn = jnp.einsum('hqd,hkd->hqk', q, k_repeated)  # [64, 4096, 4096]
    
    # Step 4: Create causal mask
    causal = jnp.tril(jnp.ones((SEQ_LEN, SEQ_LEN), dtype=jnp.bool_))
    
    # Step 5: Apply causal mask (mask out future positions)
    attn = jnp.where(causal, attn, -1e30)
    
    # Step 6: Apply softmax
    attn = jax.nn.softmax(attn, axis=-1)
    
    # Step 7: Compute output
    out = jnp.einsum('hqk,hkd->hqd', attn, v_repeated)  # [64, 4096, 128]
    
    return out
