import jax
import jax.numpy as jnp
import jax.pallas as pl
import jax.pallas as plp
import jax.interpreters.pallas as pallas
from jax.interpreters import pallas as pl
from jax.pallas.lib import SlabAllocator
import jax.numpy as jnp

# Configuration from the spec
CONFIG = {
    "batch": 4,
    "head_dim": 128,
    "model": "Llama-3.1-70B",
    "name": "llama3_70b_sparse_attention",
    "num_kv_heads": 8,
    "num_query_heads": 64,
    "operator": "sparse_attention",
    "seq_len": 4096
}

def attention_kernel(
    q_ref,
    k_ref,
    v_ref,
    out_ref,
    *,
    seq_len,
    num_query_heads,
    num_kv_heads,
    head_dim,
):
    """Pallas kernel for GQA attention with causal mask."""
    # Get program IDs
    q_head_idx = pl.program_id(0)  # Query head index
    seq_idx = pl.program_id(1)     # Sequence position index
    
    # Block sizes
    BLOCK_SEQ = 64  # Tile size for sequence dimension
    BLOCK_HD = 128  # Head dimension (full)
    
    # Calculate tile bounds
    seq_start = seq_idx * BLOCK_SEQ
    seq_end = min(seq_start + BLOCK_SEQ, seq_len)
    
    # Number of query heads per KV head
    num_q_per_kv = num_query_heads // num_kv_heads
    
    # Get the KV head index (map query head to KV head)
    kv_head_idx = q_head_idx // num_q_per_kv
    
    # Load q for this query head and sequence tile
    q_block = q_ref[q_head_idx, seq_start:seq_end, :]
    
    # Load k and v for the corresponding KV head
    k_block = k_ref[kv_head_idx, :, :]  # Full K for this KV head
    v_block = v_ref[kv_head_idx, :, :]  # Full V for this KV head
    
    # Compute attention scores: q @ k^T
    # q_block: [BLOCK_SEQ, head_dim]
    # k_block: [seq_len, head_dim]
    # attn_scores: [BLOCK_SEQ, seq_len]
    
    # We need to compute attention for all positions in the tile
    # and apply causal mask
    
    # For each position in the query tile, compute attention over all keys
    for i in range(seq_end - seq_start):
        q_pos = q_block[i, :]  # [head_dim]
        
        # Compute dot product with all keys
        # attn_scores[i, j] = q_pos @ k[j]
        attn_scores = jnp.zeros(seq_len, dtype=jnp.float32)
        
        for j in range(seq_len):
            k_pos = k_block[j, :]  # [head_dim]
            # Compute dot product in float32 for numerical stability
            score = jnp.dot(q_pos.astype(jnp.float32), k_pos.astype(jnp.float32))
            
            # Apply causal mask: only allow attention to positions <= current position
            # Global position in sequence
            global_pos = seq_start + i
            
            # Causal mask: can only attend to positions <= current position
            if j <= global_pos:
                attn_scores = attn_scores.at[j].set(score)
            else:
                attn_scores = attn_scores.at[j].set(-1e30)
        
        # Apply softmax
        attn_probs = jax.nn.softmax(attn_scores)
        
        # Compute weighted sum of values
        out_val = jnp.zeros(head_dim, dtype=jnp.float32)
        for j in range(seq_len):
            v_pos = v_block[j, :].astype(jnp.float32)
            out_val = out_val + attn_probs[j] * v_pos
        
        # Store output
        out_ref[q_head_idx, seq_start + i, :] = out_val.astype(q_ref.dtype)


def workload(q, k, v):
    """Sparse attention workload with GQA and causal mask."""
    # Get configuration values
    S = CONFIG["seq_len"]
    H_q = CONFIG["num_query_heads"]
    H_kv = CONFIG["num_kv_heads"]
    head_dim = CONFIG["head_dim"]
    
    # Number of query heads per KV head
    num_q_per_kv = H_q // H_kv
    
    # Repeat k and v along axis 0 by num_q_per_kv
    k_repeated = jnp.repeat(k, num_q_per_kv, axis=0)
    v_repeated = jnp.repeat(v, num_q_per_kv, axis=0)
    
    # Compute attention scores: q @ k^T
    # q: [H_q, S, head_dim], k: [H_q, S, head_dim]
    # attn: [H_q, S, S]
    attn = jnp.einsum('hqd,hkd->hqk', q, k_repeated)
    
    # Create causal mask
    causal = jnp.tril(jnp.ones((S, S), dtype=jnp.bool_))
    
    # Apply causal mask (set masked positions to large negative value)
    attn = jnp.where(causal, attn, -1e30)
    
    # Apply softmax along the last axis
    attn = jax.nn.softmax(attn, axis=-1)
    
    # Compute output: attn @ v
    # attn: [H_q, S, S], v: [H_q, S, head_dim]
    # out: [H_q, S, head_dim]
    out = jnp.einsum('hqk,hkd->hqd', attn, v_repeated)
    
    return out
