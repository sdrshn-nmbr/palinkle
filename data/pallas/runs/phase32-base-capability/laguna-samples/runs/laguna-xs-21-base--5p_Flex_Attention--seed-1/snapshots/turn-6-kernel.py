import jax
import jax.numpy as jnp
import jax.pallas as pl
import jax.pallas as plp
from jax import lax
import jax.random as jrandom

# Constants from the config
BATCH = 4
NUM_HEADS = 64
SEQ_LEN = 4096
HEAD_DIM = 128

def _flex_attention_kernel(
    q_ref: pl.Reref,
    k_ref: pl.Reref,
    v_ref: pl.Reref,
    rel_pos_bias_ref: pl.Reref,
    out_ref: pl.Wref,
):
    """Pallas kernel for flex attention."""
    # Get indices
    b = pl.program_id(0)  # batch index
    h = pl.program_id(1)  # head index
    q_idx = pl.program_id(2)  # query position index
    
    # Block sizes
    block_q = q_ref.shape[2]  # sequence dimension block size
    block_k = k_ref.shape[2]
    block_v = v_ref.shape[2]
    
    # Compute sm_scale = 1/sqrt(head_dim)
    sm_scale = (HEAD_DIM ** -0.5).astype(jnp.float32)
    
    # For each query position in the block, compute attention
    for i in range(block_q):
        q_pos = q_idx * block_q + i
        if q_pos >= SEQ_LEN:
            break
            
        # Load query vector
        q_vec = q_ref[b, h, q_pos, :]  # shape (head_dim,)
        
        # Compute attention scores for all keys
        scores = []
        for j in range(block_k):
            k_pos = j
            if k_pos >= SEQ_LEN:
                break
            
            # Load key vector
            k_vec = k_ref[b, h, k_pos, :]  # shape (head_dim,)
            
            # Compute dot product and scale
            score = jnp.dot(q_vec, k_vec) * sm_scale
            
            # Add relative position bias
            score = score + rel_pos_bias_ref[h, q_pos, k_pos]
            
            # Apply causal mask
            # score = jnp.where(q_pos >= k_pos, score, -1e30)
            score = score if q_pos >= k_pos else -1e30
            
            scores.append(score)
        
        # Stack scores and compute softmax
        scores = jnp.stack(scores)
        # scores = jnp.where(causal_mask, scores, -1e30)
        
        # Softmax along key dimension
        attn_weights = jax.nn.softmax(scores)
        
        # Compute weighted sum of values
        out_val = 0.0
        for j in range(len(attn_weights)):
            v_pos = j
            if v_pos >= SEQ_LEN:
                break
            v_vec = v_ref[b, h, v_pos, :]
            out_val = out_val + attn_weights[j] * v_vec
        
        # Store output
        out_ref[b, h, q_pos, :] = out_val

def workload(q, k, v, rel_pos_bias):
    """Flex attention workload for TPU Pallas."""
    # Grid dimensions: batch, heads, query positions
    grid = (BATCH, NUM_HEADS, SEQ_LEN // 64)  # Use 64 as block size for sequence
    
    # Define block specs
    block_size = 64
    
    def index_map(batch_idx, head_idx, seq_idx):
        return (batch_idx, head_idx, seq_idx * block_size, slice(None))
    
    def bias_index_map(batch_idx, head_idx, seq_idx):
        return (head_idx, seq_idx * block_size, slice(None))
    
    # Compute output shape
    out_shape = jax.ShapeDtypeStruct(q.shape, q.dtype)
    
    # Run the pallas kernel
    result = pl.pallas_call(
        _flex_attention_kernel,
        out_shape=out_shape,
        grid=grid,
        in_specs=(
            pl.BlockSpec((BATCH, NUM_HEADS, 1, HEAD_DIM), lambda b, h, _, __: (b, h, 0, 0)),
            pl.BlockSpec((BATCH, NUM_HEADS, 1, HEAD_DIM), lambda b, h, _, __: (b, h, 0, 0)),
            pl.BlockSpec((BATCH, NUM_HEADS, 1, HEAD_DIM), lambda b, h, _, __: (b, h, 0, 0)),
            pl.BlockSpec((NUM_HEADS, SEQ_LEN, SEQ_LEN), lambda h, _, __: (h, 0, 0)),
        ),
        out_specs=pl.BlockSpec((BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM), lambda b, h, s, _: (b, h, s, 0)),
        compiler_params={"tile_q": 64, "tile_k": 64},
    )(q, k, v, rel_pos_bias)
    
    return result
