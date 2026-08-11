import jax
import jax.numpy as jnp
import math
import jax.random as jrandom
import jax.lax as lax
import jax.nn as nn
import pallas as pl
import pallas_core as pcore
from jax.experimental import pallas as pl
from jax.experimental.pallas import BlockShape
import jax.interpreters.pallas as pallas
from jax.experimental import shard_map
import jax.sharding as js

# Constants from the canonical AST
DEFAULT_MASK_VALUE = -0.7 * float(jnp.max(jnp.array([1.0], dtype=jnp.float32)))

def workload(queries, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs):
    """Ragged paged attention workload.
    
    Args:
        queries: shape [max_num_batched_tokens, num_q_heads, head_dim]
        kv_pages: shape [num_pages, 2, page_size, head_dim]
        kv_lens: shape [num_seqs]
        page_indices: shape [num_seqs, pages_per_seq]
        cu_q_lens: shape [num_seqs + 1]
        num_seqs: shape [1]
    
    Returns:
        outputs: shape [max_num_batched_tokens, num_q_heads, head_dim]
    """
    # Configuration constants
    head_dim = 128
    max_num_batched_tokens = 4096
    max_num_seqs = 64
    page_size = 16
    pages_per_seq = 256
    
    # Extract shapes
    _, num_combined_kv_heads, _ = kv_pages.shape
    num_kv_heads = num_combined_kv_heads // 2
    
    _, num_q_heads, _ = queries.shape
    num_query_per_kv = num_q_heads // num_kv_heads
    
    max_seqs = max_num_batched_tokens // max_num_seqs
    tokens_per_seq = max_num_batched_tokens // max_seqs
    
    sm_scale = 1.0 / math.sqrt(head_dim)
    mask_value = DEFAULT_MASK_VALUE
    
    outputs = []
    
    for i in range(max_seqs):
        # Get sequence-specific data
        q_start = cu_q_lens[i]
        kv_len = kv_lens[i]
        indices = page_indices[i]
        
        # Slice query for this sequence
        q = jnp.dynamic_slice(
            queries,
            (q_start, 0, 0),
            (tokens_per_seq, num_q_heads, head_dim)
        )
        
        # Extract and reshape K and V from kv_pages
        k = jnp.reshape(
            kv_pages[indices, :, 0, :],
            (kv_len, num_kv_heads, head_dim)
        )
        v = jnp.reshape(
            kv_pages[indices, :, 1, :],
            (kv_len, num_kv_heads, head_dim)
        )
        
        # Repeat K and V for multi-query attention
        k = jnp.repeat(k, num_query_per_kv, axis=1)
        v = jnp.repeat(v, num_query_per_kv, axis=1)
        
        # Compute attention scores
        attn = jnp.einsum('qhd,khd->hqk', q, k, preferred_element_type=jnp.float32)
        attn = attn * sm_scale
        
        # Create masks
        q_span = kv_len - tokens_per_seq + jnp.broadcasted_iota(jnp.int32, attn.shape, 1)
        kv_span = jnp.broadcasted_iota(jnp.int32, attn.shape, 2)
        mask = (q_span < kv_span) | (kv_span >= kv_len)
        
        # Apply mask
        attn = jnp.where(mask, mask_value, attn)
        
        # Softmax
        attn = nn.softmax(attn, axis=-1).astype(v.dtype)
        
        # Compute output
        out = jnp.einsum('hqk,khd->qhd', attn, v, dtype=queries.dtype)
        
        # Check validity
        is_valid = i < num_seqs[0]
        out = jnp.where(is_valid, out, 0.0)
        
        outputs.append(out)
    
    return jnp.concatenate(outputs, axis=0)
